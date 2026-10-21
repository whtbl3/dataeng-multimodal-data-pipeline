"""DuckDB storage: schema, write strategies and Parquet export (spec §4). Roadmap #1."""

import json
from pathlib import Path

import duckdb


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    provenance_struct = """
    STRUCT(
        model           VARCHAR,
        model_version   VARCHAR,
        params          JSON,
        source_hash     VARCHAR,
        code_version    VARCHAR,
        run_id          VARCHAR,
        processed_at    TIMESTAMP
    ) NOT NULL
    """
    # shared by dlq_audio and dlq_frame: add a DLQ column here, both tables get it
    dlq_columns = f"""
        dlq_id          VARCHAR PRIMARY KEY,
        video_id        VARCHAR NOT NULL,
        record_id       VARCHAR,
        stage           VARCHAR NOT NULL,
        reason_code     VARCHAR NOT NULL,
        error_detail    VARCHAR,
        raw_payload     JSON,
        provenance      {provenance_struct},
        failed_at       TIMESTAMP NOT NULL,
        attempts        INTEGER NOT NULL,
        status          VARCHAR NOT NULL
    """
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS audio_segments (
        video_id        VARCHAR NOT NULL,
        modality        VARCHAR NOT NULL,
        text            VARCHAR NOT NULL,
        quality_flag    VARCHAR NOT NULL,
        segment_id      VARCHAR PRIMARY KEY,
        t_start         DOUBLE NOT NULL,
        t_end           DOUBLE NOT NULL,
        quality_score   DOUBLE NOT NULL,
        raw_payload     JSON NOT NULL,
        provenance      {provenance_struct}
    )
    """)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS slides (
        video_id        VARCHAR NOT NULL,
        slide_id        VARCHAR PRIMARY KEY,
        modality        VARCHAR NOT NULL,
        text            VARCHAR NOT NULL,      -- image-only slide: "" not NULL
        t_start         DOUBLE NOT NULL,
        t_end           DOUBLE NOT NULL,
        quality_score   DOUBLE NOT NULL,
        quality_flag    VARCHAR NOT NULL,
        keyframe_path   VARCHAR NOT NULL,
        raw_payload     JSON NOT NULL,
        provenance      {provenance_struct}
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS aligned (
        video_id            VARCHAR NOT NULL,
        segment_id          VARCHAR PRIMARY KEY,
        slide_id            VARCHAR NOT NULL,
        match_type          VARCHAR NOT NULL,
        overlap_s           DOUBLE  NOT NULL,
        nearest_distance_s  DOUBLE,              -- NULL on overlap rows
        align_gap_s         DOUBLE  NOT NULL,
        align_gap_ratio     DOUBLE  NOT NULL,
        slide_quality_flag  VARCHAR NOT NULL,
        run_id              VARCHAR NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id        VARCHAR PRIMARY KEY,
        video_id      VARCHAR NOT NULL,
        started_at    TIMESTAMP NOT NULL,
        finished_at   TIMESTAMP,              -- NULL while running (or killed)
        status        VARCHAR NOT NULL,       -- running | success | partial | failed
        code_version  VARCHAR NOT NULL,
        params        JSON,
        metrics       JSON
    )
    """)
    conn.execute(f"CREATE TABLE IF NOT EXISTS dlq_audio ({dlq_columns})")
    conn.execute(f"CREATE TABLE IF NOT EXISTS dlq_frame ({dlq_columns})")


def connect(db_path) -> duckdb.DuckDBPyConnection:
    conn: duckdb.DuckDBPyConnection = duckdb.connect(str(db_path))
    create_schema(conn)
    return conn


def insert_rows(
    conn: duckdb.DuckDBPyConnection, table: str, rows: list[dict], verb: str = "INSERT"
) -> None:
    cols = list(rows[0])
    sql = f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    values = [[r[c] for c in cols] for r in rows]
    conn.executemany(sql, values)


def upsert_rows(conn: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> None:
    """Insert, or replace the row with the same primary key. For history tables (runs)."""
    insert_rows(conn, table, rows, verb="INSERT OR REPLACE")


def overwrite_video_tables(
    conn: duckdb.DuckDBPyConnection, video_id: str, tables: dict[str, list[dict]]
) -> None:
    try:
        conn.begin()
        for table, rows in tables.items():
            conn.execute(f"DELETE FROM {table} WHERE video_id=?", [video_id])
            if len(rows) > 0:
                insert_rows(conn, table, rows)
        # once, after every table: commit inside the loop never ran when all lists were
        # empty, so the DELETE was dropped on close and old rows survived
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def overwrite_video(
    conn: duckdb.DuckDBPyConnection, table: str, video_id: str, rows: list[dict]
) -> None:
    tables = {table: rows}
    overwrite_video_tables(conn, video_id, tables)


def latest_run(conn: duckdb.DuckDBPyConnection, video_id: str) -> dict | None:
    """The most recent runs row of a video, params/metrics parsed from JSON; None if none.

    Output: {"run_id": "1801...", "status": "success", "started_at": datetime(...), ...,
             "params": {...}, "metrics": {"audio": {...}, ..., "summary": {...}}}
    """
    rel = conn.execute(
        "SELECT * FROM runs WHERE video_id = ? ORDER BY started_at DESC, run_id DESC LIMIT 1",
        [video_id],
    )
    cols = [d[0] for d in rel.description]
    found = rel.fetchone()
    if found is None:
        return None
    run = dict(zip(cols, found))
    for key in ("params", "metrics"):
        # JSON columns come back as str; a "running" row still has metrics NULL
        run[key] = json.loads(run[key]) if run[key] else {}
    return run


def dlq_reason_counts(conn: duckdb.DuckDBPyConnection, video_id: str) -> dict[str, int]:
    """OPEN DLQ rows of a video per table and reason_code, e.g. {"dlq_audio/EMPTY_TEXT": 2}.

    Reads the tables as they are now. Rows already replayed or ignored are not problems.
    """
    rows = conn.execute(
        """
        SELECT 'dlq_audio/' || reason_code AS k, count(*) FROM dlq_audio
        WHERE video_id = $v AND status = 'open' GROUP BY 1
        UNION ALL
        SELECT 'dlq_frame/' || reason_code, count(*) FROM dlq_frame
        WHERE video_id = $v AND status = 'open' GROUP BY 1
        ORDER BY 1
        """,
        {"v": video_id},
    ).fetchall()
    return dict(rows)


def open_dlq_rows(
    conn: duckdb.DuckDBPyConnection, table: str, video_id: str, max_attempts: int
) -> list[dict]:
    """DLQ rows still worth a replay: status open and fewer than max_attempts tries."""
    rel = conn.execute(
        f"SELECT * FROM {table} WHERE video_id = ? AND status = 'open' "
        "AND attempts < ? ORDER BY dlq_id",
        [video_id, max_attempts],
    )
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, r)) for r in rel.fetchall()]


def dlq_needs_human(
    conn: duckdb.DuckDBPyConnection, video_id: str, max_attempts: int
) -> int:
    """Open DLQ rows (both tables) that failed max_attempts replays: replay stops there."""
    return conn.execute(
        """
        SELECT (SELECT count(*) FROM dlq_audio WHERE video_id = $v AND status = 'open' AND attempts >= $m)
             + (SELECT count(*) FROM dlq_frame WHERE video_id = $v AND status = 'open' AND attempts >= $m)
        """,
        {"v": video_id, "m": max_attempts},
    ).fetchone()[0]


def replay_write(
    conn: duckdb.DuckDBPyConnection,
    main_table: str,
    fixed_rows: list[dict],
    dlq_table: str,
    dlq_rows: list[dict],
) -> None:
    """Fixed rows into the main table and updated DLQ rows back, in ONE transaction.

    Upsert by primary key, never DELETE by video_id: a replay touches only its own rows,
    and running it twice leaves the same tables.
    """
    try:
        conn.begin()
        if fixed_rows:
            upsert_rows(conn, main_table, fixed_rows)
        if dlq_rows:
            upsert_rows(conn, dlq_table, dlq_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def export_parquet(
    conn: duckdb.DuckDBPyConnection, video_id: str, out_dir: Path
) -> dict[str, Path]:
    """Write one Parquet file per table with the rows of `video_id`, sorted by primary key."""
    EXPORT_TABLES = {
        "audio_segments": "segment_id",
        "dlq_audio": "dlq_id",
        "slides": "slide_id",
        "dlq_frame": "dlq_id",
        "aligned": "segment_id",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for table, pk in EXPORT_TABLES.items():
        path = out_dir / f"{table}.parquet"
        conn.execute(
            f"COPY (SELECT * FROM {table} WHERE video_id = ? ORDER BY {pk}) "
            f"TO '{path}' (FORMAT parquet)",
            [video_id],
        )
        written[table] = path
    return written
