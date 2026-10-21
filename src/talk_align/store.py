"""DuckDB storage: schema, write strategies and Parquet export (spec §4). Roadmap #1."""

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
                conn.commit()
    except Exception:
        conn.rollback()
        raise


def overwrite_video(
    conn: duckdb.DuckDBPyConnection, table: str, video_id: str, rows: list[dict]
) -> None:
    tables = {table: rows}
    overwrite_video_tables(conn, video_id, tables)


def export_parquet(
    conn: duckdb.DuckDBPyConnection, video_id: str, out_dir: Path | None = None
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
