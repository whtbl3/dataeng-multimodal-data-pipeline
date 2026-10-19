"""DuckDB storage: schema, write strategies and Parquet export (spec §4). Roadmap #1."""

from pathlib import Path

import duckdb


def create_schema(conn: duckdb.DuckDBPyConnection):
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
    audio_segment_table = f"""
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
    """
    dlq_audio_table = f"""
    CREATE TABLE IF NOT EXISTS dlq_audio (
        dlq_id          VARCHAR PRIMARY KEY,
        video_id        VARCHAR NOT NULL,
        record_id       VARCHAR,
        stage           VARCHAR NOT NULL,
        reason_code     VARCHAR NOT NULL,
        error_detail    VARCHAR,
        raw_payload     JSON,
        provenance      {provenance_struct},
        failed_at       TIMESTAMP NOT NULL,
        attempts        INTEGER NOT NULL DEFAULT 0,
        status          VARCHAR NOT NULL DEFAULT 'open'
    )
    """
    conn.execute(audio_segment_table)
    conn.execute(dlq_audio_table)


def connect(db_path) -> duckdb.DuckDBPyConnection:
    conn: duckdb.DuckDBPyConnection = duckdb.connect(str(db_path))
    create_schema(conn)
    return conn


def insert_rows(conn: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> None:
    cols = list(rows[0])
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    values = [[r[c] for c in cols] for r in rows]
    conn.executemany(sql, values)


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
