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
    conn.execute(audio_segment_table)


def connect(db_path) -> duckdb.DuckDBPyConnection:
    conn: duckdb.DuckDBPyConnection = duckdb.connect(str(db_path))
    create_schema(conn)
    return conn


def insert_rows(conn: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> None:
    cols = list(rows[0])
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    values = [[r[c] for c in cols] for r in rows]
    conn.executemany(sql, values)


def overwrite_video(
    conn: duckdb.DuckDBPyConnection, table: str, video_id: str, rows: list[dict]
) -> None:
    conn.begin()
    try:
        conn.execute(f"DELETE FROM {table} WHERE video_id = ?", [video_id])
        insert_rows(conn, table, rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def export_parquet(
    conn: duckdb.DuckDBPyConnection, video_id: str, out_dir: Path | None = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "audio_segments.parquet"
    # table = conn.execute(
    #     "SELECT * FROM audio_segments WHERE video_id = ? ORDER BY segment_id", video_id
    # ).to_arrow_table(table, path)
    path = out_dir / "audio_segments.parquet"
    conn.execute(
        f"COPY (SELECT * FROM audio_segments WHERE video_id = ? ORDER BY segment_id) "
        f"TO '{path}' (FORMAT parquet)",
        [video_id],
    )
    return path
