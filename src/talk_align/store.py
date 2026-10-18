"""DuckDB storage: schema, write strategies and Parquet export (spec §4). Roadmap #1."""
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