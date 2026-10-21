from datetime import datetime
from pathlib import Path

import duckdb

from talk_align import idempotency
from talk_align.cli import main

VOLATILE = idempotency.VOLATILE


def write_table(out: Path, rows: list[tuple], table: str = "audio_segments") -> None:
    """rows: (segment_id, text, run_id, code_version, processed_at minute)."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t (segment_id VARCHAR, text VARCHAR, run_id VARCHAR, provenance "
        "STRUCT(model VARCHAR, code_version VARCHAR, run_id VARCHAR, processed_at TIMESTAMP))"
    )
    for seg, text, run_id, code, minute in rows:
        con.execute(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            [
                seg,
                text,
                run_id,
                {
                    "model": "small",
                    "code_version": code,
                    "run_id": run_id,
                    "processed_at": datetime(2026, 10, 22, 9, minute),
                },
            ],
        )
    con.execute(f"COPY t TO '{out / (table + '.parquet')}' (FORMAT parquet)")
    con.close()


def test_volatile_columns_are_ignored(tmp_path):
    # same rows, other run_id / processed_at (top level and inside provenance), other order
    write_table(
        tmp_path / "a",
        [("s1", "hi", "run-1", "abc", 0), ("s2", "yo", "run-1", "abc", 0)],
    )
    write_table(
        tmp_path / "b",
        [("s2", "yo", "run-2", "abc", 5), ("s1", "hi", "run-2", "abc", 5)],
    )
    result = idempotency.compare_dirs(tmp_path / "a", tmp_path / "b")
    assert idempotency.is_identical(result)
    assert result["audio_segments"]["differing_columns"] == {}


def test_changed_text_is_found_in_both_directions(tmp_path):
    write_table(tmp_path / "a", [("s1", "Delta Lake", "r1", "abc", 0)])
    write_table(tmp_path / "b", [("s1", "Deita Lake", "r2", "abc", 0)])
    r = idempotency.compare_dirs(tmp_path / "a", tmp_path / "b")["audio_segments"]
    assert (r["only_in_1"], r["only_in_2"]) == (1, 1)
    assert r["differing_columns"] == {"text": 2}


def test_extra_row_in_dir2_needs_the_second_direction(tmp_path):
    # dir1 EXCEPT dir2 = 0 rows: a one-way check would call this idempotent
    write_table(tmp_path / "a", [("s1", "hi", "r1", "abc", 0)])
    write_table(
        tmp_path / "b", [("s1", "hi", "r2", "abc", 0), ("s9", "zombie", "r2", "abc", 0)]
    )
    r = idempotency.compare_dirs(tmp_path / "a", tmp_path / "b")["audio_segments"]
    assert (r["only_in_1"], r["only_in_2"]) == (0, 1)


def test_duplicate_row_is_caught_by_except_all(tmp_path):
    # plain EXCEPT removes duplicates and would see no difference; EXCEPT ALL counts them
    write_table(tmp_path / "a", [("s1", "hi", "r1", "abc", 0)])
    write_table(
        tmp_path / "b", [("s1", "hi", "r2", "abc", 0), ("s1", "hi", "r2", "abc", 0)]
    )
    r = idempotency.compare_dirs(tmp_path / "a", tmp_path / "b")["audio_segments"]
    assert (r["rows_1"], r["rows_2"], r["only_in_2"]) == (1, 2, 1)


def test_other_code_version_differs_unless_ignored(tmp_path):
    write_table(tmp_path / "a", [("s1", "hi", "r1", "6e03ab6", 0)])
    write_table(tmp_path / "b", [("s1", "hi", "r2", "a335a90", 0)])
    r = idempotency.compare_dirs(tmp_path / "a", tmp_path / "b")
    assert r["audio_segments"]["differing_columns"] == {"provenance.code_version": 2}
    ignored = idempotency.compare_dirs(
        tmp_path / "a", tmp_path / "b", VOLATILE + ("code_version",)
    )
    assert idempotency.is_identical(ignored)


def test_missing_table_is_not_idempotent(tmp_path):
    write_table(tmp_path / "a", [("s1", "hi", "r1", "abc", 0)])
    write_table(tmp_path / "a", [("s1", "hi", "r1", "abc", 0)], table="slides")
    write_table(tmp_path / "b", [("s1", "hi", "r2", "abc", 0)])
    result = idempotency.compare_dirs(tmp_path / "a", tmp_path / "b")
    assert result["slides"] == {"missing_in": "dir2"}
    assert not idempotency.is_identical(result)


def test_cli_exit_codes(tmp_path, capsys):
    write_table(tmp_path / "a", [("s1", "hi", "r1", "abc", 0)])
    write_table(tmp_path / "b", [("s1", "hi", "r2", "abc", 0)])
    write_table(tmp_path / "c", [("s1", "bye", "r3", "abc", 0)])
    assert main(["check-idempotent", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    assert "IDEMPOTENT" in capsys.readouterr().out
    assert main(["check-idempotent", str(tmp_path / "a"), str(tmp_path / "c")]) == 1
    assert "NOT IDEMPOTENT" in capsys.readouterr().out
