import json
from pathlib import Path

import duckdb
import pytest

from talk_align import store
from talk_align.audio_lane import build_rows
from tests.test_audio_lane import PROVENANCE, make_seg

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_lakehouse.json"


def test_schema_has_the_ten_audio_columns():
    con = store.connect(":memory:")
    cols = {row[0] for row in con.execute("DESCRIBE audio_segments").fetchall()}
    # sample segment
    seg = {
        "start": 1.0,
        "end": 2.0,
        "text": " hi",
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
    }
    ok_rows, _dlq, _metrics = build_rows("x", [seg], 600.0, PROVENANCE)
    con.close()
    assert cols == set(ok_rows[0])


def test_insert_writes_all_rows():
    fixture = json.loads(FIXTURE.read_text())
    con = store.connect(":memory:")
    ok_rows, _dlq, _metrics = build_rows("lakehouse", fixture["segments"], fixture["duration"], PROVENANCE)
    store.insert_rows(con, "audio_segments", ok_rows)
    count = con.execute("SELECT count(*) FROM audio_segments").fetchone()[0]
    con.close()
    assert count == len(ok_rows)


def test_insert_twice_fails():
    fixture = json.loads(FIXTURE.read_text())
    con = store.connect(":memory:")
    ok_rows, _dlq, _metrics = build_rows("lakehouse", fixture["segments"], fixture["duration"], PROVENANCE)
    store.insert_rows(con, "audio_segments", ok_rows)
    with pytest.raises(duckdb.ConstraintException):
        store.insert_rows(con, "audio_segments", ok_rows)


# Simulator rows
def rows_at(video_id, starts):
    segs = [
        {
            "start": s,
            "end": s + 2.0,
            "text": " hi",
            "avg_logprob": -0.2,
            "no_speech_prob": 0.01,
        }
        for s in starts
    ]
    ok_rows, _dlq, _metrics = build_rows(video_id, segs, 600.0, PROVENANCE)
    return ok_rows


def ids(con, video_id):
    return [
        r[0]
        for r in con.execute(
            "SELECT segment_id FROM audio_segments WHERE video_id = ? ORDER BY 1",
            [video_id],
        ).fetchall()
    ]


def test_rerun_same_rows_does_not_fail():
    conn = store.connect(":memory:")
    store.overwrite_video(
        conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0])
    )
    store.overwrite_video(
        conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0])
    )
    assert ids(conn, "talk01") == ["talk01_a_00001000", "talk01_a_00004000"]


def test_rerun_removes_zombies_and_keeps_other_videos():
    conn = store.connect(":memory:")
    store.overwrite_video(
        conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0, 9.0])
    )  # "model small"
    store.overwrite_video(
        conn, "audio_segments", "talk02", rows_at("talk02", [1.0])
    )  # video khác
    store.overwrite_video(
        conn, "audio_segments", "talk01", rows_at("talk01", [4.0, 6.5])
    )  # "model base"
    assert ids(conn, "talk01") == [
        "talk01_a_00004000",
        "talk01_a_00006500",
    ]  # chỉ còn dòng của lần chạy mới
    assert ids(conn, "talk02") == ["talk02_a_00001000"]  # vẫn còn nguyên


def test_failed_insert_keeps_old_rows():
    conn = store.connect(":memory:")
    store.overwrite_video(
        conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0, 9.0])
    )
    broken = rows_at(
        "talk01", [2.0, 2.0]
    )  # 2 dòng TRÙNG id → INSERT sẽ lỗi SAU KHI DELETE đã chạy
    with pytest.raises(duckdb.ConstraintException):
        store.overwrite_video(conn, "audio_segments", "talk01", broken)
    assert ids(conn, "talk01") == [
        "talk01_a_00001000",
        "talk01_a_00004000",
        "talk01_a_00009000",
    ]  # 3 dòng cũ phải còn nguyên


def test_export_writes_one_sorted_file(
    tmp_path,
):  # tmp_path: pytest tự cấp 1 thư mục tạm, tự xóa sau
    con = store.connect(":memory:")
    store.overwrite_video(
        con, "audio_segments", "talk01", rows_at("talk01", [9.0, 1.0, 4.0])
    )
    store.overwrite_video(con, "audio_segments", "talk02", rows_at("talk02", [1.0]))
    path = store.export_parquet(con, "talk01", tmp_path)
    exported = duckdb.sql(f"SELECT segment_id FROM '{path}'").fetchall()
    assert exported == [
        ("talk01_a_00001000",),
        ("talk01_a_00004000",),
        ("talk01_a_00009000",),
    ]  # chỉ talk01, và đã SẮP XẾP theo segment_id


def test_dlq_rows_are_written():
    con = store.connect(":memory:")
    ok, dlq, _ = build_rows(
        "v",
        [make_seg(), make_seg(start=5.0, text="", no_speech_prob=0.1)],
        600.0,
        PROVENANCE,
    )
    store.overwrite_video_tables(con, "v", {"audio_segments": ok, "dlq_audio": dlq})
    assert con.execute("SELECT count(*) FROM audio_segments").fetchone()[0] == 1
    assert con.execute("SELECT reason_code FROM dlq_audio").fetchall() == [("EMPTY_TEXT",)]
    con.close()


def test_rerun_without_errors_clears_old_dlq_rows():
    con = store.connect(":memory:")
    _, dlq, _ = build_rows("v", [make_seg(text="", no_speech_prob=0.1)], 600.0, PROVENANCE)
    store.overwrite_video_tables(con, "v", {"audio_segments": [], "dlq_audio": dlq})  # lần 1: có lỗi
    ok, _, _ = build_rows("v", [make_seg()], 600.0, PROVENANCE)
    store.overwrite_video_tables(con, "v", {"audio_segments": ok, "dlq_audio": []})  # lần 2: hết lỗi
    assert con.execute("SELECT count(*) FROM dlq_audio").fetchone()[0] == 0
    con.close()


def test_failure_in_dlq_rolls_back_audio_too():
    con = store.connect(":memory:")
    old_ok, _, _ = build_rows("v", [make_seg(start=1.0)], 600.0, PROVENANCE)
    store.overwrite_video_tables(con, "v", {"audio_segments": old_ok, "dlq_audio": []})
    new_ok, dlq, _ = build_rows(
        "v",
        [make_seg(start=9.0), make_seg(start=5.0, text="", no_speech_prob=0.1)],
        600.0,
        PROVENANCE,
    )
    with pytest.raises(duckdb.ConstraintException):
        store.overwrite_video_tables(
            con, "v", {"audio_segments": new_ok, "dlq_audio": dlq + dlq}  # dlq trùng id → lỗi
        )
    ids = [r[0] for r in con.execute("SELECT segment_id FROM audio_segments").fetchall()]
    assert ids == ["v_a_00001000"]  # audio quay về dòng CŨ
    con.close()
