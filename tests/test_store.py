import json
from pathlib import Path

import duckdb
import pytest

from talk_align import store
from talk_align.audio_lane import build_rows
from tests.test_audio_lane import PROVENANCE

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_lakehouse.json"

def test_schema_has_the_ten_audio_columns():
    con = store.connect(":memory:")
    cols = {row[0] for row in con.execute("DESCRIBE audio_segments").fetchall()}
    # sample segment
    seg = {"start": 1.0, "end": 2.0, "text": " hi", "avg_logprob": -0.2, "no_speech_prob": 0.01}
    rows = build_rows("x", [seg], PROVENANCE)
    con.close()
    assert cols == set(rows[0])

def test_insert_writes_all_rows():
    fixture = json.loads(FIXTURE.read_text())
    con = store.connect(":memory:")
    rows = build_rows("lakehouse", fixture["segments"], PROVENANCE)
    store.insert_rows(con, "audio_segments", rows)
    count = con.execute("SELECT count(*) FROM audio_segments").fetchone()[0]
    con.close()
    assert count == len(rows)

def test_insert_twice_fails():                     
    fixture = json.loads(FIXTURE.read_text())
    con = store.connect(":memory:")
    rows = build_rows("lakehouse", fixture["segments"], PROVENANCE)
    store.insert_rows(con, "audio_segments", rows)
    with pytest.raises(duckdb.ConstraintException):
        store.insert_rows(con, "audio_segments", rows)
    con.close()

# Simulator rows
def rows_at(video_id, starts):
    segs = [{"start": s, "end": s + 2.0, "text": " hi", "avg_logprob": -0.2, "no_speech_prob": 0.01}
            for s in starts]
    return build_rows(video_id, segs, PROVENANCE)

def ids(con, video_id):
    return [r[0] for r in con.execute(
        "SELECT segment_id FROM audio_segments WHERE video_id = ? ORDER BY 1", [video_id]).fetchall()]

def test_rerun_same_rows_does_not_fail():
    conn = store.connect(":memory:")
    store.overwrite_video(conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0]))
    store.overwrite_video(conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0]))
    assert ids(conn, "talk01") == ["talk01_a_00001000", "talk01_a_00004000"]

def test_rerun_removes_zombies_and_keeps_other_videos():
    conn = store.connect(":memory:")
    store.overwrite_video(conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0, 9.0]))   # "model small"
    store.overwrite_video(conn, "audio_segments", "talk02", rows_at("talk02", [1.0]))             # video khác
    store.overwrite_video(conn, "audio_segments", "talk01", rows_at("talk01", [4.0, 6.5]))        # "model base"
    assert ids(conn, "talk01") == ["talk01_a_00004000", "talk01_a_00006500"]               # chỉ còn dòng của lần chạy mới
    assert ids(conn, "talk02") == ["talk02_a_00001000"]                 # vẫn còn nguyên

def test_failed_insert_keeps_old_rows():
    conn = store.connect(":memory:")
    store.overwrite_video(conn, "audio_segments", "talk01", rows_at("talk01", [1.0, 4.0, 9.0]))
    broken = rows_at("talk01", [2.0, 2.0])             # 2 dòng TRÙNG id → INSERT sẽ lỗi SAU KHI DELETE đã chạy
    with pytest.raises(duckdb.ConstraintException):
        store.overwrite_video(conn, "audio_segments", "talk01", broken)
    assert ids(conn, "talk01") == ["talk01_a_00001000", "talk01_a_00004000", "talk01_a_00009000"]                 # 3 dòng cũ phải còn nguyên