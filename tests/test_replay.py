import json
from datetime import datetime
from pathlib import Path

from talk_align import audio_lane, frame_lane, pipeline, replay, store
from talk_align.audio_lane import build_rows
from talk_align.config import MAX_REPLAY_ATTEMPTS, FrameParams
from talk_align.frame_lane import build_slide_rows, plan_keyframes
from talk_align.report import warnings
from tests.test_audio_lane import PROVENANCE, make_seg
from tests.test_frame_lane import tb

# provenance of the replay itself: other run, later time
REPLAY_PROV = {
    **PROVENANCE,
    "run_id": "replay-1",
    "processed_at": datetime(2026, 10, 23, 9, 0),
}


def audio_dlq(con, segs, duration=344.561):
    """Write what the audio lane would write for these segments; return the DLQ rows."""
    ok, dlq, _ = build_rows("v", segs, duration, PROVENANCE)
    store.overwrite_video_tables(con, "v", {"audio_segments": ok, "dlq_audio": dlq})
    return store.open_dlq_rows(con, "dlq_audio", "v", MAX_REPLAY_ATTEMPTS)


def replay_audio_once(con, duration):
    rows = store.open_dlq_rows(con, "dlq_audio", "v", MAX_REPLAY_ATTEMPTS)
    fixed, updated, m = replay.replay_audio_rows(rows, "v", duration, REPLAY_PROV)
    store.replay_write(con, "audio_segments", fixed, "dlq_audio", updated)
    return m


def dlq_state(con, table="dlq_audio"):
    return con.execute(f"SELECT reason_code, attempts, status FROM {table}").fetchall()


def test_row_fixed_by_the_right_duration_goes_to_the_main_table():
    # first run had a wrong duration (300 s): t_end 346 looked out of bounds
    con = store.connect(":memory:")
    rows = audio_dlq(con, [make_seg(start=340.0, end=346.0)], duration=300.0)
    assert [r["reason_code"] for r in rows] == ["TIME_OUT_OF_BOUNDS"]

    m = replay_audio_once(con, duration=400.0)

    assert (m["fixed"], m["still_failing"]) == (1, 0)
    ids = [
        r[0] for r in con.execute("SELECT segment_id FROM audio_segments").fetchall()
    ]
    assert ids == ["v_a_00340000"]
    assert dlq_state(con) == [("TIME_OUT_OF_BOUNDS", 1, "replayed")]
    run_id = con.execute("SELECT provenance.run_id FROM audio_segments").fetchone()[0]
    assert run_id == "replay-1"  # the fixed row says which code made it


def test_replaying_twice_changes_nothing_the_second_time():
    con = store.connect(":memory:")
    audio_dlq(con, [make_seg(start=340.0, end=346.0)], duration=300.0)
    replay_audio_once(con, duration=400.0)
    before = con.execute("SELECT * FROM audio_segments").fetchall(), dlq_state(con)

    m = replay_audio_once(con, duration=400.0)  # nothing open any more

    assert m["replayed_in"] == 0
    assert (
        con.execute("SELECT * FROM audio_segments").fetchall(),
        dlq_state(con),
    ) == before


def test_still_failing_row_counts_attempts_then_needs_a_human():
    con = store.connect(":memory:")
    audio_dlq(con, [make_seg(start=50.0, text="", no_speech_prob=0.1)])  # EMPTY_TEXT
    for attempt in range(1, MAX_REPLAY_ATTEMPTS + 1):
        m = replay_audio_once(con, duration=344.561)
        assert m["still_failing"] == 1
        assert dlq_state(con) == [("EMPTY_TEXT", attempt, "open")]

    assert (
        replay_audio_once(con, duration=344.561)["replayed_in"] == 0
    )  # not picked again
    assert store.dlq_needs_human(con, "v", MAX_REPLAY_ATTEMPTS) == 1
    assert store.dlq_reason_counts(con, "v") == {"dlq_audio/EMPTY_TEXT": 1}


def test_row_that_is_silence_under_new_rules_is_ignored(monkeypatch):
    con = store.connect(":memory:")
    audio_dlq(con, [make_seg(start=50.0, text="", no_speech_prob=0.1)])
    monkeypatch.setattr(audio_lane, "SILENCE_THRESHOLD", 0.05)  # rule changed since

    m = replay_audio_once(con, duration=344.561)

    assert m["ignored"] == 1
    assert dlq_state(con) == [("EMPTY_TEXT", 1, "ignored")]
    assert con.execute("SELECT count(*) FROM audio_segments").fetchone()[0] == 0
    assert store.dlq_reason_counts(con, "v") == {}  # ignored rows are not problems


def test_replay_never_deletes_other_rows_of_the_video():
    con = store.connect(":memory:")
    audio_dlq(
        con, [make_seg(start=1.0), make_seg(start=340.0, end=346.0)], duration=300.0
    )
    replay_audio_once(con, duration=400.0)
    ids = [
        r[0]
        for r in con.execute(
            "SELECT segment_id FROM audio_segments ORDER BY 1"
        ).fetchall()
    ]
    assert ids == ["v_a_00001000", "v_a_00340000"]  # upsert, not DELETE by video_id


def frame_dlq(con, tmp_path):
    """One slide whose keyframe could not be read: what the frame lane writes."""
    planned = plan_keyframes("v", [{"t_start": 20.0, "t_end": 40.0}])
    planned[0]["keyframe_path"] = tmp_path / "v_s_00020000.png"
    raw = {
        "v_s_00020000": {
            "error": "UnidentifiedImageError: x",
            "reason_code": "CORRUPT_KEYFRAME",
            "stage": "ocr",
        }
    }
    ok, dlq, _ = build_slide_rows("v", planned, raw, FrameParams(), PROVENANCE)
    store.overwrite_video_tables(con, "v", {"slides": ok, "dlq_frame": dlq})
    return store.open_dlq_rows(con, "dlq_frame", "v", MAX_REPLAY_ATTEMPTS)


def test_frame_dlq_payload_keeps_the_slide_span(tmp_path):
    con = store.connect(":memory:")
    [row] = frame_dlq(con, tmp_path)
    payload = json.loads(row["raw_payload"])
    assert payload["slide"]["t_start"] == 20.0 and payload["slide"]["t_end"] == 40.0
    assert payload["slide"]["keyframe_at"] == 30.0
    assert payload["ocr"]["reason_code"] == "CORRUPT_KEYFRAME"


def test_corrupt_keyframe_is_cut_again_and_fixed(monkeypatch, tmp_path):
    con = store.connect(":memory:")
    rows = frame_dlq(con, tmp_path)
    cut = []
    monkeypatch.setattr(
        frame_lane,
        "extract_keyframe",
        lambda video, slide: cut.append(slide["keyframe_at"]),
    )
    monkeypatch.setattr(
        frame_lane,
        "ocr_keyframes",
        lambda planned: {
            planned[0]["slide_id"]: {
                "image_h": 720,
                "boxes": [tb("1980s Businesses", 0.97)],
            }
        },
    )

    fixed, updated, m = replay.replay_frame_rows(
        rows, "v", Path("v.mp4"), FrameParams(), REPLAY_PROV
    )
    store.replay_write(con, "slides", fixed, "dlq_frame", updated)

    assert cut == [30.0]  # same time as the first try
    assert m["fixed"] == 1
    row = con.execute("SELECT slide_id, t_start, t_end, text FROM slides").fetchone()
    assert row == ("v_s_00020000", 20.0, 40.0, "1980s Businesses")
    assert dlq_state(con, "dlq_frame") == [("CORRUPT_KEYFRAME", 1, "replayed")]


def test_replay_refuses_another_video(monkeypatch, tmp_path):
    db = tmp_path / "t.duckdb"
    con = store.connect(db)
    audio_dlq(
        con, [make_seg(start=50.0, text="", no_speech_prob=0.1)]
    )  # source_hash "sha256:fixture"
    con.close()
    other = tmp_path / "v.mp4"
    other.write_bytes(b"another video")
    monkeypatch.setattr(pipeline, "git_code_version", lambda: "test")
    try:
        pipeline.run_replay(other, "audio", db_path=db)
        raise AssertionError("expected SourceMismatchError")
    except pipeline.SourceMismatchError as e:
        assert "source_hash differs" in str(e)


def test_report_warns_when_a_row_needs_a_human():
    run = {"status": "success", "code_version": "abc", "metrics": {"summary": {}}}
    assert any("needs a human" in w for w in warnings(run, {}, needs_human=2))
    assert warnings(run, {}, needs_human=0) == []
