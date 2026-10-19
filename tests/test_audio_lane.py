import json
from datetime import datetime
from pathlib import Path

import pytest

from talk_align.audio_lane import build_rows, quality_flag, quality_score

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_lakehouse.json"
PROVENANCE = {
    "model": "faster-whisper",
    "model_version": "small",
    "params": '{"beam_size": 5}',
    "source_hash": "sha256:fixture",
    "code_version": "test",
    "run_id": "run-test",
    "processed_at": datetime(2026, 10, 17, 12, 0, 0),  # cố định, KHÔNG dùng now()
}


def test_build_rows_runs_on_fixture():
    fixture = json.loads(FIXTURE.read_text())
    ok_rows, _dlq, _metrics = build_rows(fixture["video_id"], fixture["segments"], fixture["duration"], PROVENANCE)
    assert [r["segment_id"] for r in ok_rows] == [
        "lakehouse_a_00000000",
        "lakehouse_a_00005000",
        "lakehouse_a_00013000",
        "lakehouse_a_00020000",
        "lakehouse_a_00027000",
    ]
    assert {r["quality_flag"] for r in ok_rows} == {"ok"}
    assert isinstance(ok_rows[0]["raw_payload"], str)
    assert json.loads(ok_rows[0]["raw_payload"]) == fixture["segments"][0]
    assert fixture["segments"][0]["text"].startswith(" ")
    assert not ok_rows[0]["text"].startswith(" ")
    assert set(ok_rows[0]) == {
        "video_id",
        "segment_id",
        "modality",
        "text",
        "t_start",
        "t_end",
        "quality_score",
        "quality_flag",
        "raw_payload",
        "provenance",
    }


def test_segment_id_uses_round_not_int():
    seg = {
        "start": 16.15,
        "end": 20.0,
        "text": " hi",
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
    }  # thêm 2 key này luôn, STEP 14 sẽ cần
    ok_rows, _dlq_rows, _metrics = build_rows("talk01", [seg], 600.0, PROVENANCE)
    assert ok_rows[0]["segment_id"] == "talk01_a_00016150"


def test_quality_score_formula():
    seg = {"avg_logprob": -0.2, "no_speech_prob": 0.01}
    assert quality_score(seg) == pytest.approx(0.810543445547202, abs=1e-3)


def test_quality_flag_threshold():
    assert quality_flag(0.12) == "quarantine"  # nhạc nền: điểm thấp
    assert quality_flag(0.29) == "quarantine"  # ngay dưới ngưỡng
    assert quality_flag(0.3) == "ok"
    assert quality_flag(0.8) == "ok"


def test_running_twice_gives_the_same_rows():
    fixture = json.loads(FIXTURE.read_text())
    segs = fixture["segments"] + [make_seg(start=50.0, text="", no_speech_prob=0.1)]
    first   = build_rows(fixture["video_id"], segs, fixture["duration"], PROVENANCE)
    second  = build_rows(fixture["video_id"], segs, fixture["duration"], PROVENANCE)
    assert first == second
    assert len(first[1]) == 1

def test_metrics_counts_segments_in():
    fixture = json.loads(FIXTURE.read_text())
    ok_rows, dlq_rows, metrics = build_rows(fixture["video_id"], fixture["segments"], fixture["duration"], PROVENANCE)
    assert metrics["segments_in"] == 5
    assert dlq_rows == []
    assert len(ok_rows) == 5

# Hàm phụ
def make_seg(start=1.0, end=2.0, text=" hi", avg_logprob=-0.2, no_speech_prob=0.01):
    return {"start": start, "end": end, "text": text,
            "avg_logprob": avg_logprob, "no_speech_prob": no_speech_prob}

def test_silence_is_dropped_and_counted():
    ok, dlq, m = build_rows("v", [make_seg(text="", no_speech_prob=0.9)], 600.0, PROVENANCE)
    assert ok == []
    assert dlq == []
    assert m["dropped_silence"] == 1

def test_silence_threshold_is_inclusive():                 # đúng 0.6 → "≥ 0.6" → bỏ
    ok, dlq, m = build_rows("v", [make_seg(text="  ", no_speech_prob=0.6)], 600.0, PROVENANCE)
    assert m["dropped_silence"] == 1

def test_empty_text_with_speech_is_not_silence():          # có người nói mà không ra chữ → KHÔNG phải im lặng
    ok, dlq, m = build_rows("v", [make_seg(text="", no_speech_prob=0.1)], 600.0, PROVENANCE)
    assert m["dropped_silence"] == 0

def test_empty_text_with_speech_goes_to_dlq():
    seg = make_seg(start=50.0, end=52.0, text="", no_speech_prob=0.1)
    ok, dlq, m = build_rows("v", [seg], 600.0, PROVENANCE)
    assert ok == []
    assert len(dlq) == 1
    row = dlq[0]
    assert row["reason_code"] == "EMPTY_TEXT"
    assert row["stage"] == "validate"
    assert row["record_id"] == "v_a_00050000"              # tự tính segment_id từ "v" và 50.0
    assert row["status"] == "open" and row["attempts"] == 0
    assert json.loads(row["raw_payload"]) == seg  # giữ nguyên segment gốc
    assert row["failed_at"] == PROVENANCE["processed_at"]   # không dùng now()

def test_dlq_row_has_the_eleven_columns():
    _, dlq, _ = build_rows("v", [make_seg(text="", no_speech_prob=0.1)], 600.0, PROVENANCE)
    assert set(dlq[0]) == {"dlq_id", "video_id", "record_id", "stage", "reason_code", "error_detail",
                           "raw_payload", "provenance", "failed_at", "attempts", "status"}

def test_end_before_start_goes_to_dlq():
    _, dlq, _ = build_rows("v", [make_seg(start=20.0, end=19.5)], 600.0, PROVENANCE)
    assert dlq[0]["reason_code"] == "INVALID_TIME_RANGE"

def test_zero_length_goes_to_dlq():
    _, dlq, _ = build_rows("v", [make_seg(start=5.0, end=5.0)], 600.0, PROVENANCE)
    assert len(dlq) == 1

def test_small_overrun_is_clipped():
    seg = make_seg(start=340.0, end=345.2)
    ok, dlq, m = build_rows("v", [seg], 344.561, PROVENANCE)
    assert ok[0]["t_end"] == 344.561
    assert m["clipped_t_end"] == 1
    assert json.loads(ok[0]["raw_payload"])["end"] == 345.2      # bản gốc không bị sửa

def test_overrun_at_tolerance_is_still_clipped():
    ok, _, _ = build_rows("v", [make_seg(start=340.0, end=345.561)], 344.561, PROVENANCE)
    assert len(ok) == 1

def test_large_overrun_goes_to_dlq():
    ok, dlq, _ = build_rows("v", [make_seg(start=340.0, end=346.0)], 344.561, PROVENANCE)
    assert ok == [] and dlq[0]["reason_code"] == "TIME_OUT_OF_BOUNDS"

def test_segment_starting_after_video_end_goes_to_dlq():
    ok, dlq, _ = build_rows("v", [make_seg(start=344.8, end=345.0)], 344.561, PROVENANCE)
    assert ok == []
    assert dlq[0]["reason_code"] == "TIME_OUT_OF_BOUNDS"

def test_positive_logprob_gives_score_out_of_range():
    ok, dlq, _ = build_rows("v", [make_seg(avg_logprob=0.5)], 600.0, PROVENANCE)
    assert ok == []
    assert dlq[0]["reason_code"] == "SCORE_OUT_OF_RANGE"

def test_missing_text_goes_to_dlq_without_crashing():
    _, dlq, _ = build_rows("v", [{"start": 3.0, "end": 4.0}], 600.0, PROVENANCE)
    assert dlq[0]["reason_code"] == "SCHEMA_VIOLATION"
    assert dlq[0]["record_id"] == "v_a_00003000"              # có start → vẫn tính được

def test_missing_start_uses_raw_index():
    segs = [make_seg(), {"end": 4.0, "text": " x"}]   # segment hỏng ở vị trí index 1
    _, dlq, _ = build_rows("v", segs, 600.0, PROVENANCE)
    assert dlq[0]["record_id"] is None
    assert dlq[0]["dlq_id"] == "dlq_validate_v_raw00001"

def test_text_none_goes_to_dlq():
    _, dlq, _ = build_rows("v", [make_seg(text=None)], 600.0, PROVENANCE)
    assert dlq[0]["reason_code"] == "SCHEMA_VIOLATION"

def test_one_bad_segment_does_not_lose_the_good_ones():
    fixture = json.loads(FIXTURE.read_text())
    segs = fixture["segments"] + [{"start": 99.0}]
    ok, dlq, m = build_rows("lakehouse", segs, fixture["duration"], PROVENANCE)
    assert len(ok) == 5
    assert len(dlq) == 1
    assert m["segments_in"] == 6