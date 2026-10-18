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
    rows = build_rows(fixture["video_id"], fixture["segments"], PROVENANCE)
    print(rows[0])
    assert [r["segment_id"] for r in rows] == [
        "lakehouse_a_00000000",
        "lakehouse_a_00005000",
        "lakehouse_a_00013000",
        "lakehouse_a_00020000",
        "lakehouse_a_00027000",
    ]
    assert {r["quality_flag"] for r in rows} == {"ok"}
    assert isinstance(rows[0]["raw_payload"], str)
    assert json.loads(rows[0]["raw_payload"]) == fixture["segments"][0]
    assert fixture["segments"][0]["text"].startswith(" ")
    assert not rows[0]["text"].startswith(" ")
    assert set(rows[0]) == {
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
    rows = build_rows("talk01", [seg], PROVENANCE)
    assert rows[0]["segment_id"] == "talk01_a_00016150"


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
    first = build_rows(fixture["video_id"], fixture["segments"], PROVENANCE)
    second = build_rows(fixture["video_id"], fixture["segments"], PROVENANCE)
    assert first == second
