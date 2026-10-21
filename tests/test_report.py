import pytest

from talk_align import pipeline, store
from talk_align.report import summarize
from tests.test_align import put_audio, put_slides


def ok_result(**overrides):
    r = {
        "audio": {
            "segments_in": 80,
            "dropped_silence": 2,
            "dlq": 3,
            "ok": 75,
            "quarantined": 1,
        },
        "frames": {"slides_in": 12, "dlq": 1, "ok": 11},
        "align": {
            "segments_aligned": 75,
            "nearest": 3,
            "weak": 4,
            "p95_align_gap_s": 1.0,
        },
    }
    return {**r, **overrides}


def test_dlq_rate_excludes_silence():
    s = summarize(ok_result())
    assert s["dlq_rate_audio"] == pytest.approx(3 / 78)  # 3 / (80 - 2) = 0.0385
    assert s["dlq_rate_frame"] == pytest.approx(1 / 12)  # 0.0833


def test_nearest_pct():
    assert summarize(ok_result())["nearest_pct"] == pytest.approx(4.0)  # 3 / 75 * 100


def test_failed_lane_gives_none_not_zero():
    s = summarize(ok_result(audio={"error": "boom"}, align={"skipped": "audio failed"}))
    assert s["dlq_rate_audio"] is None
    assert s["nearest_pct"] is None
    assert s["dlq_rate_frame"] == pytest.approx(1 / 12)  # frames is still measured


def test_all_silence_does_not_divide_by_zero():
    s = summarize(
        ok_result(
            audio={
                "segments_in": 5,
                "dropped_silence": 5,
                "dlq": 0,
                "ok": 0,
                "quarantined": 0,
            }
        )
    )
    assert s["dlq_rate_audio"] == 0.0


def test_align_reports_gap_percentiles(tmp_path):
    db = tmp_path / "t.duckdb"
    con = store.connect(db)
    put_audio(con, "v", [(1.0, 3.0), (5.0, 13.0)])  # second segment crosses 6.0
    put_slides(con, "v", [(0.0, 6.0), (6.0, 20.0)])
    con.close()
    m = pipeline.run_align("v", "run-1", db_path=db)
    # [1,3): inside [0,6)        -> gap 0
    # [5,13): [0,6) 1, [6,20) 7  -> takes [6,20), gap 8 - 7 = 1
    # quantile_cont of [0, 1] at 0.95 = 0 + 0.95 * (1 - 0) = 0.95
    assert m["p95_align_gap_s"] == pytest.approx(0.95)
    assert m["p50_align_gap_s"] == pytest.approx(0.5)
