from datetime import datetime

import pytest

from talk_align import pipeline, store
from talk_align.report import render_report, summarize, warnings
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


def run_row(**over):
    """A finished, clean runs row as store.latest_run returns it (metrics parsed)."""
    row = {
        "run_id": "r1",
        "video_id": "v",
        "status": "success",
        "code_version": "abc1234",
        "started_at": datetime(2026, 10, 21, 15, 0, 0),
        "finished_at": datetime(2026, 10, 21, 15, 2, 26),
        "metrics": {
            "audio": {
                "segments_in": 78,
                "ok": 78,
                "dlq": 0,
                "dropped_silence": 0,
                "clipped_t_end": 0,
                "quarantined": 0,
            },
            "frames": {
                "slides_in": 12,
                "ok": 12,
                "dlq": 0,
                "merged_transitions": 4,
                "quarantined": 0,
            },
            "align": {"segments_aligned": 78, "nearest": 0, "weak": 0},
            "summary": {
                "dlq_rate_audio": 0.0,
                "dlq_rate_frame": 0.0,
                "nearest_pct": 0.0,
                "weak": 0,
                "p50_align_gap_s": 0.0,
                "p95_align_gap_s": 1.0,
                "p50_align_gap_ratio": 0.0,
                "p95_align_gap_ratio": 0.13,
            },
        },
    }
    return {**row, **over}  # metrics is rebuilt on every call: tests never share it


def test_clean_run_has_no_warnings():
    assert warnings(run_row(), {}) == []


def test_dirty_code_is_warned():
    assert any(
        "uncommitted" in w for w in warnings(run_row(code_version="abc-dirty"), {})
    )


def test_many_nearest_rows_warn_about_frames():
    r = run_row()
    r["metrics"]["summary"]["nearest_pct"] = 12.0
    assert any("lost slides" in w for w in warnings(r, {}))


def test_nearest_at_the_threshold_is_not_warned():
    r = run_row()
    r["metrics"]["summary"]["nearest_pct"] = 5.0  # "> 5 %", not ">="
    assert warnings(r, {}) == []


def test_shifted_boundaries_and_weak_matches_are_warned():
    r = run_row()
    r["metrics"]["summary"]["p95_align_gap_s"] = 4.0
    r["metrics"]["summary"]["weak"] = 3
    ws = warnings(r, {})
    assert any("shifted" in w for w in ws)
    assert any("3 weak" in w for w in ws)


def test_dlq_reasons_are_listed():
    r = run_row()
    r["metrics"]["summary"]["dlq_rate_audio"] = 2 / 78
    ws = warnings(r, {"dlq_audio/EMPTY_TEXT": 2, "dlq_frame/CORRUPT_KEYFRAME": 1})
    assert any("audio DLQ rate 2.6%" in w for w in ws)
    assert any("dlq_audio/EMPTY_TEXT 2" in w and "CORRUPT_KEYFRAME 1" in w for w in ws)


def test_failed_lane_and_killed_run_are_warned():
    r = run_row(status="partial")
    r["metrics"]["audio"] = {"error": "RuntimeError: out of memory"}
    r["metrics"]["align"] = {"skipped": "audio failed"}
    ws = warnings(r, {})
    assert "audio failed: RuntimeError: out of memory" in ws
    assert "align skipped: audio failed" in ws
    killed = run_row(status="running", finished_at=None, metrics={})
    assert any("never finished" in w for w in warnings(killed, {}))


def test_report_has_sections_and_no_warning_line():
    text = render_report(run_row(), {})
    for heading in (
        "# Run report: v",
        "## Audio",
        "## Frames",
        "## Align",
        "## Warnings",
    ):
        assert heading in text
    assert "2m26s" in text
    assert "- none" in text


def test_report_shows_dash_not_zero_for_a_failed_lane():
    r = run_row(status="partial")
    r["metrics"]["audio"] = {"error": "RuntimeError: x"}
    r["metrics"]["summary"]["dlq_rate_audio"] = None
    text = render_report(r, {})
    assert "## Audio\nfailed: RuntimeError: x" in text


def test_report_of_a_killed_run_renders():
    text = render_report(run_row(status="running", finished_at=None, metrics={}), {})
    assert "not finished" in text and "never finished" in text
