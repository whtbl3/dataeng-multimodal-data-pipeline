import json
from pathlib import Path

import pytest

from talk_align import pipeline, store


def test_audio_failure_does_not_stop_frames(monkeypatch, tmp_path):
    calls = []

    def boom(*args, **kwargs):
        raise RuntimeError("whisper out of memory")

    def fake_frames(*args, **kwargs):
        calls.append("frames")
        return {"slides_in": 12}

    def fake_align(*args, **kwargs):
        calls.append("align")
        return {}

    monkeypatch.setattr(pipeline, "run_audio", boom)
    monkeypatch.setattr(pipeline, "run_frames", fake_frames)
    monkeypatch.setattr(pipeline, "run_align", fake_align)

    result = pipeline.run_all(Path("x.mp4"), "run-1", db_path=tmp_path / "t.duckdb")

    assert calls == ["frames"]  # frames still ran; align was NOT called
    assert result["status"] == "partial"
    assert result["audio"]["error"] == "RuntimeError: whisper out of memory"
    assert "skipped" in result["align"]
    assert "Traceback" in result["audio"]["traceback"]


def test_all_lanes_ok_is_success(monkeypatch, tmp_path):
    seen = []

    def fake(name):
        def f(*args, **kwargs):
            seen.append((name, args[1]))  # args[1] = run_id
            return {"ok": 1}

        return f

    for lane in ("run_audio", "run_frames", "run_align"):
        monkeypatch.setattr(pipeline, lane, fake(lane))

    result = pipeline.run_all(Path("x.mp4"), "run-1", db_path=tmp_path / "t.duckdb")

    assert result["status"] == "success"
    assert [name for name, _ in seen] == ["run_audio", "run_frames", "run_align"]
    assert {run_id for _, run_id in seen} == {"run-1"}  # all three share ONE run_id


def test_both_lanes_failing_is_failed(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    align_called = []
    monkeypatch.setattr(pipeline, "run_audio", boom)
    monkeypatch.setattr(pipeline, "run_frames", boom)
    monkeypatch.setattr(pipeline, "run_align", lambda *a, **k: align_called.append(1))

    result = pipeline.run_all(Path("x.mp4"), "run-1", db_path=tmp_path / "t.duckdb")

    assert result["status"] == "failed"
    assert result["align"] == {"skipped": "audio and frames failed"}
    assert align_called == []


def test_align_failure_is_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "run_audio", lambda *a, **k: {"ok": 78})
    monkeypatch.setattr(pipeline, "run_frames", lambda *a, **k: {"ok": 12})

    def boom(*args, **kwargs):
        raise pipeline.MissingInputError("0 slides")

    monkeypatch.setattr(pipeline, "run_align", boom)

    result = pipeline.run_all(Path("x.mp4"), "run-1", db_path=tmp_path / "t.duckdb")

    assert result["status"] == "partial"
    assert result["align"]["error"].startswith("MissingInputError")


def fake_ok(*args, **kwargs):
    return {"ok": 1}


def read_runs(db_path):
    con = store.connect(db_path)
    rows = con.execute(
        "SELECT run_id, status, finished_at IS NOT NULL, metrics FROM runs ORDER BY run_id"
    ).fetchall()
    con.close()
    return rows


def test_run_is_recorded(monkeypatch, tmp_path):
    for lane in ("run_audio", "run_frames", "run_align"):
        monkeypatch.setattr(pipeline, lane, fake_ok)
    db = tmp_path / "t.duckdb"
    pipeline.run_all(Path("x.mp4"), "run-1", db_path=db)
    [(run_id, status, finished, metrics)] = read_runs(db)  # exactly one row
    assert (run_id, status, finished) == ("run-1", "success", True)
    assert json.loads(metrics)["frames"] == {"ok": 1}


def test_runs_keep_history(monkeypatch, tmp_path):
    for lane in ("run_audio", "run_frames", "run_align"):
        monkeypatch.setattr(pipeline, lane, fake_ok)
    db = tmp_path / "t.duckdb"
    pipeline.run_all(Path("x.mp4"), "run-1", db_path=db)
    pipeline.run_all(Path("x.mp4"), "run-2", db_path=db)  # same video, second run
    assert [r[0] for r in read_runs(db)] == ["run-1", "run-2"]  # upsert keeps both


def test_killed_run_stays_running(monkeypatch, tmp_path):
    def killed(*args, **kwargs):
        raise KeyboardInterrupt  # like Ctrl+C: a BaseException, not an Exception

    monkeypatch.setattr(pipeline, "run_audio", killed)
    db = tmp_path / "t.duckdb"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_all(Path("x.mp4"), "run-1", db_path=db)
    [(_, status, finished, _)] = read_runs(db)
    assert (status, finished) == ("running", False)
