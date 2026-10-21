from pathlib import Path

from talk_align import pipeline


def test_audio_failure_does_not_stop_frames(monkeypatch):
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

    result = pipeline.run_all(Path("x.mp4"), "run-1")

    assert calls == ["frames"]  # frames still ran; align was NOT called
    assert result["status"] == "partial"
    assert result["audio"]["error"] == "RuntimeError: whisper out of memory"
    assert "skipped" in result["align"]
    assert "Traceback" in result["audio"]["traceback"]


def test_all_lanes_ok_is_success(monkeypatch):
    seen = []

    def fake(name):
        def f(*args, **kwargs):
            seen.append((name, args[1]))  # args[1] = run_id
            return {"ok": 1}

        return f

    for lane in ("run_audio", "run_frames", "run_align"):
        monkeypatch.setattr(pipeline, lane, fake(lane))

    result = pipeline.run_all(Path("x.mp4"), "run-1")

    assert result["status"] == "success"
    assert [name for name, _ in seen] == ["run_audio", "run_frames", "run_align"]
    assert {run_id for _, run_id in seen} == {"run-1"}  # all three share ONE run_id


def test_both_lanes_failing_is_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    align_called = []
    monkeypatch.setattr(pipeline, "run_audio", boom)
    monkeypatch.setattr(pipeline, "run_frames", boom)
    monkeypatch.setattr(pipeline, "run_align", lambda *a, **k: align_called.append(1))

    result = pipeline.run_all(Path("x.mp4"), "run-1")

    assert result["status"] == "failed"
    assert result["align"] == {"skipped": "audio and frames failed"}
    assert align_called == []


def test_align_failure_is_partial(monkeypatch):
    monkeypatch.setattr(pipeline, "run_audio", lambda *a, **k: {"ok": 78})
    monkeypatch.setattr(pipeline, "run_frames", lambda *a, **k: {"ok": 12})

    def boom(*args, **kwargs):
        raise pipeline.MissingInputError("0 slides")

    monkeypatch.setattr(pipeline, "run_align", boom)

    result = pipeline.run_all(Path("x.mp4"), "run-1")

    assert result["status"] == "partial"
    assert result["align"]["error"].startswith("MissingInputError")
