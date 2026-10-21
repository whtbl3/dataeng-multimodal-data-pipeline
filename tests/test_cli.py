from pathlib import Path

import pytest

from talk_align import cli, pipeline
from talk_align.cli import main


def test_unknown_video_file_is_a_clean_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frames", "khong_co.mp4"])
    assert exc.value.code == 2  # argparse uses exit code 2 for command-line errors
    assert "File not exist" in capsys.readouterr().err


def test_align_without_lanes_raises_missing_input(tmp_path):
    with pytest.raises(pipeline.MissingInputError, match="0 audio rows, 0 slides"):
        pipeline.run_align("khong_co", "run-test", db_path=tmp_path / "t.duckdb")


def test_run_exit_code_is_1_when_not_success(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")  # exists, never really read: run_all is faked
    monkeypatch.setattr(
        pipeline,
        "run_all",
        lambda path, run_id: {
            "run_id": run_id,
            "video_id": "v",
            "status": "partial",
            "audio": {"error": "RuntimeError: x", "traceback": "Traceback ..."},
            "frames": {"ok": 12},
            "align": {"skipped": "audio failed"},
        },
    )
    assert main(["run", str(video)]) == 1


def test_report_without_any_run_is_a_clean_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DB_PATH", tmp_path / "t.duckdb")
    with pytest.raises(SystemExit) as exc:
        main(["report", "khong_co"])
    assert exc.value.code == 2
    assert "No run recorded for khong_co" in capsys.readouterr().err


def test_report_prints_the_latest_run(monkeypatch, tmp_path, capsys):
    db = tmp_path / "t.duckdb"
    monkeypatch.setattr(cli, "DB_PATH", db)
    for lane in ("run_audio", "run_frames", "run_align"):
        monkeypatch.setattr(pipeline, lane, lambda *a, **k: {"ok": 1})
    pipeline.run_all(Path("v.mp4"), "run-1", db_path=db)
    assert main(["report", "v"]) == 0
    out = capsys.readouterr().out
    assert "# Run report: v" in out and "run-1" in out
