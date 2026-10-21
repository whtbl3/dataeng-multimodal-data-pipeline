import pytest

from talk_align import pipeline
from talk_align.cli import main


def test_unknown_video_file_is_a_clean_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frames", "khong_co.mp4"])
    assert exc.value.code == 2  # argparse uses exit code 2 for command-line errors
    assert "File not exist" in capsys.readouterr().err


def test_align_without_lanes_raises_missing_input(tmp_path):
    with pytest.raises(pipeline.MissingInputError, match="0 audio rows, 0 slides"):
        pipeline.run_align("khong_co", "run-test", db_path=tmp_path / "t.duckdb")
