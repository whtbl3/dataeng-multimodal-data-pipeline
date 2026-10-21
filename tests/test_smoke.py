"""End-to-end on the real video: run the whole pipeline twice, exports must match.

Slow (~5 min: Whisper twice) and needs data/input/lakehouse.mp4, so it is marked `slow`
and left out of the default `pytest` run. Run it with: uv run pytest -m slow
"""

import pytest

from talk_align import idempotency, pipeline, store
from talk_align.config import INPUT_DIR

VIDEO = INPUT_DIR / "lakehouse.mp4"


@pytest.mark.slow
@pytest.mark.skipif(not VIDEO.exists(), reason="needs data/input/lakehouse.mp4")
def test_two_runs_give_the_same_rows(tmp_path):
    db = tmp_path / "smoke.duckdb"
    dirs = []
    for i in (1, 2):
        result = pipeline.run_all(VIDEO, f"smoke-{i}", db_path=db)
        assert result["status"] == "success", result
        con = store.connect(db)
        dirs.append(store.export_parquet(con, "lakehouse", tmp_path / f"run{i}"))
        con.close()

    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    comparison = idempotency.compare_dirs(out1, out2)
    assert idempotency.is_identical(comparison), idempotency.render_comparison(
        comparison, idempotency.VOLATILE
    )
