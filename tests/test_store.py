import json
import pytest
import duckdb
from pathlib import Path

from talk_align import store
from talk_align.audio_lane import build_rows
from tests.test_audio_lane import PROVENANCE

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_lakehouse.json"

def test_schema_has_the_ten_audio_columns():
    con = store.connect(":memory:")
    cols = {row[0] for row in con.execute("DESCRIBE audio_segments").fetchall()}
    # sample segment
    seg = {"start": 1.0, "end": 2.0, "text": " hi", "avg_logprob": -0.2, "no_speech_prob": 0.01}
    rows = build_rows("x", [seg], PROVENANCE)
    con.close()
    assert cols == set(rows[0])

def test_insert_writes_all_rows():
    fixture = json.loads(FIXTURE.read_text())
    con = store.connect(":memory:")
    rows = build_rows("lakehouse", fixture["segments"], PROVENANCE)
    store.insert_rows(con, "audio_segments", rows)
    count = con.execute("SELECT count(*) FROM audio_segments").fetchone()[0]
    con.close()
    assert count == len(rows)

def test_insert_twice_fails():                     
    fixture = json.loads(FIXTURE.read_text())
    con = store.connect(":memory:")
    rows = build_rows("lakehouse", fixture["segments"], PROVENANCE)
    store.insert_rows(con, "audio_segments", rows)
    with pytest.raises(duckdb.ConstraintException):
        store.insert_rows(con, "audio_segments", rows)
    con.close()