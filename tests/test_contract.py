# tests/test_contract.py
import pytest
from pydantic import ValidationError

from talk_align.contract import AudioSegmentRow, reason_code_for


def valid_row(**overrides):
    row = {"video_id": "v", "segment_id": "v_a_00001000", "modality": "audio", "text": "hi",
           "t_start": 1.0, "t_end": 2.0, "quality_score": 0.8, "quality_flag": "ok",
           "raw_payload": "{}", "provenance": {}}
    return {**row, **overrides}                    # ghi đè những cột muốn làm hỏng

def test_valid_row_passes():
    AudioSegmentRow.model_validate(valid_row())    # không lỗi là pass

def test_score_above_one_is_score_out_of_range():
    with pytest.raises(ValidationError) as exc:
        AudioSegmentRow.model_validate(valid_row(quality_score=1.3))
    assert reason_code_for(exc.value) == "SCORE_OUT_OF_RANGE"

def test_bad_flag_is_schema_violation():
    with pytest.raises(ValidationError) as exc:
        AudioSegmentRow.model_validate(valid_row(quality_flag="maybe"))
    assert reason_code_for(exc.value) == "SCHEMA_VIOLATION"

def test_extra_column_is_schema_violation():
    with pytest.raises(ValidationError) as exc:
        AudioSegmentRow.model_validate(valid_row(color="red"))
    assert reason_code_for(exc.value) == "SCHEMA_VIOLATION"