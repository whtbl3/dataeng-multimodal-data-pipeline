"""Data contract (step ③): validate rows, map errors to reason_code, route to DLQ. Roadmap #2."""

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AudioSegmentRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str
    segment_id: str
    modality: Literal["audio"]
    text: str
    raw_payload: str
    t_start: float = Field(ge=0)
    t_end: float
    quality_score: float = Field(ge=0, le=1)
    quality_flag: Literal["ok", "quarantine"]
    provenance: dict


def make_error_detail(err: ValidationError) -> str:
    return "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in err.errors())


def reason_code_for(err: ValidationError) -> str:
    """Map a Pydantic error to our DLQ reason_code."""
    for e in err.errors():
        if e["loc"][0] == "quality_score":
            return "SCORE_OUT_OF_RANGE"
        else:
            return "SCHEMA_VIOLATION"


class SlideRow(BaseModel):
    """One row of the slides table (spec §4). Text may be empty: image-only slide."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    slide_id: str
    modality: Literal["frame"]
    text: str
    t_start: float = Field(ge=0)
    t_end: float
    quality_score: float = Field(ge=0, le=1)
    quality_flag: Literal["ok", "quarantine"]
    keyframe_path: str
    raw_payload: str
    provenance: dict


class AlignedRow(BaseModel):
    """One aligned row. A violation here is a bug in the align SQL, so it is not caught."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    segment_id: str
    slide_id: str
    match_type: Literal["overlap", "nearest"]
    overlap_s: float = Field(ge=0)
    nearest_distance_s: float | None = Field(
        ge=0
    )  # required key; None allowed; if set, >= 0
    align_gap_s: float = Field(ge=0)
    align_gap_ratio: float = Field(ge=0, le=1)
    slide_quality_flag: Literal["ok", "quarantine"]
    run_id: str


def aligned_row(**overrides):
    row = {
        "video_id": "v",
        "segment_id": "v_a_00001000",
        "slide_id": "v_s_00000000",
        "match_type": "overlap",
        "overlap_s": 2.0,
        "nearest_distance_s": None,
        "align_gap_s": 0.0,
        "align_gap_ratio": 0.0,
        "slide_quality_flag": "ok",
        "run_id": "r",
    }
    return {**row, **overrides}


def test_aligned_contract_accepts_a_valid_row():
    AlignedRow.model_validate(aligned_row())


def test_aligned_contract_rejects_bad_ratio():
    with pytest.raises(ValidationError):
        AlignedRow.model_validate(aligned_row(align_gap_ratio=1.3))


def test_aligned_contract_rejects_negative_nearest_distance():
    with pytest.raises(ValidationError):
        AlignedRow.model_validate(
            aligned_row(match_type="nearest", nearest_distance_s=-2.0)
        )
