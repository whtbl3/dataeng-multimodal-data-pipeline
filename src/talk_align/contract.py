"""Data contract (step ③): validate rows, map errors to reason_code, route to DLQ. Roadmap #2."""

from typing import Literal

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