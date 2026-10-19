# tests/test_frame_lane.py
from dataclasses import asdict
from talk_align.config import FrameParams

def test_frame_params_are_all_fields():
    assert set(asdict(FrameParams())) == {"sample_fps", "scene_threshold", "min_duration"}