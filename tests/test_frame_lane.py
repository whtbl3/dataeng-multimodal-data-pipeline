# tests/test_frame_lane.py
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from talk_align.config import FrameParams
from talk_align.frame_lane import (
    box_height,
    build_slide_spans,
    clean_boxes,
    make_slide_id,
    plan_keyframes,
)

# tests/test_frame_lane.py
SCENES = Path(__file__).parent / "fixtures" / "scenes_lakehouse.json"
OCR = Path(__file__).parent / "fixtures" / "ocr_lakehouse.json"


def ocr(slide_id):
    return json.loads(OCR.read_text())["ocr"][slide_id]


def test_frame_params_are_all_fields():
    assert set(asdict(FrameParams())) == {
        "sample_fps", "scene_threshold", "min_duration", "min_box_chars", "min_box_height_ratio",
    }

def test_spec_example():                       # đúng ví dụ trong spec §3 / learning-notes §3
    spans, m = build_slide_spans([0.0, 12.0, 300.0, 300.4, 330.2], 344.561, 1.5)
    assert [(s["t_start"], s["t_end"]) for s in spans] == [(0.0, 12.0), (12.0, 300.4), (300.4, 330.2), (330.2, 344.561)]
    assert m["merged_transitions"] == 1

def test_lakehouse_gives_twelve_slides():
    d = json.loads(SCENES.read_text())
    spans, m = build_slide_spans(d["times"], d["duration"], 1.5)
    assert [s["t_start"] for s in spans] == [0.0, 6.0, 14.0, 20.0, 40.0, 83.0, 91.0, 150.0, 194.0, 225.0, 254.0, 340.0]
    assert (m["marks_in"], m["merged_transitions"], m["slides"]) == (m["marks_in"], m["merged_transitions"], m["slides"]) == (16, 4, 12)

def test_slides_tile_the_whole_video():        # không hở, không chồng lên nhau
    d = json.loads(SCENES.read_text())
    spans, _ = build_slide_spans(d["times"], d["duration"], 1.5)
    assert spans[0]["t_start"] == 0.0
    assert spans[-1]["t_end"] == d["duration"]
    for a, b in itertools.pairwise(spans):         # từng cặp slide liền nhau
        assert a["t_end"] == b["t_start"]

def test_missing_zero_is_added_and_input_is_cleaned():  
    spans, _ = build_slide_spans([20.0, 5.0, 5.0], 30.0, 1.5)      # thiếu 0, chưa sắp xếp, trùng
    assert [s["t_start"] for s in spans] == [0.0, 5.0, 20.0]

def test_zero_is_never_dropped():
    spans, m = build_slide_spans([0.0, 1.0, 10.0], 30.0, 1.5)
    assert [s["t_start"] for s in spans] == [0.0, 10.0]   # 1.0 is dropped, 0.0 is kept
    assert m["merged_transitions"] == 1

def test_every_slide_is_at_least_min_duration():
    for times in ([0.0, 1.0, 10.0], [0.0, 0.5, 1.0, 10.0], json.loads(SCENES.read_text())["times"]):
        spans, _ = build_slide_spans(times, 344.561, 1.5)
        assert all(s["t_end"] - s["t_start"] >= 1.5 for s in spans), times

def test_plan_keyframes_adds_id_midpoint_and_path():
    planned = plan_keyframes("v", [{"t_start": 16.15, "t_end": 20.0}])
    p = planned[0]
    assert p["slide_id"] == "v_s_00016150"
    assert p["keyframe_at"] == pytest.approx(18.075)
    assert str(p["keyframe_path"]).endswith("keyframes/v/v_s_00016150.png")

def test_box_height_is_max_y_minus_min_y():
    b = {"box": [[934.0, 203.0], [1090.0, 206.0], [1089.0, 256.0], [933.0, 253.0]], "text": "x", "conf": 0.9}
    assert box_height(b) == 53.0          # 256 − 203


def test_clean_slide_keeps_everything_in_order():
    r = ocr("lakehouse_s_00020000")
    kept = clean_boxes(r["boxes"], r["image_h"], FrameParams())
    assert [b["text"].strip() for b in kept] == ["1980s", "Businesses need", "more than", "relational", "databases"]

def test_footer_and_short_junk_are_dropped():
    r = ocr("lakehouse_s_00091000")
    kept = clean_boxes(r["boxes"], r["image_h"], FrameParams())
    texts = [b["text"] for b in kept]
    assert len(kept) == 10
    assert "小" not in texts
    assert not any("Databricks Inc" in t for t in texts)
    assert texts[0] == "Data Lakes"

def test_boundary_ratio():                    # đúng 0.03 → giữ hay bỏ? quyết định "<" hay "<=" giống quality_flag
    b = {"box": [[0, 0], [100, 0], [100, 21.6], [0, 21.6]], "text": "abc", "conf": 0.9}   # 21.6 / 720 = 0.03
    assert len(clean_boxes([b], 720, FrameParams())) == True