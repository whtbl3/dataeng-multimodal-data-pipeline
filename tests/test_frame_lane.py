# tests/test_frame_lane.py
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from talk_align.config import FrameParams
from talk_align.frame_lane import (
    box_height,
    build_slide_rows,
    build_slide_spans,
    clean_boxes,
    plan_keyframes,
    slide_flag,
    slide_quality,
    slide_text,
)
from tests.test_audio_lane import PROVENANCE

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

def tb(text, conf):                                   # box giả, chỉ cần text và conf
    return {"box": [[0, 0], [1, 0], [1, 50], [0, 50]], "text": text, "conf": conf}


def test_slide_quality_is_weighted_by_characters():   # đúng ví dụ trong learning-notes §3
    kept = [
        tb("Lakehouse Architecture", 0.97),           # 22 ký tự
        tb("• Object storage + table format", 0.91),  # 31 ký tự (• tính là 1)
        tb("• Iceberg / Deita", 0.58),                # 17 ký tự
    ]
    # (22×0.97 + 31×0.91 + 17×0.58) / (22+31+17) = 59.41 / 70 = 0.8487
    assert slide_quality(kept) == pytest.approx(0.8487, abs=1e-3)


def test_slide_text_strips_and_joins_lines():
    assert slide_text([tb(" Transaction support", 0.9), tb("BI Support ", 0.9)]) == "Transaction support\nBI Support"


def test_empty_slide_is_kept_as_quarantine():
    assert slide_text([]) == ""
    assert slide_quality([]) == 0.0
    assert slide_flag(slide_quality([])) == "quarantine"


def test_real_slide_scores_ok():
    r = ocr("lakehouse_s_00091000")
    kept = clean_boxes(r["boxes"], r["image_h"], FrameParams())
    assert slide_flag(slide_quality(kept)) == "ok"
    assert slide_text(kept).startswith("Data Lakes\nCons:")

def fixture():
    return json.loads(OCR.read_text())


def test_build_slide_rows_on_fixture():
    d = fixture()
    ok, dlq, m = build_slide_rows(d["video_id"], d["slides"], d["ocr"], FrameParams(), PROVENANCE)
    assert len(ok) == 4 and dlq == []
    assert set(ok[0]) == {
        "video_id", "slide_id", "modality", "text", "t_start", "t_end",
        "quality_score", "quality_flag", "keyframe_path", "raw_payload", "provenance",
    }
    by_id = {r["slide_id"]: r for r in ok}
    assert by_id["lakehouse_s_00340000"]["text"] == "databricks"
    assert (m["boxes_in"], m["boxes_dropped"]) == (28, 3)     # 5+12+10+1; 小 + 2 footer


def test_raw_payload_keeps_dropped_boxes():
    d = fixture()
    ok, _, _ = build_slide_rows(d["video_id"], d["slides"], d["ocr"], FrameParams(), PROVENANCE)
    row = next(r for r in ok if r["slide_id"] == "lakehouse_s_00091000")
    assert len(json.loads(row["raw_payload"])["boxes"]) == 12  # 12 thô, không phải 10 đã lọc


def test_slide_with_no_boxes_is_kept_as_quarantine():
    slide = {"slide_id": "v_s_00000000", "t_start": 0.0, "t_end": 5.0, "keyframe_path": "x.png"}
    ok, dlq, m = build_slide_rows(
        "v", [slide], {"v_s_00000000": {"image_h": 720, "boxes": []}}, FrameParams(), PROVENANCE
    )
    assert (ok[0]["text"], ok[0]["quality_score"], ok[0]["quality_flag"]) == ("", 0.0, "quarantine")
    assert dlq == [] and m["empty_text"] == 1


def test_running_twice_gives_the_same_slide_rows():
    d = fixture()
    assert build_slide_rows(d["video_id"], d["slides"], d["ocr"], FrameParams(), PROVENANCE) == \
           build_slide_rows(d["video_id"], d["slides"], d["ocr"], FrameParams(), PROVENANCE)