"""Frame lane (step ②): scene change keyframes -> OCR -> slides rows. Roadmap #3-4."""

import re
import shutil
import subprocess
from pathlib import Path

from talk_align.config import KEYFRAME_DIR, FrameParams
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


def detect_scene_times(video_path: Path, params: FrameParams) -> list[float]:
    """Return raw scene-change times in seconds, always starting with 0.0.

    Input:  Path("data/input/lakehouse.mp4"), FrameParams()
    Output: [0.0, 5.0, 6.0, 13.0, ..., 344.0]
    """
    video_filter = (
        f"fps={params.sample_fps},"
        f"select='eq(n,0)+gt(scene,{params.scene_threshold})',"
        f"showinfo"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",   # no -loglevel error: showinfo logs at info level
        "-i", str(video_path),
        "-vf", video_filter,
        "-an", "-f", "null", "-",               # drop audio, write no file
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # showinfo writes to the log, and ffmpeg's log goes to stderr (stdout is empty)
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", result.stderr)]

def build_slide_spans(
    times: list[float], duration: float, min_duration: float
) -> tuple[list[dict], dict]:
    """Turn raw scene-change times into slide spans [t_start, t_end). Pure function.

    Every slide lasts at least `min_duration`; 0.0 is always the first mark.

    Input:  times=[0.0, 5.0, 6.0, 13.0, 14.0, ...], duration=344.561, min_duration=1.5
    Output: ([{"t_start": 0.0, "t_end": 6.0}, {"t_start": 6.0, "t_end": 14.0}, ...],
             {"marks_in": 16, "merged_transitions": 4, "slides": 12})

    Example (lakehouse.mp4, 2026-10-19):
        1. Clean
           times = [0.0, 5.0, 6.0, 13.0, 14.0, 20.0, 39.0, 40.0, 83.0, 91.0,
                    150.0, 194.0, 225.0, 254.0, 340.0, 344.0]
           → marks: unchanged (already has 0.0, no duplicates, sorted) → 16 marks

        2. Merge (next = next mark, or duration for the last one)
           0.0   → always kept
           5.0   → next 6.0    : 1.0   < 1.5  DROP  merged=1
           6.0   → next 13.0   : 7.0          keep  (6.0 from 0.0 ≥ 1.5)
           13.0  → next 14.0   : 1.0          DROP  merged=2
           14.0, 20.0                         keep
           39.0  → next 40.0   : 1.0          DROP  merged=3
           40.0, 83.0, 91.0, 150.0, 194.0, 225.0, 254.0, 340.0   keep
           344.0 → duration 344.561 : 0.561   DROP  merged=4   (fade to black at the end)
           → kept: 12 marks

        3. Spans
           [0, 6) [6, 14) [14, 20) [20, 40) [40, 83) [83, 91) [91, 150) [150, 194)
           [194, 225) [225, 254) [254, 340) [340, 344.561)   → 12 slides
    """
    # 1. Clean: always start at 0.0, no duplicates, ascending, nothing past the end
    marks = sorted({t for t in times if t < duration} | {0.0})

    # 2. Merge: drop a mark whose next mark (or the end of the video) is too close
    kept = []
    merged = 0
    for i, t in enumerate(marks):
        next_t = marks[i + 1] if i + 1 < len(marks) else duration
        if t != 0.0:
            too_close_to_next = next_t - t < min_duration
            too_close_to_zero = kept == [0.0] and t - 0.0 < min_duration
            if too_close_to_next or too_close_to_zero:
                merged += 1
                continue
        kept.append(t)

    # 3. Spans: each slide ends where the next kept mark starts (LEAD), the last at duration
    spans = []
    for i, t in enumerate(kept):
        t_end = kept[i + 1] if i + 1 < len(kept) else duration
        spans.append({"t_start": t, "t_end": t_end})

    return spans, {"marks_in": len(marks), "merged_transitions": merged, "slides": len(spans)}

def make_slide_id(video_id: str, t_start: float) -> str:
    """Deterministic slide id from the start time in ms (round, never int).

    Input:  "lakehouse", 16.15
    Output: "lakehouse_s_00016150"
    """
    return f"{video_id}_s_{round(t_start * 1000):08d}"


def plan_keyframes(video_id: str, spans: list[dict]) -> list[dict]:
    """Add slide_id, keyframe_at and keyframe_path to each span. Pure function.

    Input:  "lakehouse", [{"t_start": 0.0, "t_end": 6.0}, ...]
    Output: [{"t_start": 0.0, "t_end": 6.0, "slide_id": "lakehouse_s_00000000",
              "keyframe_at": 3.0,
              "keyframe_path": Path("data/keyframes/lakehouse/lakehouse_s_00000000.png")}, ...]
    """
    planned = []
    for span in spans:
        slide_id = make_slide_id(video_id, span["t_start"])
        planned.append({
            **span,
            "slide_id": slide_id,
            # middle of the slide: at t_start a fade may still be running (marks are 1 s apart)
            "keyframe_at": (span["t_start"] + span["t_end"]) / 2,
            "keyframe_path": KEYFRAME_DIR / video_id / f"{slide_id}.png",
        })
    return planned


def extract_keyframes(video_path: Path, video_id: str, planned: list[dict]) -> None:
    """Write one PNG per slide. Replaces this video's keyframe folder (no stale files).

    Input:  Path("data/input/lakehouse.mp4"), "lakehouse", output of plan_keyframes
    Output: None; data/keyframes/lakehouse/<slide_id>.png for each slide
    """
    out_dir = KEYFRAME_DIR / video_id
    # partition overwrite on disk: images of an older run (other params) must not survive
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for slide in planned:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(slide["keyframe_at"]),       # before -i: fast seek
            "-i", str(video_path),
            "-frames:v", "1",
            str(slide["keyframe_path"]),
        ]
        subprocess.run(cmd, check=True)

def ocr_keyframes(planned: list[dict]) -> dict[str, dict]:
    """Run OCR on each slide's keyframe. Keeps every box (filtering happens later).

    Input:  output of plan_keyframes (needs "slide_id", "keyframe_path")
    Output: {"lakehouse_s_00000000": {"image_h": 720,
                                      "boxes": [{"box": [[x, y], ...], "text": "...", "conf": 0.997}, ...]},
             ...}
    """
    engine = RapidOCR()  # load the model once, reuse for every image
    raw_ocr: dict[str, dict] = {}
    for slide in planned:
        path = slide["keyframe_path"]
        with Image.open(path) as img:
            _width, height = img.size  # kept so later steps never need to reopen the image
        result, _elapse = engine(str(path))
        boxes = [
            {
                "box": [[float(x), float(y)] for x, y in box],
                "text": text,
                "conf": float(conf),
            }
            for box, text, conf in (result or [])  # None when the image has no text
        ]
        raw_ocr[slide["slide_id"]] = {"image_h": height, "boxes": boxes}
    return raw_ocr

def box_top(b: dict) -> float:
    return min(p[1] for p in b["box"])


def box_left(b: dict) -> float:
    return min(p[0] for p in b["box"])


def box_height(b: dict) -> float:
    ys = [p[1] for p in b["box"]]
    return max(ys) - min(ys)


def clean_boxes(boxes: list[dict], image_h: int, params: FrameParams) -> list[dict]:
    """Drop junk boxes, return the rest in reading order. Pure function.

    Junk: fewer than min_box_chars characters after strip ('小'), or a box whose height
    is a small share of the image (footer '©2022 Databricks...', 19/720 = 0.026).
    Reading order: top to bottom, then left to right (two-column slides interleave).
    """
    kept = [
        b
        for b in boxes
        if len(b["text"].strip()) >= params.min_box_chars
        and box_height(b) / image_h >= params.min_box_height_ratio
    ]
    return sorted(kept, key=lambda b: (box_top(b), box_left(b)))