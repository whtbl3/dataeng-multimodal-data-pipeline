"""Frame lane (step ②): scene change keyframes -> OCR -> slides rows. Roadmap #3-4."""

import re
import subprocess
from pathlib import Path

from talk_align.config import FrameParams


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