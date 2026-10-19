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