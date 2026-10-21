"""Helpers shared by both lanes that read facts about the input video."""

import subprocess
from pathlib import Path


def probe_duration(video_path: Path) -> float:
    """Return the video length in seconds, read by ffprobe from the original file.

    Input:  Path("data/input/lakehouse.mp4")
    Output: 344.561
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())
