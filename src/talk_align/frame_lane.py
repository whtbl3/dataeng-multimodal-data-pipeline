"""Frame lane (step ②): scene change keyframes -> OCR -> slides rows. Roadmap #3-4."""
from pathlib import Path
from talk_align.config import FrameParams

def detect_scene_times(video_path: Path, params: FrameParams) -> list[float]:
    raise NotImplementedError