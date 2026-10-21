"""Pipeline steps, one function per lane. No argparse here: errors are exceptions.

Each step opens its own connection, writes its tables in one transaction and returns its
metrics. `run` (roadmap #6) calls the three steps with ONE run_id.
"""

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from talk_align import store
from talk_align.align import align_video
from talk_align.audio_lane import build_rows, transcribe
from talk_align.config import BASE_DIR, DB_PATH, AudioParams, FrameParams
from talk_align.contract import AlignedRow
from talk_align.frame_lane import (
    build_slide_rows,
    build_slide_spans,
    detect_scene_times,
    extract_keyframes,
    ocr_keyframes,
    plan_keyframes,
)
from talk_align.media import probe_duration


class MissingInputError(Exception):
    """A step needs rows that an earlier step has not written yet."""


def new_run_id() -> str:
    return str(uuid.uuid4())


def file_sha256(video_path: Path) -> str:
    with video_path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    return f"sha256:{digest.hexdigest()}"


def git_code_version() -> str:
    """Short commit, plus "-dirty" when this project has uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain", "."],
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{commit}-dirty" if changes else commit


def build_provenance(
    video_path: Path, model: str, model_version: str, params, run_id: str
) -> dict:
    """Who made these rows: model, its version, every param that changes output, code, run.

    params: any dataclass (AudioParams, FrameParams). run_id comes from the caller, so one
    `run` can share it across the three steps.
    """
    return {
        "model": model,
        "model_version": model_version,
        "params": json.dumps(asdict(params), sort_keys=True),
        "source_hash": file_sha256(video_path),
        "run_id": run_id,
        "code_version": git_code_version(),
        "processed_at": datetime.now(),
    }


def run_audio(video_path: Path, run_id: str, db_path: Path | str = DB_PATH) -> dict:
    """video -> Whisper -> audio_segments + dlq_audio. Returns the lane metrics."""
    video_id = video_path.stem
    params = AudioParams()
    provenance = build_provenance(
        video_path, "faster-whisper", params.model, params, run_id
    )
    raw_segments, duration = transcribe(video_path, params)
    ok_rows, dlq_rows, metrics = build_rows(
        video_id, raw_segments, duration, provenance
    )

    conn = store.connect(db_path)  # open only now: Whisper above takes minutes
    try:
        store.overwrite_video_tables(
            conn, video_id, {"audio_segments": ok_rows, "dlq_audio": dlq_rows}
        )
    finally:
        conn.close()
    return metrics


def run_frames(video_path: Path, run_id: str, db_path: Path | str = DB_PATH) -> dict:
    """video -> scene marks -> keyframes -> OCR -> slides + dlq_frame. Returns metrics."""
    video_id = video_path.stem
    params = FrameParams()
    provenance = build_provenance(
        video_path,
        "rapidocr_onnxruntime",
        version("rapidocr_onnxruntime"),
        params,
        run_id,
    )
    duration = probe_duration(video_path)
    times = detect_scene_times(video_path, params)
    spans, span_metrics = build_slide_spans(times, duration, params.min_duration)
    planned = plan_keyframes(video_id, spans)
    extract_keyframes(video_path, video_id, planned)
    raw_ocr = ocr_keyframes(planned)
    ok_rows, dlq_rows, slide_metrics = build_slide_rows(
        video_id, planned, raw_ocr, params, provenance
    )

    conn = store.connect(db_path)
    try:
        store.overwrite_video_tables(
            conn, video_id, {"slides": ok_rows, "dlq_frame": dlq_rows}
        )
    finally:
        conn.close()
    return {**span_metrics, **slide_metrics}


def run_align(video_id: str, run_id: str, db_path: Path | str = DB_PATH) -> dict:
    """audio_segments x slides -> aligned. Raises MissingInputError if a lane has not run."""
    conn = store.connect(db_path)
    try:
        n_audio = conn.execute(
            "SELECT count(*) FROM audio_segments WHERE video_id = ?", [video_id]
        ).fetchone()[0]
        n_slides = conn.execute(
            "SELECT count(*) FROM slides WHERE video_id = ?", [video_id]
        ).fetchone()[0]
        if n_audio == 0 or n_slides == 0:
            raise MissingInputError(
                f"video_id {video_id}: {n_audio} audio rows, {n_slides} slides "
                "(run `talk-align audio` and `talk-align frames` first)"
            )

        rows, metrics = align_video(conn, video_id)
        rows = [{**r, "video_id": video_id, "run_id": run_id} for r in rows]
        for row in rows:
            AlignedRow.model_validate(
                row
            )  # a violation is a bug in the align SQL: crash
        store.overwrite_video_tables(conn, video_id, {"aligned": rows})
    finally:
        conn.close()
    return metrics
