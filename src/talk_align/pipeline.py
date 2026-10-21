"""Pipeline steps, one function per lane. No argparse here: errors are exceptions.

Each step opens its own connection, writes its tables in one transaction and returns its
metrics. `run` (roadmap #6) calls the three steps with ONE run_id.
"""

import hashlib
import json
import subprocess
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from talk_align import replay, store
from talk_align.align import align_video
from talk_align.audio_lane import build_rows, transcribe
from talk_align.config import (
    BASE_DIR,
    DB_PATH,
    MAX_REPLAY_ATTEMPTS,
    AudioParams,
    FrameParams,
)
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
from talk_align.report import summarize


class MissingInputError(Exception):
    """A step needs rows that an earlier step has not written yet."""


class SourceMismatchError(Exception):
    """The video given to replay is not the one that produced the DLQ rows."""


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
        p50_s, p95_s, p50_r, p95_r = conn.execute(
            """
            SELECT quantile_cont(align_gap_s, 0.5),
                   quantile_cont(align_gap_s, 0.95),
                   quantile_cont(align_gap_ratio, 0.5),
                   quantile_cont(align_gap_ratio, 0.95)
            FROM aligned WHERE video_id = ?
            """,
            [video_id],
        ).fetchone()
        metrics.update(
            p50_align_gap_s=p50_s,
            p95_align_gap_s=p95_s,
            p50_align_gap_ratio=p50_r,
            p95_align_gap_ratio=p95_r,
        )
    finally:
        conn.close()
    return metrics


def run_replay(video_path: Path, modality: str, db_path: Path | str = DB_PATH) -> dict:
    """Replay open DLQ rows of one modality ("audio" | "frame") for this video.

    No Whisper: audio rows are re-validated from raw_payload; frame rows with an OCR-stage
    error get their keyframe cut again and OCR on that one image. Fixed rows are upserted
    (never DELETE by video_id), then `aligned` is rebuilt if anything changed.
    Output: {"replayed_in", "fixed", "still_failing", "ignored", "needs_human", "realigned"}
    """
    video_id = video_path.stem
    run_id = new_run_id()
    if modality == "audio":
        dlq_table, main_table = "dlq_audio", "audio_segments"
        params = AudioParams()
        provenance = build_provenance(
            video_path, "faster-whisper", params.model, params, run_id
        )
    else:
        dlq_table, main_table = "dlq_frame", "slides"
        params = FrameParams()
        provenance = build_provenance(
            video_path,
            "rapidocr_onnxruntime",
            version("rapidocr_onnxruntime"),
            params,
            run_id,
        )

    conn = store.connect(db_path)
    try:
        rows = store.open_dlq_rows(conn, dlq_table, video_id, MAX_REPLAY_ATTEMPTS)
    finally:
        conn.close()

    # replaying against another video would "fix" rows with the wrong content
    other = {r["provenance"]["source_hash"] for r in rows} - {provenance["source_hash"]}
    if other:
        raise SourceMismatchError(
            f"{video_path} is not the video that produced the {dlq_table} rows of "
            f"{video_id} (source_hash differs)"
        )

    if modality == "audio":
        fixed, updated, metrics = replay.replay_audio_rows(
            rows, video_id, probe_duration(video_path), provenance
        )
    else:
        fixed, updated, metrics = replay.replay_frame_rows(
            rows, video_id, video_path, params, provenance
        )

    conn = store.connect(db_path)
    try:
        store.replay_write(conn, main_table, fixed, dlq_table, updated)
        metrics["needs_human"] = store.dlq_needs_human(
            conn, video_id, MAX_REPLAY_ATTEMPTS
        )
    finally:
        conn.close()

    metrics["realigned"] = False
    if fixed:  # aligned was built without these rows
        try:
            run_align(video_id, run_id, db_path=db_path)
            metrics["realigned"] = True
        except MissingInputError:
            pass  # the other lane has not run yet: nothing to align
    return metrics


def save_run(db_path: Path | str, run_row: dict) -> None:
    """Upsert one runs row. Open and close at once: never hold DuckDB during Whisper."""
    conn = store.connect(db_path)
    try:
        store.upsert_rows(conn, "runs", [run_row])
    finally:
        conn.close()


def run_all(video_path: Path, run_id: str, db_path: Path | str = DB_PATH) -> dict:
    """Run audio, frames, then align with ONE run_id. A failing lane never stops the other.

    runs table is written twice: "running" before, the final status after. A killed run
    (Ctrl+C, OOM kill) keeps its "running" row with finished_at NULL, so it is visible.
    """
    video_id = video_path.stem
    run_row = {
        "run_id": run_id,
        "video_id": video_id,
        "started_at": datetime.now(),
        "finished_at": None,
        "status": "running",
        "code_version": git_code_version(),
        "params": json.dumps(
            {"audio": asdict(AudioParams()), "frames": asdict(FrameParams())},
            sort_keys=True,
        ),
        "metrics": None,
    }
    save_run(db_path, run_row)  # ① before: a trace even if we never reach the end

    result: dict = {"run_id": run_id, "video_id": video_id}

    for lane, step in (("audio", run_audio), ("frames", run_frames)):
        try:
            result[lane] = step(video_path, run_id, db_path=db_path)
        except Exception as e:  # noqa: BLE001 - orchestration: record and go on
            result[lane] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    failed = [lane for lane in ("audio", "frames") if "error" in result[lane]]
    if failed:
        result["align"] = {"skipped": f"{' and '.join(failed)} failed"}
    else:
        try:
            result["align"] = run_align(video_id, run_id, db_path=db_path)
        except Exception as e:  # noqa: BLE001
            result["align"] = {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    if len(failed) == 2:
        result["status"] = "failed"
    elif failed or "error" in result["align"]:
        result["status"] = "partial"
    else:
        result["status"] = "success"

    result["summary"] = summarize(result)
    run_row["finished_at"] = datetime.now()  # ② after: same run_id -> replaces row ①
    run_row["status"] = result["status"]
    run_row["metrics"] = json.dumps(
        {step: result[step] for step in ("audio", "frames", "align", "summary")},
        default=str,
    )
    save_run(db_path, run_row)
    return result
