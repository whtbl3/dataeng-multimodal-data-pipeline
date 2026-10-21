"""Command-line entry point: uv run talk-align <command> (spec 5). Roadmap #1."""

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import duckdb

from talk_align import store
from talk_align.align import align_video
from talk_align.audio_lane import build_rows, transcribe
from talk_align.config import BASE_DIR, DB_PATH, OUT_DIR, AudioParams, FrameParams
from talk_align.frame_lane import (
    build_slide_rows,
    build_slide_spans,
    detect_scene_times,
    extract_keyframes,
    ocr_keyframes,
    plan_keyframes,
)
from talk_align.media import probe_duration

# def save_fixture_temp(video_id: str, raw_segments: list[dict], duration: float):
#     fixture_dir =Path("tests/fixtures")
#     fixture_dir.mkdir(parents=True, exist_ok=True)
#     fixture_file = fixture_dir / f"whisper_{video_id}.json"

#     data = {
#         "video_id": video_id,
#         "duration": duration,
#         "segments": raw_segments[:5] # First 5 segments
#     }

#     with open(fixture_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     print(f"Saved temp fixture at {fixture_file}")


def file_sha256(video_path: Path) -> str:
    with video_path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    return f"sha256:{digest.hexdigest()}"


def git_code_version() -> str:
    cmd_commit = ["git", "rev-parse", "--short", "HEAD"]
    commit = subprocess.run(
        cmd_commit,
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    cmd_list_changes = ["git", "status", "--porcelain", "."]
    changes = subprocess.run(
        cmd_list_changes,
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return f"{commit}-dirty" if changes else commit


def build_provenance(video_path: Path, model: str, model_version: str, params) -> dict:
    """Who made these rows: model, its version, every param that changes output, code, run.

    params: any dataclass (AudioParams, FrameParams); asdict keeps provenance and the run in sync.
    """
    return {
        "model": model,
        "model_version": model_version,
        "params": json.dumps(asdict(params), sort_keys=True),
        "source_hash": file_sha256(video_path),
        "run_id": str(uuid.uuid4()),
        "code_version": git_code_version(),
        "processed_at": datetime.now(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="talk-align", description="Talk Align CLI")
    # Subcomamnd available
    subparser = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )
    # Subcommand: Audio
    audio_parser = subparser.add_parser("audio", help="Process audio from video")
    audio_parser.add_argument("video", type=Path, help="Path to input video file")
    # Subcommand: Export
    export_parser = subparser.add_parser(
        "export", help="Export processed data as parquet file"
    )
    export_parser.add_argument("video_id", help="ID of the video to export")
    export_parser.add_argument(
        "-o", "--out", type=Path, help="Output directory or file path"
    )
    # Subcommand: Frames
    frame_parser = subparser.add_parser("frames", help="Extract one keyframe per slide")
    frame_parser.add_argument("video", type=Path, help="Path to input video file")
    # Subcommand: Align
    align_parser = subparser.add_parser("align", help="Match each audio segment to one slide")
    align_parser.add_argument("video_id", help="ID of the video to align")

    args = parser.parse_args(argv)
    if args.command == "audio":
        # save_fixture_temp(video_id=video_id, raw_segments=raw_segments, duration=duration)
        video_path: Path = args.video
        if not video_path.exists():
            parser.error(f"File not exist: {video_path}")
        video_id: str = video_path.stem

        params = AudioParams()
        provenance = build_provenance(
            video_path, "faster-whisper", params.model, params
        )

        raw_segments, duration = transcribe(video_path, params)

        ok_rows, dlq_rows, metrics = build_rows(
            video_id, raw_segments, duration, provenance
        )

        conn: duckdb.DuckDBPyConnection = store.connect(db_path=DB_PATH)
        # store.insert_rows(conn, "audio_segments", rows)
        # store.overwrite_video(
        #     conn, table="audio_segments", video_id=video_id, rows=rows
        # )
        store.overwrite_video_tables(
            conn, video_id, {"audio_segments": ok_rows, "dlq_audio": dlq_rows}
        )
        conn.close()
        print(metrics)

        # print(len(rows), rows[0]["segment_id"], rows[0]["provenance"]["run_id"])
    elif args.command == "export":
        conn = store.connect(db_path=DB_PATH)
        video_id = args.video_id
        runs = conn.execute(
            """
            SELECT DISTINCT provenance.run_id, provenance.processed_at FROM audio_segments WHERE video_id = $v
            UNION
            SELECT DISTINCT provenance.run_id, provenance.processed_at FROM dlq_audio WHERE video_id = $v
            UNION
            SELECT DISTINCT provenance.run_id, provenance.processed_at FROM slides WHERE video_id = $v
            UNION
            SELECT DISTINCT provenance.run_id, provenance.processed_at FROM dlq_frame WHERE video_id = $v
            """,
            {"v": video_id},
        ).fetchall()  # [(run_id, processed_at), ...]
        if not runs:
            parser.error(f"No rows for video_id: {video_id}")
        # latest run; run_id breaks a tie so the choice is always the same
        run_id = max(runs, key=lambda r: (r[1], r[0]))[0]
        out_dir = args.out or OUT_DIR / run_id
        output_path = store.export_parquet(conn, video_id, out_dir)
        conn.close()
        print(output_path)
    elif args.command == "frames":
        video_path: Path = args.video
        if not video_path.exists():
            parser.error(f"File not exist: {video_path}")
        video_id: str = video_path.stem
        params = FrameParams()
        duration = probe_duration(video_path)
        times = detect_scene_times(video_path, params)
        spans, span_metrics = build_slide_spans(times, duration, params.min_duration)
        planned = plan_keyframes(video_id, spans)
        extract_keyframes(video_path, video_id, planned)
        raw_ocr = ocr_keyframes(planned)
        provenance = build_provenance(
            video_path, "rapidocr_onnxruntime", version("rapidocr_onnxruntime"), params
        )
        ok_rows, dlq_rows, metrics = build_slide_rows(
            video_id, planned, raw_ocr, params, provenance
        )

        # frame lane only: audio_segments / dlq_audio of this video stay untouched (spec §3)
        conn = store.connect(db_path=DB_PATH)
        store.overwrite_video_tables(
            conn, video_id, {"slides": ok_rows, "dlq_frame": dlq_rows}
        )
        conn.close()

        print({**span_metrics, **metrics})
    
    elif args.command == "align":
        conn = store.connect(db_path=DB_PATH)
        rows = align_video(conn, args.video_id)
        conn.close()
        print(len(rows))

    return 0


if __name__ == "__main__":
    sys.exit(main())
