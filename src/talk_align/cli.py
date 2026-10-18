"""Command-line entry point: uv run talk-align <command> (spec 5). Roadmap #1."""

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb

from talk_align import store
from talk_align.audio_lane import build_rows, transcribe
from talk_align.config import BASE_DIR, DB_PATH, AudioParams

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
    cmd_commit = [
        "git",
        "rev-parse",
        "--short",
        "HEAD"
    ]
    commit = subprocess.run(
        cmd_commit,
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    cmd_list_changes = [
        "git",
        "status",
        "--porcelain",
        "."
    ]
    changes = subprocess.run(
        cmd_list_changes,
        cwd=BASE_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    
    return f"{commit}-dirty" if changes else commit

def build_provenance(video_path: Path, params: AudioParams) -> dict:
    provenance = {
        "model": "faster-whisper",
        "model_version": params.model,
        "params": json.dumps(asdict(params), sort_keys=True),
        "source_hash": file_sha256(video_path),
        "run_id": str(uuid.uuid4()),
        "code_version": git_code_version(),
        "processed_at": datetime.now()
    }
    return provenance

def main(argv=None):
    parser = argparse.ArgumentParser(prog="talk-align", description="Talk Align CLI")
    # Subcomamnd available
    subparser = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )
    # Subcommand: Audio
    audio_parser = subparser.add_parser("audio", help="Process audio from video")
    audio_parser.add_argument("video", type=Path, help="Path to input video file")

    # save_fixture_temp(video_id=video_id, raw_segments=raw_segments, duration=duration)

    args = parser.parse_args(argv)
    video_path: Path = args.video
    video_id: str = video_path.stem
    
    params = AudioParams()
    provenance = build_provenance(video_path, params)
    
    raw_segments, _duration = transcribe(video_path, AudioParams())

    rows = build_rows(video_id, raw_segments, provenance)

    conn: duckdb.DuckDBPyConnection = store.connect(db_path=DB_PATH)
    # store.insert_rows(conn, "audio_segments", rows)
    store.overwrite_video(conn, table="audio_segments", video_id=video_id, rows=rows)
    conn.close()

    # print(len(rows), rows[0]["segment_id"], rows[0]["provenance"]["run_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
