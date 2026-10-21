"""Command-line entry point: uv run talk-align <command> (spec §5).

The I/O shell: read arguments, check inputs, call pipeline steps, print results.
"""

import argparse
import sys
from pathlib import Path

from talk_align import pipeline, store
from talk_align.config import DB_PATH, OUT_DIR


def resolve_video(parser: argparse.ArgumentParser, video_path: Path) -> Path:
    if not video_path.exists():
        parser.error(f"File not exist: {video_path}")
    return video_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talk-align", description="Talk Align CLI")
    # Subcommands available
    sub = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    audio = sub.add_parser("audio", help="Process audio from video")
    audio.add_argument("video", type=Path, help="Path to input video file")

    frames = sub.add_parser("frames", help="Extract one keyframe per slide and OCR it")
    frames.add_argument("video", type=Path, help="Path to input video file")

    align = sub.add_parser("align", help="Match each audio segment to one slide")
    align.add_argument("video_id", help="ID of the video to align")

    export = sub.add_parser("export", help="Export processed data as parquet files")
    export.add_argument("video_id", help="ID of the video to export")
    export.add_argument("-o", "--out", type=Path, help="Output directory")

    run = sub.add_parser("run", help="Run audio, frames and align with one run_id")
    run.add_argument("video", type=Path, help="Path to input video file")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audio":
        video_path = resolve_video(parser, args.video)
        print(pipeline.run_audio(video_path, pipeline.new_run_id()))

    elif args.command == "frames":
        video_path = resolve_video(parser, args.video)
        print(pipeline.run_frames(video_path, pipeline.new_run_id()))

    elif args.command == "align":
        try:
            print(pipeline.run_align(args.video_id, pipeline.new_run_id()))
        except pipeline.MissingInputError as e:
            parser.error(str(e))

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
            conn.close()
            parser.error(f"No rows for video_id: {video_id}")
        # latest run; run_id breaks a tie so the choice is always the same
        run_id = max(runs, key=lambda r: (r[1], r[0]))[0]
        out_dir = args.out or OUT_DIR / run_id
        paths = store.export_parquet(conn, video_id, out_dir)
        conn.close()
        print(paths)
    elif args.command == "run":
        video_path = resolve_video(parser, args.video)
        result = pipeline.run_all(video_path, pipeline.new_run_id())
        print(f"run_id: {result['run_id']}  status: {result['status']}")
        for step in ("audio", "frames", "align"):
            r = result[step]
            if "error" in r:
                print(f"  {step}: ERROR {r['error']}")
                print(
                    r["traceback"], file=sys.stderr
                )  # full traceback only when it failed
            elif "skipped" in r:
                print(f"  {step}: skipped ({r['skipped']})")
            else:
                print(f"  {step}: {r}")
        if "summary" in result:
            print(f"  summary: {result['summary']}")
        return 0 if result["status"] == "success" else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
