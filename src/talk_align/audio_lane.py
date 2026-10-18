import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from faster_whisper import WhisperModel
from pydantic import ValidationError
from talk_align.config import QUARANTINE_THRESHOLD, SILENCE_THRESHOLD, TIME_TOLERANCE_S, WORK_DIR, AudioParams
from talk_align.contract import AudioSegmentRow, reason_code_for, make_error_detail

def transcribe(video_path: Path, params: AudioParams) -> tuple[list[dict], float]:
    work_dir = WORK_DIR / video_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "audio.wav"

    if not audio_path.exists():
        cmd_make_wav = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        subprocess.run(cmd_make_wav, check=True)

    cmd_get_duration = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result_duration = subprocess.run(
        cmd_get_duration, check=True, capture_output=True, text=True
    )
    duration = float(result_duration.stdout.strip())

    model = WhisperModel(
        model_size_or_path=params.model,
        device=params.device,
        compute_type=params.compute_type,
    )

    segments_generator, _ = model.transcribe(
        str(audio_path),
        beam_size=params.beam_size,
        language=params.language,
        temperature=params.temperature,
    )

    raw_segments = [asdict(s) for s in segments_generator]

    return raw_segments, duration


def quality_score(seg) -> float:
    # seg["avg_logprob"]: độ tự tin của Whisper về chữ nó nghe ra. Giá trị âm, càng gần 0 càng tự tin. exp() đổi nó về khoảng 0 đến 1.
    # 1 - seg["no_speech_prob"]: xác suất đoạn này có tiếng người nói.
    return math.exp(seg["avg_logprob"]) * (1 - seg["no_speech_prob"])


def quality_flag(score) -> str:
    label = "ok" if score >= QUARANTINE_THRESHOLD else "quarantine"
    return label

def make_segment_id(video_id: str, start: float) -> str:
    return f"{video_id}_a_{round(start * 1000):08d}"

def make_dlq_row(video_id, item, reason_code, error_detail, provenance) -> dict:
    record_id = make_segment_id(video_id, item["start"])
    dlq_dict = {
        "dlq_id": f"dlq_validate_{record_id}",
        "video_id": video_id,
        "record_id": record_id,
        "stage": "validate",
        "reason_code": reason_code,
        "error_detail": error_detail,
        "raw_payload": json.dumps(item, ensure_ascii=False),
        "provenance": provenance,
        "failed_at": provenance["processed_at"],
        "attempts": 0,
        "status": "open"
    }
    return dlq_dict


def build_rows(video_id: str, raw_segments: list[dict], duration: float, provenance: dict) -> tuple[list[dict], list[dict], dict]:
    ok_rows = []
    dlq_rows = []
    metrics = {"segments_in": len(raw_segments), "dropped_silence": 0, "clipped_t_end": 0}
    for item in raw_segments:
        if item["text"].strip() == "" and item["no_speech_prob"] >= SILENCE_THRESHOLD:
            metrics["dropped_silence"] += 1
            continue
        if item["text"].strip() == "":
            dlq_rows.append(make_dlq_row(
                video_id,
                item,
                "EMPTY_TEXT",
                f"text is empty but no_speech_prob={item['no_speech_prob']:.2f} < {SILENCE_THRESHOLD}",
                provenance
            ))
            continue
        if item["start"] >= item["end"]:
            dlq_rows.append(make_dlq_row(
                video_id,
                item,
                "INVALID_TIME_RANGE",
                f"{item['start']} >= {item["end"]}",
                provenance
            ))
            continue
        if item["start"] >= duration:
            dlq_rows.append(make_dlq_row(
                video_id, item, "TIME_OUT_OF_BOUNDS",
                f"t_start={item['start']} >= duration={duration}",
                provenance,
            ))
            continue
        t_end = item["end"]
        overshoot = t_end - duration                 # vượt bao nhiêu giây (âm = không vượt)
        if overshoot > TIME_TOLERANCE_S:             # vượt quá nhiều → DLQ
            dlq_rows.append(make_dlq_row(
                video_id, item, "TIME_OUT_OF_BOUNDS",
                f"t_end={t_end} exceeds duration={duration} by {overshoot:.3f}s > {TIME_TOLERANCE_S}s",
                provenance,
            ))
            continue

        if overshoot > 0:                            # vượt một chút → cắt, đếm
            t_end = duration
            metrics["clipped_t_end"] += 1
        score = quality_score(item)
        segment = {
            "raw_payload": json.dumps(item, ensure_ascii=False),
            "video_id": video_id,
            "modality": "audio",
            "text": item["text"].strip(),
            "t_start": item["start"],
            "t_end": t_end,
            "segment_id": make_segment_id(video_id, item["start"]),
            "quality_score": score,
            "quality_flag": quality_flag(score),
            "provenance": provenance,
        }
        try:
            AudioSegmentRow.model_validate(segment)
        except ValidationError as e:
            dlq_rows.append(
                make_dlq_row(
                    video_id,
                    item,
                    reason_code_for(e),
                    make_error_detail(e),
                    provenance
                )
            )
            continue
        ok_rows.append(segment)
    return ok_rows, dlq_rows, metrics