"""DLQ replay (roadmap #7): run open DLQ rows through the lane rules again, without Whisper.

Each DLQ row kept its raw_payload, so a replay only re-validates it (audio), or cuts the
keyframe again and re-runs OCR on that one image (frame). Per row the outcome is:

    fixed          -> row upserted into the main table, DLQ row status "replayed"
    still failing  -> DLQ row stays "open", attempts + 1, reason/detail of this try
    no content now -> DLQ row status "ignored" (e.g. it is silence under today's rules)

Every outcome adds 1 to attempts; at MAX_REPLAY_ATTEMPTS the row is no longer picked
and the report says it needs a human.
"""

import json
from pathlib import Path

from talk_align import frame_lane
from talk_align.audio_lane import build_rows
from talk_align.config import FrameParams


def _after_try(
    dlq_row: dict, status: str, provenance: dict, new: dict | None = None
) -> dict:
    """The DLQ row after one replay: attempts + 1, new status, reason of this try."""
    row = {**dlq_row, "attempts": dlq_row["attempts"] + 1, "status": status}
    if new is not None:  # still failing: keep why it failed THIS time
        row["reason_code"] = new["reason_code"]
        row["error_detail"] = new["error_detail"]
        row["stage"] = new["stage"]
        row["failed_at"] = provenance["processed_at"]
    return row


def _empty_metrics(n: int) -> dict:
    return {"replayed_in": n, "fixed": 0, "still_failing": 0, "ignored": 0}


def replay_audio_rows(
    dlq_rows: list[dict], video_id: str, duration: float, provenance: dict
) -> tuple[list[dict], list[dict], dict]:
    """Re-validate open dlq_audio rows from raw_payload. Pure function.

    Input:  rows from store.open_dlq_rows(..., "dlq_audio", ...), video duration (s),
            provenance of this replay (code of today)
    Output: (fixed audio_segments rows, updated dlq_audio rows, metrics)
    """
    fixed: list[dict] = []
    updated: list[dict] = []
    m = _empty_metrics(len(dlq_rows))
    for dlq_row in dlq_rows:
        item = json.loads(dlq_row["raw_payload"])
        ok, dlq, _ = build_rows(video_id, [item], duration, provenance)
        if ok:
            fixed += ok
            updated.append(_after_try(dlq_row, "replayed", provenance))
            m["fixed"] += 1
        elif dlq:
            updated.append(_after_try(dlq_row, "open", provenance, dlq[0]))
            m["still_failing"] += 1
        else:  # dropped as silence by today's rules: nothing to put back
            updated.append(_after_try(dlq_row, "ignored", provenance))
            m["ignored"] += 1
    return fixed, updated, m


def replay_frame_rows(
    dlq_rows: list[dict],
    video_id: str,
    video_path: Path,
    params: FrameParams,
    provenance: dict,
) -> tuple[list[dict], list[dict], dict]:
    """Retry open dlq_frame rows. OCR-stage errors cut the keyframe again from the video.

    Input:  rows from store.open_dlq_rows(..., "dlq_frame", ...), the source video
    Output: (fixed slides rows, updated dlq_frame rows, metrics)
    """
    fixed: list[dict] = []
    updated: list[dict] = []
    m = _empty_metrics(len(dlq_rows))
    for dlq_row in dlq_rows:
        payload = json.loads(dlq_row["raw_payload"])
        if "slide" not in payload:  # written before slide spans were stored
            new = {
                "reason_code": dlq_row["reason_code"],
                "stage": "replay",
                "error_detail": "raw_payload has no slide span; run the frames lane again",
            }
            updated.append(_after_try(dlq_row, "open", provenance, new))
            m["still_failing"] += 1
            continue

        slide = {
            **payload["slide"],
            "keyframe_path": Path(payload["slide"]["keyframe_path"]),
        }
        if dlq_row["stage"] == "ocr":
            # CORRUPT_KEYFRAME / OCR_FAILED: the image is the problem, make a new one
            frame_lane.extract_keyframe(video_path, slide)
            raw = frame_lane.ocr_keyframes([slide])[slide["slide_id"]]
        else:  # validate: the OCR output itself is fine to re-check
            raw = payload["ocr"]

        ok, dlq, _ = frame_lane.build_slide_rows(
            video_id, [slide], {slide["slide_id"]: raw}, params, provenance
        )
        if ok:
            fixed += ok
            updated.append(_after_try(dlq_row, "replayed", provenance))
            m["fixed"] += 1
        else:
            updated.append(_after_try(dlq_row, "open", provenance, dlq[0]))
            m["still_failing"] += 1
    return fixed, updated, m
