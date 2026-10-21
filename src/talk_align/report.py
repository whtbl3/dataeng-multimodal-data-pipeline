"""Run report (step ⑥): metrics of a run, written to runs and report.md. Roadmap #6."""

"""Run report (step ⑥): roll lane metrics up into the numbers that answer "was this run ok?".

Pure functions: no database, no files. Roadmap #6.
"""


def _lane(result: dict, name: str) -> dict | None:
    """Metrics of one step, or None when it failed or was skipped (nothing was measured)."""
    m = result.get(name) or {}
    if not m or "error" in m or "skipped" in m:
        return None
    return m


def _rate(part: int, whole: int) -> float:
    return part / whole if whole > 0 else 0.0  # nothing to process -> nothing failed


def summarize(result: dict) -> dict:
    """Roll up the output of run_all.

    Input:  {"audio": {...}, "frames": {...}, "align": {...}} (a step may hold error/skipped)
    Output: {"dlq_rate_audio": 0.038, "dlq_rate_frame": 0.083, "nearest_pct": 4.0, ...}
            None = could not be measured (lane failed or skipped), never 0
    """
    audio = _lane(result, "audio")
    frames = _lane(result, "frames")
    align = _lane(result, "align")

    summary: dict = {
        "dlq_rate_audio": None,
        "dropped_silence": None,
        "clipped_t_end": None,
        "quarantined_audio": None,
        "dlq_rate_frame": None,
        "merged_transitions": None,
        "segments_aligned": None,
        "nearest_pct": None,
        "weak": None,
        "p50_align_gap_s": None,
        "p95_align_gap_s": None,
        "p50_align_gap_ratio": None,
        "p95_align_gap_ratio": None,
    }

    if audio is not None:
        seg_in, dlq = audio.get("segments_in"), audio.get("dlq")
        if seg_in is not None and dlq is not None:
            content = seg_in - audio.get("dropped_silence", 0)  # silence cannot fail
            summary["dlq_rate_audio"] = _rate(dlq, content)
        summary["dropped_silence"] = audio.get("dropped_silence")
        summary["clipped_t_end"] = audio.get("clipped_t_end")
        summary["quarantined_audio"] = audio.get("quarantined")

    if frames is not None:
        slides_in, dlq = frames.get("slides_in"), frames.get("dlq")
        if slides_in is not None and dlq is not None:
            summary["dlq_rate_frame"] = _rate(dlq, slides_in)
        summary["merged_transitions"] = frames.get("merged_transitions")

    if align is not None:
        n, nearest = align.get("segments_aligned"), align.get("nearest")
        summary["segments_aligned"] = n
        if n is not None and nearest is not None:
            summary["nearest_pct"] = _rate(nearest, n) * 100
        summary["weak"] = align.get("weak")
        for key in (
            "p50_align_gap_s",
            "p95_align_gap_s",
            "p50_align_gap_ratio",
            "p95_align_gap_ratio",
        ):
            summary[key] = align.get(key)

    return summary
