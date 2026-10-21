"""Run report (step ⑥): roll lane metrics up into the numbers that answer "was this run ok?",
then render them as Markdown with warnings that say which number is bad.

Pure functions: no database, no files (store.latest_run / dlq_reason_counts read the DB).
Roadmap #6.
"""

from talk_align.config import GAP_WARN_P95_S, NEAREST_WARN_PCT, WEAK_MATCH_RATIO


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


def warnings(run: dict, dlq_by_reason: dict[str, int]) -> list[str]:
    """Plain-language warnings for one runs row; [] when nothing needs a look.

    Input:  a runs row from store.latest_run (metrics already parsed), and
            store.dlq_reason_counts, e.g. {"dlq_audio/EMPTY_TEXT": 2}
    Output: ["audio failed: RuntimeError: ...", "code_version abc-dirty: ...", ...]
    """
    found: list[str] = []
    metrics = run.get("metrics") or {}
    summary = metrics.get("summary") or {}

    if run["status"] == "running":
        found.append(
            "run never finished (status running): killed, or still in progress"
        )
    for step in ("audio", "frames", "align"):
        m = metrics.get(step) or {}
        if "error" in m:
            found.append(f"{step} failed: {m['error']}")
        elif "skipped" in m:
            found.append(f"{step} skipped: {m['skipped']}")

    # DLQ holds processing bugs only (spec §3), so any row is worth a look
    for lane, key in (("audio", "dlq_rate_audio"), ("frames", "dlq_rate_frame")):
        rate = summary.get(key)
        if rate:
            found.append(f"{lane} DLQ rate {rate:.1%}: pipeline bugs, not noisy data")
    if dlq_by_reason:
        listed = ", ".join(f"{k} {n}" for k, n in sorted(dlq_by_reason.items()))
        found.append(f"DLQ rows by reason: {listed}")

    nearest_pct = summary.get("nearest_pct")
    if nearest_pct is not None and nearest_pct > NEAREST_WARN_PCT:
        found.append(
            f"nearest matches {nearest_pct:.1f}% > {NEAREST_WARN_PCT:g}%: "
            "frame lane may have lost slides"
        )
    weak = summary.get("weak")
    if weak:
        found.append(f"{weak} weak matches (align_gap_ratio > {WEAK_MATCH_RATIO:g})")
    p95_gap = summary.get("p95_align_gap_s")
    if p95_gap is not None and p95_gap > GAP_WARN_P95_S:
        found.append(
            f"p95 align_gap_s {p95_gap:.1f} s > {GAP_WARN_P95_S:g} s: "
            "slide boundaries look shifted"
        )

    if run.get("code_version", "").endswith("-dirty"):
        found.append(
            f"code_version {run['code_version']} is dirty: rows come from uncommitted code"
        )
    return found


def _fmt(value, spec: str = "") -> str:
    """'-' for a value that was not measured (None), so it never reads as 0."""
    return "-" if value is None else format(value, spec)


def _duration(run: dict) -> str:
    if run.get("finished_at") is None:
        return "not finished"
    seconds = int((run["finished_at"] - run["started_at"]).total_seconds())
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _lane_line(metrics: dict, step: str, body: str) -> str:
    m = metrics.get(step) or {}
    if "error" in m:
        return f"failed: {m['error']}"
    if "skipped" in m:
        return f"skipped: {m['skipped']}"
    return body


def render_report(run: dict, dlq_by_reason: dict[str, int]) -> str:
    """Markdown report of one runs row: header, one section per step, warnings."""
    m = run.get("metrics") or {}
    s = m.get("summary") or {}
    a, f, al = m.get("audio") or {}, m.get("frames") or {}, m.get("align") or {}

    audio = (
        f"segments {_fmt(a.get('segments_in'))} · ok {_fmt(a.get('ok'))} · "
        f"dlq {_fmt(a.get('dlq'))} ({_fmt(s.get('dlq_rate_audio'), '.1%')}) · "
        f"silence dropped {_fmt(a.get('dropped_silence'))} · "
        f"t_end clipped {_fmt(a.get('clipped_t_end'))} · "
        f"quarantined {_fmt(a.get('quarantined'))}"
    )
    frames = (
        f"slides {_fmt(f.get('slides_in'))} · dlq {_fmt(f.get('dlq'))} "
        f"({_fmt(s.get('dlq_rate_frame'), '.1%')}) · "
        f"merged transitions {_fmt(f.get('merged_transitions'))} · "
        f"quarantined {_fmt(f.get('quarantined'))}"
    )
    align = (
        f"aligned {_fmt(al.get('segments_aligned'))} · nearest {_fmt(al.get('nearest'))} "
        f"({_fmt(s.get('nearest_pct'), '.1f')}%) · weak {_fmt(al.get('weak'))}\n"
        f"align_gap_s p50 {_fmt(s.get('p50_align_gap_s'), '.2f')} · "
        f"p95 {_fmt(s.get('p95_align_gap_s'), '.2f')}  |  "
        f"align_gap_ratio p50 {_fmt(s.get('p50_align_gap_ratio'), '.2f')} · "
        f"p95 {_fmt(s.get('p95_align_gap_ratio'), '.2f')}"
    )
    found = warnings(run, dlq_by_reason)
    warning_lines = "\n".join(f"- {w}" for w in found) if found else "- none"

    return (
        f"# Run report: {run['video_id']}\n\n"
        f"- run_id: {run['run_id']}\n"
        f"- status: {run['status']} · {_duration(run)} · code {run['code_version']}\n"
        f"- started: {run['started_at']:%Y-%m-%d %H:%M:%S}\n\n"
        f"## Audio\n{_lane_line(m, 'audio', audio)}\n\n"
        f"## Frames\n{_lane_line(m, 'frames', frames)}\n\n"
        f"## Align\n{_lane_line(m, 'align', align)}\n\n"
        f"## Warnings\n{warning_lines}\n"
    )
