"""Alignment (step ⑤): interval join audio_segments x slides -> aligned. Roadmap #5."""

import duckdb

from talk_align.config import WEAK_MATCH_RATIO


def align_video(
    conn: duckdb.DuckDBPyConnection, video_id: str
) -> tuple[list[dict], dict]:
    """Match each 'ok' audio segment of video_id to exactly one slide.

    Input:  an open connection holding audio_segments and slides, "lakehouse"
    Output: (rows, metrics); one row per segment:
            segment_id, slide_id, match_type, overlap_s, nearest_distance_s,
            align_gap_s, align_gap_ratio, slide_quality_flag
    Rule:   score = LEAST(ends) - GREATEST(starts); > 0 is the overlap, <= 0 is minus the
            distance. Highest score wins; tie -> later slide -> slide_id.
    Gap:    align_gap_s = segment length - overlap_s; ratio = gap / length
            (length > 0 is guaranteed: INVALID_TIME_RANGE rows never reach audio_segments)
    """
    sql = """
        WITH pairs AS (
            SELECT a.segment_id,
                   a.t_start             AS seg_t_start,
                   a.t_end - a.t_start   AS seg_len,
                   s.slide_id,
                   s.t_start             AS slide_t_start,
                   s.quality_flag        AS slide_quality_flag,
                   LEAST(a.t_end, s.t_end) - GREATEST(a.t_start, s.t_start) AS overlap
            FROM audio_segments a
            JOIN slides s
              ON a.video_id = s.video_id      -- every slide of the SAME video
            WHERE a.video_id = ?
              AND a.quality_flag = 'ok'       -- spec: quarantined audio is not aligned
                                              -- slides are NOT filtered: flag is carried below
        )
        SELECT segment_id,
               slide_id,
               CASE WHEN overlap > 0 THEN 'overlap' ELSE 'nearest' END   AS match_type,
               GREATEST(overlap, 0)                                      AS overlap_s,
               CASE WHEN overlap <= 0 THEN -overlap END                  AS nearest_distance_s,
               seg_len - GREATEST(overlap, 0)                            AS align_gap_s,
               (seg_len - GREATEST(overlap, 0)) / seg_len                AS align_gap_ratio,
               slide_quality_flag
        FROM pairs
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY segment_id
            ORDER BY overlap DESC, slide_t_start DESC, slide_id   -- raw score, full order
        ) = 1
        ORDER BY seg_t_start
    """
    rel = conn.execute(sql, [video_id])
    cols = [d[0] for d in rel.description]
    rows = [dict(zip(cols, r)) for r in rel.fetchall()]

    metrics = {
        "segments_aligned": len(rows),
        "nearest": sum(1 for r in rows if r["match_type"] == "nearest"),
        "weak": sum(1 for r in rows if r["align_gap_ratio"] > WEAK_MATCH_RATIO),
        "max_align_gap_s": max((r["align_gap_s"] for r in rows), default=0.0),
        "max_align_gap_ratio": max((r["align_gap_ratio"] for r in rows), default=0.0),
    }
    return rows, metrics
