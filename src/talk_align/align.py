"""Alignment (step ⑤): interval join audio_segments x slides -> aligned. Roadmap #5."""

import duckdb


def align_video(conn: duckdb.DuckDBPyConnection, video_id: str) -> list[dict]:
    """Match each 'ok' audio segment of video_id to exactly one slide.

    Input:  an open connection holding audio_segments and slides, "lakehouse"
    Output: one dict per segment:
            {"segment_id", "slide_id", "match_type", "overlap_s", "nearest_distance_s"}
            match_type "overlap" -> overlap_s > 0, nearest_distance_s None
            match_type "nearest" -> overlap_s 0,   nearest_distance_s = gap in seconds
    Rule:   score = LEAST(ends) - GREATEST(starts); > 0 is the overlap, <= 0 is minus the
            distance. Highest score wins, so an overlapping slide always beats any other,
            and a segment in a gap takes the nearest slide. Tie -> later slide -> slide_id.
    """
    sql = """
        WITH pairs AS (
            SELECT a.segment_id,
                   a.t_start AS seg_t_start,
                   s.slide_id,
                   s.t_start AS slide_t_start,
                   LEAST(a.t_end, s.t_end) - GREATEST(a.t_start, s.t_start) AS overlap
            FROM audio_segments a
            JOIN slides s
              ON a.video_id = s.video_id      -- every slide of the SAME video, overlapping or not
            WHERE a.video_id = ?
              AND a.quality_flag = 'ok'       -- spec: quarantined audio is not aligned
        )
        SELECT segment_id,
               slide_id,
               CASE WHEN overlap > 0 THEN 'overlap' ELSE 'nearest' END AS match_type,
               GREATEST(overlap, 0)                                    AS overlap_s,
               CASE WHEN overlap <= 0 THEN -overlap END                AS nearest_distance_s
        FROM pairs
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY segment_id
            ORDER BY overlap DESC, slide_t_start DESC, slide_id   -- raw score, full order
        ) = 1
        ORDER BY seg_t_start
    """
    rel = conn.execute(sql, [video_id])
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, r)) for r in rel.fetchall()]
