"""Alignment (step ⑤): interval join audio_segments x slides -> aligned. Roadmap #5."""
import duckdb


def align_video(conn: duckdb.DuckDBPyConnection, video_id: str) -> list[dict]:
    """Match each 'ok' audio segment of video_id to exactly one slide.

    Input:  an open connection holding audio_segments and slides, "lakehouse"
    Output: one dict per segment, [{"segment_id", "slide_id", "overlap"}, ...]
            (78 rows for lakehouse; 5 segments cross a slide change and keep one slide)
    Rule:   biggest overlap; tie -> the LATER slide; still tied -> smaller slide_id
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
              ON a.video_id = s.video_id
             AND a.t_start < s.t_end
             AND a.t_end > s.t_start
            WHERE a.video_id = ?
              AND a.quality_flag = 'ok'
        )
        SELECT segment_id, slide_id, overlap
        FROM pairs
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY segment_id
            ORDER BY overlap DESC, slide_t_start DESC, slide_id   -- full order: never a tie
        ) = 1
        ORDER BY seg_t_start                                      -- same output order every run
    """
    rel = conn.execute(sql, [video_id])
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, r)) for r in rel.fetchall()]