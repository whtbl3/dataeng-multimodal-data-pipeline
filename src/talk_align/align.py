"""Alignment (step ⑤): interval join audio_segments x slides -> aligned. Roadmap #5."""
import duckdb


def align_video(conn: duckdb.DuckDBPyConnection, video_id: str) -> list[dict]:
    """Return every (segment, slide) pair that overlaps in time.

    Input:  an open connection holding audio_segments and slides, "lakehouse"
    Output: [{"segment_id": "...", "slide_id": "...", "overlap": 7.0}, ...]
            (83 pairs for lakehouse: 5 segments cross a slide change)
    """
    sql = """
        SELECT a.segment_id,
               s.slide_id,
               -- end that comes FIRST minus start that comes LAST
               LEAST(a.t_end, s.t_end) - GREATEST(a.t_start, s.t_start) AS overlap
        FROM audio_segments a
        JOIN slides s
          ON a.video_id = s.video_id      -- never pair one video's audio with another's slides
         AND a.t_start < s.t_end          -- half-open [start, end): touching edges
         AND a.t_end > s.t_start          --   do not count as overlap
        WHERE a.video_id = ?
          AND a.quality_flag = 'ok'       -- spec: quarantined audio is not aligned
        ORDER BY a.t_start, s.t_start     -- same order every run
    """
    rel = conn.execute(sql, [video_id])
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, r)) for r in rel.fetchall()]