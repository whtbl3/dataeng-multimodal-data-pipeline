from talk_align import store
from talk_align.align import align_video
from talk_align.audio_lane import build_rows
from talk_align.config import FrameParams
from talk_align.frame_lane import build_slide_rows, plan_keyframes
from tests.test_audio_lane import PROVENANCE, make_seg
from tests.test_frame_lane import tb


def put_audio(con, video_id, spans, duration=1000.0):  # spans = [(t_start, t_end), ...]
    segs = [make_seg(start=s, end=e) for s, e in spans]
    ok, _, _ = build_rows(video_id, segs, duration, PROVENANCE)
    store.overwrite_video_tables(con, video_id, {"audio_segments": ok, "dlq_audio": []})


def put_slides(con, video_id, spans):  # spans = [(t_start, t_end), ...]
    planned = plan_keyframes(video_id, [{"t_start": s, "t_end": e} for s, e in spans])
    ocr = {
        p["slide_id"]: {"image_h": 720, "boxes": [tb("Slide text", 0.9)]}
        for p in planned
    }
    ok, _, _ = build_slide_rows(video_id, planned, ocr, FrameParams(), PROVENANCE)
    store.overwrite_video_tables(con, video_id, {"slides": ok, "dlq_frame": []})


def align(spans_audio, spans_slides):
    """Build a tiny in-memory video 'v' and align it."""
    con = store.connect(":memory:")
    put_audio(con, "v", spans_audio)
    put_slides(con, "v", spans_slides)
    rows, metrics = align_video(con, "v")
    con.close()
    return rows, metrics


# STEP 1: helpers


def test_helpers_put_rows():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 3.0), (5.0, 8.0)])
    put_slides(con, "v", [(0.0, 4.0), (4.0, 10.0)])
    assert con.execute("SELECT count(*) FROM audio_segments").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM slides").fetchone()[0] == 2
    con.close()


# STEP 2: overlap rules


def test_touching_intervals_do_not_overlap():
    rows, _ = align([(6.0, 10.0)], [(0.0, 6.0), (6.0, 14.0)])
    # [0,6) only touches at 6: score 6 - 6 = 0 -> loses to [6,14) with 4
    assert [(r["slide_id"], r["match_type"]) for r in rows] == [
        ("v_s_00006000", "overlap")
    ]


def test_other_videos_are_not_joined():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 3.0)])
    put_slides(con, "v", [(0.0, 10.0)])
    put_slides(con, "w", [(0.0, 10.0)])  # same times, other video
    rows, _ = align_video(con, "v")
    con.close()
    assert [r["slide_id"] for r in rows] == ["v_s_00000000"]


# STEP 3: one slide per segment, tie-break


def test_crossing_segment_takes_the_bigger_overlap():
    rows, _ = align([(5.0, 13.0)], [(0.0, 6.0), (6.0, 14.0)])
    # [0,6): 6 - 5 = 1  ·  [6,14): 13 - 6 = 7  -> 7 wins
    assert [(r["slide_id"], r["overlap_s"]) for r in rows] == [("v_s_00006000", 7.0)]


def test_tie_goes_to_the_later_slide():  # spec / learning-notes §8 example
    rows, _ = align([(745.0, 755.0)], [(721.0, 750.0), (750.0, 780.0)])
    assert [r["slide_id"] for r in rows] == ["v_s_00750000"]


def test_each_segment_appears_once():
    rows, _ = align(
        [(1.0, 5.0), (5.0, 13.0), (13.0, 20.0)], [(0.0, 6.0), (6.0, 14.0), (14.0, 30.0)]
    )
    ids = [r["segment_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 3


# STEP 4: nearest


def test_segment_in_a_gap_goes_to_the_nearest_slide():
    rows, _ = align(
        [(12.0, 14.0)], [(0.0, 10.0), (20.0, 30.0)]
    )  # slide [10, 20) is missing
    r = rows[0]
    # [0,10): 10 - 12 = -2 (2 s away)  ·  [20,30): 14 - 20 = -6 (6 s away)
    assert (
        r["slide_id"],
        r["match_type"],
        r["overlap_s"],
        r["nearest_distance_s"],
    ) == (
        "v_s_00000000",
        "nearest",
        0.0,
        2.0,
    )


def test_nearest_tie_goes_to_the_later_slide():  # 4 s on both sides
    rows, _ = align([(14.0, 16.0)], [(0.0, 10.0), (20.0, 30.0)])
    assert rows[0]["slide_id"] == "v_s_00020000"


def test_overlap_rows_have_no_nearest_distance():
    rows, _ = align([(1.0, 3.0)], [(0.0, 10.0)])
    assert rows[0]["match_type"] == "overlap" and rows[0]["nearest_distance_s"] is None


def test_every_ok_segment_is_aligned():  # invariant: no segment disappears
    rows, _ = align(
        [(1.0, 3.0), (12.0, 14.0), (25.0, 27.0)], [(0.0, 10.0), (20.0, 30.0)]
    )
    assert [r["match_type"] for r in rows] == ["overlap", "nearest", "overlap"]


# STEP 5: gap, weak match, slide quality flag


def test_gap_for_the_spec_example():  # [745, 755) -> later slide, overlap 5
    rows, m = align([(745.0, 755.0)], [(721.0, 750.0), (750.0, 780.0)])
    # length 10, overlap 5 -> gap 5, ratio 5 / 10 = 0.5
    assert (rows[0]["align_gap_s"], rows[0]["align_gap_ratio"]) == (5.0, 0.5)
    assert m["weak"] == 0  # weak is "> 0.5", so exactly 0.5 is NOT weak


def test_nearest_row_is_a_weak_match():
    rows, m = align([(12.0, 14.0)], [(0.0, 10.0), (20.0, 30.0)])
    # overlap_s 0 -> gap = whole length 2 -> ratio 1.0
    assert rows[0]["align_gap_ratio"] == 1.0
    assert (m["nearest"], m["weak"]) == (1, 1)


def test_slide_quality_flag_is_carried_not_filtered():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 3.0)])
    put_slides(con, "v", [(0.0, 10.0)])
    con.execute("UPDATE slides SET quality_flag = 'quarantine'")
    rows, _ = align_video(con, "v")
    con.close()
    # quarantined slide is still matched (no fake nearest), the flag is just copied
    assert len(rows) == 1 and rows[0]["slide_quality_flag"] == "quarantine"


def test_aligned_rows_are_written_and_replaced():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 3.0), (5.0, 8.0)])
    put_slides(con, "v", [(0.0, 4.0), (4.0, 10.0)])
    rows, _ = align_video(con, "v")
    rows = [{**r, "video_id": "v", "run_id": "run-1"} for r in rows]
    store.overwrite_video_tables(con, "v", {"aligned": rows})

    put_slides(con, "v", [(0.0, 10.0)])  # frame lane runs again: only 1 slide now
    rows, _ = align_video(con, "v")
    rows = [{**r, "video_id": "v", "run_id": "run-2"} for r in rows]
    store.overwrite_video_tables(con, "v", {"aligned": rows})

    got = con.execute(
        "SELECT count(DISTINCT slide_id), count(*), count(DISTINCT run_id) FROM aligned"
    ).fetchone()
    con.close()
    # both segments now point to v_s_00000000; nothing from run-1 survives
    assert got == (1, 2, 1)
