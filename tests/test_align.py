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
    ocr = {p["slide_id"]: {"image_h": 720, "boxes": [tb("Slide text", 0.9)]} for p in planned}
    ok, _, _ = build_slide_rows(video_id, planned, ocr, FrameParams(), PROVENANCE)
    store.overwrite_video_tables(con, video_id, {"slides": ok, "dlq_frame": []})


def test_helpers_put_rows():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 3.0), (5.0, 8.0)])
    put_slides(con, "v", [(0.0, 4.0), (4.0, 10.0)])
    assert con.execute("SELECT count(*) FROM audio_segments").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM slides").fetchone()[0] == 2
    con.close()

# def test_overlap_pairs_and_amounts():
#     con = store.connect(":memory:")
#     put_audio(con, "v", [(5.0, 13.0)])
#     put_slides(con, "v", [(0.0, 6.0), (6.0, 14.0), (14.0, 20.0)])
#     pairs = align_video(con, "v")
#     con.close()
#     assert sorted((p["slide_id"], p["overlap"]) for p in pairs) == [
#         ("v_s_00000000", 1.0),   # LEAST(13, 6) − GREATEST(5, 0)  = 6 − 5  = 1
#         ("v_s_00006000", 7.0),   # LEAST(13, 14) − GREATEST(5, 6) = 13 − 6 = 7
#     ]                            # [14, 20): 13 > 14 sai → không chồng, không có dòng


def test_touching_intervals_do_not_overlap():
    con = store.connect(":memory:")
    put_audio(con, "v", [(6.0, 10.0)])
    put_slides(con, "v", [(0.0, 6.0), (6.0, 14.0)])
    pairs = align_video(con, "v")
    con.close()
    # [0, 6) chỉ CHẠM câu ở biên 6: 6 < 6 sai → không chồng
    assert [p["slide_id"] for p in pairs] == ["v_s_00006000"]


def test_other_videos_are_not_joined():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 3.0)])
    put_slides(con, "v", [(0.0, 10.0)])
    put_slides(con, "w", [(0.0, 10.0)])   # cùng thời gian nhưng của video khác
    pairs = align_video(con, "v")
    con.close()
    assert len(pairs) == 1                # thiếu a.video_id = s.video_id thì ra 2

def test_crossing_segment_takes_the_bigger_overlap():
    con = store.connect(":memory:")
    put_audio(con, "v", [(5.0, 13.0)])
    put_slides(con, "v", [(0.0, 6.0), (6.0, 14.0)])
    rows = align_video(con, "v")
    con.close()
    # [0,6): 6−5 = 1   ·   [6,14): 13−6 = 7   → chọn 7
    assert [(r["slide_id"], r["overlap"]) for r in rows] == [("v_s_00006000", 7.0)]


def test_tie_goes_to_the_later_slide():  # đúng ví dụ spec / learning-notes §8
    con = store.connect(":memory:")
    put_audio(con, "v", [(745.0, 755.0)])
    put_slides(con, "v", [(721.0, 750.0), (750.0, 780.0)])
    rows = align_video(con, "v")
    con.close()
    # [721,750): 750−745 = 5   ·   [750,780): 755−750 = 5   → hòa → slide bắt đầu SAU
    assert [r["slide_id"] for r in rows] == ["v_s_00750000"]


def test_each_segment_appears_once():
    con = store.connect(":memory:")
    put_audio(con, "v", [(1.0, 5.0), (5.0, 13.0), (13.0, 20.0)])
    put_slides(con, "v", [(0.0, 6.0), (6.0, 14.0), (14.0, 30.0)])
    ids = [r["segment_id"] for r in align_video(con, "v")]
    con.close()
    # [1,5) → 1 slide · [5,13) → 2 slide · [13,20) → 2 slide  ⇒ 5 cặp, 3 câu
    assert len(ids) == len(set(ids)) == 3