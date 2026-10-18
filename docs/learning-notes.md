# Learning notes: talk-align (Lab 1)
Date: 2026-10-16 · Nguồn: ghi chép của tác giả + review khi bootstrap

Tóm tắt kiến thức đã nạp khi chốt intent và spec. **[+review]** = bổ sung hoặc sửa trong
lúc review. Ví dụ dùng video thật `lakehouse.mp4` (344.561 s) và fixture
`tests/fixtures/whisper_talk01.json`. Chi tiết quy định nằm ở `docs/project-spec.md`;
file này giải thích *vì sao*.

## 0. Bức tranh tổng

```
lakehouse.mp4 ─┬─ ① audio lane: ffmpeg → faster-whisper → validate ─┐
               └─ ② frame lane: scene change + t=0 → OCR → validate ┤
                                                                    ▼
        ③ contract: ok → bảng chính · lỗi xử lý → dlq_* · rác → metric
        ④ DuckDB (partition overwrite theo video_id)
        ⑤ align: overlap + ROW_NUMBER → aligned
        ⑥ report + export Parquet
```
Hai lane độc lập. Model chỉ chạy ở ① ②; mọi bước sau đọc `raw_payload`, nên replay hay
đổi luật không cần chạy lại Whisper/OCR.

## 1. Data contract

Cột chung: `video_id, modality, t_start, t_end, quality_score, quality_flag, provenance`
+ id tất định.
- Khoảng thời gian nửa mở `[t_start, t_end)`.

## 2. Audio lane

### Id tất định
- `id` của Whisper chỉ là số thứ tự của một lần chạy, nên không dùng làm khóa.
- `segment_id = f"{video_id}_a_{round(t_start*1000):08d}"`.
- **[+review] `round`, không phải `int`:**
  ```
  16.15 * 1000  → 16149.999999999998
  int(...)      → 16149 → lakehouse_a_00016149  ✗
  round(...)    → 16150 → lakehouse_a_00016150  ✓
  ```
  Khoảng 1% mốc có 2 chữ số thập phân bị lỗi này. Cùng một segment có thể ra 2 id, phá
  idempotency. (Ví dụ `12.48` dùng lúc đầu là sai: `int(12.48*1000)` ra đúng `12480`.)

### quality_score
`exp(avg_logprob) × (1 − no_speech_prob)`, ngưỡng quarantine **0.3**.
```
câu tốt:   exp(-0.31) × 0.98 = 0.72  → ok
nhạc nền:  exp(-1.20) × 0.40 = 0.12  → quarantine (vẫn giữ trong bảng)
```

### Input → output (segment đầu của fixture)
```
input  (raw Whisper): {"id": 1, "start": 12.48, "end": 16.15,
                       "text": " Welcome to this talk...", "avg_logprob": -0.31,
                       "no_speech_prob": 0.02, ...}
output (audio_segments):
  segment_id    = "lakehouse_a_00012480"
  modality      = "audio"
  t_start/t_end = 12.48 / 16.15
  quality_score = 0.719, quality_flag = "ok"
  raw_payload   = <JSON gốc ở trên>
  provenance    = {model: "faster-whisper", model_version: "small", params: {...},
                   source_hash, code_version, run_id, processed_at}
```

### provenance
Mọi thứ làm output khác đi đều phải nằm trong provenance. Lọc reprocess:
`WHERE provenance.model_version='small' AND quality_score < 0.5`.
- Tên đúng là `compute_type`, không phải `compute_size`.
- `temperature=0`: mặc định faster-whisper thử lại với temperature ngẫu nhiên khi gặp
  đoạn khó nghe, nên hai lần chạy có thể ra chữ khác. Đặt 0 và ghi vào `params`.
- **[+review]** `language="en"` cố định, không auto-detect. `code_version` thêm `-dirty`
  khi working tree có thay đổi.
- **[+review]** Máy 2 core, 3 GB RAM, GTX 1650 → `small` trên CPU int8. `large-v3` dễ OOM;
  GPU cần CUDA/cuDNN trong WSL và cho kết quả khác CPU.

## 3. Frame lane

### Ranh giới slide
- Dùng scene change thay vì cắt mỗi 2 s: khoảng 15–20 ảnh thay vì 300, không phải OCR
  thừa rồi dedup.
- Ngưỡng `gt(scene,0.3)` để chịu được animation hoặc cảnh quay người nói.
- `t_end = LEAD(t_start)`; slide cuối thì `t_end` = duration (ffprobe).
- **[+review] Thêm keyframe t=0**, vì `gt(scene,0.3)` không bao giờ trả về frame đầu.
- Mốc cách mốc kế < 1.5 s là chuyển cảnh: gộp, đếm `merged_transitions`. Không vào DLQ
  (replay vẫn ra như cũ), không đánh DEGRADED (không mất slide nào).

```
input  (mốc ffmpeg + t=0): [0.0, 12.0, 300.0, 300.4, 330.2], duration = 344.561
       300.4 − 300.0 = 0.4 s < 1.5 → bỏ mốc 300.0
output (slides):  [0.0, 12.0)  [12.0, 300.4)  [300.4, 330.2)  [330.2, 344.561)
       merged_transitions = 1
```

### OCR và quality_score của slide
`rapidocr-onnxruntime`; mỗi keyframe cho danh sách box `(tọa độ, chữ, confidence)`.
- Bỏ box rác trước khi tính: dưới 3 ký tự, hoặc quá nhỏ (logo, số trang).
- Điểm = trung bình confidence **cân theo số ký tự**, vì điểm cần trả lời "bao nhiêu
  phần chữ đáng tin":
```
"Lakehouse Architecture"            22 ký tự  0.97
"• Object storage + table format"   31 ký tự  0.91
"• Iceberg / Deita"                 17 ký tự  0.58
min 0.58 (một box mờ kéo cả slide xuống) · trung vị 0.91 (giấu lỗi "Deita")
trung bình 0.82 · cân theo ký tự 0.85 ✓
```
- Lưu điểm từng box trong `raw_payload`: lọc nhanh bằng 0.85, khi soát vẫn thấy "Deita"
  chỉ có 0.58.
- Ngưỡng quarantine của slide: calibrate từ phân phối điểm ở lần chạy đầu.
- **[+review] Slide OCR không đọc ra chữ vẫn là nội dung thật** (slide chỉ có ảnh): giữ
  với `text=""`, điểm 0, `quarantine`. **Không có `BLURRY_IMAGE` trong DLQ**: xóa slide
  thì audio của đoạn đó ghép nhầm sang slide bên cạnh mà không ai báo lỗi.

## 4. Phân loại bản ghi xấu (quyết định trung tâm)

```
Nội dung có thật không?
├─ không (im lặng, chuyển cảnh)        → bỏ, đếm metric
└─ có → xử lý có lỗi không?
         ├─ có (crash, file hỏng, sai contract) → DLQ + reason_code, sửa rồi replay
         └─ không, chỉ chất lượng kém           → giữ, quality_flag = quarantine
```

| Input | Kết quả |
|---|---|
| `text=""`, `no_speech_prob=0.9` | Bỏ, `dropped_silence += 1` |
| Slide giả `[300.0, 300.4)` | Gộp, `merged_transitions += 1` |
| Nhạc nền, điểm 0.12 | Giữ, `quarantine` |
| Slide chỉ có ảnh **[+review]** | Giữ, `text=""`, `quarantine` |
| Keyframe hỏng làm OCR crash | DLQ `CORRUPT_KEYFRAME`; replay: cắt lại keyframe từ video. **Không được bỏ** |
| `t_end <= t_start` | DLQ `INVALID_TIME_RANGE` |

Nhờ vậy **DLQ rate chỉ đo bug của pipeline**, không đo độ ồn của dữ liệu.

## 5. Reason codes

Viết hoa, mô tả lỗi chứ không mô tả cách sửa.

| Input | Kết quả |
|---|---|
| `t_start=20.0, t_end=19.5` | `INVALID_TIME_RANGE` |
| `text=""`, `no_speech_prob=0.1` | `EMPTY_TEXT` (có nói mà không ra chữ: lỗi thật) |
| `t_end=345.2`, duration 344.561 (vượt 0.64 s) | Cắt về 344.561, `clipped_t_end += 1` (padding của Whisper) |
| `t_end=346.0` (vượt 1.44 s) | `TIME_OUT_OF_BOUNDS` |
| `quality_score=1.3` | `SCORE_OUT_OF_RANGE`: bug code của mình, tìm dòng bị ảnh hưởng bằng `code_version` |

**[+review]** Thêm `SCHEMA_VIOLATION`, `TRANSCRIBE_FAILED`, `CORRUPT_KEYFRAME`,
`OCR_FAILED`, và ngưỡng im lặng 0.6 (bằng mặc định của Whisper). Lỗi `Field(ge=0)` của
Pydantic không tự có `reason_code`, phải map sang `SCHEMA_VIOLATION`.

## 6. Cột của DLQ

`dlq_id, video_id, record_id, stage, reason_code, error_detail, raw_payload, provenance,
failed_at, attempts, status`
- **`raw_payload` quan trọng nhất**: replay chỉ cần đọc lại, sửa và validate, không chạy
  lại Whisper cả video. (Từng quên cột này.)
- `stage` (transcribe, validate, write) vì mỗi bước có cách sửa khác nhau.
- `status` = `open | replayed | ignored`; replay 3 lần vẫn lỗi thì dừng và gọi người.

## 7. Cách ghi và idempotency khi ghi

Id tất định là điều kiện cần, chưa đủ; cách ghi mới quyết định:
```
INSERT thường (Parquet, không khóa) → 2 dòng trùng
INSERT, có PRIMARY KEY              → lần 2 báo lỗi, replay crash
UPSERT (INSERT OR REPLACE)          → 1 dòng ✓
```

**Bẫy MERGE khi chạy lại cả video:**
```
input:  bản small cũ 150 segment · large-v3 ra 140 · chỉ 60 id trùng
MERGE:  60 update + 80 insert + 90 dòng cũ giữ nguyên = 230 dòng ✗
        → 90 dòng zombie chồng thời gian lên dòng mới → join slide fan-out lặng lẽ
```
Đáp án: **partition overwrite** trong một transaction.
```sql
BEGIN; DELETE FROM audio_segments WHERE video_id='lakehouse';
INSERT INTO audio_segments SELECT * FROM new_run; COMMIT;
```
```
crash sau DELETE, không có transaction → 0 dòng, video như chưa từng xử lý (Missing)
crash sau DELETE, có transaction       → rollback, vẫn còn 150 dòng cũ
```
- Sửa từng dòng (replay DLQ) → upsert theo `segment_id`.
- Chạy lại cả video → DELETE + INSERT theo `video_id`.
- INNER JOIN không thay được: join là phép đọc, không xóa được dòng cũ.
- **[+review]** Đã thử trên DuckDB 1.5.6: DELETE rồi INSERT lại cùng PK trong một
  transaction chạy được. Lý do chọn DuckDB: ADR 0001.

## 8. Alignment

### Overlap và tie-break
`overlap = LEAST(a.t_end, s.t_end) − GREATEST(a.t_start, s.t_start)`. Viết ngược dấu
thì luôn ra số âm, rank chọn sai slide.
```
câu nói   [745, 755)
slide_17  [721, 750)  overlap = 750 − 745 = 5 s
slide_18  [750, 780)  overlap = 755 − 750 = 5 s     → hòa
ORDER BY overlap DESC, t_start DESC, slide_id       → slide_18 (slide sau)
```
- Chỉ `ORDER BY overlap DESC` thì khi hòa, kết quả không tất định.
- Chọn **slide sau**: câu vắt qua mốc chuyển slide thường đã nói về slide mới.
  **[+review]** Luật nearest ban đầu lấy slide trước; giờ cả hai dùng cùng luật.

### align_gap
Cùng lệch 2.5 s nhưng mức tệ khác nhau, nên giữ cả số giây và tỉ lệ:
```
câu A dài 20 s, lệch 2.5 s → ratio 0.125 (phần lớn vẫn đúng slide)
câu B dài  3 s, lệch 2.5 s → ratio 0.83  (gần như chắc ghép sai)

align_gap_s     = (t_end − t_start) − overlap
align_gap_ratio = align_gap_s / (t_end − t_start)   ; > 0.5 → weak match
ví dụ [745, 755) → slide_18: gap_s = 10 − 5 = 5, ratio = 0.5 (chạm ngưỡng, chưa weak)
```
- `align_gap_s`: mốc scene change có lệch đều không (p95 ≈ 3 s thì chỉnh mốc).
  `align_gap_ratio`: dòng này có đáng tin để dùng không.
- Không chạm slide nào → slide gần nhất, `match_type = 'nearest'`.
- **[+review]** Với dòng nearest, `align_gap_s` luôn bằng độ dài câu nên mất khoảng
  cách → thêm `nearest_distance_s`.
- **[+review]** Có t=0 và `t_end = LEAD` thì slide lát kín timeline: **có dòng nearest
  nghĩa là frame lane mất slide**. Ngưỡng cảnh báo 5%.
- **[+review]** Join chỉ lọc quarantine phía audio. Lọc cả slide quarantine sẽ sinh
  nearest giả; `aligned.slide_quality_flag` để tự lọc khi cần.

## 9. Idempotency khi kiểm tra kết quả

Cùng video + tham số + code → **cùng tập dòng** (không cần cùng thứ tự), bỏ qua
`run_id`, `processed_at`. Không đòi file giống từng byte vì Parquet có metadata, row
group và nén khác nhau.
```sql
SELECT ... FROM 'run1/audio_segments.parquet' EXCEPT SELECT ... FROM 'run2/...';  -- 0 dòng
SELECT ... FROM 'run2/...' EXCEPT SELECT ... FROM 'run1/...';                     -- 0 dòng
```
Phải chạy cả hai chiều; một chiều không bắt được dòng thừa ở run2. Nên so mọi cột trừ
hai cột đổi theo lần chạy.

## 10. Môi trường và setup **[+review]**

- Hook bảo vệ secret từng bị tắt lặng lẽ vì `python3` không có trên PATH; giờ chạy qua
  `uv run`.
- Repo trong WSL: bỏ đường dẫn `D:\` và winget; ffmpeg cài bằng apt; video nằm trong
  `data/input/` (gitignored).
- `segments` của faster-whisper là generator: model chỉ chạy khi lặp qua nó.
- `transcribe()` chưa nhận `video_id` trong khi `TODO(4)` cần nó: tác giả chọn cách xử lý.

## Câu hỏi tự kiểm tra

| Câu hỏi | Test kiểm tra |
|---|---|
| Vì sao `segment_id` không dùng `id` của Whisper, và vì sao `round`? | `test_segment_id_is_deterministic_milliseconds_not_whisper_id` |
| `quality_score` tính từ hai trường nào? | `test_quality_score_is_exp_logprob_times_speech_prob` |
| Vì sao rerun cả video phải DELETE + INSERT? | `test_whole_video_rerun_leaves_only_the_new_rows` |
| Crash giữa DELETE và INSERT thì sao? | `test_failed_insert_after_delete_keeps_the_old_rows` |
| Replay 2 lần ra mấy dòng? | `test_replaying_twice_gives_one_row_with_the_latest_values` |
| Slide 0.4 s do chuyển cảnh xử lý ra sao, vì sao không vào DLQ? | (roadmap #3) |
| Slide không có chữ vào đâu, vì sao không vào DLQ? | (roadmap #4) |
| Có dòng `nearest` trong `aligned` nghĩa là gì? | (roadmap #5) |
