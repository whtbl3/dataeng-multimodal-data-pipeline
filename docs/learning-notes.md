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
- `transcribe()` không nhận `video_id`: tác giả chọn suy ra từ `video_path.stem` để đặt
  thư mục `data/work/<stem>/`. Hệ quả: nếu sau này có `--video-id` khác tên file, wav vẫn
  nằm theo tên file.

## 11. Bài học khi làm roadmap #1 (2026-10-18)

Những điều tự phát hiện khi chạy thật, không có trong spec lúc đầu.

### Whisper
- **`avg_logprob` và `no_speech_prob` tính theo cửa sổ ~30 s (`seek`), không theo câu.**
  Các câu cùng `seek` có cùng hai giá trị này, nên `quality_score` giống nhau cả nhóm:
  ```
  id  seek  start  avg_logprob          quality_score
  1   0     0.0    -0.12359793012340864  0.799
  2   0     5.0    -0.12359793012340864  0.799   ← giống câu 1
  4   0     20.0   -0.12359793012340864  0.799
  5   2700  27.0   -0.04130014218389988  0.959   ← seek khác → điểm khác
  ```
  Câu nghe nhầm ("asset transactions" thay vì "ACID transactions", 261 s) không có điểm
  thấp riêng. Hướng sửa sau: `word_timestamps=True`, dùng `probability` của từng từ.
- **Mốc thời gian của `lakehouse.mp4` toàn tròn giây** (0.0, 5.0, 13.0...), nên mọi
  `segment_id` kết thúc bằng `000`. Ảnh hưởng độ chính xác khi align (roadmap #5).
- 78 segment; câu cuối kết thúc 341.0 s trong khi `duration` 344.561 s → `duration` phải
  lấy từ ffprobe của video gốc. Wav sau khi chuyển đổi còn lệch nhẹ (344.560938).
- **`faster-whisper 1.2.1` + `av 19` → `TypeError: open() got an unexpected keyword
  argument 'metadata_errors'`.** `faster-whisper` chỉ ghi `av>=11`; đã thử 15–18 đều chạy.
  Khóa `av<19` trong `pyproject.toml`.
- `Segment` là dataclass: dùng `dataclasses.asdict(seg)`; `seg._asdict()` đã deprecated.

### Python
- **float để tính, số nguyên làm id.** `16.15` thật ra là `16.1499999...`:
  ```
  int(16.15 * 1000)   → 16149   ✗ (cắt)
  round(16.15 * 1000) → 16150   ✓ (làm tròn)
  f"{16.15 * 1000}"   → "16149.999999999998"   ✗ (float trong id)
  ```
  So float trong test bằng `pytest.approx`, không dùng `==`.
- Fixture tròn giây không bắt được lỗi `int`: cần test riêng với `start = 16.15`.
- `Path(__file__).resolve().parents[2]` = gốc project khi file nằm ở
  `src/talk_align/config.py`. `.parent` chỉ lên 1 cấp → từng ghi wav vào `src/`.
- **Không `mkdir` trong `config.py`:** nó tạo cả thư mục sai (`data/data/work`) lúc import
  nên che lỗi đường dẫn. Tạo thư mục ở hàm cần ghi file.
- argparse: tham số tùy chọn có `default=` thì không bao giờ là `None`; nhánh
  `if args.out is None` từng không bao giờ chạy.

### Kiến trúc
- `build_rows` là hàm thuần: `provenance` (có `run_id`, `processed_at`, sha256, git) tạo
  một lần ở CLI rồi truyền vào. Test chạy 0.15 s và chạy 2 lần ra giống hệt.
- Tham số model gom vào `AudioParams` (frozen dataclass) dùng chung cho `transcribe` và
  `provenance.params`, để hai nơi không lệch nhau.
- `code_version`: `git status --porcelain .` với `cwd=BASE_DIR`; dấu `.` để repo cha
  (`/home/dev/projects`) có thay đổi ở project khác không làm `talk-align` thành `-dirty`.
- `store.connect(db_path)` phải dùng tham số, không dùng `DB_PATH`: từng làm test ghi vào
  `data/lab1.duckdb` thật. CLI mới là nơi chọn `DB_PATH`.
- DuckDB chỉ cho một tiến trình ghi: mở connection ngay trước khi ghi, không mở trước khi
  chạy Whisper vài phút.
- Ghi lại một video, đã thử:
  ```
  không PK, chạy 2 lần         → 6 dòng (trùng, không lỗi)
  có PK, chạy 2 lần            → ConstraintException: Duplicate key
  có PK, small rồi base        → 4 dòng, không lỗi (zombie)
  DELETE + INSERT trong 1 tx   → đúng dòng mới; INSERT lỗi thì rollback, giữ dòng cũ
  ```
- CLI chưa có test: thiếu `video_id` khi gọi `overwrite_video` chỉ lộ ra sau khi Whisper
  chạy xong. Ứng viên cho test sau (mock `transcribe`).

### Cách làm
- **Outside-in:** bắt đầu từ lệnh CLI, viết vỏ hàm `raise NotImplementedError`, chạy để
  lỗi chỉ chỗ tiếp theo. Lệnh ngoài (ffmpeg, ffprobe, git) chạy bằng tay trước rồi mới đưa
  vào `subprocess.run(..., check=True)`.
- Bước chậm (Whisper) chạy một lần, lưu `tests/fixtures/whisper_lakehouse.json` (5 câu);
  từ đó `build_rows` phát triển bằng pytest.
- Fixture nằm ở `tests/` (commit, test chạy được ở mọi máy), không ở `data/`.

## 12. Bài học khi làm roadmap #2 (2026-10-19)

### Thứ tự luật trong `build_rows`
Mỗi luật là một khối `if ...: <đếm hoặc thêm DLQ>; continue`. Dòng nào qua hết mới vào
`ok_rows`. Thứ tự đã chốt:
```
1. text rỗng + no_speech_prob ≥ 0.6   → bỏ, dropped_silence
2. text rỗng                          → EMPTY_TEXT        (tới đây chắc chắn < 0.6)
3. t_end ≤ t_start                    → INVALID_TIME_RANGE
4. t_start ≥ duration                 → TIME_OUT_OF_BOUNDS (trước khi cắt!)
5. t_end − duration > 1.0             → TIME_OUT_OF_BOUNDS
6. 0 < t_end − duration ≤ 1.0         → cắt t_end = duration, clipped_t_end
7. Pydantic AudioSegmentRow           → SCORE_OUT_OF_RANGE / SCHEMA_VIOLATION
   (cả vòng bọc try/except KeyError, TypeError, AttributeError, ValueError → SCHEMA_VIOLATION)
```
- Luật 4 đứng trước luật 6: `start=344.8, end=345.0, duration=344.561` nếu cắt trước sẽ
  ra `t_end=344.561 < t_start`. Thứ tự luật là một phần của logic, cần test riêng.
- `t_end` là biến riêng (có thể bị cắt); `raw_payload` vẫn giữ `end` gốc.

### Thiết kế
- **Bất biến "không mất segment":** `segments_in == ok + dlq + dropped_silence`. Một test
  đưa đủ 8 loại segment kiểm tra đẳng thức này: nhánh nào quên thêm DLQ hoặc quên đếm là
  test fail.
- `make_dlq_row` dùng chung cho mọi reason_code; `make_segment_id` tách riêng để công thức
  id chỉ ở một chỗ. Khi thiếu `start`, `make_dlq_row` không được ném lỗi lần nữa (lỗi
  trong `except` không còn gì bắt).
- `failed_at` lấy từ `provenance.processed_at`: thử đổi sang `now()` thì test "chạy 2 lần"
  fail ngay.
- Chỉ bắt lỗi do dữ liệu; bắt `Exception` sẽ biến một lỗi gõ tên hàm (`NameError`) thành
  78 dòng DLQ.
- Pydantic: `e.errors()` trả về mọi lỗi cùng lúc, mỗi lỗi có `loc` (cột) và `type`.
  `reason_code_for` map: lỗi ở `quality_score` → `SCORE_OUT_OF_RANGE` (thường là bug công
  thức), còn lại → `SCHEMA_VIOLATION`. `extra="forbid"` bắt cột thừa trước DuckDB.

### Ghi DuckDB
- Ghi `audio_segments` và `dlq_audio` trong **một** transaction
  (`overwrite_video_tables`); test cho `dlq_audio` lỗi sau khi `audio_segments` đã ghi
  xong → cả hai rollback.
- Bảng rỗng: **luôn DELETE**, chỉ INSERT khi có dòng. `insert_rows([])` lỗi ở `rows[0]`;
  bỏ cả DELETE thì dòng DLQ cũ của lỗi đã sửa vẫn nằm lại.
- `overwrite_video(conn, table, ...)` cũ giữ lại, gọi sang hàm mới: 10 chỗ gọi trong test
  không phải sửa, logic transaction chỉ ở một chỗ.
- `CREATE TABLE IF NOT EXISTS` tự thêm bảng `dlq_audio` vào `lab1.duckdb` cũ, không cần
  xóa file.
- Export: mỗi bảng một file, bảng rỗng vẫn ra file 0 dòng đủ cột. `run_id` lấy từ cả hai
  bảng (`UNION ALL`), vì video có thể có toàn bộ segment vào DLQ.

### Kết quả trên `lakehouse.mp4`
`{'segments_in': 78, 'dropped_silence': 0, 'clipped_t_end': 0, 'ok': 78, 'dlq': 0,
'quarantined': 0}`: video sạch, mọi luật chỉ được kiểm bằng segment tự tạo trong test.

## 13. Scene detect trên video thật (2026-10-19, trước roadmap #3)

Ngưỡng `gt(scene,0.3)` ở §3 được chọn trước khi đo. Đo trên `lakehouse.mp4` (10335 frame):
```
so từng frame:   điểm cao nhất 0.099 → > 0.3: 0 mốc → cả video 1 slide
                 > 0.05: 15 frame, dồn cụm (13.2, 13.3, 13.4 ...) = một cú fade kéo dài
lấy mẫu 1 fps:   > 0.3 : 9 mốc  [5, 13, 14, 20, 39, 40, 83, 91, 344]
                 > 0.15: 15 mốc [5, 6, 13, 14, 20, 39, 40, 83, 91, 150, 194, 225, 254, 340, 344]
```
- Chuyển slide dạng **fade**: mỗi frame chỉ khác frame trước một chút, nên so từng frame
  không bao giờ vượt ngưỡng. Lấy mẫu 1 fps thì hai mẫu cách nhau 1 s, cả cú fade dồn vào
  một bước nhảy.
- Đối chiếu số trang ở góc slide: mốc 150, 194, 225, 254 khớp slide 9→10→11→12→13. Ngưỡng
  0.3 bỏ sót cả 4 (giây 91 → 340 thành một slide).
- Không bắt được: đổi chữ trong cùng slide (slide 9 "Pros" → "Cons", ~giây 95–115) vì nền
  và hình giữ nguyên. Chấp nhận; OCR ở roadmap #4 có thể cho thấy.
- Mốc ở 344 là fade về đen cuối video (duration 344.561).
- Bài học: ngưỡng trong spec là giả định; **đo phân phối trên dữ liệu thật trước khi
  code** (ffmpeg `metadata=print` in `lavfi.scene_score` từng frame).

## 14. Bài học khi làm roadmap #3 (2026-10-19)

### ffmpeg
- `showinfo` ghi vào log → đọc `result.stderr` (stdout rỗng). **Không** thêm
  `-loglevel error`: `showinfo` log ở mức info, `-loglevel error` cho ra `[]` mà không báo
  lỗi gì.
- Cắt 1 ảnh: `-ss <t>` đặt **trước** `-i` để nhảy thẳng tới t (nhanh), `-frames:v 1`.
- `probe_duration` chuyển sang `media.py`: hai lane độc lập, không lane nào import lane kia.

### Gộp mốc
```
mốc   kế       khoảng   kết quả
5.0   6.0      1.0      bỏ (fade)
13.0  14.0     1.0      bỏ
39.0  40.0     1.0      bỏ
344.0 344.561  0.561    bỏ (so với duration: fade về đen cuối)
→ 16 mốc thô → 12 slide, merged_transitions = 4
```
- Test `[0.0, 1.0, 10.0]` từng được điền theo output của code (`[0, 1, 10]`, slide đầu
  dài 1 s). Bất biến "mọi slide ≥ min_duration" lộ ra lỗ: `0.0` luôn được giữ mà không
  kiểm tra khoảng cách. Sửa: mốc cách `0.0` < `min_duration` thì bỏ mốc đó.
- Hai bất biến của frame lane: slide lát kín `[0, duration)` (từng cặp liền nhau
  `a.t_end == b.t_start`, `itertools.pairwise`), và mọi slide ≥ `min_duration`.
- `@dataclass`: dòng không có type hint (`scene_threshold = 0.15`) **không phải field**:
  `asdict` bỏ qua, constructor không nhận → provenance thiếu tham số mà không lỗi.

### Keyframe
- Lấy ở giữa slide: 12/12 ảnh rõ, không ảnh nào chồng hình do fade.
- Partition overwrite áp dụng cho file: xóa `data/keyframes/<video_id>/` trước khi ghi,
  nếu không ảnh của lần chạy với tham số cũ (vd mốc 150 khi ngưỡng 0.15) còn nằm lại và
  OCR đọc nhầm.
- Hạn chế để ý ở #4/#9: slide có bullet hiện dần (`254s`, ảnh ở 297 s chưa đủ bullet) và
  đổi nội dung trong cùng slide (`40s`, `91s` chỉ còn "Cons"): một ảnh giữa slide không
  chứa hết chữ của slide đó.

## 15. Bài học khi làm roadmap #4 (2026-10-21)

### OCR thật trên `lakehouse`
```
slide 91s (12 box)
  h=45 conf=0.997 'Data Lakes'
  h=33 conf=0.661 '小'                                         ← rác 1 ký tự
  h=23 conf=0.977 'Structured,Semi-StructuredandUnstructuredData'  ← chữ thật nhỏ nhất: 23/720 = 0.032
  h=19 conf=0.942 '2022Databricks Inc.—All rights reserved'   ← footer mọi slide: 19/720 = 0.026
```
- 0.4–1.5 s mỗi ảnh, RAM tối đa ~524 MB. Nạp model (`RapidOCR()`) một lần cho cả batch.
- Ảnh không có chữ → `result` là `None`, không phải `[]` → `result or []`.
- File mất / không phải ảnh: rapidocr báo `LoadImageError`; Pillow báo
  `FileNotFoundError` / `UnidentifiedImageError` (đều là `OSError`).
- Box ra dạng `[4 góc, text, conf]` kiểu Python thường; đổi sang dict có tên key ngay khi
  đọc. Lưu `image_h` cùng box để output OCR tự đủ khi replay.
- `Path` không `json.dump` được → `str()` trước khi ghi fixture.

### Lọc và điểm
- `conf` đo "ký tự có đúng không", không đo dấu cách: 12/12 slide điểm 0.963–0.996 dù có
  "Oneplatformtounifyallyourdata", "BigData", "2000S". Ngưỡng 0.3 không bao giờ chạm.
- Cân theo ký tự (ví dụ §3): trung bình thường 0.82, cân theo ký tự 0.85.
- Slide rỗng: kiểm tra `Σ ký tự == 0` trước khi chia (tránh `ZeroDivisionError`) →
  `text=""`, điểm 0, quarantine, không DLQ.
- **Đừng test đúng ngưỡng khi ngưỡng là kết quả phép chia float:** `21.6 / 720 =
  0.030000000000000002`, nên box được giữ dù viết `>=` hay `>`. Test hai giá trị rõ ràng
  hai bên ngưỡng (21.0 → 0.0292, 22.0 → 0.0306). Và `len(...) == True` pass chỉ vì
  `1 == True`.
- `raw_payload` giữ cả box bị lọc: đổi ngưỡng lọc thì chạy lại `build_slide_rows`, không
  OCR lại.

### Lỗi thư viện ngoài
- Được bắt `Exception` cho `OCR_FAILED` vì lỗi nằm trong engine (không biết trước loại),
  nhưng `try` chỉ bọc đúng dòng gọi engine; phần đổi box sang dict nằm ngoài để bug của
  mình vẫn crash.

### Ghi và export
- `dlq_audio`, `dlq_frame` dùng chung biến cột → thêm cột một chỗ.
- `build_provenance(video_path, model, model_version, params)` chung cho 2 lane;
  `model_version` của rapidocr lấy từ `importlib.metadata.version`, không viết cứng.
- Ghi frame lane không đụng bảng audio (hai lane độc lập).
- Export: mỗi lane có `run_id` riêng → chọn `run_id` có `processed_at` mới nhất trong 4
  bảng; `any_value` cho thư mục đổi qua lại giữa các lần chạy.
- Lỗi `run_id = run_id[0]` (dòng sót lại từ khi `fetchone()` trả tuple) lấy ký tự đầu của
  UUID → thư mục `data/out/6/`. Đặt tên biến đúng kiểu dữ liệu (`row` vs `run_id`) giúp
  tránh.
- Kiểm tra slide lát kín bằng SQL: `LEAD(t_start) OVER (ORDER BY t_start)` → 0 khoảng hở.

## Câu hỏi tự kiểm tra

| Câu hỏi | Test kiểm tra |
|---|---|
| Vì sao `segment_id` không dùng `id` của Whisper, và vì sao `round`? | `test_segment_id_uses_round_not_int` |
| `quality_score` tính từ hai trường nào? Ngưỡng 0.3 là `<` hay `<=`? | `test_quality_score_formula`, `test_quality_flag_threshold` |
| Vì sao `build_rows` không được gọi `datetime.now()`? | `test_running_twice_gives_the_same_rows` |
| INSERT 2 lần vào bảng có PRIMARY KEY thì sao? | `test_insert_twice_fails` |
| Vì sao rerun cả video phải DELETE + INSERT? | `test_rerun_removes_zombies_and_keeps_other_videos` |
| Crash giữa DELETE và INSERT thì sao? | `test_failed_insert_keeps_old_rows` |
| Text rỗng lúc nào bị bỏ, lúc nào vào DLQ? Đúng 0.6 thì sao? | `test_silence_is_dropped_and_counted`, `test_silence_threshold_is_inclusive`, `test_empty_text_with_speech_goes_to_dlq` |
| `t_end` vượt `duration` bao nhiêu thì cắt, bao nhiêu thì DLQ? | `test_small_overrun_is_clipped`, `test_overrun_at_tolerance_is_still_clipped`, `test_large_overrun_goes_to_dlq` |
| Câu bắt đầu sau khi video hết thì sao? | `test_segment_starting_after_video_end_goes_to_dlq` |
| Segment thiếu `start` thì `dlq_id` là gì? | `test_missing_start_uses_raw_index` |
| Một segment hỏng có làm mất segment tốt không? | `test_one_bad_segment_does_not_lose_the_good_ones` |
| Có segment nào có thể "biến mất" không? | `test_every_segment_is_accounted_for` |
| Vì sao 2 bảng phải chung 1 transaction? | `test_failure_in_dlq_rolls_back_audio_too` |
| Lần chạy mới hết lỗi thì dòng DLQ cũ đi đâu? | `test_rerun_without_errors_clears_old_dlq_rows` |
| Replay 2 lần ra mấy dòng? | (roadmap #7) |
| Slide 0.4 s do chuyển cảnh xử lý ra sao, vì sao không vào DLQ? | `test_spec_example`, `test_every_slide_is_at_least_min_duration` |
| Mốc 0.0 quá gần mốc kế tiếp thì bỏ mốc nào? | `test_zero_is_never_dropped` |
| Có đoạn video nào không thuộc slide nào không? | `test_slides_tile_the_whole_video` |
| Slide không có chữ vào đâu, vì sao không vào DLQ? | `test_empty_slide_is_kept_as_quarantine`, `test_slide_with_no_boxes_is_kept_as_quarantine` |
| Box nào bị lọc? Footer có bị lọc không? | `test_footer_and_short_junk_are_dropped` |
| Vì sao điểm slide cân theo số ký tự? | `test_slide_quality_is_weighted_by_characters` |
| Vì sao `raw_payload` giữ cả box bị lọc? | `test_raw_payload_keeps_dropped_boxes` |
| Ảnh keyframe hỏng thì sao? Các slide khác có mất không? | `test_corrupt_keyframe_goes_to_dlq`, `test_error_entry_becomes_dlq_row_and_good_slides_survive` |
| Chạy frame lane có xóa dữ liệu audio không? | `test_frame_write_keeps_audio_rows` |
| Có dòng `nearest` trong `aligned` nghĩa là gì? | (roadmap #5) |
