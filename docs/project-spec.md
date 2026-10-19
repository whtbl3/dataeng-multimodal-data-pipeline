# Project spec: talk-align
Status: draft · Derived from: docs/intent.md

## 1. Goal
Biến một video talk (5–10 phút) thành các bản ghi audio và slide có timestamp, có
`quality_score`, theo một data contract, rồi ghép chúng theo time overlap. Bản ghi lỗi
không làm crash pipeline mà vào DLQ để replay. Đây là lab tự học multimodal data
engineering: tác giả tự viết code pipeline, spec này là khung.

## 2. Scope (in / out)
**In:** batch một video mỗi lần chạy; audio lane (ffmpeg → faster-whisper); frame lane
(ffmpeg scene change → OCR); Pydantic contract; DLQ theo modality; interval join trong
DuckDB; run report; replay DLQ; kiểm tra idempotency.

**Out:** streaming, incremental ingestion, embedding, vector search, LLM, UI, cloud,
nhiều video một lần chạy, CI/CD (không làm; mọi kiểm tra chạy local).

## 3. Architecture (box level)
```
data/input/talk01.mp4
   │
   ├─ ① audio lane: ffmpeg (wav 16 kHz mono) → faster-whisper → validate ─┐
   │                                                                      │
   └─ ② frame lane: ffmpeg scene change + t=0 → keyframes → OCR → validate ┤
                                                                          ▼
                 ③ contract (Pydantic): ok → bảng chính · lỗi → dlq_* · rác → metric
                                                                          │
                 ④ store: data/lab1.duckdb (partition overwrite theo video_id)
                                                                          │
                 ⑤ align: interval join (overlap + ROW_NUMBER) → aligned
                                                                          │
                 ⑥ report + export Parquet → data/out/<run_id>/
```
- Hai lane độc lập: lane này lỗi thì lane kia vẫn chạy xong và ghi kết quả.
- Model chỉ chạy ở ① và ②. Mọi bước sau đọc `raw_payload`, nên replay và chỉnh luật
  không cần chạy lại Whisper/OCR.

### Phân loại bản ghi xấu
```
Nội dung có thật không?
├─ không (im lặng, chuyển cảnh)        → bỏ, đếm metric
└─ có → xử lý có lỗi không?
         ├─ có (crash, file hỏng, sai contract) → dlq_* + reason_code, sửa rồi replay
         └─ không, chỉ chất lượng kém           → giữ, quality_flag = quarantine
```
- Không bao giờ bỏ một dòng nội dung thật. Keyframe hỏng làm OCR crash thì vào DLQ;
  khi replay, cắt lại keyframe từ video.
- DLQ rate vì thế chỉ đo bug của pipeline, không đo độ ồn của dữ liệu.

## 4. Key data entities
Mọi bảng nằm trong `data/lab1.duckdb`. Cột chung theo contract: `video_id`, `modality`,
`t_start`, `t_end` (giây, float, khoảng nửa mở `[t_start, t_end)`), `quality_score`
∈ [0, 1], `quality_flag` (`ok` | `quarantine`), `provenance`.

### `provenance` (STRUCT)
| Field | Ý nghĩa |
|---|---|
| `model`, `model_version` | vd `faster-whisper` / `small`; `rapidocr` / phiên bản package |
| `params` | audio: `device`, `compute_type`, `beam_size`, `language`, `temperature`; frame: `sample_fps`, `scene_threshold`, `min_duration`, `min_box_chars`, `min_box_height_ratio` |
| `source_hash` | sha256 của file video |
| `code_version` | git commit, thêm `-dirty` nếu working tree có thay đổi |
| `run_id`, `processed_at` | khác nhau mỗi lần chạy, bỏ qua khi so idempotency |

Mọi thứ làm output khác đi đều phải nằm trong `provenance`.

### `audio_segments` (PK `segment_id`)
`video_id, segment_id, modality='audio', text, t_start, t_end, quality_score,
quality_flag, raw_payload, provenance`
- `segment_id = f"{video_id}_a_{round(t_start * 1000):08d}"`, vd `talk01_a_00012480`.
  Dùng `round`, không dùng `int`: `int(16.15 * 1000)` ra `16149` do sai số float
  (`16.15 * 1000 == 16149.999999999998`); khoảng 1% mốc có 2 chữ số thập phân bị lỗi này.
  Không dùng `id` của Whisper (chỉ là số thứ tự trong một lần chạy).
- `quality_score = exp(avg_logprob) * (1 - no_speech_prob)`; `< 0.3` → `quarantine`.
- `raw_payload`: JSON segment gốc từ Whisper.

### `slides` (PK `slide_id`)
`video_id, slide_id, modality='frame', text, t_start, t_end, quality_score,
quality_flag, keyframe_path, raw_payload, provenance`
- `slide_id = f"{video_id}_s_{round(t_start * 1000):08d}"`.
- `t_start`: mốc scene change trên luồng **lấy mẫu 1 fps**
  (`fps=1,select='eq(n,0)+gt(scene,0.15)'`, có keyframe t=0). Mốc chính xác tới 1 s.
  So từng frame với ngưỡng 0.3 không dùng được: chuyển slide dạng fade nên điểm từng frame
  tối đa 0.099 trên `lakehouse.mp4` (xem learning-notes §13).
  `t_end = LEAD(t_start)`; slide cuối thì `t_end` = duration (ffprobe).
- Mốc cách mốc kế < `min_duration` (1.5 s) là chuyển cảnh: bỏ mốc đó, slide trước kéo
  dài tới mốc kế, đếm `merged_transitions`. Không vào DLQ, không mất slide nào.
- `text`: chữ OCR của các box đã lọc, sắp từ trên xuống, trái sang phải.
- Lọc box rác trước khi tính điểm: dưới `min_box_chars` (3) ký tự hoặc thấp hơn
  `min_box_height_ratio` chiều cao frame (logo, số trang).
- `quality_score = Σ(confidence × số ký tự) / Σ(số ký tự)` trên các box đã lọc.
  Ngưỡng quarantine: **calibrate ở lần chạy đầu từ phân phối điểm**, tạm dùng 0.3.
- `raw_payload`: JSON mọi box OCR (kể cả box bị lọc) với tọa độ, chữ, confidence.
- `keyframe_path`: `data/keyframes/<video_id>/<slide_id>.png`.
- Slide OCR không đọc được chữ nào vẫn là nội dung thật (vd slide chỉ có ảnh): giữ với
  `text=""`, `quality_score=0`, `quarantine`. Không đưa vào DLQ, vì mất slide thì audio
  ghép nhầm sang slide bên cạnh.

### `dlq_audio`, `dlq_frame` (PK `dlq_id`)
`dlq_id, video_id, record_id, stage, reason_code, error_detail, raw_payload,
provenance, failed_at, attempts, status`
- `dlq_id` tất định từ `(video_id, stage, record_id)`; khi chưa có `record_id` (crash
  trước khi biết `t_start`) thì dùng thứ tự trong output thô. Đã cài:
  `dlq_validate_<record_id>` (vd `dlq_validate_v_a_00050000`; `record_id` đã chứa
  `video_id`), hoặc `dlq_validate_<video_id>_raw<index:05d>` khi thiếu `start`
  (`record_id` = NULL).
- `failed_at` = `provenance.processed_at` của lần chạy, không gọi `now()` trong lane, để
  `build_rows` vẫn là hàm thuần và chạy 2 lần ra cùng dòng DLQ.
- `stage`: audio `extract | transcribe | validate | write`; frame
  `extract | ocr | validate | write`.
- `status`: `open | replayed | ignored`. Replay 3 lần vẫn lỗi (`attempts >= 3`) thì
  dừng, report ghi "needs human".

### `reason_code`
Viết hoa, mô tả lỗi chứ không mô tả cách sửa.

| Code | Khi nào | Ghi chú |
|---|---|---|
| `INVALID_TIME_RANGE` | `t_end <= t_start` | |
| `EMPTY_TEXT` | `text == ""` và `no_speech_prob` < `silence_threshold` (0.6) | Nếu ≥ ngưỡng thì là im lặng: bỏ, đếm `dropped_silence` |
| `TIME_OUT_OF_BOUNDS` | `t_start >= duration`, hoặc `t_end > duration + 1.0` | Vượt ≤ 1 s (padding của Whisper): cắt về duration, đếm `clipped_t_end`. Kiểm tra `t_start` trước khi cắt, để không sinh `t_end < t_start` |
| `SCORE_OUT_OF_RANGE` | `quality_score` ngoài [0, 1] | Bug của mình: sửa code, tìm dòng bị ảnh hưởng qua `code_version` |
| `SCHEMA_VIOLATION` | Pydantic báo lỗi kiểu/field khác; hoặc segment thô thiếu key / sai kiểu (`KeyError`, `TypeError`, `AttributeError`, `ValueError`) | Lỗi Pydantic không tự có reason_code, phải map. Không bắt `Exception` chung, để bug code (`NameError`...) vẫn crash |
| `TRANSCRIBE_FAILED` | Whisper crash | |
| `CORRUPT_KEYFRAME` | Không đọc được file keyframe | Replay: cắt lại keyframe từ video |
| `OCR_FAILED` | OCR crash | |

### `aligned` (PK `segment_id`)
`video_id, segment_id, slide_id, match_type, overlap_s, align_gap_s, align_gap_ratio,
nearest_distance_s, slide_quality_flag, run_id`
- Đầu vào mặc định: audio có `quality_flag = 'ok'`.
- `overlap = LEAST(a.t_end, s.t_end) - GREATEST(a.t_start, s.t_start)`, điều kiện
  `a.t_start < s.t_end AND a.t_end > s.t_start`.
- Mỗi segment giữ đúng 1 dòng:
  `ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY overlap DESC, s.t_start DESC, s.slide_id)`.
  Hòa thì lấy slide sau: câu nói vắt qua mốc chuyển slide thường đã nói về slide mới.
- Không overlap: `match_type = 'nearest'`, chọn slide có khoảng cách nhỏ nhất, hòa thì
  cùng luật (`t_start DESC`, `slide_id`).
- `align_gap_s = (a.t_end - a.t_start) - overlap`;
  `align_gap_ratio = align_gap_s / (a.t_end - a.t_start)` (0 = gọn trong slide,
  1 = không chạm slide). `align_gap_ratio > 0.5` → ghép yếu (weak match).
- `nearest_distance_s`: khoảng cách tới slide gần nhất, chỉ có ở dòng `nearest`
  (NULL ở dòng `overlap`).
- Slides lát kín timeline từ t=0, nên dòng `nearest` chỉ xuất hiện khi frame lane mất
  slide. Đó là tín hiệu frame lane có vấn đề.

### `runs` (PK `run_id`)
`run_id, video_id, started_at, finished_at, status, code_version, params, metrics`
- `metrics` (JSON): số dòng mỗi bảng, DLQ rate mỗi modality, `dropped_silence`,
  `clipped_t_end`, `merged_transitions`, số quarantine, p50/p95/max của `align_gap_s`
  và `align_gap_ratio`, số và % dòng `nearest`, số weak match.

### Cách ghi (xem ADR 0001)
| Tình huống | Cách ghi |
|---|---|
| Chạy cả video | `DELETE ... WHERE video_id = ?` + `INSERT` cho mọi bảng của video đó, trong **1 transaction** |
| Replay DLQ (từng dòng) | `INSERT OR REPLACE` theo PK, rồi `UPDATE dlq_* SET status = 'replayed'` |

MERGE khi chạy lại cả video để lại dòng "zombie" của lần chạy cũ (id theo `t_start` đổi
khi model đổi), nên join bị fan-out mà không ai báo lỗi.

## 5. Interfaces (APIs, files, UIs)
CLI (`uv run talk-align ...`, argparse):

| Lệnh | Việc |
|---|---|
| `run <video> [--video-id ID]` | ① → ⑥ cho một video; `video_id` mặc định = tên file không đuôi |
| `audio <video>` / `frames <video>` | Chạy riêng một lane rồi ghi vào store |
| `align <video_id>` | Dựng lại `aligned` từ bảng hiện có |
| `report <video_id>` | In report của lần chạy gần nhất |
| `replay --modality audio\|frame` | Replay các dòng DLQ `open` |
| `export <video_id> [--out DIR]` | Export mọi bảng ra Parquet |
| `check-idempotent <dir1> <dir2>` | `EXCEPT` hai chiều trên mọi cột trừ `run_id`, `processed_at` |

Files (cả `data/` đều gitignored):
```
data/input/<video>.mp4        video gốc, copy vào repo (không đọc từ /mnt/d)
data/work/<video_id>/         wav, file trung gian
data/keyframes/<video_id>/    PNG từng slide
data/lab1.duckdb              nguồn sự thật
data/out/<run_id>/            *.parquet + report.md
```

## 6. Non-functional requirements
- **Determinism:** `temperature=0` (tắt fallback ngẫu nhiên), `beam_size` cố định,
  `language="en"` (không auto-detect), cùng device. Ghi tất cả vào `provenance.params`.
- **Idempotency:** cùng video + tham số + code → cùng tập dòng ở mọi bảng
  (`check-idempotent` ra 0 dòng cả hai chiều). Chạy 1 hay 10 lần đều như nhau.
- **Atomicity:** crash giữa chừng thì bảng quay về trạng thái trước lần chạy, không bao
  giờ có video với 0 dòng.
- **Resource:** chạy được trên 2 core, 3 GB RAM; peak RAM < 2.5 GB. Video 10 phút xong
  trong khoảng 15 phút (ước lượng, đo ở lần chạy đầu).
- **Offline:** không gọi network ngoài lần tải model đầu tiên. Không paid API.
- **Privacy:** video và output chỉ nằm local, không commit vào git.

## 7. Definition of Done
- [ ] `make check` passes
- [ ] Every acceptance criterion has a test (fixtures JSON từ Whisper/OCR, không cần model)
- [ ] Docs and this spec updated
- [ ] No secrets in the diff
- [ ] Không có video, model hay file trong `data/` trong diff

## 8. Decision log
| Decision | Status | Link |
|---|---|---|
| DuckDB file là nguồn sự thật; partition overwrite cho cả video, upsert cho replay | Final | [0001](decisions/0001-duckdb-source-of-truth.md) |
| Id tất định từ `video_id` + `t_start` (ms) | Final | [0002](decisions/0002-deterministic-record-ids.md) |
| Slide không có chữ là nội dung thật: giữ, `text=""`, điểm 0, `quarantine`; không có `BLURRY_IMAGE` | Final | §4 `slides` |
| Join chỉ lọc quarantine phía audio; `aligned.slide_quality_flag` để người dùng tự lọc | Final | §4 `aligned` |
| DLQ chỉ chứa lỗi xử lý; chất lượng kém → `quality_flag` | Final | docs/intent.md |
| Whisper `small`, CPU int8, `temperature=0`, `language="en"` | Final | docs/tech-stack.md |
| Scene detect: lấy mẫu 1 fps + ngưỡng 0.15 (thay cho từng frame + 0.3); calibrate lại ở roadmap #9. Chấp nhận sót đổi chữ trong cùng slide | Final | learning-notes §13 |
| Không làm CI/CD; `pytest` + `ruff` chạy local trước khi commit | Final | roadmap #10 |
| Khóa `av<19`: av 19 bỏ `metadata_errors` mà faster-whisper 1.2.1 dùng | Final | docs/learning-notes.md §11 |
| `quality_score` audio hiện chung theo cửa sổ Whisper (`seek`), chưa theo câu; xem lại khi làm `word_timestamps` | Open | docs/learning-notes.md §11 |
