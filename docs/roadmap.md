# Roadmap

| # | Feature | Status | Spec |
|---|---|---|---|
| 1 | Walking skeleton: `talk-align audio` → Whisper → `audio_segments` (id tất định, provenance) trong DuckDB → `talk-align export` → Parquet; 12 test trên fixture (`pytest` + `ruff` xanh; chưa có `make check`) | Implemented | — |
| 2 | Audio contract + `dlq_audio`: reason_code, bỏ im lặng, cắt `t_end`, `quality_flag`, Pydantic contract, ghi 2 bảng trong 1 transaction, export cả `dlq_audio.parquet`; 37 test | Implemented | — |
| 3 | Frame lane: scene change + keyframe t=0, `t_end = LEAD`, gộp `min_duration` | Planned | — |
| 4 | OCR + `quality_score` của slide + `dlq_frame` | Planned | — |
| 5 | Align: overlap join, tie-break, nearest, `align_gap_*` | Planned | — |
| 6 | Bảng `runs` + report (DLQ rate, phân phối align_gap, nearest %) | Planned | — |
| 7 | Replay DLQ: upsert, `attempts`, `status` | Planned | — |
| 8 | `check-idempotent` + smoke test `@slow` trên video thật | Planned | — |
| 9 | Calibrate ngưỡng quarantine của slide trên video thật, cập nhật spec | Planned | — |
| 10 | ~~CI trên GitHub Actions~~: không làm CI/CD; kiểm tra chạy local (`pytest`, `ruff`) | Dropped | — |

Status: Planned · In spec · In progress · Implemented · Dropped
