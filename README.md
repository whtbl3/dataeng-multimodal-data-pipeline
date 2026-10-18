# talk-align

Lab tự học multimodal data engineering: biến một video talk thành các bản ghi audio và
slide có timestamp, có `quality_score`, rồi ghép chúng theo time overlap trong DuckDB.

## Cấu trúc thư mục

```
talk-align/
├── pyproject.toml          lệnh `talk-align` → talk_align.cli:main
├── src/talk_align/
│   ├── cli.py              ĐIỂM BẮT ĐẦU: đọc lệnh, gọi lần lượt các hàm bên dưới   (#1)
│   ├── config.py           đường dẫn data/, tham số mặc định (AudioParams, ngưỡng)  (#1)
│   ├── audio_lane.py       video → wav → Whisper → dòng audio_segments            (#1)
│   ├── store.py            DuckDB: tạo bảng, ghi, export Parquet                  (#1)
│   ├── contract.py         kiểm tra dòng, lỗi → DLQ                               (#2)
│   ├── frame_lane.py       video → keyframe → OCR → dòng slides                   (#3–4)
│   ├── align.py            ghép audio ↔ slide theo thời gian                      (#5)
│   ├── report.py           thống kê một lần chạy                                  (#6)
│   └── replay.py           chạy lại dòng trong DLQ                                (#7)
├── tests/
│   ├── conftest.py         fixture dùng chung cho test
│   └── fixtures/           output Whisper/OCR lưu ra JSON để test không cần model
├── docs/
│   ├── project-spec.md     ĐỀ BÀI: bảng nào, cột nào, công thức nào
│   ├── learning-notes.md   VÌ SAO: giải thích kèm ví dụ input → output
│   └── roadmap.md          thứ tự các feature
└── data/                   (gitignored, không commit)
    ├── input/              video gốc, vd lakehouse.mp4
    ├── work/<video_id>/    wav và file trung gian
    ├── keyframes/<video_id>/  ảnh PNG từng slide
    ├── out/<run_id>/       Parquet + report.md
    └── lab1.duckdb         database (tự tạo khi chạy lần đầu)
```

Số trong ngoặc là roadmap item. Roadmap #1 chỉ cần 4 file đầu.

## Luồng gọi hàm của roadmap #1

```
uv run talk-align audio data/input/lakehouse.mp4
  └─ cli.main
       ├─ video_id = "lakehouse"
       ├─ audio_lane.transcribe(video)          → (raw_segments, duration)
       ├─ audio_lane.build_rows(video_id, ...)  → (rows, dlq_rows, metrics)
       ├─ store.connect("data/lab1.duckdb")     → con
       └─ store.overwrite_video(con, "audio_segments", video_id, rows)
```

## Cài đặt

```bash
uv add duckdb faster-whisper pyarrow pydantic
uv add --dev pytest ruff
uv run talk-align --help
```
Cần ffmpeg (`sudo apt install ffmpeg`).
