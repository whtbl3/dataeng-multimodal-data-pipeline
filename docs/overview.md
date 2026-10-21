# Tổng quan talk-align: vấn đề, hướng xử lý, giải pháp

Nhìn lại project theo từng bước thiết kế. Mỗi bước gồm ba phần: **vấn đề** gặp phải,
**hướng xử lý** chọn, và **giải pháp** đã (hoặc sẽ) làm. Chi tiết luật và công thức nằm
ở `project-spec.md`, lý do và số liệu đo được nằm ở `learning-notes.md`.

Bước 0 đến 9 đã làm xong (roadmap #1 đến #6). Bước 10 đến 12 là kế hoạch (roadmap #7
đến #9).

---

## Bước 0. Bài toán tổng

**Vấn đề:** có một video thuyết trình. Muốn biết người nói **nói gì, lúc nào, và lúc đó
đang chiếu slide nào**.

**Hướng xử lý:** biến video thành dữ liệu có cấu trúc, lưu vào database để truy vấn được.

**Giải pháp:** một pipeline chạy bằng một lệnh: `talk-align run <video>`.

---

## Bước 1. Video chứa hai loại dữ liệu khác nhau

**Vấn đề:** một file video có cả **hình** (slide) lẫn **tiếng** (người nói). Mỗi loại cần
công cụ riêng để đọc.

**Hướng xử lý:** tách thành hai luồng xử lý độc lập, sau đó ghép kết quả của hai luồng
lại.

**Giải pháp:**
- Luồng audio: tiếng → các câu nói có thời gian.
- Luồng frame: hình → các slide có thời gian.
- Bước ghép: mỗi câu nói gắn với một slide.
- Hai luồng độc lập, nên luồng này lỗi thì luồng kia vẫn chạy xong.

---

## Bước 2. Luồng audio: biến tiếng nói thành chữ

**Vấn đề:** máy không hiểu được tiếng nói. Cần chuyển thành chữ, kèm thời điểm bắt đầu và
kết thúc của từng câu.

**Hướng xử lý:** dùng một model nhận dạng giọng nói (speech-to-text) chạy được trên máy cá
nhân.

**Giải pháp:**
- ffmpeg tách phần tiếng ra file wav đúng định dạng model cần (16 kHz, 1 kênh).
- Whisper (`small`, CPU) chuyển tiếng thành chữ.
- Kết quả: 78 câu, mỗi câu có chữ, thời gian, và độ tin cậy.

---

## Bước 3. Luồng frame: biến hình ảnh thành slide

**Vấn đề:** video có hơn 10.000 khung hình, nhưng chỉ có khoảng 12 slide. Cần biết slide
nào hiện từ lúc nào tới lúc nào, và trên slide viết gì.

**Hướng xử lý:** tìm những thời điểm hình thay đổi nhiều (chuyển slide), chụp một ảnh cho
mỗi slide, rồi đọc chữ trên ảnh.

**Giải pháp:**
- ffmpeg so sánh các khung hình cách nhau 1 giây để tìm mốc chuyển slide.
- Bỏ những mốc quá gần nhau (dưới 1.5 giây), vì đó chỉ là hiệu ứng chuyển cảnh.
- Chụp ảnh ở giữa mỗi slide, dùng OCR (rapidocr) đọc chữ, rồi bỏ các chữ rác (ký tự lạ,
  dòng bản quyền).
- Kết quả: 12 slide, mỗi slide có thời gian, chữ, và điểm chất lượng.

---

## Bước 4. Kết quả của model không phải lúc nào cũng đúng

**Vấn đề:** model có thể trả về dữ liệu xấu: câu rỗng, thời gian ngược, vượt quá độ dài
video, ảnh hỏng... Nếu đưa thẳng vào database thì dữ liệu sẽ sai, hoặc cả chương trình bị
dừng.

**Hướng xử lý:** đặt một lớp kiểm tra (contract) giữa model và database. Mỗi bản ghi xấu
được xử lý riêng, không làm ảnh hưởng tới các bản ghi khác.

**Giải pháp:** mỗi bản ghi đi tới **một trong ba nơi**:
- **Bỏ đi** nếu không có nội dung thật (ví dụ đoạn im lặng), nhưng vẫn đếm lại số lượng.
- **Đưa vào DLQ** (hàng đợi lỗi) nếu có nội dung nhưng xử lý bị lỗi, kèm lý do, để sửa sau.
- **Giữ lại** nếu ổn. Chất lượng kém thì gắn cờ `quarantine` chứ không xóa.

Không bản ghi nào được biến mất mà không để lại dấu vết.

---

## Bước 5. Lưu ở đâu để dễ phân tích

**Vấn đề:** cần một nơi lưu dữ liệu vừa đơn giản (không phải cài server), vừa đủ mạnh để
ghép dữ liệu theo thời gian và tính thống kê.

**Hướng xử lý:** dùng một database phân tích chạy trực tiếp trên máy, chỉ là một file.

**Giải pháp:** **DuckDB** (`data/lab1.duckdb`), với 6 bảng:

| Bảng | Chứa gì |
|---|---|
| `audio_segments` | câu nói |
| `dlq_audio` | câu bị lỗi, kèm lý do và dữ liệu gốc |
| `slides` | slide |
| `dlq_frame` | slide bị lỗi |
| `aligned` | câu nói nào thuộc slide nào |
| `runs` | lịch sử các lần chạy |

Khi cần mang dữ liệu sang công cụ khác thì xuất ra file Parquet.

---

## Bước 6. Chạy lại nhiều lần mà không bị trùng hay hỏng dữ liệu

**Vấn đề:** pipeline sẽ được chạy lại nhiều lần (sửa lỗi, đổi tham số). Nếu chỉ ghi thêm
thì dữ liệu bị lặp. Nếu chương trình bị dừng giữa chừng thì video có thể mất sạch dữ liệu.

**Hướng xử lý:** mỗi lần chạy thay toàn bộ dữ liệu của video đó, và việc thay phải xảy ra
"tất cả hoặc không có gì".

**Giải pháp:**
- Xóa dữ liệu cũ của video rồi ghi dữ liệu mới, tất cả trong một transaction. Lỗi giữa
  chừng thì quay về như cũ.
- Mã của câu và slide được tính từ thời gian (ví dụ `lakehouse_a_00013000`), nên chạy lại
  vẫn ra cùng một mã.

---

## Bước 7. Ghép câu nói với slide

**Vấn đề:** một câu nói có thể nằm vắt qua hai slide. Có câu lại không trùng slide nào
(khi luồng frame bị mất slide).

**Hướng xử lý:** với mỗi câu, xét mọi slide, rồi chọn đúng một slide theo một luật cố định.

**Giải pháp:** một câu SQL duy nhất:
- Chọn slide trùng thời gian với câu nhiều nhất.
- Nếu trùng bằng nhau thì chọn slide sau.
- Nếu không trùng slide nào thì chọn slide gần nhất, và đánh dấu `nearest`.
- Ghi lại độ lệch của từng câu để biết dòng ghép nào đáng tin.

---

## Bước 8. Biết một lần chạy có tốt hay không

**Vấn đề:** chạy xong thì kết quả chỉ in ra màn hình rồi mất. Không trả lời được "lần chạy
hôm qua có lỗi không, chạy bằng code phiên bản nào?".

**Hướng xử lý:** ghi lại mỗi lần chạy vào một bảng riêng, chỉ thêm vào chứ không xóa, và
tạo báo cáo tự chỉ ra điểm bất thường.

**Giải pháp:**
- Bảng `runs`: mỗi lần chạy một dòng, gồm trạng thái, thời gian, phiên bản code, và các
  chỉ số.
- Bảng này tách riêng khỏi các bảng dữ liệu, vì nó cần giữ **lịch sử**, còn các bảng dữ
  liệu chỉ cần **bản mới nhất**. Hai loại bảng nối với nhau qua `run_id`.
- Lệnh `talk-align report` tạo báo cáo gồm tỉ lệ lỗi, độ lệch, và cảnh báo.

---

## Bước 9. Kiểm tra code nhanh mà không phải chờ model

**Vấn đề:** Whisper mất vài phút cho mỗi lần chạy. Nếu mỗi lần sửa code lại phải chờ thì
rất chậm.

**Hướng xử lý:** tách phần **gọi model** ra khỏi phần **tính toán**, rồi lưu sẵn kết quả
của model để dùng lại khi test.

**Giải pháp:**
- Chạy model một lần, lưu kết quả thành file mẫu (fixture) trong `tests/fixtures/`.
- Phần tính toán được test trên file mẫu, mỗi lần chỉ mất dưới 1 giây.
- Hiện có 109 test.

---

## Kết quả hiện tại

Một lệnh, khoảng 2 phút 20 giây cho video 5 phút 44 giây: 78 câu nói, 12 slide, 78 dòng
ghép, 0 lỗi, không có cảnh báo.

---

## Bước 10. Sửa và chạy lại các bản ghi bị lỗi (replay DLQ), roadmap #7, chưa làm

**Vấn đề:** bản ghi lỗi đã nằm trong DLQ, nhưng mới chỉ được **cất đi**. Sau khi sửa code
hoặc sửa nguyên nhân, cần đưa chúng trở lại bảng chính. Chạy lại cả video thì rất tốn:
Whisper mất vài phút chỉ để sửa vài dòng.

**Hướng xử lý:** mỗi dòng DLQ đã lưu sẵn **dữ liệu gốc** (`raw_payload`). Chỉ cần lấy dữ
liệu gốc đó, cho đi qua lại bước kiểm tra, mà không phải chạy lại model. Dòng nào ổn thì
đưa vào bảng chính; dòng nào vẫn lỗi thì giữ lại và đếm số lần thử.

**Giải pháp:**
- Lệnh `talk-align replay --modality audio|frame`, chỉ lấy các dòng DLQ có `status = open`.
- Với audio: đọc lại `raw_payload` rồi chạy lại các luật kiểm tra.
- Với frame: nếu lỗi là `CORRUPT_KEYFRAME` thì chụp lại ảnh từ video rồi OCR lại.
- Dòng sửa được: ghi vào bảng chính bằng **upsert theo mã** (chỉ thay đúng dòng đó, không
  xóa cả video), rồi đánh dấu dòng DLQ là `replayed`.
- Dòng vẫn lỗi: tăng `attempts` lên 1. Thử tới 3 lần vẫn lỗi thì dừng, và report ghi "cần
  người xem".
- Chạy replay 1 lần hay 10 lần đều phải ra cùng một kết quả.

**Lưu ý:** video `lakehouse` hiện có 0 dòng lỗi, nên phải tự tạo lỗi để thử (làm hỏng một
ảnh slide, hoặc dùng câu nói giả trong test).

---

## Bước 11. Chứng minh chạy lại ra cùng kết quả (idempotency), roadmap #8, chưa làm

**Vấn đề:** cả pipeline được thiết kế để "chạy lại bao nhiêu lần cũng ra cùng kết quả",
nhưng đến giờ mới chỉ **tin** là như vậy, chưa **kiểm chứng** trên dữ liệu thật.

**Hướng xử lý:** chạy cùng một video hai lần, xuất ra hai bộ file, rồi so sánh từng dòng.
Bỏ qua những cột chắc chắn khác nhau giữa các lần chạy (`run_id`, thời điểm chạy).

**Giải pháp:**
- Lệnh `talk-align check-idempotent <thư_mục_1> <thư_mục_2>`.
- So sánh bằng `EXCEPT` **theo cả hai chiều**: dòng có ở lần 1 mà không có ở lần 2, và
  ngược lại. Hai chiều đều phải ra 0 dòng. So một chiều thì không phát hiện được dòng
  **thừa** ở lần 2.
- Nếu có khác biệt, in ra cột nào, dòng nào khác (ví dụ OCR đọc khác một chữ, Whisper cắt
  câu khác).
- Thêm một **smoke test** chạy cả pipeline trên video thật, tách riêng khỏi các test nhanh.

**Có thể sẽ phát hiện ra:** những chỗ chưa thật sự tất định, ví dụ thứ tự các ô chữ OCR
hoặc sai số float. Đó chính là mục đích của bước này.

---

## Bước 12. Chỉnh các ngưỡng theo dữ liệu thật (calibrate), roadmap #9, chưa làm

**Vấn đề:** nhiều ngưỡng vẫn là giả định, hoặc đã tỏ ra không phù hợp:
- Ngưỡng quarantine của slide là 0.3, nhưng cả 12 slide đều có điểm từ 0.96 trở lên, nên
  ngưỡng này không bao giờ có tác dụng.
- OCR cho điểm cao kể cả khi chữ bị dính liền (`Oneplatformtounifyallyourdata`).
- Ngưỡng lọc chữ rác 0.03 nằm rất sát chữ thật (dòng bản quyền 0.026, chữ thật nhỏ nhất
  0.032).
- Ảnh chụp ở giữa slide có thể thiếu những bullet hiện ra sau, hoặc thiếu phần nội dung
  thay đổi trong cùng slide ("Pros" rồi "Cons").

**Hướng xử lý:** đo phân phối các con số trên dữ liệu thật, thử nhiều giá trị ngưỡng, rồi
so sánh kết quả giữa các lần chạy bằng bảng `runs`.

**Giải pháp:**
- Cho phép truyền tham số khi chạy (ví dụ `--scene-threshold 0.2`), và tạo tham số **một
  lần** trong `run_all` để truyền xuống cả hai luồng. Như vậy `runs` và `provenance` luôn
  ghi đúng tham số đã dùng.
- Chạy với vài giá trị khác nhau, rồi so các chỉ số trong `runs`: số slide, % nearest, độ
  lệch p95.
- Thêm một chỉ số chất lượng chữ cho OCR, ví dụ tỉ lệ "từ" dài bất thường, để bắt chữ dính
  liền.
- Thử chụp nhiều ảnh cho mỗi slide (giữa và gần cuối), rồi lấy ảnh có nhiều chữ hơn.
- Ghi các ngưỡng đã chọn cùng lý do vào spec.

---

## Thứ tự đề xuất khi implement

1. **Bước 11 (idempotency)** trước: làm được ngay trên dữ liệu thật, không cần tạo lỗi giả,
   và có thể lộ ra những chỗ cần sửa trước khi làm hai bước kia.
2. **Bước 10 (replay)**: cần tạo dữ liệu lỗi để thử.
3. **Bước 12 (calibrate)** cuối cùng: làm khi pipeline đã được chứng minh là tất định, để
   biết mỗi thay đổi trong kết quả là do đổi ngưỡng, chứ không phải do chạy lại ngẫu nhiên.
