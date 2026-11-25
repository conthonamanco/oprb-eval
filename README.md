# Đánh giá VLM và OCR nhẹ trên ảnh tài liệu bị che

So sánh **5 model** OCR/VLM nhẹ, dùng nguyên trọng số pretrained, trên **3.000 ảnh** lấy ngẫu nhiên từ
9.000 ảnh test của [OPRB](https://huggingface.co/datasets/kpurkayastha/OPRB): 2.000 ảnh `text` và 1.000 ảnh `nontext`.
Thước đo: CER, WER, Precision/Recall/F1 và thời gian suy luận.

| Model | Nhóm | Cách đọc chữ |
|---|---|---|
| Tesseract 5 (`tessdata_best`) | OCR | LSTM theo dòng, chạy CPU |
| PP-OCRv5 mobile, bản tiếng Anh | OCR | Detector + CTC theo dòng |
| docTR FAST-base + PARSeq | OCR | Detector theo từ + decoder tự hồi quy |
| LightOnOCR-2-1B | VLM chuyên OCR | Đọc cả trang một lần |
| Qwen3-VL-2B-Instruct | VLM tổng quát | Đọc cả trang theo prompt |

**Trạng thái:** đã có code cho cả 5 model và notebook chạy. Đã chạy pilot hai model CPU; chưa chạy pilot trên T4, chưa có kết quả.

## Cấu trúc

```
configs/
  models.toml      đúng 5 model; là một phần của fingerprint mỗi lần chạy
  smoke.toml       model giả để thử pipeline
src/oprb_eval/
  data.py          tải, ghép cặp, đọc GT, lấy mẫu phân tầng
  text.py          chuẩn hóa văn bản, hai giao thức notebook / normalized
  metrics.py       CER, WER, P/R/F1
  models.py        giao diện model và danh sách adapter
  adapters/        mỗi họ model một file: tesseract, ppocr, doctr_words, vlm
  reading_order.py khử nghiêng, gom dòng, tách cột cho OCR trả về khung chữ
  preprocess.py    phóng to, khử nghiêng theo khuyến nghị chính thức của model
  pilot.py         kiểm tra kỹ thuật (lỗi, tốc độ, dtype), không đọc GT
  runner.py        suy luận (batch, shard, resume) và đo thời gian
  env.py           ghi lại GPU và phiên bản thư viện của mỗi phiên
  score.py         chấm điểm, tổng hợp
  cli.py           các lệnh download, stats, pilot, run, latency, score
tests/             unit test, chỉ cần thư viện chuẩn
notebooks/run.ipynb  chạy một model trên Kaggle/Colab
report/main.tex    khung báo cáo
data/, results/    sinh ra khi chạy; dữ liệu và dự đoán thô không commit
```

## Chạy trên Kaggle hoặc Colab

Cách dễ nhất: mở `notebooks/run.ipynb` trên Kaggle/Colab, sửa `MODEL` và `STAGE` rồi chạy.
Notebook tự tải dữ liệu vào ổ tạm, chia việc theo số GPU/CPU, resume và chấm điểm.

Chạy tay thì đứng ở thư mục gốc của repo. Không cài lại `torch`: Kaggle và Colab đã có bản khớp với CUDA của máy.

```bash
pip install -q -e ".[data]"
python -m unittest discover -s tests
python -m oprb_eval download                 # khoảng 1,75 GB
python -m oprb_eval stats --sample 3000       # kèm danh sách ảnh được chọn

# Thử pipeline bằng model giả
python -m oprb_eval run   --model smoke --config configs/smoke.toml --sample 40
python -m oprb_eval score --model smoke --sample 40
```

Chạy 3.000 ảnh trên Kaggle T4×2, mỗi GPU một nửa:

```bash
CUDA_VISIBLE_DEVICES=0 python -m oprb_eval run --model qwen3_vl_2b --sample 3000 --shard 0/2 --batch-size 8 &
CUDA_VISIBLE_DEVICES=1 python -m oprb_eval run --model qwen3_vl_2b --sample 3000 --shard 1/2 --batch-size 8 &
wait
python -m oprb_eval score   --model qwen3_vl_2b --sample 3000
python -m oprb_eval latency --model qwen3_vl_2b --sample 100
```

Phiên bị ngắt thì giữ lại thư mục `results/` và chạy lại đúng lệnh cũ: ảnh đã xong được bỏ qua, ảnh lỗi được thử lại.

## Cách lấy mẫu

- `--sample N` là **tổng** số ảnh, seed mặc định 42; bỏ `--sample` thì dùng cả 9.000 ảnh.
- Lấy ngẫu nhiên phân tầng, giữ đúng tỉ lệ gốc: 2 `text` : 1 `nontext`, và tỉ lệ loại vết × mật độ trong từng vùng.
  Vì vậy trung bình trên mẫu ước lượng thẳng trung bình trên 9.000 ảnh, không cần trọng số.
- Cả 5 model dùng cùng một danh sách ảnh; `stats --sample 3000` ghi danh sách ra `results/pages_sample3000.txt`.
- Mẫu nhỏ là phần đầu của mẫu lớn cùng seed, nên tăng N không phải chạy lại ảnh đã xong.
- Trung bình CER, WER và F1 kèm khoảng tin cậy bootstrap 95%.

## Nguyên tắc

- Dùng nguyên trọng số pretrained, không train hay fine-tune. Chỉ tải và dùng tập test.
- Model chỉ nhận ảnh trong `test/*/img/`. GT chỉ được đọc khi chấm điểm và thống kê dữ liệu.
- Prompt và tham số lấy theo tài liệu chính thức hoặc giá trị cố định đặt trước khi chạy, không chọn theo điểm.
- Thiếu ảnh thì từ chối chạy và chấm; ảnh lỗi bị chấm như chuỗi rỗng và được đếm riêng
  (`n_failed`); cột `*_ok_mean` là trung bình chỉ tính ảnh thành công.
- Thời gian đo riêng, từng ảnh một, trên cùng loại máy (phiên T4; OCR CPU không dùng GPU); không lấy từ lượt chạy theo batch.
- Tiền xử lý ảnh chỉ áp dụng khi tài liệu của model khuyến nghị.
- Hai VLM dùng greedy decoding; trang lặp một mẫu suốt 256 token thì dừng và được đánh dấu `truncated`.
- PP-OCR và docTR dùng chung một cách sắp thứ tự đọc; Tesseract và VLM tự sắp.
- Đổi cấu hình model thì phải chạy lại từ đầu; repo không cho trộn kết quả.
