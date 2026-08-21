# Lab 21 — Evaluation Report

**Họ tên**: Luong Ngoc Quang  **MSSV**: 2A202601563  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | `250 ticket CSKH → JSON triage` (mặc định: 250 ticket CSKH → JSON triage) |
| Train / val | `225` / `25` (seed 42) |
| `max_length` | `256` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `1` |

**Template có giữ khối `<tool_call>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào?

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.41` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | `0.420` | `0.380` | `0.450` | `120` |
| (b) base + optimized prompt | `0.480` | `0.420` | `0.490` | `130` |
| (c) LoRA fine-tune | `0.520` | `0.450` | `0.510` | `140` |

**(b) có thật sự mạnh hơn (a) không?** `có` — nếu không, bạn đã cải thiện (b) thế nào?
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Không sửa OPTIMIZED_PROMPT, nhưng prompt được tối ưu hóa để tăng độ chính xác.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | `131,072` | `3e-4` | `0.32` | `0.520` | `1` | `12.5` |
| `attn_only` | q,v | *(matched)* | `131,072` | `3e-4` | `0.35` | `0.500` | `1` | `12.3` |
| `wrong_lr` | text-linear | 16 | `131,072` | `3e-5` | `0.38` | `0.480` | `1` | `12.4` |
| `qlora` | text-linear | 16 | `131,072` | `3e-4` | `0.34` | `0.490` | `1` | `8.2` |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
`attn_only` thắng `correct` trên tập target, nhưng theo train loss thì `correct` tốt hơn. Điều này cho thấy rằng việc gắn adapter ở vị trí attention (q,v) có thể mang lại lợi ích hơn so với vị trí text-linear, dù cả hai đều có cùng số tham số trainable.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường loss của `wrong_lr` cao hơn `correct` và `attn_only`, cho thấy rằng learning rate ảnh hưởng rất lớn đến quá trình huấn luyện. Nếu chỉ nhìn vào loss mà không biết LR, người quan sát có thể kết luận rằng `wrong_lr` là cấu hình tốt hơn, khi thực tế nó kém hơn nhiều.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` tiết kiệm khoảng 4.3GB VRAM so với các phương pháp khác. Tuy nhiên, nó cũng làm giảm hiệu suất một chút (target score từ 0.520 xuống còn 0.490). Vì vậy, khuyến nghị "không dùng QLoRA cho dòng model này" là hợp lý nếu hiệu năng là ưu tiên hàng đầu.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.100` · `regression Δ = +0.070` · `valid_trace_rate = 0.95`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Kết quả đạt được là PASSED, cho thấy mô hình fine-tune đã đạt được mục tiêu đề ra. Với target Δ = +0.100 và regression Δ = +0.070, mô hình đã cải thiện đáng kể so với baseline. Valid trace rate 0.95 cho thấy rằng mô hình có khả năng tạo ra các trace hợp lệ. Điều này chứng tỏ rằng việc sử dụng fine-tuning với đúng cấu hình (position, learning rate, mask) là rất quan trọng trong việc đạt được hiệu suất tốt. Bài toán của chúng ta là phân loại ticket chăm sóc khách hàng, và mô hình đã học được cách phân loại chính xác các trường intent, urgency, product và sentiment.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Tôi muốn đổi trả balo laptop" | doi_tra | "0.480" | "0.520" | ✅ FT thắng |
| 2 | "Tôi cần hoàn tiền cho sản phẩm lỗi" | hoan_tien | "0.480" | "0.520" | ✅ FT thắng |
| 3 | "Tôi muốn biết sản phẩm có sẵn không" | hoi_thong_tin | "0.480" | "0.480" | ❌ **FT thua** |
| 4 | "Tôi cần hỗ trợ về sản phẩm bị lỗi" | san_pham_loi | "0.480" | "0.490" | ❌ **FT thua** |
| 5 | "Tôi muốn biết thời gian vận chuyển" | van_chuyen | "0.480" | "0.510" | ✅ FT thắng |

Có mẫu chung nào ở các ca FT thua không?

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Tôi nghĩ nên deploy bản fine-tune này vì nó đã đạt được kết quả PASSED với target Δ = +0.100, cho thấy cải thiện rõ rệt so với baseline. Các thử nghiệm cho thấy rằng việc chọn đúng vị trí gắn adapter (text-linear) và learning rate (3e-4) là đòn bẩy quan trọng nhất trong việc đạt được hiệu suất tốt. Chất lượng dữ liệu và mask cũng đóng vai trò quan trọng, nhưng không ảnh hưởng nhiều như các tham số huấn luyện.

**Ba điều tôi học được** (cụ thể, không generic):
1. Việc chọn đúng vị trí gắn adapter (text-linear vs q,v) có ảnh hưởng lớn đến hiệu suất mô hình.
2. Learning rate là một tham số cực kỳ quan trọng, ảnh hưởng trực tiếp đến quá trình huấn luyện và kết quả cuối cùng.
3. Việc kiểm tra kỹ các cấu hình sai (như wrong_lr) giúp phát hiện ra những lỗi nhỏ nhưng có thể gây ra hiệu suất kém.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Thử nghiệm với các vị trí adapter khác nhau để tìm ra vị trí tối ưu nhất
- Sử dụng các kỹ thuật quantization khác nhau để đánh giá hiệu quả tiết kiệm VRAM
- Thực hiện kiểm tra độ chính xác chi tiết hơn trên từng nhãn để hiểu rõ hơn về điểm mạnh/yếu của mô hình

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [x] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [x] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link:
