# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> - Ở temperature thấp (0.0 và 0.5), các phản hồi mang tính nhất quán cao, tập trung vào các thông tin lịch sử hoặc địa lý phổ biến (như hình dạng chữ S, xuất khẩu cà phê/hạt điều) với câu từ trang trọng và cấu trúc chuẩn xác.
> - Ở temperature cao (1.0 và 1.5), mô hình bắt đầu đưa ra các phản hồi đa dạng và sáng tạo hơn, sử dụng cấu trúc câu phóng khoáng hơn. Tuy nhiên, ở mức quá cao (1.5), văn bản có thể trở nên thiếu tự nhiên, lặp từ hoặc có nguy cơ xuất hiện thông tin sai lệch (hallucination).

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature từ **0.0 đến 0.2**. 
> Lý do là vì chatbot hỗ trợ khách hàng yêu cầu tính chính xác tối đa, nhất quán và đáng tin cậy. Việc hạ thấp temperature giúp giảm thiểu nguy cơ mô hình "bịa đặt" thông tin (hallucination) và đảm bảo các câu trả lời về chính sách, giá cả, hoặc hướng dẫn kỹ thuật luôn tuân thủ đúng tài liệu chuẩn của doanh nghiệp.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Theo bảng giá `PRICING_1M_TOKENS`:
> - Giá GPT-4o: $5.00/1M input tokens và $20.00/1M output tokens.
> - Giá GPT-4o-mini: $0.150/1M input tokens và $0.600/1M output tokens.
> - Tỷ lệ giá (GPT-4o / GPT-4o-mini) cho cả Input và Output đều là: $5.00 / $0.150 = $20.00 / $0.600 = **33.33 lần**.
> Do đó, với workload trên và với bất kỳ tỉ lệ phân bổ input/output tokens nào, GPT-4o luôn đắt hơn GPT-4o-mini đúng **33.33 lần**.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> - **GPT-4o xứng đáng:** Khi ứng dụng yêu cầu khả năng suy luận logic cực kỳ cao, phân tích đa bước phức tạp, hoặc trích xuất thông tin có cấu trúc phức tạp từ văn bản dài (như thẩm định hợp đồng pháp lý, lập trình các thuật toán phức tạp, phân tích báo cáo tài chính chuyên sâu).
> - **GPT-4o-mini tốt hơn:** Khi cần thực hiện các tác vụ đơn giản, lặp đi lặp lại với số lượng lớn (high volume, low complexity) như: phân loại cảm xúc (sentiment analysis), tóm tắt văn bản ngắn, phân loại email/nhãn ticket hỗ trợ khách hàng, hoặc chatbot giao tiếp thông thường với người dùng.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> - **Streaming quan trọng nhất** trong các ứng dụng chat tương tác trực tiếp với người dùng (chatbot, trợ lý ảo) hoặc khi tạo văn bản dài (bài blog, báo cáo). Việc stream từng token khi chúng được sinh ra (Time to First Token - TTFT cực thấp) giúp mang lại cảm giác phản hồi tức thì, nâng cao trải nghiệm người dùng bằng cách giảm sự chờ đợi trong khi mô hình đang xử lý toàn bộ câu trả lời dài.
> - **Non-streaming phù hợp hơn** trong các quy trình xử lý ngầm (background job), tích hợp API hệ thống (system-to-system), hoặc các ứng dụng yêu cầu xử lý/phân tích toàn bộ nội dung hoàn chỉnh trước khi chuyển sang bước tiếp theo (ví dụ: chấm điểm bài viết, biên dịch văn bản, trích xuất dữ liệu JSON có cấu trúc để đưa vào database). Trong các trường hợp này, việc streaming là dư thừa và làm tăng độ phức tạp trong lập trình xử lý kết quả.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/ -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
