# Tổng hợp "Tricks" & Chiến lược cho dự án LifeMate (Zalo AI Challenge)

Dựa trên tài liệu Slide hướng dẫn Track 1 - LifeMate, dưới đây là những kỹ thuật "ăn điểm" và chiến lược tối ưu nhất mà nhóm phát triển Baseline đã đúc kết được. Những mẹo này tập trung vào việc tối ưu hệ thống đánh giá, mẹo huấn luyện LLM và xử lý lỗi ASR.

## 1. Tối ưu Điểm số (Scoring) & Tương tác `ToolEnv`
* **Mục tiêu tối thượng là `State Hash` cuối cùng:** Nếu State Hash ở bước cuối cùng sai, toàn bộ mẫu đó sẽ nhận **0 điểm**. Việc có thừa hoặc thiếu các bước đọc (read calls) làm giảm độ chính xác (Precision) nhưng không làm chết điểm hoàn toàn. -> **Hành động:** Tập trung tuyệt đối vào việc đảm bảo các lệnh `write` (lệnh thay đổi trạng thái) là chính xác.
* **[Cực kỳ quan trọng] Khóa van trước `ToolEnv` (Validate first):** Mọi lệnh gọi vào `ToolEnv` đều bị hệ thống ghi nhận, kể cả lệnh bị lỗi (sai schema, sai ID). Các lệnh lỗi này bị tính là "extra calls" và làm giảm F1 score. -> **Trick:** Hãy tự viết một hàm `validate(call)` (kiểm tra schema, check x-ref) ngay trong code của bạn. Nếu sai, trả lỗi về thẳng LLM thay vì ném vào `ToolEnv`.
* **Cơ chế chặn (Guards):**
    * **Không lặp lại:** Chặn hoàn toàn việc mô hình gọi lại 2 lệnh giống hệt nhau liên tiếp.
    * **Giới hạn thử lại (Retry cap):** Ví dụ, nếu một lệnh `read` (tìm kiếm) trả về rỗng 2 lần, hãy ngắt luôn để mô hình không bị kẹt trong vòng lặp tìm kiếm (Ví dụ: tìm chuyến bay).

## 2. Kỹ thuật Huấn luyện LLM (Training Strategy)
* **Tùy chỉnh Trọng số Loss (Loss Weighting):** Phần văn bản tự do (free text - ví dụ nội dung tin nhắn) chiếm rất nhiều token và chiếm 41-66% loss, làm LLM "lười" học các logic gọi hàm quan trọng hơn. -> **Trick:** Giảm trọng số loss của phần "free text" xuống `0.3`, và giữ nguyên trọng số `1.0` cho tên hàm, x-ref IDs, và token kết thúc (stop).
* **Dạy mô hình cách "Sửa sai" (Recovery Training):** Đừng chỉ dạy LLM bằng các kịch bản hoàn hảo. Trong tập huấn luyện, hãy chủ động chèn các lỗi (như tìm kiếm sai/không ra kết quả) với `loss = 0`, sau đó phạt `loss = 1` ở hành động tiếp theo để dạy LLM **biết phải làm gì sau khi tìm kiếm thất bại** (Ví dụ: hỏi lại người dùng hoặc thử từ khóa khác).

## 3. Kiến trúc Hệ thống & Tối ưu Bầu chọn (Ensemble)
* **Vote theo từng bước (Vote per step):** Thay vì cho 3 mô hình LoRA chạy 3 luồng độc lập đến cuối cùng rồi vote trên Hash (cách này lãng phí 2 lần chạy `ToolEnv` vốn rất tốn kém), hãy cho 3 mô hình dự đoán lệnh tiếp theo, **bầu chọn (majority vote) ngay tại bước đó**, rồi lấy lệnh thắng cuộc nạp vào 1 `ToolEnv` duy nhất. Cách này giúp tăng 1.8 điểm Validation.
* **Prompt Engineering với dữ liệu thực tế:** Đừng để LLM tự mò. Hãy thêm hẳn một bảng "Ngày trong tuần" (weekday table) và map cứng tên các ứng dụng theo thói quen của người dùng trong tập TRAIN (VD: người dùng nói "bên ví", map thẳng thành `ewallet`). Chi tiết nhỏ này tăng tận 5.1 điểm!

## 4. Vũ khí bí mật cho việc leo Top (Next Steps từ Slide 35)
* **Suy luận có ngân sách (Chain-of-Thought / Short verified notes):** Cho LLM sinh ra một dòng lý luận ngắn (max 256 tokens) trước khi thực thi lệnh `write`. Đặc biệt, dùng code python để kiểm tra chéo các con số (tiền, thời gian) trong dòng suy luận đó trước khi chốt hạ.
* **Nhiều giả thuyết ASR (Multiple Hypotheses):** Vì ASR luôn có sai sót, hãy thử truyền vào prompt cho LLM **cả 2 phiên bản transcript** (một từ model ASR tinh chỉnh, một từ model base). Điều này giúp LLM có thêm góc nhìn để "đoán" đúng các thực thể (tên riêng, số lượng).
* **Mô phỏng App mới (Synthetic unseen apps):** 45% dữ liệu public test dùng các App chưa từng xuất hiện trong tập Train. Hãy tự sinh thêm dữ liệu huấn luyện giả lập cho các app mới (dựa trên schema) nhưng vẫn giữ nguyên văn phong của tập TRAIN để rèn khả năng Zero-shot cho LLM.