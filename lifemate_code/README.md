# Zalo AI Challenge 2026 — LifeMate: Voice-to-Action Agentic AI (v8.0)
**Đội thi (Team Name):** `AIO_HacDaiBangBang`

Tài liệu kỹ thuật mô tả toàn bộ ý tưởng giải pháp, kiến trúc mô hình, quy trình sinh dữ liệu tổng hợp (Synthetic Data Generation), huấn luyện (Training) và suy luận (Offline Deterministic Inference) của đội **`AIO_HacDaiBangBang`**.

---

## 1. Tổng quan Ý tưởng Giải pháp (Solution Architecture)

Bài toán **LifeMate** yêu cầu chuyển đổi câu lệnh giọng nói tiếng Việt đa vùng miền (Bắc - Trung - Nam, có pha trộn thuật ngữ tiếng Anh/code-switching và địa chỉ email/số hiệu đọc bằng lời) thành chuỗi lệnh gọi công cụ (`tool_calls`) chính xác trên môi trường giả lập `ToolEnv`, đảm bảo khớp hoàn toàn `state_hash` cuối cùng trong giới hạn `< 6.0B` tham số và `< 30 giây/mẫu`.

Giải pháp **LifeMate v8.0** gồm 2 tầng kết nối chặt chẽ:

1. **Tầng 1 — Dual-ASR Đa Phương Ngữ & Tiền Xử Lý Ngữ Âm (`lifemate/asr.py`):**
   - Kết hợp 2 mô hình ASR bổ trợ lẫn nhau với giải mã tất định (`beam_size=5`, `do_sample=False`, `temperature=0.0`):
     - **ASR 1 (`openai/whisper-large-v3-turbo` fine-tuned):** Mạnh về nhận diện từ tiếng Anh chêm vào (`check-in`, `playlist`, `book`, `COD`, mã chuyến bay/tàu `RV`, `SE`, mã chứng khoán `FPT`, `VCB`) với `initial_prompt` động sinh từ danh sách ứng dụng đã cài (`context.installed_apps`).
     - **ASR 2 (`vinai/PhoWhisper-medium` fine-tuned):** Chuyên biệt cho thanh điệu và phương ngữ tiếng Việt 3 miền Bắc - Trung - Nam.
   - **Chuẩn hóa thực thể nói (`clean_asr_text`):** Tự động khử lặp từ/ảo giác phụ đề và chuẩn hóa địa chỉ email đọc bằng lời (`"finance a còng example chấm test"` $\rightarrow$ `finance@example.test`, `"logistics a còng example chấm test"` $\rightarrow$ `logistics@example.test`, `"legal a còng example chấm test"` $\rightarrow$ `legal@example.test`).
   - **Bảo toàn ngữ cảnh hội thoại dài:** Phiên âm đầy đủ tới `30` lượt âm thanh lịch sử (`max_history_user_audios=30`), giải quyết trọn vẹn các mẫu hội thoại dài 18–24 lượt (`needle-in-a-haystack`).

2. **Tầng 2 — Catalog-Driven Multi-Step Tool-Calling Agent (`lifemate/agent.py` & `lifemate/prompting.py`):**
   - **Mô hình LLM:** `Qwen/Qwen3-4B-Instruct-2507` được tinh chỉnh bằng **PureLoRA** trên dữ liệu hội thoại Multi-turn Tool-Calling, sau đó hợp nhất trọng số (`merge`) thành mô hình độc lập `4.022B` tham số với `max_position_embeddings = 16384`.
   - **Ghép Kép Bản Dịch (Dual-ASR Fusion):** Đưa cả 2 bản dịch `[ASR-1 Whisper-v3-turbo]` và `[ASR-2 PhoWhisper-medium]` vào prompt để LLM đối chiếu chéo với dữ liệu thực tế trả về từ các lệnh `[READ]`.

---

## 2. Chi tiết Mô hình & Ngân sách Tham số (< 6.0B Budget)

| Thành phần | Mô hình Gốc (Pre-26/10/2026) | Phiên bản trong `/code/saved_models/` | Số Tham số | Context Length |
| :--- | :--- | :--- | :---: | :---: |
| **ASR 1 (Primary)** | `openai/whisper-large-v3-turbo` | `saved_models/asr_turbo_ft_v1` | `0.809 B` | 448 tokens |
| **ASR 2 (Secondary)** | `vinai/PhoWhisper-medium` | `saved_models/asr_pho_ft_v1` | `0.769 B` | 448 tokens |
| **LLM Agent** | `Qwen/Qwen3-4B-Instruct-2507` | `saved_models/qwen3_4b_sft_merged` | `4.022 B` | `16,384` tokens |
| **TỔNG CỘNG** | **Toàn bộ hệ thống Inference** | **100% Offline trong Docker** | **`5.600 B`** | **Đạt chuẩn `< 6.0B` (LLM `< 5.0B`)** |

* **Lưu ý tuân thủ quy chế:**
  - `config.json` của LLM được cố định `max_position_embeddings = 16384` (đúng quy định $\le 16\text{K}$ tokens).
  - Trọng số LoRA đã được merge trực tiếp vào `saved_models/qwen3_4b_sft_merged`, không đưa bản sao base model thừa vào Docker.

---

## 3. Cơ chế Gọi Tool & Hệ thống Chốt chặn (Precision-Safe Guardrails)

1. **Dynamic Tool Filtering (`filter_installed_tools`):**
   - Chỉ nạp vào prompt các công cụ thuộc ứng dụng trong `item["context"]["installed_apps"]` cộng với `sys.ask_missing_info`.
   - Gắn nhãn rõ `[READ]` / `[WRITE]` và danh sách `[THAM SỐ x-ref CẦN TRA CỨU TRƯỚC]` vào mô tả từng công cụ.
2. **Out-of-Scope Early Exit:**
   - Nếu ở bước đầu tiên, mọi công cụ mô hình đề xuất đều thuộc ứng dụng có thật trong hệ sinh thái nhưng **không được cài đặt** trên thiết bị của mẫu đó, Agent lập tức dừng và trả về `[]` (đạt điểm tuyệt đối trên các mẫu Out-of-Scope).
3. **Chốt chặn `env.validate()` trước khi thực thi `env.call()`:**
   - Mọi lệnh gọi hàm đều đi qua bộ chuẩn hóa kiểu dữ liệu theo Catalog (`_sanitize_arguments`: chuẩn hóa ISO-8601 `YYYY-MM-DDTHH:MM:SS+07:00`, `HH:MM`, ép kiểu số nguyên/số thực, sửa hoa/thường cho `enum`, quy đổi mã sân bay IATA `SGN/HAN/DAD` sang tên thành phố cho các công cụ đường bộ/thời tiết) rồi kiểm tra bằng `env.validate(name, args)`.
   - Nếu vi phạm schema, lỗi được phản hồi ngay trong bộ nhớ để LLM tự sửa mà **không** gọi `env.call()`, tránh làm bẩn lịch sử `tool_calls`.
4. **Chốt chặn Chống Bịa ID (`x-ref` Read-Before-Write Guard):**
   - Kiểm tra mọi tham số mang thuộc tính `x-ref` (như `device_id`, `contact_id`, `beneficiary_id`, `habit_id`, `item_id`). Nếu giá trị ID chưa từng xuất hiện trong lịch sử hội thoại hoặc kết quả `[READ]` của phiên hiện tại, Agent chặn lệnh `[WRITE]` trong bộ nhớ và yêu cầu LLM gọi công cụ `[READ]` tương ứng trước.
5. **Gợi ý Khôi phục Khi Tra cứu Rỗng (`annotate_tool_result`):**
   - Khi các công cụ tìm kiếm (`event_tickets.search`, `smart_home.get_status`, `recipes.search`, `ecommerce.search`, `grocery.search`) trả về danh sách rỗng `[]` do bộ lọc quá hẹp, hệ thống đính kèm gợi ý `_system_note` hướng dẫn LLM gọi lại với bộ lọc rộng hơn (`{}`) hoặc từ khóa gốc trước khi kết luận không tìm thấy.
6. **Chốt chặn Số tiền Mơ hồ (`amount_is_missing`):**
   - Phát hiện các câu lệnh chuyển tiền/nạp ví chỉ nói định lượng mơ hồ (*"chuyển một ít tiền"*, *"nạp chút tiền"*) mà không có số tiền cụ thể để tự động chuyển sang `sys.ask_missing_info(params=["amount"])`, chống hiện tượng LLM tự bịa số tiền `500000`.

---

## 4. Quy trình Sinh Dữ liệu Tổng hợp & Huấn luyện (Training Code & Data)

Toàn bộ mã nguồn huấn luyện nằm tại `/code/scripts/` và `/code/training_code/`:

### 4.1. Phân chia dữ liệu & Fine-tune cặp ASR
* **Chia tập Train/Val (`scripts/make_splits.py`):** Chia tập `train` (1,545 mẫu) thành `train_split.jsonl` (1,295 mẫu) và `val_split.jsonl` (250 mẫu) phân tầng theo ứng dụng và loại hành động với `seed=42`.
* **Fine-tune ASR (`scripts/finetune_asr.py`):**
  - Tinh chỉnh `openai/whisper-large-v3-turbo` $\rightarrow$ `asr_turbo_ft_v1` (giảm WER trên tập Val xuống `11.06%`).
  - Tinh chỉnh `vinai/PhoWhisper-medium` $\rightarrow$ `asr_pho_ft_v1` (giảm WER trên tập Val xuống `11.03%`).

### 4.2. Sinh dữ liệu tổng hợp (Synthetic Data Generation)
1. **Source A — ASR Noise & Colloquial Paraphrase Augmentation (`scripts/gen_data_source_a.py`):**
   - Kết hợp bản dịch thực tế có nhiễu từ cặp ASR (`asr_train_ft_cache.json`) với dữ liệu gốc để LLM học cách tự sửa lỗi đồng âm/code-switching thông qua kết quả trả về của tool `[READ]`.
2. **Source B — Rule-Based & Multi-App Synthetic Trajectories (`scripts/gen_data_source_b.py`):**
   - Sinh tự động các kịch bản kiểm tra biên: thiếu tham số bắt buộc (`sys.ask_missing_info`), yêu cầu ngoài phạm vi (`out-of-scope -> []`), đa ứng dụng (Multi-App) và chuỗi `Read -> Write` xác thực qua `ToolEnv`.
3. **Tập huấn luyện hội tụ v8 (`scripts/build_sft_data.py --v8-focused`):**
   - Kết hợp chuỗi suy luận đa bước từ nhãn chuẩn (`gold_train_replay`) cùng tập mẫu tăng cường tập trung vào các nhóm lỗi khó (`v8_focused_augmentation` + `source_b`), kiểm định 100% qua `schema_validate`.

### 4.3. Huấn luyện LLM PureLoRA (`scripts/train_sft_lora.py`)
* Huấn luyện `Qwen/Qwen3-4B-Instruct-2507` với `PureLoRA` (`r=16`, `lora_alpha=32`, `lr=1.5e-4`, `seed=42`, chỉ tính hàm mất mát trên các token `<tool_call>...</tool_call>` của `assistant`).
* Hợp nhất trọng số vào mô hình gốc bằng `scripts/prepare_docker_models.py` để tạo `saved_models/qwen3_4b_sft_merged`.

### 4.4. Tái lập toàn bộ quá trình Training bằng 1 lệnh
```bash
bash /code/training_code/train_all.sh
```

---

## 5. Tính Tái lập (Reproducibility) & Cố định Seed

* Mọi kịch bản huấn luyện (`train_all.sh`) và suy luận (`predict.py`, `predict_notebook.ipynb`) đều cố định:
  - `seed_everything(42)` (`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
  - `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
  - Giải mã tất định 100%: ASR dùng `beam_size=5, do_sample=False, temperature=0.0`; LLM dùng Greedy Decoding (`do_sample=False`).
* File `/result/submission.jsonl` (xuất từ `/code/predict.sh`) và `/result/jupyter_submission.jsonl` (xuất từ `/code/predict_notebook.ipynb`) dùng chung hàm nạp mô hình và vòng lặp suy luận nên cho kết quả `tool_calls` và `state_hash` **giống hệt nhau 100%**.

---

## 6. Liên kết Tải Dữ liệu & Checkpoint Huấn luyện (HuggingFace / Google Drive)

* **Base Models sử dụng (phát hành trước 26/10/2026):**
  - `openai/whisper-large-v3-turbo`: https://huggingface.co/openai/whisper-large-v3-turbo
  - `vinai/PhoWhisper-medium`: https://huggingface.co/vinai/PhoWhisper-medium
  - `Qwen/Qwen3-4B-Instruct-2507`: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
* **Fine-tuned Checkpoints & Synthetic SFT Data:**
  - Được đóng gói sẵn trong `/code/saved_models/` (cho inference offline) và có thể tái tạo trực tiếp từ `/code/training_code/train_all.sh`.

