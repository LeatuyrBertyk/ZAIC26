# [2026] Hướng dẫn dùng docker để nộp bài cho Zalo AI Challenge 2026 - LifeMate - Agentic AI That Turns Your Voice Into Action

## 1. Giới thiệu
Tài liệu này hướng dẫn các đội thi Zalo AI Challenge 2026 đóng gói và nộp giải pháp dưới dạng Docker Image, bao gồm đầy đủ mã nguồn inference, mô hình, mã nguồn và dữ liệu huấn luyện (Training Code & Data), cùng các tài liệu phục vụ kiểm chứng kết quả.

Docker Image phải được hoàn thiện và nộp trước khi công bố bộ private test. Quá trình inference phải hoạt động hoàn toàn offline để BTC có thể chạy lại, kiểm chứng và xác định kết quả cuối cùng.

*   **Docker** là một nền tảng giúp người dùng đóng gói và chạy chương trình của mình trên các môi trường khác nhau một cách nhanh nhất dựa trên các container.
*   **Docker Image** là một dạng tập hợp các tệp của ứng dụng, được tạo ra bởi Docker engine. Nội dung của các Docker image sẽ không bị thay đổi khi di chuyển. Docker image được dùng để chạy các Docker container.
*   **Docker Container** là một dạng runtime của các Docker image, dùng để làm môi trường chạy ứng dụng.

Hướng dẫn chi tiết các bạn có thể tham khảo tại: [https://docs.docker.com/get-started/](https://docs.docker.com/get-started/)

## 2. Cài đặt Docker trên Ubuntu
Đối với các hệ điều hành khác, tham khảo cách cài đặt tại đây: [https://docs.docker.com/install/overview/](https://docs.docker.com/install/overview/)

1.  **Cài đặt docker**
    ```bash
    sudo apt-get install docker.io
    ```
2.  **Kiểm tra phiên bản docker**
    ```bash
    sudo docker --version
    ```
3.  **Chạy thử docker hello world**
    ```bash
    sudo docker run hello-world
    ```
4.  **Một số câu lệnh phổ biến**
    *   Liệt kê các images hiện có: `sudo docker images`
    *   Liệt kê các container hiện có: `sudo docker ps -a`

## 3. Cài đặt Nvidia Docker / Driver
Để sử dụng được GPU trên docker, bạn cần cài đặt Nvidia docker. Hướng dẫn cài đặt tại đây: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

**Lưu ý:** Để tránh lỗi khi chấm, các đội cần cài đặt các thư viện đồng bộ môi trường với server BTC:
*   **GPU:** 1 × NVIDIA GeForce RTX 5090 (32 GB), CPU 16 nhân Intel(R) Xeon(R) Gold 6459C, RAM 64 GB
*   **Driver Version:** 595.58.03
*   **CUDA Version:** 13.2.0

Base image phải dùng CUDA **không cao hơn 13.2** (driver của BTC chạy được các image CUDA 12.x và 13.x <= 13.2). Ví dụ nvidia docker image: `nvidia/cuda:13.2.0-devel-ubuntu22.04`.

## 4. Nộp bài cho Zalo AI Challenge
Các bước cần thiết để tạo được một Docker Image:
1.  Tạo một docker container mới (hoặc sử dụng docker có sẵn).
2.  Đưa model và source code đang có vào docker.
3.  Install các packages và libraries cần thiết mà solution của bạn sử dụng để chạy.
4.  Viết một file script tên `predict.sh`. File này chứa command các bước để chạy test. Nhận input từ `/data` và output ra `/result/submission.jsonl` (tuỳ theo format quy định của đề bài). Ngoài ra, xuất ra các thông số thời gian load model, thời gian predict ở stdout.
5.  Nộp file predict `predict_notebook.ipynb` để đo thời gian inference (xem phụ lục).
6.  Commit các thay đổi trong docker container.
7.  Save docker container thành 1 file image kèm checksum và nộp lên website cuộc thi.

## 5. Cấu trúc thư mục code
Ví dụ source code của bạn ở folder `/mnt/zac2026/` với cấu trúc như sau:
```text
/mnt/zac2026/
|---- predict.py
|---- preprocessing.py
|---- saved_models/
|     |---- models.safetensors # model cần đc copy vào trong docker
|---- requirements.txt
|---- predict.sh
|---- start_jupyter.sh
|---- predict_notebook.ipynb # sử dụng để đánh giá thời gian
|---- training_code/         # chứa toàn bộ mã nguồn training và README.
      |---- README.md
```

**Dữ liệu input (`/data`):** BTC mount thư mục chứa split private test vào `/data`, cùng cấu trúc với gói public test.
```text
/data/
|---- private_test/
      |---- private_test.jsonl                 # input từng mẫu: id, audio, context, history
      |---- private_test_tools_catalog.json    # schema các tool
      |---- private_test_environment/<id>/state.json # KHÔNG được đọc trực tiếp, chỉ qua zaic_tools
      |---- audio/<id>/<id>.wav                # yêu cầu cuối
      |---- <id>_u1.wav, <id>_u2.wav...        # các lượt user trong history
|---- zaic_tools/                              # thư viện chạy tool của private test
```

### Sample `predict.py` tối thiểu:
```python
import json, sys, time
from pathlib import Path

ROOT = Path("/data")
SPLIT = "private_test"
sys.path.insert(0, str(ROOT / SPLIT)) # chứa split zaic_tools của split, KHÔNG dùng bản đóng trong image
from zaic_tools import ToolEnv

def infer(item):
    t1 = time.perf_counter()
    env = ToolEnv(SPLIT, item["id"], root=ROOT)
    
    # Logic inference: ASR -> LLM -> env.call(tool, arguments) -> LLM -> ...
    
    tool_calls = env.get_tool_calls()
    state_hash = env.get_state_hash()
    ms = int((time.perf_counter() - t1) * 1000)
    return tool_calls, state_hash, ms

Path("/result").mkdir(parents=True, exist_ok=True)
with open(ROOT / SPLIT / f"{SPLIT}.jsonl", encoding="utf-8") as fin, \
     open("/result/submission.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        tool_calls, state_hash, ms = infer(item)
        fout.write(json.dumps({
            "id": item["id"], 
            "tool_calls": tool_calls, 
            "state_hash": state_hash, 
            "time": ms
        }, ensure_ascii=False) + "\n")
```

### Các bước thực hiện chi tiết:
**Bước 1: Khởi động docker container và đặt tên container với name zac2026**
```bash
docker run --gpus '"device=0"' --network host -it --name zac2026 nvidia/cuda:13.2.0-devel-ubuntu22.04 /bin/bash
```
*Bạn bắt buộc phải sử dụng base image có CUDA 13.2. Cờ `--network host` chỉ dùng lúc dựng image.*

**Bước 2: Kiểm tra container có sử dụng được GPU hay không**
```bash
nvidia-smi
```

**Bước 3: Copy source code vào trong container**
Mở một terminal mới trên máy host:
```bash
sudo docker cp /mnt/zac2026/ zac2026:/code/
```
*Mọi model phải nằm sẵn trong image, không tải từ HuggingFace lúc chạy.*

**Bước 4: Cài python và các package cần thiết trong container**
```bash
apt update; apt-get -y install libgl1-mesa-glx libglib2.0-0 vim ffmpeg; apt -y install python3-pip
```

**Bước 5: Cài đặt các thư viện cần thiết**
```bash
cd /code
pip install jupyterlab
pip install -r requirements.txt
```
*Lưu ý: Môi trường bắt buộc phải cài đặt `jupyterlab` để chấm thời gian inference.*

**Bước 6: Chỉnh sửa file `/code/predict.sh`**
Ví dụ có 2 bước để chạy:
```bash
python3 /code/preprocessing.py
python3 /code/predict.py
```

**Bước 7: Chỉnh sửa file `/code/start_jupyter.sh`**
```bash
jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --allow-root --no-browser
```

**Bước 8: Chỉnh sửa file `predict_notebook.ipynb`** (Xem phụ lục)

**Bước 9: Lưu lại các thay đổi trên container (Commit)**
Trên terminal máy host:
```bash
sudo docker commit zac2026 zac2026-teamname:v1
```

**Bước 10: Kiểm tra docker lần cuối**
*10.1 Kiểm tra hàm predict để xuất submission.jsonl*
```bash
sudo docker run --gpus '"device=0"' -v /mnt/zac2026/data:/data:ro -v /mnt/zac2026/result:/result zac2026-teamname:v1 /bin/bash /code/predict.sh
```
*10.2 Kiểm tra jupyter có chạy được không*
```bash
sudo docker run -it --gpus '"device=0"' -p 9777:9777 -v /mnt/zac2026/data:/data:ro -v /mnt/zac2026/result:/result zac2026-teamname:v1 /bin/bash /code/start_jupyter.sh
```
Truy cập `localhost:9777` với mật khẩu `zac2026`.

**Bước 11: Xuất image ra và nộp bài**
```bash
sudo docker save zac2026-teamname:v1 | gzip > zac2026_TeamName.tar.gz
```

## 6. Bổ sung bắt buộc về Training Code & Tài liệu mô tả
1.  **README.md mô tả ý tưởng:** Cần có file trình bày ý tưởng solution, training/inference: model ASR, LLM (tên, tham số, context length), cách gọi tool, sinh dữ liệu tổng hợp.
2.  **Cung cấp đầy đủ Training Code & Data:** BTC sẽ reproduce lại quá trình training. Các đội có thể upload data/models lên HuggingFace và ghi URL tải xuống trong README. Base model không được đưa vào Docker.
3.  **Cố định seed:** Phải set seed cố định trong training và inference để đảm bảo tính tái lập (reproducibility).

## 7. Phụ lục: Cấu trúc viết Jupyter notebook để đo lường thời gian
Notebook (`predict_notebook.ipynb`) phải gồm ít nhất 4 ô (cell) tương ứng với 4 bước:

**Bước 1: Set seed cố định**
```python
import os, torch, random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
```

**Bước 2: Nạp mô hình và các tài nguyên**
```python
# Không dùng API ngoài
asr_model = ...
model = ...
```

**Bước 3: Đọc nội dung test cases**
```python
import json, sys
from pathlib import Path
ROOT = Path("/data")
SPLIT = next(p.name for p in ROOT.iterdir() if (p / f"{p.name}_tools_catalog.json").exists())
sys.path.insert(0, str(ROOT / SPLIT))
from zaic_tools import ToolEnv

with open(ROOT / SPLIT / f"{SPLIT}.jsonl", encoding="utf-8") as f:
    test_cases = [json.loads(line) for line in f]
```

**Bước 4: Thực hiện dự đoán, in kết quả và thời gian**
```python
import time
all_result = []

for item in test_cases:
    t1 = time.perf_counter()
    env = ToolEnv(SPLIT, item["id"], root=ROOT)
    
    # run_agent(item, env)
    
    tool_calls, state_hash = env.get_tool_calls(), env.get_state_hash()
    ms = int((time.perf_counter() - t1) * 1000)
    all_result.append({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash, "time": ms})

Path("/result").mkdir(parents=True, exist_ok=True)
with open("/result/jupyter_submission.jsonl", "w", encoding="utf-8") as f:
    for r in all_result:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```
*Lưu ý: Kết quả từ jupyter phải có prefix là `jupyter_submission.jsonl`. Điểm của file này phải giống hệt file `submission.jsonl` thì thời gian mới được công nhận.*

## 8. Upload docker
1.  **Lấy checksum MD5:**
    ```bash
    md5sum zac2026_TeamName.tar.gz
    ```
2.  **Upload docker lên google drive:** Tải file `.tar.gz` lên Google Drive và chỉnh chế độ share thành "Anyone with the link".
3.  **Nộp link và checksum:** Nộp lại thông tin theo form/thông báo của BTC.