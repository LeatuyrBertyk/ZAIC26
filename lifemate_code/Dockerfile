# Zalo AI Challenge 2026 — LifeMate Official Submission Dockerfile
# Base image strictly matches BTC requirement (CUDA 13.2.0, Ubuntu 22.04, RTX 5090 / Driver 595.58.03)
FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TOKENIZERS_PARALLELISM=false

# Step 4 in BTC Guide: Install system libraries, ffmpeg, and Python 3 pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsndfile1 \
        vim \
        ffmpeg \
        python3 \
        python3-pip \
        python3-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Step 5 in BTC Guide: Install jupyterlab and requirements.txt
COPY requirements.txt /code/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir jupyterlab && \
    pip3 install --no-cache-dir -r /code/requirements.txt

# Step 3 in BTC Guide: Copy source code, training_code/, and saved_models/ into /code
COPY . /code/

RUN chmod +x /code/predict.sh /code/start_jupyter.sh /code/training_code/train_all.sh && \
    mkdir -p /result

# Lock offline mode after build so runtime never accesses HuggingFace or external network
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

CMD ["/bin/bash", "/code/predict.sh"]

