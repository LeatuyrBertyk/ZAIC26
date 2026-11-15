#!/usr/bin/env bash
# Script đóng gói và xuất Docker Image nộp bài chuẩn quy định BTC cho đội: AIO_HacDaiBangBang
# Tài liệu tham chiếu: zalo_ai_challenge_2026_docker_guide.md (Mục 5, 7, 8)
set -euo pipefail

TEAM_NAME="AIO_HacDaiBangBang"
IMAGE_TAG="zac2026-aio_hacdaibangbang:v1"
TAR_FILE="zac2026_${TEAM_NAME}.tar.gz"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo " ĐÓNG GÓI DOCKER ZALO AI CHALLENGE 2026 — ĐỘI: ${TEAM_NAME}"
echo "=================================================================="

# 1. Kiểm tra thư mục saved_models/ trước khi đóng gói vào image
if [ ! -f "${CODE_DIR}/saved_models/qwen3_4b_sft_merged/config.json" ]; then
    echo "⚙️ [Bước 0] Đang chuẩn bị saved_models/ (ASR + Merge PureLoRA vào Qwen3-4B)..."
    python3 -X utf8 "${CODE_DIR}/scripts/prepare_docker_models.py" --out-dir "${CODE_DIR}/saved_models"
fi

# 2. Thực hiện đúng theo các bước trong Mục 5 của zalo_ai_challenge_2026_docker_guide.md:
echo "🚀 [Bước 1-5] Khởi tạo container zac2026 từ nvidia/cuda:13.2.0-devel-ubuntu22.04 và cài đặt môi trường..."
sudo docker rm -f zac2026 2>/dev/null || true
sudo docker run -d --gpus '"device=0"' --network host --name zac2026 nvidia/cuda:13.2.0-devel-ubuntu22.04 sleep infinity

echo "📂 [Bước 3] Copy toàn bộ mã nguồn, training_code/ và saved_models/ vào zac2026:/code/ ..."
sudo docker exec zac2026 mkdir -p /code /result
sudo docker cp "${CODE_DIR}/." zac2026:/code/

echo "📦 [Bước 4-5] Cài đặt gói hệ thống, jupyterlab và requirements.txt trong container zac2026..."
sudo docker exec zac2026 bash -c "
    export DEBIAN_FRONTEND=noninteractive &&
    apt-get update &&
    apt-get -y install libgl1-mesa-glx libglib2.0-0 libsndfile1 vim ffmpeg python3 python3-pip &&
    cd /code &&
    pip install jupyterlab &&
    pip install -r requirements.txt &&
    chmod +x /code/predict.sh /code/start_jupyter.sh /code/training_code/train_all.sh
"

echo "💾 [Bước 9] Commit container zac2026 thành image: ${IMAGE_TAG} ..."
sudo docker commit \
    --change 'ENV PYTHONUTF8=1' \
    --change 'ENV PYTHONHASHSEED=42' \
    --change 'ENV CUBLAS_WORKSPACE_CONFIG=:4096:8' \
    --change 'ENV HF_HUB_OFFLINE=1' \
    --change 'ENV TRANSFORMERS_OFFLINE=1' \
    --change 'ENV HF_DATASETS_OFFLINE=1' \
    --change 'WORKDIR /code' \
    zac2026 "${IMAGE_TAG}"
sudo docker stop zac2026 && sudo docker rm zac2026

echo "📦 [Bước 11 & Mục 8] Xuất file ${TAR_FILE} và tính mã MD5 checksum..."
sudo docker save "${IMAGE_TAG}" | gzip > "${TAR_FILE}"
md5sum "${TAR_FILE}" | tee "${TAR_FILE}.md5"

echo "=================================================================="
echo "✅ HOÀN TẤT! File nộp bài của đội ${TEAM_NAME}: ${TAR_FILE}"
echo "   Mã MD5 đã được lưu tại: ${TAR_FILE}.md5"
echo "=================================================================="

