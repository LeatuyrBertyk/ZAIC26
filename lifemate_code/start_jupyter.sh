#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ -d "/code" ]; then
    cd /code
else
    cd "$(dirname "${BASH_SOURCE[0]}")"
fi

jupyter lab --port 9777 --ip 0.0.0.0 --NotebookApp.password='zac2026' --NotebookApp.token='zac2026' --allow-root --no-browser

