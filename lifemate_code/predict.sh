#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CODE_DIR="/code"
if [ ! -f "${CODE_DIR}/predict.py" ]; then
    CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

python3 -X utf8 "${CODE_DIR}/preprocessing.py"
python3 -X utf8 "${CODE_DIR}/predict.py" "$@"

