#!/usr/bin/env bash
# End-to-End Deterministic Training Pipeline for ZAC 2026 LifeMate (seed=42)
set -euo pipefail

export PYTHONUTF8=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${CODE_DIR}/../train+public_test}"
cd "${CODE_DIR}"

echo "=== [Step 1/6] Creating Deterministic Stratified Train/Val Split (seed=42) ==="
python3 -X utf8 scripts/make_splits.py --data-root "${DATA_ROOT}" --seed 42

echo "=== [Step 2/6] Fine-tuning Dual-ASR Models (Whisper-large-v3-turbo & PhoWhisper-medium) ==="
python3 -X utf8 scripts/finetune_asr.py \
    --model-name openai/whisper-large-v3-turbo \
    --data-root "${DATA_ROOT}" \
    --out-dir saved_models/asr_turbo_ft_v1 \
    --seed 42

python3 -X utf8 scripts/finetune_asr.py \
    --model-name vinai/PhoWhisper-medium \
    --data-root "${DATA_ROOT}" \
    --out-dir saved_models/asr_pho_ft_v1 \
    --seed 42

echo "=== [Step 3/6] Generating Synthetic Multi-Turn Tool-Calling Data (Sources A & B) ==="
python3 -X utf8 scripts/gen_data_source_a.py --data-root "${DATA_ROOT}" --seed 42
python3 -X utf8 scripts/gen_data_source_b.py --data-root "${DATA_ROOT}" --seed 42

echo "=== [Step 4/6] Building Focused v8 SFT Dataset ==="
python3 -X utf8 scripts/build_sft_data.py \
    --data-root "${DATA_ROOT}" \
    --include-source-b \
    --source-b-count 250 \
    --v8-focused \
    --out outputs/sft/sft_train_v8.jsonl \
    --seed 42

echo "=== [Step 5/6] Fine-tuning Qwen3-4B-Instruct-2507 with PureLoRA (seed=42) ==="
python3 -X utf8 scripts/train_sft_lora.py \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --train-jsonl outputs/sft/sft_train_v8.jsonl \
    --val-jsonl outputs/sft/sft_val.jsonl \
    --out-dir saved_models/qwen3_4b_sft_lora_v8 \
    --epochs 1 \
    --lr 1.5e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --seed 42

echo "=== [Step 6/6] Merging PureLoRA into Standalone Model & Verifying <6B Budget ==="
python3 -X utf8 scripts/prepare_docker_models.py \
    --out-dir saved_models \
    --asr-turbo saved_models/asr_turbo_ft_v1 \
    --asr-pho saved_models/asr_pho_ft_v1 \
    --lora-dir saved_models/qwen3_4b_sft_lora_v8

echo "=== Training & Model Packaging Complete! ==="

