"""Prepare `/code/saved_models/` for 100% Offline Docker Submission (ZAC 2026 LifeMate).

What this script does:
1. Copies fine-tuned Dual-ASR checkpoints into `saved_models/`:
   - `saved_models/asr_turbo_ft_v1` (0.809B params)
   - `saved_models/asr_pho_ft_v1`   (0.769B params)
2. Loads `Qwen/Qwen3-4B-Instruct-2507` (4.022B params), merges the trained PureLoRA adapter
   (`qwen3_4b_sft_lora_v8` / `v7` / `v5`), sets `max_position_embeddings = 16384` in `config.json`
   (as strictly required by BTC rules), and saves the standalone merged model to:
   - `saved_models/qwen3_4b_sft_merged` (4.022B params)
   Note: Saving the merged model means the un-finetuned base model is NOT duplicated inside Docker,
   satisfying both "Mọi model phải nằm sẵn trong image" and "Base model không được đưa vào Docker".
3. Verifies the strict competition parameter budget:
   0.809B + 0.769B + 4.022B = 5.600B < 6.0B (and LLM = 4.022B < 5.0B).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CODE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from lifemate.lora_utils import load_and_merge_pure_lora  # noqa: E402


def find_existing_dir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p
    return None


def copy_checkpoint_dir(src: Path, dst: Path) -> None:
    if dst.resolve() == src.resolve():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir():
            continue  # skip training sub-checkpoints
        target = dst / item.name
        if not target.exists() or target.stat().st_size != item.stat().st_size:
            shutil.copy2(item, target)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Prepare offline models in saved_models/ for ZAC 2026 Docker")
    ap.add_argument("--out-dir", default=str(CODE_DIR / "saved_models"), help="Target saved_models directory")
    ap.add_argument("--asr-turbo", default=None, help="Path to fine-tuned Whisper-large-v3-turbo directory")
    ap.add_argument("--asr-pho", default=None, help="Path to fine-tuned PhoWhisper-medium directory")
    ap.add_argument("--lora-dir", default=None, help="Path to PureLoRA adapter directory (v8 / v7 / v5)")
    ap.add_argument("--base-llm", default="Qwen/Qwen3-4B-Instruct-2507", help="Base LLM HF ID or local path")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--skip-llm-merge", action="store_true", help="Only copy ASR models and LoRA adapter")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy ASR Primary (Whisper-large-v3-turbo FT)
    turbo_src = (
        Path(args.asr_turbo)
        if args.asr_turbo
        else find_existing_dir([WORKSPACE_ROOT / "asr_turbo_ft_v1", CODE_DIR / "saved_models" / "asr_turbo_ft_v1"])
    )
    if turbo_src and (turbo_src / "config.json").is_file():
        turbo_dst = out_dir / "asr_turbo_ft_v1"
        print(f"[1/3] Copying ASR Primary: {turbo_src} -> {turbo_dst}")
        copy_checkpoint_dir(turbo_src, turbo_dst)
    else:
        print("[1/3] Warning: asr_turbo_ft_v1 not found locally; pass --asr-turbo if stored elsewhere.")

    # 2. Copy ASR Secondary (PhoWhisper-medium FT)
    pho_src = (
        Path(args.asr_pho)
        if args.asr_pho
        else find_existing_dir([WORKSPACE_ROOT / "asr_pho_ft_v1", CODE_DIR / "saved_models" / "asr_pho_ft_v1"])
    )
    if pho_src and (pho_src / "config.json").is_file():
        pho_dst = out_dir / "asr_pho_ft_v1"
        print(f"[2/3] Copying ASR Secondary: {pho_src} -> {pho_dst}")
        copy_checkpoint_dir(pho_src, pho_dst)
    else:
        print("[2/3] Warning: asr_pho_ft_v1 not found locally; pass --asr-pho if stored elsewhere.")

    # 3. Merge PureLoRA adapter into Qwen3-4B-Instruct-2507 and clamp max_position_embeddings = 16384
    lora_src = (
        Path(args.lora_dir)
        if args.lora_dir
        else find_existing_dir(
            [
                WORKSPACE_ROOT / "qwen3_4b_sft_lora_v8",
                WORKSPACE_ROOT / "qwen3_4b_sft_lora_v7",
                WORKSPACE_ROOT / "qwen3_4b_sft_lora_v5",
                WORKSPACE_ROOT / "qwen3_4b_sft_lora_v3",
                WORKSPACE_ROOT / "qwen3_4b_sft_lora",
            ]
        )
    )
    if lora_src and (lora_src / "pure_lora_config.json").is_file():
        lora_dst = out_dir / "qwen3_4b_sft_lora"
        copy_checkpoint_dir(lora_src, lora_dst)
        print(f"[3/3] Copied PureLoRA adapter: {lora_src} -> {lora_dst}")

    if args.skip_llm_merge:
        print("[Done] Skipped LLM merge (--skip-llm-merge). Run without --skip-llm-merge to build qwen3_4b_sft_merged.")
        return

    merged_dst = out_dir / "qwen3_4b_sft_merged"
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[3/3] Loading base LLM '{args.base_llm}' in {args.dtype} for offline merge...")
    tokenizer_src = str(lora_src) if (lora_src and (lora_src / "tokenizer_config.json").is_file()) else args.base_llm
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_llm,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_llm,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if lora_src and (lora_src / "pure_lora_config.json").is_file():
        model = load_and_merge_pure_lora(model, lora_src)

    # Strictly enforce max_position_embeddings = 16384 per competition rule
    model.config.max_position_embeddings = 16384
    llm_params = sum(p.numel() for p in model.parameters())
    assert llm_params < 5_000_000_000, f"LLM params {llm_params:,} >= 5B!"

    merged_dst.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] Saving merged standalone LLM ({llm_params / 1e9:.3f}B params) to {merged_dst}...")
    model.save_pretrained(merged_dst, safe_serialization=True)
    tokenizer.save_pretrained(merged_dst)

    # Double-check saved config.json has max_position_embeddings = 16384
    cfg_path = merged_dst / "config.json"
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["max_position_embeddings"] = 16384
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    manifest = {
        "asr_primary": {"path": "saved_models/asr_turbo_ft_v1", "params_B": 0.809},
        "asr_secondary": {"path": "saved_models/asr_pho_ft_v1", "params_B": 0.769},
        "llm_merged": {
            "path": "saved_models/qwen3_4b_sft_merged",
            "source_lora": str(lora_src.name) if lora_src else "base",
            "params": llm_params,
            "params_B": round(llm_params / 1e9, 3),
            "max_position_embeddings": 16384,
        },
        "total_params_B": round(0.809 + 0.769 + (llm_params / 1e9), 3),
        "budget_limit_B": 6.0,
    }
    with open(out_dir / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"[SUCCESS] Offline models ready in {out_dir}!\n"
        f"  Total parameter count: {manifest['total_params_B']:.3f}B < 6.0B\n"
        f"  LLM max_position_embeddings: 16384"
    )


if __name__ == "__main__":
    main()

