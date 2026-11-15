"""Pre-inference environment & model verification script for ZAC 2026 LifeMate Docker submission.

Executed as step 1 of `/code/predict.sh`:
  python3 /code/preprocessing.py
  python3 /code/predict.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def detect_data_root() -> Path:
    env_root = os.environ.get("DATA_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    if Path("/data").exists():
        return Path("/data")
    fallback = CODE_DIR.parent / "train+public_test"
    if fallback.exists():
        return fallback
    return Path("/data")


def detect_split(root: Path) -> str:
    env_split = os.environ.get("SPLIT")
    if env_split and (root / env_split / f"{env_split}_tools_catalog.json").exists():
        return env_split
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / f"{p.name}_tools_catalog.json").exists():
            return p.name
    return "private_test"


def verify_and_patch_llm_config(model_dir: Path) -> None:
    """Ensure LLM config.json strictly satisfies max_position_embeddings <= 16384 per ZAC 2026 rules."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return
    try:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        max_pos = int(cfg.get("max_position_embeddings", 16384))
        if max_pos > 16384:
            cfg["max_position_embeddings"] = 16384
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[Preprocessing] Clamped max_position_embeddings from {max_pos} -> 16384 in {cfg_path}")
    except Exception as e:
        print(f"[Preprocessing] Note: Could not update {cfg_path} ({e})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    root = detect_data_root()
    if not root.exists():
        print(f"[Preprocessing] Warning: DATA_ROOT {root} does not exist yet.")
        return

    split = detect_split(root)
    split_dir = root / split
    jsonl_path = split_dir / f"{split}.jsonl"
    catalog_path = split_dir / f"{split}_tools_catalog.json"

    n_samples = 0
    if jsonl_path.is_file():
        with open(jsonl_path, encoding="utf-8") as f:
            n_samples = sum(1 for line in f if line.strip())

    saved_models = CODE_DIR / "saved_models"
    for sub in ("qwen3_4b_sft_merged", "qwen3_4b_base"):
        if (saved_models / sub).is_dir():
            verify_and_patch_llm_config(saved_models / sub)

    Path("/result").mkdir(parents=True, exist_ok=True) if os.access("/", os.W_OK) else None
    print(
        f"[Preprocessing] Ready | root={root} | split={split} | "
        f"samples={n_samples} | catalog_exists={catalog_path.is_file()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

