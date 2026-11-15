"""Official Docker Inference Entrypoint for Zalo AI Challenge 2026 — LifeMate.

Follows BTC Docker specification (`zalo_ai_challenge_2026_docker_guide.md`):
- Reads input from `/data/<split>` (auto-detects `private_test` or `public_test`).
- Imports `ToolEnv` directly from `/data/<split>/zaic_tools` (never from bundled code).
- Loads pre-packaged offline models from `/code/saved_models/` (< 6.0B total parameters, context <= 16,384).
- Prints model load time and prediction time to `stdout`.
- Writes `/result/submission.jsonl` with `{"id", "tool_calls", "state_hash", "time"}` per sample.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from lifemate.agent import LifeMateAgent, NoopLLMBackend, TransformersLLMBackend, VLLMBackend  # noqa: E402
from lifemate.asr import ASREngine  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_data_root(custom_root: str | None = None) -> Path:
    if custom_root:
        return Path(custom_root)
    env_root = os.environ.get("DATA_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    if Path("/data").exists():
        return Path("/data")
    fallback = CODE_DIR.parent / "train+public_test"
    if fallback.exists():
        return fallback
    return Path("/data")


def resolve_split(root: Path, custom_split: str | None = None) -> str:
    if custom_split:
        return custom_split
    env_split = os.environ.get("SPLIT")
    if env_split and (root / env_split / f"{env_split}_tools_catalog.json").exists():
        return env_split
    return next(
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / f"{p.name}_tools_catalog.json").exists()
    )


def resolve_model_paths(code_dir: Path = CODE_DIR) -> tuple[str, str, str]:
    """Locate offline ASR and LLM checkpoints inside `/code/saved_models/` or workspace root."""
    saved = code_dir / "saved_models"
    parent = code_dir.parent

    asr_primary = "openai/whisper-large-v3-turbo"
    for cand in (saved / "asr_turbo_ft_v1", parent / "asr_turbo_ft_v1"):
        if (cand / "config.json").is_file():
            asr_primary = str(cand)
            break

    asr_secondary = "vinai/PhoWhisper-medium"
    for cand in (saved / "asr_pho_ft_v1", parent / "asr_pho_ft_v1"):
        if (cand / "config.json").is_file():
            asr_secondary = str(cand)
            break

    llm_model = "Qwen/Qwen3-4B-Instruct-2507"
    llm_candidates = (
        saved / "qwen3_4b_sft_merged",
        saved / "qwen3_4b_sft_lora_v8",
        saved / "qwen3_4b_sft_lora_v7",
        saved / "qwen3_4b_sft_lora_v5",
        saved / "qwen3_4b_sft_lora",
        parent / "qwen3_4b_sft_merged",
        parent / "qwen3_4b_sft_lora_v8",
        parent / "qwen3_4b_sft_lora_v7",
        parent / "qwen3_4b_sft_lora_v5",
        parent / "qwen3_4b_sft_lora",
    )
    for cand in llm_candidates:
        if (cand / "config.json").is_file() or (cand / "pure_lora_config.json").is_file():
            llm_model = str(cand)
            break

    return asr_primary, asr_secondary, llm_model


def load_models(
    asr_mode: str = "transformers",
    llm_backend: str = "transformers",
    device: str | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 384,
    seed: int = 42,
) -> tuple[ASREngine, Any, float]:
    """Load Dual-ASR + 4B LLM into GPU memory and return `(asr_engine, llm_backend, load_seconds)`."""
    t_load0 = time.perf_counter()
    try:
        import torch

        has_cuda = torch.cuda.is_available()
        has_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    except ImportError:
        has_cuda = False
        has_bf16 = False
    if device is None:
        device = "cuda" if has_cuda else "cpu"
    if device == "cuda" and dtype == "bfloat16" and not has_bf16:
        dtype = "float16"

    asr_primary, asr_secondary, llm_model_path = resolve_model_paths(CODE_DIR)
    print(
        f"[INIT] Loading models on device={device} (dtype={dtype}):\n"
        f"  - ASR Primary  : {asr_primary}\n"
        f"  - ASR Secondary: {asr_secondary}\n"
        f"  - LLM Agent    : {llm_model_path}",
        flush=True,
    )

    asr = ASREngine(
        mode=asr_mode,
        primary_model=asr_primary,
        secondary_model=asr_secondary if asr_mode not in ("noop", "oracle") else None,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        cache_path=None,
        max_history_user_audios=30,
        beam_size=5,
    )
    if asr_mode not in ("noop", "oracle"):
        asr.preload()

    if llm_backend == "noop":
        llm = NoopLLMBackend()
    elif llm_backend == "vllm":
        llm = VLLMBackend(
            model_name_or_path=llm_model_path,
            dtype="half" if dtype == "float16" else dtype,
            gpu_memory_utilization=0.72,
            max_model_len=16384,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    else:
        llm = TransformersLLMBackend(
            model_name_or_path=llm_model_path,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            max_context_len=16384,
        )

    load_seconds = time.perf_counter() - t_load0
    print(f"[TIMING] Model load time: {load_seconds:.2f} s ({int(load_seconds * 1000)} ms)", flush=True)
    return asr, llm, load_seconds


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="ZAC 2026 LifeMate Official Docker Predictor")
    ap.add_argument("--data-root", default=None, help="Override /data path")
    ap.add_argument("--split", default=None, help="Override split name (auto-detected by default)")
    ap.add_argument("--out", default="/result/submission.jsonl", help="Output submission JSONL path")
    ap.add_argument("--asr-mode", default="transformers", choices=["transformers", "faster_whisper", "oracle", "noop"])
    ap.add_argument("--llm-backend", default="transformers", choices=["transformers", "vllm", "noop"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "half"])
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--time-budget", type=float, default=25.0)
    ap.add_argument("--step-reserve", type=float, default=2.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Step 1: Fix all random seeds
    seed_everything(args.seed)

    # Step 2: Load ASR & LLM models and log model loading time
    asr, llm, model_load_s = load_models(
        asr_mode=args.asr_mode,
        llm_backend=args.llm_backend,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    # Step 3: Resolve /data split and import split's zaic_tools.ToolEnv
    ROOT = resolve_data_root(args.data_root)
    SPLIT = resolve_split(ROOT, args.split)
    split_dir = ROOT / SPLIT
    if (split_dir / "zaic_tools").exists():
        sys.path.insert(0, str(split_dir))
    elif (ROOT / "zaic_tools").exists():
        sys.path.insert(0, str(ROOT))
    else:
        sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv  # noqa: E402

    catalog = json.load(open(split_dir / f"{SPLIT}_tools_catalog.json", encoding="utf-8"))
    with open(split_dir / f"{SPLIT}.jsonl", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        test_cases = test_cases[: args.limit]

    agent = LifeMateAgent(
        llm=llm,
        catalog=catalog,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )

    # Step 4: Run inference per sample and write /result/submission.jsonl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w", encoding="utf-8").close()

    t_pred0 = time.perf_counter()
    total_ms = 0

    for idx, item in enumerate(test_cases, 1):
        t1 = time.perf_counter()
        env = ToolEnv(SPLIT, item["id"], root=ROOT)
        deadline = t1 + args.time_budget - args.step_reserve

        try:
            transcript_map = asr.transcribe_sample(item, split_dir, history_dual=False)
        except Exception as e:
            print(f"[ASR ERROR] {item['id']}: {e}", flush=True)
            transcript_map = {}

        try:
            agent.run_sample(env, item, transcript_map, deadline=deadline)
        except Exception as e:
            print(f"[AGENT ERROR] {item['id']}: {e}", flush=True)

        tool_calls, state_hash = env.get_tool_calls(), env.get_state_hash()
        ms = int((time.perf_counter() - t1) * 1000)
        total_ms += ms

        row = {
            "id": item["id"],
            "tool_calls": tool_calls,
            "state_hash": state_hash,
            "time": ms,
        }
        with open(out_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        if idx % 25 == 0 or idx == len(test_cases):
            avg_ms = total_ms / max(1, idx)
            print(
                f"[PREDICT {idx}/{len(test_cases)}] id={item['id']} | "
                f"calls={len(tool_calls)} | time={ms} ms | avg={avg_ms:.1f} ms",
                flush=True,
            )

    pred_seconds = time.perf_counter() - t_pred0
    avg_sample_ms = total_ms / max(1, len(test_cases))
    print(
        f"[TIMING] Summary:\n"
        f"  - Model load time   : {model_load_s:.2f} s ({int(model_load_s * 1000)} ms)\n"
        f"  - Total predict time: {pred_seconds:.2f} s ({total_ms} ms across {len(test_cases)} samples)\n"
        f"  - Avg time / sample : {avg_sample_ms:.1f} ms ({avg_sample_ms / 1000.0:.2f} s)\n"
        f"  - Output file       : {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
