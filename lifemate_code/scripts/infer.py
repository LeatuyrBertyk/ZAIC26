"""End-to-End Inference Script for ZAC 2026 LifeMate (generates `submission.jsonl`).

Examples:
  1. CPU smoke-test on public_test (verifies ToolEnv + state_hash + schema validator):
     python -X utf8 scripts/infer.py --split public_test --asr-mode noop --llm-backend noop --out outputs/noop_public_submission.jsonl

  2. Validate LLM Agent on 250-sample Validation Split using Oracle transcripts (fast GPU test):
     python -X utf8 scripts/infer.py --split train --input-jsonl outputs/splits/val_split.jsonl --asr-mode oracle --llm-backend vllm --llm-model Qwen/Qwen3-4B-Instruct-2507 --out outputs/val_pred.jsonl

  3. Full End-to-End Baseline on public_test (ASR + LLM -> submission.jsonl for Leaderboard):
     python -X utf8 scripts/infer.py --split public_test --asr-mode faster_whisper --llm-backend vllm --llm-model Qwen/Qwen3-4B-Instruct-2507 --out submission.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)
sys.path.insert(0, str(WORKSPACE))

from lifemate.agent import (  # noqa: E402
    LifeMateAgent,
    MultiLoRAEnsembleBackend,
    NoopLLMBackend,
    OpenAICompatBackend,
    TransformersLLMBackend,
    VLLMBackend,
)
from lifemate.asr import ASREngine  # noqa: E402


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def build_llm_backend(args):
    if args.llm_backend == "noop":
        return NoopLLMBackend()
    if args.llm_backend == "vllm":
        return VLLMBackend(
            model_name_or_path=args.llm_model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=16384,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            enable_prefix_caching=not args.no_prefix_caching,
        )
    if args.llm_backend == "transformers":
        extra_adapters = [p.strip() for p in (args.llm_ensemble or "").split(",") if p.strip()]
        all_adapters = [args.llm_model] + [p for p in extra_adapters if p != args.llm_model]
        if len(all_adapters) >= 2 and all((Path(p) / "pure_lora_config.json").is_file() for p in all_adapters):
            return MultiLoRAEnsembleBackend(
                adapter_paths=all_adapters,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                max_context_len=16384,
            )
        return TransformersLLMBackend(
            model_name_or_path=args.llm_model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_context_len=16384,
        )
    if args.llm_backend == "openai_api":
        return OpenAICompatBackend(
            base_url=args.api_base,
            api_key=args.api_key,
            model=args.llm_model,
            max_tokens=args.max_new_tokens,
        )
    raise ValueError(f"Unknown --llm-backend: {args.llm_backend}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="ZAC 2026 LifeMate End-to-End Inference")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Folder containing train/ and public_test/")
    ap.add_argument("--split", default="public_test", choices=["train", "public_test", "private_test"])
    ap.add_argument("--input-jsonl", default=None, help="Optional custom input JSONL (e.g. outputs/splits/val_split.jsonl)")
    ap.add_argument("--out", default="submission.jsonl", help="Output submission JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Optional max samples to run (0 = all)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output JSONL if interrupted")

    # ASR flags
    ap.add_argument("--asr-mode", default="faster_whisper", choices=["faster_whisper", "transformers", "oracle", "noop"])
    ap.add_argument("--asr-model", default="Systran/faster-whisper-large-v3-turbo")
    ap.add_argument("--asr-secondary", default=None, help="Optional 2nd ASR model, e.g. vinai/PhoWhisper-medium")
    ap.add_argument("--asr-cache", default=None, help="Path to JSON cache for ASR transcripts")
    ap.add_argument("--asr-base-cache", default=None, help="Optional zero-shot base ASR cache for 3rd hypothesis (Trick #4.2)")

    # LLM flags
    ap.add_argument("--llm-backend", default="vllm", choices=["vllm", "transformers", "openai_api", "noop"])
    ap.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--llm-ensemble", default=None, help="Optional comma-separated extra PureLoRA adapters for Step-Level Voting (Trick #3.1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="half", help="half / bfloat16 / float16")
    ap.add_argument("--gpu-mem-util", type=float, default=0.72)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-steps", type=int, default=10, help="Gold has at most 6 calls per sample; 10 leaves room for retries")
    ap.add_argument("--time-budget", type=float, default=25.0, help="Per-sample seconds (ASR + agent). BTC limit is 30 s => 0 points")
    ap.add_argument("--step-reserve", type=float, default=2.5, help="Do not start a new LLM step with less than this many seconds left")
    ap.add_argument("--no-prefix-caching", action="store_true", help="Disable vLLM prefix caching")
    ap.add_argument("--timing-out", default=None, help="Optional JSONL: per-sample seconds (agent+ASR) for latency analysis")
    ap.add_argument("--history-dual-asr", action="store_true", help="Also run the 2nd ASR on history turns (v4 behaviour)")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr-only", action="store_true", help="Only run ASR and save transcript cache, then exit (frees VRAM before LLM)")
    args = ap.parse_args()

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv, schema_validate  # noqa: E402

    input_path = Path(args.input_jsonl) if args.input_jsonl else (split_dir / f"{args.split}.jsonl")
    catalog = json.load(open(split_dir / f"{args.split}_tools_catalog.json", encoding="utf-8"))
    samples = [json.loads(line) for line in open(input_path, encoding="utf-8") if line.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"[Resume] Found {len(done_ids)} completed samples in {out_path}")

    cache_path = (
        Path(args.asr_cache)
        if args.asr_cache
        else (WORKSPACE / "outputs" / "asr_cache" / f"{args.split}_transcripts.json")
    )
    asr = ASREngine(
        mode=args.asr_mode,
        primary_model=args.asr_model,
        secondary_model=args.asr_secondary,
        device=args.device,
        compute_type="float16" if args.device == "cuda" else "int8",
        cache_path=cache_path if args.asr_mode not in ("oracle", "noop") else None,
        base_cache_path=Path(args.asr_base_cache) if args.asr_base_cache else None,
    )

    if args.asr_only:
        t0 = time.perf_counter()
        for idx, sample in enumerate(samples, 1):
            asr.transcribe_sample(sample, split_dir, history_dual=args.history_dual_asr)
            if idx % 25 == 0 or idx == len(samples):
                asr.save_cache()
                elapsed = time.perf_counter() - t0
                print(f"[ASR-Only {idx}/{len(samples)}] cached={len(asr.cache)} wavs | elapsed={elapsed:.1f}s")
        asr.save_cache()
        asr.unload()
        print(f"ASR-only complete! Saved {len(asr.cache)} transcripts to {cache_path}")
        return

    # Pre-transcribe any uncached wavs and unload ASR models BEFORE loading the 4B LLM to avoid GPU OOM!
    if args.asr_mode not in ("oracle", "noop"):
        cache_before = len(asr.cache)
        for sample in samples:
            if sample["id"] not in done_ids:
                try:
                    asr.transcribe_sample(sample, split_dir, history_dual=args.history_dual_asr)
                except Exception as e:
                    print(f"[ASR PRE-PASS ERROR] {sample['id']}: {e}")
        if len(asr.cache) != cache_before or asr._primary is not None or asr._secondary is not None:
            asr.save_cache()
            asr.unload()
            print(f"[ASR Pre-pass] Cached {len(asr.cache)} wavs and freed ASR GPU memory before loading LLM.")

    llm = build_llm_backend(args)
    agent = LifeMateAgent(llm=llm, catalog=catalog, max_steps=args.max_steps, max_new_tokens=args.max_new_tokens)

    t0 = time.perf_counter()
    processed = 0
    slow = 0
    file_mode = "a" if (args.resume and done_ids) else "w"

    timing_f = open(args.timing_out, "a", encoding="utf-8") if args.timing_out else None
    with open(out_path, file_mode, encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, 1):
            qid = sample["id"]
            if qid in done_ids:
                continue

            s_start = time.perf_counter()
            try:
                transcript_map = asr.transcribe_sample(sample, split_dir, history_dual=args.history_dual_asr)
            except Exception as e:  # a broken wav must not kill the whole run
                print(f"[ASR ERROR] {qid}: {e}")
                transcript_map = {}

            # Rule: Exactly ONE ToolEnv session created per sample!
            env = ToolEnv(args.split, qid, root=data_root)
            deadline = s_start + args.time_budget - args.step_reserve
            try:
                agent.run_sample(env, sample, transcript_map, deadline=deadline)
            except Exception as e:  # keep whatever was executed; the line is still written
                print(f"[AGENT ERROR] {qid}: {e}")

            s_elapsed = time.perf_counter() - s_start
            if s_elapsed > 28.0:
                slow += 1
                print(f"[SLOW] {qid}: {s_elapsed:.1f}s (> 28 s, danger zone for the 30 s limit)")

            if timing_f:
                timing_f.write(json.dumps({"id": qid, "seconds": round(s_elapsed, 3)}) + "\n")
                timing_f.flush()
            row = {
                "id": qid,
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash(),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

            if processed % 25 == 0 or idx == len(samples):
                asr.save_cache()
                elapsed = time.perf_counter() - t0
                avg_s = elapsed / max(1, processed)
                print(
                    f"[{idx}/{len(samples)}] {qid} | calls={len(row['tool_calls'])} | "
                    f"step={time.perf_counter()-s_start:.2f}s | avg={avg_s:.2f}s/sample"
                )

    asr.save_cache()

    # Post-run schema check WITHOUT opening a second ToolEnv (rule: exactly one ToolEnv per sample).
    # Allowed tools are taken from the sample's own context.installed_apps.
    allowed_by_id = {
        s["id"]: {t for a in ((s.get("context") or {}).get("installed_apps") or {}).values() for t in a.get("tools", [])}
        | {"sys.ask_missing_info"}
        for s in samples
    }
    bad = 0
    lines_checked = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lines_checked += 1
            allowed = allowed_by_id.get(r["id"])
            for check in schema_validate(r.get("tool_calls") or [], catalog, allowed):
                if not check["ok"]:
                    bad += 1
                    print(f"[SCHEMA ERROR] {r['id']} -> {check['name']}: {check['error']}")

    total_time = time.perf_counter() - t0
    print(
        f"Done! {lines_checked}/{len(samples)} lines in {out_path} | "
        f"{bad} schema errors | {slow} samples > 28 s | elapsed={total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
