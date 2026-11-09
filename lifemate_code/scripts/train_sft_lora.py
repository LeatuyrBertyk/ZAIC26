"""Supervised Fine-Tuning (Pure-PyTorch LoRA SFT) v7.0 for Qwen3-4B-Instruct-2507 on ZAC 2026 LifeMate.

Key Upgrades in v7.0:
1. Custom Token-Level Loss Weighting (`--free-text-weight 0.3`, Trick #2.1): Down-weights free-text argument values
   (`content`, `body`, `subject`, `title`, `note`) and closing prose to `0.3` while keeping `1.0` on tool names,
   `x-ref` IDs, numbers/dates, structural JSON, and stop tokens (`<|im_end|>`).
2. Recovery Training Support (Trick #2.2): Assistant turns marked with `"_supervise": False` (injected search/x-ref errors)
   get `labels = -100` (`loss = 0`), while the subsequent recovery turns get `loss = 1.0`.
3. Periodic Step & Per-Epoch Checkpoints (`--save-steps`, `epoch_1`, `epoch_2`): Prevents Colab loss on disconnect and
   provides distinct checkpoints for Step-Level Majority Voting Ensemble (`MultiLoRAEnsembleBackend`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from lifemate.lora_utils import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    apply_pure_lora_state,
    inject_pure_lora,
    load_pure_lora_state,
    save_pure_lora,
)

FREE_TEXT_ARG_KEYS = ("content", "body", "subject", "title", "note", "description")


def apply_template(tokenizer, messages: list[dict], tools: list[dict], add_gen: bool) -> str:
    clean_msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
    kwargs = {"tokenize": False, "add_generation_prompt": add_gen, "tools": tools}
    try:
        return tokenizer.apply_chat_template(clean_msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(clean_msgs, **kwargs)


def _find_subseq(haystack: list[int], needle: list[int], start: int, end: int) -> tuple[int, int] | None:
    """Find token subsequence `needle` inside `haystack[start:end]`."""
    m = len(needle)
    if m == 0 or end - start < m:
        return None
    for pos in range(start, end - m + 1):
        if haystack[pos : pos + m] == needle:
            return pos, pos + m
    return None


class MultiTurnToolDataset(Dataset):
    """Tokenizes multi-turn trajectories with Assistant-Only loss masking (`-100`), Recovery Training (`_supervise=False`),
    and Token-Level Loss Weighting (`0.3` on free-text spans, `1.0` on tool names/x-refs/stop tokens)."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 16384, free_text_weight: float = 0.3):
        self.samples: list[dict[str, torch.Tensor]] = []
        raw_rows = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]

        skipped = 0
        seq_lens: list[int] = []
        sup_toks: list[int] = []
        downweighted_toks = 0

        for idx, row in enumerate(raw_rows, 1):
            messages = row["messages"]
            tools = row["tools"]

            user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
            if not user_indices:
                skipped += 1
                continue
            last_user_idx = user_indices[-1]

            full_text = apply_template(tokenizer, messages, tools, add_gen=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            labels = [-100] * len(full_ids)
            weights = [0.0] * len(full_ids)
            n_sup = 0

            for i in range(last_user_idx + 1, len(messages)):
                msg = messages[i]
                if msg.get("role") != "assistant":
                    continue
                if msg.get("_supervise") is False:
                    # Recovery Training (Trick #2.2): injected error turn has loss = 0
                    continue

                prefix_text = apply_template(tokenizer, messages[:i], tools, add_gen=True)
                upto_text = apply_template(tokenizer, messages[: i + 1], tools, add_gen=False)
                start_pos = len(tokenizer.encode(prefix_text, add_special_tokens=False))
                end_pos = min(len(tokenizer.encode(upto_text, add_special_tokens=False)), len(full_ids))
                if start_pos >= end_pos:
                    continue

                tcalls = msg.get("tool_calls") or []
                is_closing_prose = not tcalls and bool(msg.get("content"))

                for pos in range(start_pos, end_pos):
                    labels[pos] = full_ids[pos]
                    # For closing prose ("Đã hoàn tất thực thi."), weight prose at free_text_weight, keep stop tokens at 1.0
                    if is_closing_prose and pos < end_pos - 2:
                        weights[pos] = free_text_weight
                        downweighted_toks += 1
                    else:
                        weights[pos] = 1.0

                # Down-weight interior tokens of free-text argument values (`content`, `body`, `subject`, `title`, `note`)
                if tcalls and free_text_weight < 1.0:
                    for tc in tcalls:
                        raw_args = (tc.get("function") or {}).get("arguments") or "{}"
                        try:
                            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except Exception:
                            args_dict = {}
                        if not isinstance(args_dict, dict):
                            continue
                        for k in FREE_TEXT_ARG_KEYS:
                            val = args_dict.get(k)
                            if isinstance(val, str) and len(val.strip()) >= 8:
                                sub_ids = tokenizer.encode(val, add_special_tokens=False)
                                if len(sub_ids) > 2:
                                    # Trim boundary tokens in case BPE merges quote + first char
                                    core_ids = sub_ids[1:-1] if len(sub_ids) >= 4 else sub_ids
                                    span = _find_subseq(full_ids, core_ids, start_pos, end_pos)
                                    if span:
                                        for p in range(span[0], span[1]):
                                            if weights[p] == 1.0:
                                                weights[p] = free_text_weight
                                                downweighted_toks += 1

                n_sup += end_pos - start_pos

            if n_sup == 0:
                skipped += 1
                continue

            seq_lens.append(len(full_ids))
            sup_toks.append(n_sup)
            self.samples.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "loss_weights": torch.tensor(weights, dtype=torch.float32),
                }
            )
            if idx % 500 == 0 or idx == len(raw_rows):
                print(f"[Tokenize {idx}/{len(raw_rows)}] valid={len(self.samples)} skipped={skipped}")

        if seq_lens:
            s_sorted = sorted(seq_lens)
            p50 = s_sorted[len(s_sorted) // 2]
            p95 = s_sorted[int(len(s_sorted) * 0.95)]
            avg_sup = sum(sup_toks) / len(sup_toks)
            pct_dw = 100.0 * downweighted_toks / max(1, sum(sup_toks))
            print(
                f"[Dataset Summary] Kept {len(self.samples)}/{len(raw_rows)} samples (skipped={skipped}) | "
                f"seq_len p50={p50}, p95={p95}, max={s_sorted[-1]} | avg_supervised_tokens={avg_sup:.1f} | "
                f"downweighted_free_text={pct_dw:.1f}% (w={free_text_weight})"
            )
        assert len(self.samples) >= int(0.95 * len(raw_rows)), (
            f"Too many samples skipped ({skipped}/{len(raw_rows)})! Increase --max-length."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.samples[i]


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_input_ids = []
        batch_mask = []
        batch_labels = []
        batch_weights = []
        for f in features:
            n = f["input_ids"].size(0)
            pad_n = max_len - n
            batch_input_ids.append(
                torch.cat([f["input_ids"], torch.full((pad_n,), self.pad_token_id, dtype=torch.long)])
            )
            batch_mask.append(torch.cat([f["attention_mask"], torch.zeros(pad_n, dtype=torch.long)]))
            batch_labels.append(torch.cat([f["labels"], torch.full((pad_n,), -100, dtype=torch.long)]))
            batch_weights.append(torch.cat([f["loss_weights"], torch.zeros(pad_n, dtype=torch.float32)]))
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_mask),
            "labels": torch.stack(batch_labels),
            "loss_weights": torch.stack(batch_weights),
        }


def _load_base_causal_lm(model_name: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    common = dict(low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **common)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **common)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Fine-tune Qwen3-4B-Instruct LoRA v7.0 on ZAC 2026 LifeMate")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train-jsonl", default=str(WORKSPACE / "outputs" / "sft" / "sft_train_v7.jsonl"))
    ap.add_argument("--output-dir", required=True, help="Directory to save LoRA adapter (e.g. on Google Drive)")
    ap.add_argument("--resume-from-adapter", default=None, help="Optional existing PureLoRA adapter dir to resume from")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--free-text-weight", type=float, default=0.3, help="Loss weight on free-text tokens (Trick #2.1)")
    ap.add_argument("--save-steps", type=int, default=150, help="Save PureLoRA checkpoint every N optimizer steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    class SparseAssistantTrainer(Trainer):
        """Memory-optimal Trainer: computes `lm_head` logits ONLY on assistant tokens (`labels != -100`)
        with custom Token-Level Loss Weighting (`free_text_weight = 0.3`)."""

        def __init__(self, *t_args, **t_kwargs):
            super().__init__(*t_args, **t_kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            loss_weights = inputs.get("loss_weights")
            raw_model = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
            if labels is not None and hasattr(raw_model, "model") and hasattr(raw_model, "lm_head"):
                backbone_out = raw_model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                )
                hidden_states = backbone_out[0]
                shift_hidden = hidden_states[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                valid_mask = shift_labels.ne(-100)
                if valid_mask.any():
                    active_hidden = shift_hidden[valid_mask]
                    active_labels = shift_labels[valid_mask]
                    active_logits = raw_model.lm_head(active_hidden).float()
                    if loss_weights is not None:
                        shift_weights = loss_weights[:, 1:].contiguous()
                        active_w = shift_weights[valid_mask].float()
                        per_tok = F.cross_entropy(active_logits, active_labels, reduction="none")
                        loss = (per_tok * active_w).sum() / active_w.sum().clamp_min(1e-6)
                    else:
                        loss = F.cross_entropy(active_logits, active_labels)
                else:
                    loss = hidden_states.sum() * 0.0
                return (loss, backbone_out) if return_outputs else loss
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    print(
        f"[SFT v7.0] Loading {args.base_model} in {dtype} | max_length={args.max_length} | "
        f"r={args.lora_r}, alpha={args.lora_alpha} | free_text_weight={args.free_text_weight}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = MultiTurnToolDataset(
        Path(args.train_jsonl),
        tokenizer,
        max_length=args.max_length,
        free_text_weight=args.free_text_weight,
    )

    model = _load_base_causal_lm(args.base_model, dtype)
    model.config.use_cache = False

    inject_pure_lora(
        model,
        target_modules=DEFAULT_TARGET_MODULES,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=0.05,
    )
    if args.resume_from_adapter and (Path(args.resume_from_adapter) / "pure_lora_config.json").is_file():
        print(f"[SFT v7.0] Resuming PureLoRA weights from {args.resume_from_adapter}...")
        st = load_pure_lora_state(Path(args.resume_from_adapter), device="cpu")
        apply_pure_lora_state(model, st)

    model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        model.to("cuda")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class SaveStepAndEpochAdapterCallback(TrainerCallback):
        def on_step_end(self, _args, state, _control, **kwargs):
            if args.save_steps > 0 and state.global_step > 0 and state.global_step % args.save_steps == 0:
                print(f"[PureLoRA Checkpoint] Saving step {state.global_step} checkpoint to {out_dir}...")
                save_pure_lora(model, out_dir, args.base_model, args.lora_r, args.lora_alpha, DEFAULT_TARGET_MODULES)
                tokenizer.save_pretrained(str(out_dir))

        def on_epoch_end(self, _args, state, _control, **kwargs):
            ep_tag = int(round(state.epoch or 1))
            ep_dir = out_dir / f"epoch_{ep_tag}"
            print(f"[PureLoRA Checkpoint] Saving epoch {ep_tag} adapter to {ep_dir} and {out_dir}...")
            save_pure_lora(model, ep_dir, args.base_model, args.lora_r, args.lora_alpha, DEFAULT_TARGET_MODULES)
            tokenizer.save_pretrained(str(ep_dir))
            save_pure_lora(model, out_dir, args.base_model, args.lora_r, args.lora_alpha, DEFAULT_TARGET_MODULES)
            tokenizer.save_pretrained(str(out_dir))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=35,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()),
        logging_steps=25,
        save_strategy="no",
        remove_unused_columns=False,
        label_names=["labels", "loss_weights"],
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = SparseAssistantTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[SaveStepAndEpochAdapterCallback()],
    )

    print("[SFT v7.0] Starting Sparse-Logits PureLoRA training with Token-Level Loss Weighting...")
    trainer.train()

    print(f"[SFT v7.0] Saving final PureLoRA adapter + tokenizer to {out_dir}...")
    save_pure_lora(model, out_dir, args.base_model, args.lora_r, args.lora_alpha, DEFAULT_TARGET_MODULES)
    tokenizer.save_pretrained(str(out_dir))
    print(f"[SFT v7.0] Done! Adapter saved at {out_dir}")


if __name__ == "__main__":
    main()
