"""Fine-tune a Whisper-family model (Whisper-large-v3-turbo, PhoWhisper-medium, ...) on the LABELLED TRAIN audio.

Why: zero-shot ASR is the weakest link of the pipeline. Examples from `public_test` (zero-shot turbo / PhoWhisper):
    "khám răng" -> "khám găng", "báo thức" -> "bảo sức / đạt pháo thức", "tối nay" -> "tối sau".
A wrong word in the transcript is a wrong argument (or a missed request => `[]`) => 0 points for the sample.
The train set has 5,036 labelled utterances (~7 h, 16 kHz mono): the final request of each sample plus every user
turn of `history`, all with `audio_text`. This is exactly the in-domain data (accents, app vocabulary, names) needed.

Rules compliance: only TRAIN audio/text is used (never public_test / private_test).
Honest evaluation: utterances that belong to the 250 validation samples are held out (WER is reported on them)
unless `--include-val` is given (use that only for the very last training run).

    python -X utf8 scripts/finetune_asr.py --base-model vinai/PhoWhisper-medium --output-dir /content/drive/MyDrive/ZaloAI2026/asr_pho_ft
    python -X utf8 scripts/finetune_asr.py --base-model openai/whisper-large-v3-turbo --output-dir .../asr_turbo_ft
    python -X utf8 scripts/finetune_asr.py --base-model vinai/PhoWhisper-medium --max-steps 20 --eval-n 16 --output-dir /tmp/smoke   # smoke test

Plain PyTorch loop (no Trainer): immune to the transformers 4.x/5.x Trainer API changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_wer(text: str) -> str:
    """lower-case, NFC, no punctuation, single spaces (digits kept: labels keep the spoken-style digits as is)."""
    t = unicodedata.normalize("NFC", str(text or "").lower())
    t = _PUNCT_RE.sub(" ", t.replace("_", " "))
    return " ".join(t.split())


def load_pairs(train_jsonl: Path, val_ids: set[str], include_val: bool) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """-> (train_pairs, val_pairs) as lists of (audio path relative to train/, transcript)."""
    train_pairs: dict[str, str] = {}
    val_pairs: dict[str, str] = {}
    for line in open(train_jsonl, encoding="utf-8"):
        if not line.strip():
            continue
        it = json.loads(line)
        is_val = it["id"] in val_ids
        utts = [(it.get("audio"), it.get("audio_text"))]
        utts += [(h.get("audio"), h.get("audio_text")) for h in it.get("history") or [] if h.get("role") == "user"]
        for audio, text in utts:
            if not audio or not (text or "").strip():
                continue
            if is_val:
                val_pairs[audio] = text.strip()
                if include_val:
                    train_pairs[audio] = text.strip()
            else:
                train_pairs[audio] = text.strip()
    # an utterance can never be both held-out and trained on, unless --include-val was asked for
    if not include_val:
        for a in list(val_pairs):
            train_pairs.pop(a, None)
    return sorted(train_pairs.items()), sorted(val_pairs.items())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Fine-tune Whisper / PhoWhisper on the labelled train utterances")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--val-jsonl", default=str(WORKSPACE / "outputs" / "splits" / "val_split.jsonl"))
    ap.add_argument("--include-val", action="store_true", help="Also train on the val utterances (final run only)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps (smoke test)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--freeze-encoder", action="store_true", help="Train only the decoder (less VRAM / faster, for L4 24GB)")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--eval-n", type=int, default=250, help="Max held-out utterances for WER (0 = skip)")
    ap.add_argument("--eval-baseline", action="store_true", help="Also measure WER of the base model before training")
    ap.add_argument("--eval-beams", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=250, help="Save intermediate checkpoint every N optimizer steps (0 = only at end)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.optimization import get_cosine_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_ids = {json.loads(l)["id"] for l in open(args.val_jsonl, encoding="utf-8") if l.strip()}
    train_pairs, val_pairs = load_pairs(train_dir / "train.jsonl", val_ids, args.include_val)
    if args.eval_n and len(val_pairs) > args.eval_n:
        rng = random.Random(args.seed)
        val_pairs = rng.sample(val_pairs, args.eval_n)
    print(f"[ASR-FT] base={args.base_model} | train utterances={len(train_pairs)} | held-out for WER={len(val_pairs)} | include_val={args.include_val}")
    assert train_pairs, "no training utterances found"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_amp = device == "cuda"

    processor = WhisperProcessor.from_pretrained(args.base_model)
    tok, fe = processor.tokenizer, processor.feature_extractor
    try:
        tok.set_prefix_tokens(language="vi", task="transcribe", predict_timestamps=False)
    except TypeError:
        tok.set_prefix_tokens(language="vi", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    dec_start = model.config.decoder_start_token_id
    model.to(device)

    def load_audio(rel: str) -> np.ndarray:
        wav, sr = sf.read(str(train_dir / rel), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        return wav[: 16000 * 30]

    @torch.no_grad()
    def evaluate(tag: str) -> float | None:
        if not val_pairs:
            return None
        import jiwer

        model.eval()
        model.config.use_cache = True
        refs, hyps = [], []
        for i in range(0, len(val_pairs), 16):
            chunk = val_pairs[i : i + 16]
            feats = fe([load_audio(a) for a, _ in chunk], sampling_rate=16000, return_tensors="pt").input_features.to(device)
            if use_amp:
                feats = feats.to(amp_dtype)
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                out = model.generate(
                    input_features=feats,
                    language="vi",
                    task="transcribe",
                    num_beams=args.eval_beams,
                    do_sample=False,
                    max_new_tokens=160,
                )
            for (_, ref), hyp in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
                refs.append(normalize_for_wer(ref))
                hyps.append(normalize_for_wer(hyp))
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        wer = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
        print(f"[ASR-FT] WER ({tag}) on {len(pairs)} held-out utterances = {100 * wer:.2f}%")
        for r, h in pairs[:4]:
            print(f"    REF: {r}\n    HYP: {h}")
        model.config.use_cache = False
        return wer

    report: dict = {"base_model": args.base_model, "train_utterances": len(train_pairs), "heldout_utterances": len(val_pairs)}
    if args.eval_baseline and args.eval_n:
        report["wer_before"] = evaluate("BEFORE fine-tuning")

    # ---------------------------------------------------------------- data
    def encode_labels(text: str) -> list[int]:
        ids = tok(text).input_ids
        if ids and ids[0] == dec_start:  # the model prepends the decoder start token itself when shifting labels right
            ids = ids[1:]
        return ids[:440]

    labels_cache = {a: encode_labels(t) for a, t in train_pairs}

    def make_batch(chunk: list[tuple[str, str]]):
        feats = fe([load_audio(a) for a, _ in chunk], sampling_rate=16000, return_tensors="pt").input_features
        lab = [labels_cache[a] for a, _ in chunk]
        width = max(len(x) for x in lab)
        lab_t = torch.full((len(lab), width), -100, dtype=torch.long)
        for i, x in enumerate(lab):
            lab_t[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        return feats, lab_t

    # ---------------------------------------------------------------- optimiser
    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad_(False)
    if not args.no_grad_ckpt:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[ASR-FT] trainable params = {sum(p.numel() for p in params) / 1e6:.0f}M | amp={'bf16' if use_bf16 else ('fp16' if use_amp else 'off')}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_pairs) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * args.warmup_ratio)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    # ---------------------------------------------------------------- train
    model.train()
    step, micro, running, t0 = 0, 0, 0.0, time.perf_counter()
    order: list[tuple[str, str]] = []
    done = False
    print(f"[ASR-FT] total optimizer steps = {total_steps} (steps/epoch = {steps_per_epoch})")
    while not done:
        order = list(train_pairs)
        random.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            chunk = order[i : i + args.batch_size]
            feats, labels = make_batch(chunk)
            feats, labels = feats.to(device), labels.to(device)
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                loss = model(input_features=feats, labels=labels).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 20 == 0 or step == total_steps:
                el = time.perf_counter() - t0
                print(f"[ASR-FT] step {step}/{total_steps} | loss={running / 20:.4f} | lr={sched.get_last_lr()[0]:.2e} | {el / step:.2f}s/step | eta={(total_steps - step) * el / step / 60:.1f} min")
                running = 0.0
            if args.save_steps > 0 and step % args.save_steps == 0 and step < total_steps:
                ckpt_dir = Path(args.output_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                print(f"[ASR-FT Checkpoint] Saving intermediate checkpoint at step {step}/{total_steps} -> {ckpt_dir}")
                model.save_pretrained(str(ckpt_dir), safe_serialization=True)
                processor.save_pretrained(str(ckpt_dir))
            if step >= total_steps:
                done = True
                break

    # ---------------------------------------------------------------- eval + save
    if args.eval_n:
        report["wer_after"] = evaluate("AFTER fine-tuning")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.half().save_pretrained(str(out), safe_serialization=True)
    processor.save_pretrained(str(out))
    report["steps"] = step
    json.dump(report, open(out / "asr_ft_report.json", "w", encoding="utf-8"), indent=2)
    print(f"[ASR-FT] saved to {out} | report: {json.dumps(report, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
