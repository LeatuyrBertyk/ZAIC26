"""Multi-Step Tool-Calling Agent Runtime v5.0 — Precision-Safe Guardrails, Catalog-Driven Sanitizer, Context & Time Budgets.

v5 changes (each one measured with `scripts/oracle_agent_check.py`, which replays gold calls through this runtime):
- Sanitizer is driven by the catalog instead of a hard-coded list: drops unknown keys / empty values, coerces types
  ("26" -> 26, "true" -> True), repairs enum casing, normalises date-time / time to the gold format
  (`YYYY-MM-DDTHH:MM:SS+07:00`, `HH:MM`). Fewer schema rejections = fewer retries, and exact matches on non-fuzzy args.
- `sys.ask_missing_info` may target a READ tool when the asked params are required params of that tool
  (gold does this, e.g. `flights.search` / `date`); person lookups (`contacts.search`, `chat_ott.list_chats`) stay blocked.
- x-ref check also accepts an ID that appears in ANY tool result of the session (history included), not only in
  results of the referenced app (gold reuses `c_778` returned by `sms.read_recent`).
- Identical WRITE re-issued -> blocked (a re-run of a write changes the state hash => score 0).
- Out-of-scope: if, before anything was executed, every proposed call targets a real app that is NOT installed,
  stop immediately with `[]` (gold for out-of-scope is `[]`, any extra call gives 0).
- Context budget: oldest history turns are dropped (then long tool results are shortened) so prompt + generation
  always fits the 16,384-token limit, instead of silently cutting the END of the prompt (old transformers backend)
  or crashing (vLLM).
- Time budget: the loop stops at `deadline` so a sample never exceeds the 30 s limit (timeout = 0 points).
"""
from __future__ import annotations

import copy
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .prompting import annotate_tool_result, build_messages, filter_installed_tools, to_openai_tools

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_DT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")
_TIME_RE = re.compile(r"^(\d{1,2})[:hH](\d{2})?(?::\d{2})?$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:([.,])\d{3})(?:\1\d{3})*$")
_EMPTY = (None, "", [], {})

# --- "missing amount" guard (measured on all 1,545 train gold samples) ---------------------------------------------
# Gold asks `sys.ask_missing_info(params=["amount"])` when the user only says "chút tiền / ít tiền" (18/18 such
# utterances; 0 false positives). A 4B model instead invents a plausible amount (500000) => the sample scores 0.
AMOUNT_GUARD_TOOLS = {"ewallet.transfer", "ewallet.topup_mobile", "banking.transfer"}
_VAGUE_AMOUNT_RE = re.compile(r"(chút|ít|một ít|một chút)\s+(tiền|xíu|đỉnh)|tiền\s+(chút|ít)|chút\s+đỉnh")
# the amount may legitimately be derived (previous result, arithmetic, conversion): never second-guess those
_DERIVED_AMOUNT_RE = re.compile(
    r"(đó|đấy|ấy|nớ|bằng|trừ|cộng|nửa|gấp|thêm|bớt|tăng|giảm|hạ|nâng|giống|như|theo|quy|đổi|hết|toàn bộ|tất cả|nguyên|"
    r"đúng|mức|bao nhiêu|nhiêu|tổng|còn lại|phần|phí|giá|lít|kg|ký|cân|mét)"
)
_DIGIT_RUN_RE = re.compile(r"\d[\d.,]*")

# Every app id seen in the train + public_test catalogs. Used to recognise "real app, but not installed" (= out of scope).
KNOWN_APP_IDS = {
    "alarm", "appliances", "banking", "bike_rental", "bill_payment", "calculator", "calendar", "chat_ott", "contacts",
    "door_lock", "ecommerce", "email", "event_tickets", "ewallet", "exchange_rate", "fitness", "flights", "food_delivery",
    "fuel_price", "gold_price", "grocery", "habit_tracker", "home_automation", "home_energy", "home_security",
    "household_chores", "learning", "library", "lottery", "loyalty", "maps", "medication", "music", "news", "notes",
    "package_tracking", "pantry", "parking", "phone", "podcast_radio", "public_transit", "recipes", "reminders",
    "ride_hailing", "robot_vacuum", "shopping_list", "smart_home", "sms", "stocks", "subscriptions", "tasks",
    "train_bus_tickets", "translator", "tv", "video_meeting", "weather",
}

# READ tools that must never be the target of sys.ask_missing_info (ask for `contact_id` on the WRITE tool instead).
ASK_DENY_READ_TARGETS = {"contacts.search", "chat_ott.list_chats"}

_EN_ROOM_MAP = {
    "living room": "phòng khách",
    "living_room": "phòng khách",
    "bedroom": "phòng ngủ",
    "master bedroom": "phòng ngủ lớn",
    "kitchen": "bếp",
    "bathroom": "phòng tắm",
    "office": "phòng làm việc",
    "study": "phòng làm việc",
    "dining room": "phòng ăn",
    "garage": "nhà xe",
    "balcony": "ban công",
}

# Read tools that are almost always called as step-1 lookup before a [WRITE] or `sys.ask_missing_info`
ACTION_PREREQ_READ_TOOLS = {
    "contacts.search",
    "chat_ott.list_chats",
    "banking.list_beneficiaries",
    "pantry.list_items",
    "habit_tracker.list_habits",
    "parking.find",
    "parking.list_sessions",
    "ecommerce.search",
    "grocery.search",
    "event_tickets.search",
    "train_bus_tickets.search",
    "flights.search",
    "translator.translate",
}

# Write tools that can legitimately be called multiple times with identical arguments (e.g., redeeming 2 cups of coffee)
MAX_IDENTICAL_WRITES = {
    "loyalty.redeem": 3,
}


def _canonical_calls_sig(calls: list[dict[str, Any]]) -> str:
    """Canonical signature of a list of proposed tool calls for step-level majority voting."""
    if not calls:
        return "[]"
    norm = []
    for c in calls:
        name = str(c.get("name") or "")
        args = c.get("arguments") or {}
        if isinstance(args, dict):
            clean_args = {k: v for k, v in sorted(args.items()) if v not in _EMPTY}
        else:
            clean_args = {}
        norm.append({"name": name, "arguments": clean_args})
    return json.dumps(norm, sort_keys=True, ensure_ascii=False)


def _vote_step_outputs(candidates: list[tuple[str, list[dict[str, Any]]]]) -> tuple[str, list[dict[str, Any]]]:
    """Majority vote across step-level LLM proposals (falls back to candidate 0 on a 3-way tie)."""
    if not candidates:
        return "", []
    if len(candidates) == 1:
        return candidates[0]

    # Exact (name + arguments) majority first
    sigs = [_canonical_calls_sig(calls) for _, calls in candidates]
    counts: dict[str, int] = defaultdict(int)
    first_idx: dict[str, int] = {}
    for i, s in enumerate(sigs):
        counts[s] += 1
        if s not in first_idx:
            first_idx[s] = i

    best_sig = max(counts.keys(), key=lambda s: (counts[s], -first_idx[s]))
    if counts[best_sig] >= 2:
        return candidates[first_idx[best_sig]]

    # Secondary vote: tool-name sequence majority (e.g. two models agree on the exact tool name, only differ slightly in fuzzy text)
    name_sigs = [json.dumps([c.get("name") for c in calls]) for _, calls in candidates]
    n_counts: dict[str, int] = defaultdict(int)
    n_first: dict[str, int] = {}
    for i, ns in enumerate(name_sigs):
        n_counts[ns] += 1
        if ns not in n_first:
            n_first[ns] = i
    best_ns = max(n_counts.keys(), key=lambda ns: (n_counts[ns], -n_first[ns]))
    if n_counts[best_ns] >= 2:
        return candidates[n_first[best_ns]]

    return candidates[0]


def _is_empty_read_result(name: str, result: Any) -> bool:
    """True when a [READ] search tool returned ok=True but every list collection in the result is empty (`[]`)."""
    if not isinstance(result, dict) or not result.get("ok"):
        return False
    list_vals = [v for k, v in result.items() if isinstance(v, list) and not k.startswith("_")]
    return bool(list_vals) and all(len(v) == 0 for v in list_vals)



def parse_tool_calls(raw_text: str) -> list[dict[str, Any]]:
    """Extract tool calls from Qwen `<tool_call>...</tool_call>` blocks or fallback JSON tool call output."""
    cleaned = _THINK_RE.sub("", raw_text or "").strip()
    calls: list[dict[str, Any]] = []

    for m in _TOOL_CALL_RE.findall(cleaned):
        try:
            obj = json.loads(m)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            if isinstance(name, str) and isinstance(args, dict):
                calls.append({"name": name, "arguments": args})
        except Exception:
            continue

    if calls:
        return calls

    if cleaned.startswith("{") or cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
            items = [parsed] if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            for obj in items:
                if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    if isinstance(args, dict):
                        calls.append({"name": obj["name"], "arguments": args})
        except Exception:
            pass
    return calls


# ----------------------------------------------------------------------------------------------------------------------
# LLM backends
# ----------------------------------------------------------------------------------------------------------------------
class BaseLLMBackend:
    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        raise NotImplementedError

    # Optional: backends that know their tokenizer return the exact prompt length (enables context budgeting).
    count_tokens = None


class NoopLLMBackend(BaseLLMBackend):
    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        return "", []


def _apply_chat_template(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": tools}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class TransformersLLMBackend(BaseLLMBackend):
    """HuggingFace Transformers backend supporting both base models and PureLoRA / PEFT adapters (< 5B, <= 16,384 context)."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 384,
        max_context_len: int = 16384,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.max_context_len = min(max_context_len, 16384)
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
        if self.device != "cuda":
            torch_dtype = torch.float32

        def _load_lm(mpath: str):
            common = dict(low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=True)
            try:
                return AutoModelForCausalLM.from_pretrained(mpath, dtype=torch_dtype, **common)
            except TypeError:
                return AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch_dtype, **common)

        pure_cfg_path = Path(model_name_or_path) / "pure_lora_config.json"
        adapter_cfg_path = Path(model_name_or_path) / "adapter_config.json"
        if pure_cfg_path.is_file():
            from .lora_utils import load_and_merge_pure_lora

            pure_cfg = json.load(open(pure_cfg_path, encoding="utf-8"))
            base_path = pure_cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B-Instruct-2507")
            print(f"[LLM] Loading base model {base_path} + merging PureLoRA adapter from {model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            base_model = _load_lm(base_path)
            base_model.to(self.device)
            self.model = load_and_merge_pure_lora(base_model, Path(model_name_or_path))
        elif adapter_cfg_path.is_file():
            from peft import PeftModel

            adapter_cfg = json.load(open(adapter_cfg_path, encoding="utf-8"))
            base_path = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B-Instruct-2507")
            print(f"[LLM] Loading base model {base_path} + merging LoRA adapter from {model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            base_model = _load_lm(base_path)
            peft_model = PeftModel.from_pretrained(base_model, model_name_or_path)
            self.model = peft_model.merge_and_unload()
            self.model.to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = _load_lm(model_name_or_path)
            self.model.to(self.device)

        self.model.eval()

    def _apply_template(self, messages: list[dict], tools: list[dict]) -> str:
        return _apply_chat_template(self.tokenizer, messages, tools)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self.tokenizer.encode(self._apply_template(messages, tools), add_special_tokens=False))

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        import torch

        prompt = self._apply_template(messages, tools)
        # No right-truncation here: the agent already fits the prompt to the context budget (v4 silently cut the
        # END of the prompt, i.e. the user's request, when it was too long).
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        for tok in ("<|im_end|>", "<|endoftext|>"):
            raw = raw.replace(tok, "")
        raw = raw.strip()
        return raw, parse_tool_calls(raw)


class VLLMBackend(BaseLLMBackend):
    """High-throughput offline vLLM backend (prefix caching: each agent step re-uses the KV cache of the previous one)."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        dtype: str = "half",
        gpu_memory_utilization: float = 0.72,
        max_model_len: int = 16384,
        max_new_tokens: int = 384,
        seed: int = 42,
        enable_prefix_caching: bool = True,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.max_model_len = min(max_model_len, 16384)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_name_or_path,
            dtype=dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            trust_remote_code=True,
            enforce_eager=False,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            seed=seed,
        )

    def _apply_template(self, messages: list[dict], tools: list[dict]) -> str:
        return _apply_chat_template(self.tokenizer, messages, tools)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self.tokenizer.encode(self._apply_template(messages, tools), add_special_tokens=False))

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        prompt = self._apply_template(messages, tools)
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        raw = outputs[0].outputs[0].text.strip()
        return raw, parse_tool_calls(raw)


class OpenAICompatBackend(BaseLLMBackend):
    """Development only (e.g. a local vLLM server). Never used at inference in the offline Docker."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        max_tokens: int = 384,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            temperature=0.0,
            max_tokens=self.max_tokens,
            seed=42,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                calls.append({"name": tc.function.name, "arguments": args})
            return msg.content or "", calls
        raw = msg.content or ""
        return raw, parse_tool_calls(raw)


class StepEnsembleLLMBackend(BaseLLMBackend):
    """Step-Level Majority Voting Ensemble across 2-3 LLM backends (Trick #3.1: +1.8 val pts on 1 single ToolEnv).
    Uses early-exit consensus: if Backend 0 and Backend 1 agree on the exact tool call, Backend 2 is skipped!"""

    def __init__(self, backends: list[BaseLLMBackend], early_exit_consensus: bool = True):
        assert backends, "StepEnsembleLLMBackend requires at least 1 backend"
        self.backends = backends
        self.early_exit_consensus = early_exit_consensus
        self.count_tokens = getattr(backends[0], "count_tokens", None)

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        candidates: list[tuple[str, list[dict]]] = []
        for idx, b in enumerate(self.backends):
            raw, calls = b.generate(messages, tools)
            candidates.append((raw, calls))
            if (
                self.early_exit_consensus
                and idx == 1
                and len(self.backends) >= 3
                and _canonical_calls_sig(candidates[0][1]) == _canonical_calls_sig(candidates[1][1])
            ):
                return candidates[0]
        return _vote_step_outputs(candidates)


class MultiLoRAEnsembleBackend(BaseLLMBackend):
    """VRAM-Optimal Step-Level Ensemble: loads ONE Qwen3-4B base model (< 5B budget) and swaps 2-3 PureLoRA adapters
    in-place (< 2 ms per swap) to vote at every step on a single ToolEnv."""

    def __init__(
        self,
        adapter_paths: list[str],
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 384,
        max_context_len: int = 16384,
        early_exit_consensus: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .lora_utils import DEFAULT_TARGET_MODULES, apply_pure_lora_state, inject_pure_lora, load_pure_lora_state

        assert adapter_paths, "MultiLoRAEnsembleBackend requires at least 1 PureLoRA adapter path"
        self.max_new_tokens = max_new_tokens
        self.max_context_len = min(max_context_len, 16384)
        self.early_exit_consensus = early_exit_consensus
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)
        if self.device != "cuda":
            torch_dtype = torch.float32

        first_dir = Path(adapter_paths[0])
        first_cfg = json.load(open(first_dir / "pure_lora_config.json", encoding="utf-8"))
        base_path = first_cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B-Instruct-2507")
        r = int(first_cfg.get("r", 32))
        alpha = int(first_cfg.get("lora_alpha", 64))
        target_modules = first_cfg.get("target_modules", list(DEFAULT_TARGET_MODULES))

        print(f"[MultiLoRA-Ensemble] Loading single base model {base_path} + {len(adapter_paths)} adapters: {adapter_paths}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(first_dir), trust_remote_code=True)
        common = dict(low_cpu_mem_usage=True, attn_implementation="sdpa", trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch_dtype, **common)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch_dtype, **common)

        inject_pure_lora(self.model, target_modules=target_modules, r=r, alpha=alpha, dropout=0.0)
        self.model.to(self.device)
        self.model.eval()

        self.states = [load_pure_lora_state(Path(p), device=self.device) for p in adapter_paths]

    def _apply_template(self, messages: list[dict], tools: list[dict]) -> str:
        return _apply_chat_template(self.tokenizer, messages, tools)

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return len(self.tokenizer.encode(self._apply_template(messages, tools), add_special_tokens=False))

    def generate(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[dict]]:
        import torch

        from .lora_utils import apply_pure_lora_state

        prompt = self._apply_template(messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        in_len = inputs["input_ids"].shape[1]

        candidates: list[tuple[str, list[dict]]] = []
        with torch.no_grad():
            for idx, st in enumerate(self.states):
                apply_pure_lora_state(self.model, st)
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                gen_ids = out_ids[0][in_len:]
                raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                for tok in ("<|im_end|>", "<|endoftext|>"):
                    raw = raw.replace(tok, "")
                raw = raw.strip()
                calls = parse_tool_calls(raw)
                candidates.append((raw, calls))

                if (
                    self.early_exit_consensus
                    and idx == 1
                    and len(self.states) >= 3
                    and _canonical_calls_sig(candidates[0][1]) == _canonical_calls_sig(candidates[1][1])
                ):
                    return candidates[0]

        return _vote_step_outputs(candidates)



# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------
def _collect_strings(obj: Any, bag: set[str]):
    if isinstance(obj, str):
        bag.add(obj)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        bag.add(str(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, bag)
    elif isinstance(obj, list):
        for x in obj:
            _collect_strings(x, bag)


def _id_family(v: str) -> str:
    """'c_778' -> 'c', 'chat_6142' -> 'chat', 'light_bedroom2' -> 'light'."""
    return v.split("_", 1)[0] if "_" in v else ""


def tz_offset_from_context(ctx: dict) -> str:
    """'+07:00' taken from context.current_time (all train/public samples are Asia/Ho_Chi_Minh)."""
    try:
        dt = datetime.fromisoformat(str(ctx.get("current_time") or ""))
        off = dt.strftime("%z")
        if off:
            return f"{off[:3]}:{off[3:]}"
    except Exception:
        pass
    return "+07:00"


def _coerce_value(v: Any, p: dict, tz: str) -> Any:
    """Best-effort repair of one argument value against its JSON-schema property (never raises)."""
    t = p.get("type")
    try:
        if t in ("integer", "number") and isinstance(v, str):
            s = v.strip()
            if _THOUSANDS_RE.match(s):  # Vietnamese "500.000" / "1,200,000" -> 500000 / 1200000
                s = re.sub(r"[.,]", "", s)
            if _INT_RE.match(s):
                v = int(s)
            elif _NUM_RE.match(s):
                f = float(s)
                v = int(f) if (t == "integer" and f.is_integer()) else f
        if t == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        if t == "boolean":
            if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                v = v.strip().lower() == "true"
        elif t == "string":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = str(v)
        elif t == "array":
            items_t = (p.get("items") or {}).get("type")
            if isinstance(v, str) and items_t in (None, "string"):
                v = [v]
    except Exception:
        return v

    if isinstance(v, str):
        enum = p.get("enum")
        if enum and v not in enum:
            low = v.strip().lower()
            for e in enum:
                if isinstance(e, str) and e.lower() == low:
                    v = e
                    break
        # Gold ALWAYS writes date-times as 'YYYY-MM-DDTHH:MM:SS+07:00' and times as 'HH:MM' (checked on all 1,545
        # train samples), and most train tools declare no `format`, so normalise by the VALUE's shape, not only `format`.
        fmt = p.get("format")
        s = v.strip()
        m_dt = _DT_RE.match(s)
        if fmt == "date":
            if len(s) > 10 and re.match(r"^\d{4}-\d{2}-\d{2}[T ]", s):
                v = s[:10]
        elif m_dt and int(m_dt.group(2)) < 24:
            off = m_dt.group(5)
            if off is None:
                off = tz
            elif off == "Z":
                off = "+00:00"
            elif ":" not in off:
                off = f"{off[:3]}:{off[3:]}"
            v = f"{m_dt.group(1)}T{int(m_dt.group(2)):02d}:{m_dt.group(3)}:{m_dt.group(4) or '00'}{off}"
        elif fmt in (None, "time") and not p.get("enum"):
            m_t = _TIME_RE.match(s)
            if m_t and int(m_t.group(1)) < 24 and (fmt == "time" or not s.isdigit()):
                v = f"{int(m_t.group(1)):02d}:{m_t.group(2) or '00'}"
    return v


# ----------------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------------
class LifeMateAgent:
    """Executes the deterministic multi-turn tool loop on a single `ToolEnv` session with Precision-Safe Guardrails."""

    def __init__(
        self,
        llm: BaseLLMBackend,
        catalog: dict,
        max_steps: int = 10,
        max_validation_retries: int = 3,
        max_empty_read_retries: int = 2,
        enforce_xref_read_first: bool = True,
        enable_zero_shot_anti_stop: bool = False,
        block_duplicate_writes: bool = True,
        stop_on_uninstalled_app: bool = True,
        auto_prereq_read: bool = True,
        max_new_tokens: int = 384,
        max_context_tokens: int = 16384,
    ):
        self.llm = llm
        self.catalog = catalog
        self.max_steps = max_steps
        self.max_validation_retries = max_validation_retries
        self.max_empty_read_retries = max_empty_read_retries
        self.enforce_xref_read_first = enforce_xref_read_first
        self.enable_zero_shot_anti_stop = enable_zero_shot_anti_stop
        self.block_duplicate_writes = block_duplicate_writes
        self.stop_on_uninstalled_app = stop_on_uninstalled_app
        self.auto_prereq_read = auto_prereq_read
        self.prompt_budget = min(max_context_tokens, 16384) - max_new_tokens - 32

        self.tool_specs: dict[str, dict] = {}
        self.xref_map: dict[str, dict[str, str]] = {}
        self.kind_map: dict[str, str] = {}

        for app in catalog.get("apps", []):
            for t in app.get("tools", []):
                tname = t["name"]
                self.tool_specs[tname] = t
                self.kind_map[tname] = t.get("kind", "write")
                props = (t.get("parameters") or {}).get("properties") or {}
                xrefs = {pname: pspec["x-ref"] for pname, pspec in props.items() if "x-ref" in pspec}
                if xrefs:
                    self.xref_map[tname] = xrefs

    # -- argument repair -----------------------------------------------------------------------------------------------
    def _sanitize_arguments(self, name: str, args: dict, tz: str = "+07:00") -> dict:
        args = copy.deepcopy(args) if isinstance(args, dict) else {}
        spec = self.tool_specs.get(name)
        if spec is None:
            return args  # unknown tool: leave it to env.validate()
        params = spec.get("parameters") or {}
        props = params.get("properties") or {}
        if not props:
            return {}

        # Model habits that are not plain type errors
        if name == "smart_home.get_status" and isinstance(args.get("room"), str):
            r_low = args["room"].strip().lower()
            if r_low in _EN_ROOM_MAP:
                args["room"] = _EN_ROOM_MAP[r_low]
        if name == "smart_home.climate_set" and args.get("power") == "off" and "mode" not in args and "power" not in props:
            args["mode"] = "off"
            args.pop("power", None)

        out = {}
        for k, v in args.items():
            if k not in props:
                continue  # unknown key: never in gold, would only break the argument match
            if v in _EMPTY:
                continue  # the validator treats empty as "not given" anyway
            out[k] = _coerce_value(v, props[k], tz)
        return out

    # -- grounding of the money amount ---------------------------------------------------------------------------------
    @staticmethod
    def amount_is_missing(name: str, args: dict, user_text: str, known_numbers: set[int] | None = None, history_text: str = "") -> bool:
        """True when the CURRENT request only says a vague quantity ("chút tiền", "ít tiền") but the model invented an
        `amount`. Gold asks `sys.ask_missing_info(params=["amount"])` in all 18/18 such train utterances (0 false positives).
        A broader "number not spoken" check was tried and REJECTED: it blocked 6 correct gold calls whose amount is
        computed from tool results (oracle 99.86 -> 99.47)."""
        if name not in AMOUNT_GUARD_TOOLS or not isinstance(args.get("amount"), (int, float)):
            return False
        return bool(_VAGUE_AMOUNT_RE.search((user_text or "").lower()))

    def _sanitize_ask(self, args: dict) -> dict:
        """sys.ask_missing_info: keep only params that exist on the target tool, de-duplicated, order kept."""
        target = args.get("tool_name")
        params = args.get("params")
        if isinstance(params, str):
            params = [params]
        params = [p for p in (params or []) if isinstance(p, str)]
        spec = self.tool_specs.get(target) if isinstance(target, str) else None
        if spec is not None:
            props = (spec.get("parameters") or {}).get("properties") or {}
            params = [p for p in params if p in props]
        seen, uniq = set(), []
        for p in params:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        out = {k: v for k, v in args.items() if k not in ("tool_name", "params")}
        out["tool_name"] = target
        out["params"] = uniq
        return out

    def _ask_target_error(self, args: dict, allowed: set[str]) -> str | None:
        target = args.get("tool_name")
        if not isinstance(target, str) or target == "sys.ask_missing_info":
            return "tool_name phải là tool đích (không phải sys.ask_missing_info)."
        if target not in allowed:
            return f"tool_name='{target}' không thuộc app đã cài."
        if not args.get("params"):
            return f"params rỗng hoặc không phải tham số của '{target}'."
        if self.kind_map.get(target) == "read":
            req = set(((self.tool_specs.get(target) or {}).get("parameters") or {}).get("required") or [])
            if target in ASK_DENY_READ_TARGETS or not set(args["params"]) <= req:
                return (
                    f"tool_name='{target}' là tool [READ]. Hãy hỏi cho tool [WRITE] đích "
                    f"(VD thiếu người nhận -> params=['contact_id'] của tool gửi/gọi)."
                )
        return None

    # -- context budget ------------------------------------------------------------------------------------------------
    def _fit_context(self, messages: list[dict], tools: list[dict], final_idx: int) -> list[dict] | None:
        count = getattr(self.llm, "count_tokens", None)
        if not callable(count):
            return messages
        try:
            if count(messages, tools) <= self.prompt_budget:
                return messages
            head, hist, tail = messages[:1], messages[1:final_idx], messages[final_idx:]
            while hist:
                hist = hist[1:]
                while hist and hist[0].get("role") != "user":
                    hist = hist[1:]
                view = head + hist + tail
                if count(view, tools) <= self.prompt_budget:
                    return view
            view = head + tail
            for limit in (2000, 800):
                view = [
                    {**m, "content": m["content"][:limit] + " …(đã cắt bớt)"}
                    if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > limit
                    else m
                    for m in view
                ]
                if count(view, tools) <= self.prompt_budget:
                    return view
        except Exception as e:  # tokenizer problems must never kill a sample
            print(f"[Agent] context budgeting failed: {e}")
            return messages
        return None

    # -- main loop -----------------------------------------------------------------------------------------------------
    def run_sample(self, env, sample: dict, transcript_map: dict[str, dict[str, str]], deadline: float | None = None) -> None:
        ctx = sample.get("context") or {}
        tz = tz_offset_from_context(ctx)
        installed_apps = ctx.get("installed_apps") or {}
        installed_prefixes = set(installed_apps.keys()) | {"sys"}
        filtered_tools = filter_installed_tools(self.catalog, installed_apps)
        allowed_tools = {t["name"] for t in filtered_tools}
        openai_tools = to_openai_tools(filtered_tools)
        messages = build_messages(sample, transcript_map)
        final_idx = len(messages) - 1

        # IDs seen in real data. Per app (strict check) + global (an ID returned by another app is still real data).
        known_by_app: dict[str, set[str]] = defaultdict(set)
        known_any: set[str] = set()
        history_text: list[str] = []
        history_tool_names: set[str] = set()
        for h in sample.get("history") or []:
            for tc, tr in zip(h.get("tool_calls") or [], h.get("tool_results") or []):
                tc_name = tc.get("name") or ""
                history_tool_names.add(tc_name)
                app_p = tc_name.split(".", 1)[0]
                _collect_strings(tc, known_by_app[app_p])
                _collect_strings(tr, known_by_app[app_p])
                _collect_strings(tr, known_any)
            if h.get("text"):
                history_text.append(h["text"])

        final_user_text = str(messages[final_idx].get("content") or "")

        executed_counts: dict[str, int] = defaultdict(int)
        empty_read_counts: dict[str, int] = defaultdict(int)
        executed_any = False
        executed_write_or_meta = False
        executed_prereq_read = False
        anti_stop_reminded = False
        retries_used = 0
        call_seq = 1

        def blocked_msg(cid: str, name: str, error: str, flag: str) -> dict:
            return {
                "role": "tool",
                "tool_call_id": cid,
                "name": name,
                "content": json.dumps({"ok": False, flag: True, "error": error}, ensure_ascii=False),
            }

        for _step in range(self.max_steps):
            if deadline is not None and time.perf_counter() > deadline:
                print(f"[Agent] {sample.get('id')}: time budget reached, stopping.")
                break

            view = self._fit_context(messages, openai_tools, final_idx)
            if view is None:
                print(f"[Agent] {sample.get('id')}: prompt cannot fit the context budget, stopping.")
                break
            try:
                raw_text, proposed_calls = self.llm.generate(view, openai_tools)
            except Exception as e:
                print(f"[Agent] {sample.get('id')}: LLM error, stopping: {e}")
                break

            if not proposed_calls:
                if (
                    self.enable_zero_shot_anti_stop
                    and executed_prereq_read
                    and not executed_write_or_meta
                    and not anti_stop_reminded
                ):
                    anti_stop_reminded = True
                    messages.append({"role": "assistant", "content": raw_text or "Đã tra cứu."})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[KIỂM TRA TRƯỚC KHI KẾT THÚC]: Bạn vừa gọi tool tra cứu nhưng chưa gọi tool [WRITE] hay `sys.ask_missing_info`. "
                                "1) Nếu còn bước hành động tiếp theo (và đủ điều kiện/số dư/chỗ trống), hãy gọi tiếp tool [WRITE] ngay. "
                                "2) Nếu kết quả có >= 2 mục trùng tên không phân biệt được hoặc thiếu tham số bắt buộc, hãy gọi `sys.ask_missing_info`. "
                                "3) Nếu yêu cầu của người dùng thuần túy chỉ là xem thông tin, hoặc điều kiện 'Nếu...' KHÔNG thỏa mãn, hoặc mặt hàng/mã không tồn tại, hãy kết thúc ngay (KHÔNG gọi thêm bất kỳ tool nào)."
                            ),
                        }
                    )
                    continue
                break

            # Out-of-scope shortcut: nothing executed yet and every call targets a real app that is not installed.
            if self.stop_on_uninstalled_app and not executed_any:
                prefixes = [str(c.get("name") or "").split(".", 1)[0] for c in proposed_calls]
                if prefixes and all(p in KNOWN_APP_IDS and p not in installed_prefixes for p in prefixes):
                    break

            assistant_tc_msgs = []
            tool_response_msgs = []
            asked_missing_info = False
            blocked_in_memory = False

            for call in proposed_calls:
                name = call.get("name")
                if not isinstance(name, str):
                    continue
                raw_args = call.get("arguments") or {}
                if name == "sys.ask_missing_info":
                    args = self._sanitize_ask(raw_args if isinstance(raw_args, dict) else {})
                else:
                    args = self._sanitize_arguments(name, raw_args, tz)
                cid = f"call_{call_seq}"
                call_seq += 1

                assistant_tc_msgs.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                    }
                )

                # Guardrail: sys.ask_missing_info must point to a sensible installed target with real params
                if name == "sys.ask_missing_info":
                    err = self._ask_target_error(args, allowed_tools)
                    if err:
                        blocked_in_memory = True
                        tool_response_msgs.append(blocked_msg(cid, name, f"LỖI (chưa thực thi): {err}", "validation_blocked"))
                        continue

                # Guardrail: schema validation BEFORE env.call() (rejected calls would count as unmatched U)
                val = env.validate(name, args)
                if not val["ok"]:
                    blocked_in_memory = True
                    tool_response_msgs.append(
                        blocked_msg(
                            cid,
                            name,
                            f"LỖI SCHEMA (chưa thực thi): {val['error']}. Hãy sửa tham số hoặc gọi sys.ask_missing_info.",
                            "validation_blocked",
                        )
                    )
                    continue

                # Guardrail: anti-hallucination x-ref check BEFORE env.call()
                if self.enforce_xref_read_first and name in self.xref_map:
                    hallucinated = []
                    for p, ref in self.xref_map[name].items():
                        vals = args.get(p)
                        vals = vals if isinstance(vals, list) else [vals]
                        ref_app = ref.split(".", 1)[0]
                        for one in vals:
                            val_str = str(one or "")
                            if not val_str or val_str in known_by_app[ref_app]:
                                continue
                            if any(val_str in txt for txt in history_text):
                                continue
                            if val_str in known_any:
                                fams = {_id_family(x) for x in known_by_app[ref_app] if _id_family(x)}
                                if not fams or _id_family(val_str) in fams:
                                    continue
                            hallucinated.append((p, val_str, ref, ref_app))
                    if hallucinated:
                        blocked_in_memory = True
                        p_bad, v_bad, ref_bad, ref_app = hallucinated[0]
                        tool_response_msgs.append(
                            blocked_msg(
                                cid,
                                name,
                                (
                                    f"CHƯA THỰC THI: '{p_bad}'='{v_bad}' thuộc '{ref_bad}' nhưng chưa được tra cứu từ app '{ref_app}'. "
                                    f"Hãy gọi tool [READ] của '{ref_app}' trước để lấy ID thật!"
                                ),
                                "xref_blocked",
                            )
                        )
                        continue

                # Guardrail: invented money amount -> ask for it (gold behaviour), after the person was looked up
                if self.amount_is_missing(name, args, final_user_text):
                    name, args = "sys.ask_missing_info", {"tool_name": name, "params": ["amount"]}
                    assistant_tc_msgs[-1]["function"] = {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}
                    val = env.validate(name, args)
                    if not val["ok"] or self._ask_target_error(args, allowed_tools):
                        blocked_in_memory = True
                        tool_response_msgs.append(blocked_msg(cid, name, "LỖI: không thể hỏi lại số tiền.", "validation_blocked"))
                        continue

                # Guardrail: identical READ again = wasted precision; identical WRITE again = wrong state hash
                kind = self.kind_map.get(name, "write")
                sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                max_allowed_same = 1 if kind == "read" else (MAX_IDENTICAL_WRITES.get(name, 1) if self.block_duplicate_writes else 99)
                if executed_counts[sig] >= max_allowed_same:
                    blocked_in_memory = True
                    msg = (
                        "Tool [READ] này đã được gọi với cùng tham số. Hãy dùng kết quả đã có."
                        if kind == "read"
                        else "Lệnh [WRITE] này ĐÃ được thực thi với đúng tham số này. Nếu yêu cầu đã xong, hãy kết thúc (không gọi thêm tool)."
                    )
                    tool_response_msgs.append(blocked_msg(cid, name, msg, "duplicate_blocked"))
                    continue

                # Guardrail (Trick #1.3): Retry cap on empty READ search (stops infinite search loops after 2 empty returns)
                if kind == "read" and self.max_empty_read_retries > 0 and empty_read_counts[name] >= self.max_empty_read_retries:
                    blocked_in_memory = True
                    tool_response_msgs.append(
                        blocked_msg(
                            cid,
                            name,
                            (
                                f"GIỚI HẠN THỬ LẠI: Tool '{name}' đã trả về kết quả rỗng {empty_read_counts[name]} lần. "
                                "KHÔNG gọi tìm kiếm lặp lại nữa! Nếu thiếu thông tin hoặc cần người dùng làm rõ, hãy gọi sys.ask_missing_info hoặc kết thúc."
                            ),
                            "retry_cap_blocked",
                        )
                    )
                    continue

                # Auto-prerequisite Read: robot_vacuum.start(rooms=[...]) requires robot_vacuum.get_status({}) first
                # (13/14 room-specific train gold samples call get_status first; fixes 11 partial-score val errors)
                if (
                    self.auto_prereq_read
                    and name == "robot_vacuum.start"
                    and args.get("rooms")
                    and "robot_vacuum.get_status" in allowed_tools
                    and "robot_vacuum.get_status" not in history_tool_names
                    and executed_counts["robot_vacuum.get_status:{}"] == 0
                    and not any(c.get("name") == "robot_vacuum.get_status" for c in proposed_calls)
                ):
                    pre_sig = "robot_vacuum.get_status:{}"
                    executed_counts[pre_sig] += 1
                    executed_any = True
                    pre_res = env.call("robot_vacuum.get_status", {})
                    _collect_strings(pre_res, known_by_app["robot_vacuum"])
                    _collect_strings(pre_res, known_any)

                # Execute on the single ToolEnv session
                executed_counts[sig] += 1
                executed_any = True
                result = env.call(name, args)
                app_prefix = name.split(".", 1)[0]
                _collect_strings(result, known_by_app[app_prefix])
                _collect_strings(result, known_any)
                if kind == "read" and _is_empty_read_result(name, result):
                    empty_read_counts[name] += 1

                if kind in ("write", "meta") or name == "phone.call":
                    executed_write_or_meta = True
                elif name in ACTION_PREREQ_READ_TOOLS:
                    executed_prereq_read = True

                tool_response_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "name": name,
                        "content": json.dumps(annotate_tool_result(name, result), ensure_ascii=False),
                    }
                )

                if name == "sys.ask_missing_info":
                    asked_missing_info = True
                    break

            if not assistant_tc_msgs:
                break
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": assistant_tc_msgs[: len(tool_response_msgs)]}
            )
            messages.extend(tool_response_msgs)

            if asked_missing_info:
                break

            if blocked_in_memory:
                retries_used += 1
                if retries_used > self.max_validation_retries:
                    break
