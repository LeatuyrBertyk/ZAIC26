"""Regression check: replay the GOLD tool calls through `LifeMateAgent` (sanitizer + guardrails) on `train.jsonl`.

A perfect LLM that always proposes the gold call should score 100. Anything below 100 is damage done by OUR OWN
runtime code (a guardrail blocking a correct call, the sanitizer rewriting a correct argument, ...), so run this
after every change to `lifemate/agent.py`. CPU only, ~1 minute:

    python -X utf8 scripts/oracle_agent_check.py
    python -X utf8 scripts/oracle_agent_check.py --show 20     # print the first 20 damaged samples
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(DATA_ROOT / "train"))

from evaluate_local import compute_call_f1, infer_category  # noqa: E402
from lifemate.agent import BaseLLMBackend, LifeMateAgent  # noqa: E402
from zaic_tools import ToolEnv  # noqa: E402
from zaic_tools.validate import catalog_tools  # noqa: E402


class OracleLLM(BaseLLMBackend):
    """Proposes the gold calls one by one (one call per assistant turn, like the SFT data)."""

    def __init__(self):
        self.gold: list[dict] = []
        self.k = 0

    def reset(self, gold: list[dict]):
        self.gold, self.k = gold, 0

    def generate(self, messages, tools):
        if self.k >= len(self.gold):
            return "Đã hoàn tất thực thi.", []
        c = self.gold[self.k]
        self.k += 1
        return "", [{"name": c["name"], "arguments": json.loads(json.dumps(c["arguments"]))}]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_ROOT / "train" / "train.jsonl"))
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    catalog = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
    tools_map = catalog_tools(catalog)
    llm = OracleLLM()
    agent = LifeMateAgent(llm=llm, catalog=catalog, max_steps=12)

    scores, damaged, by_cat = [], [], Counter()
    for line in open(args.input, encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        gold = item["label"]["tool_calls"]
        env_gt = ToolEnv("train", item["id"], root=DATA_ROOT)
        for c in gold:
            env_gt.call(c["name"], c["arguments"])

        env = ToolEnv("train", item["id"], root=DATA_ROOT)
        llm.reset(gold)
        tmap = {h["audio"]: {"primary": h.get("audio_text") or ""} for h in item.get("history") or [] if h.get("audio")}
        tmap[item["audio"]] = {"primary": item.get("audio_text") or ""}
        agent.run_sample(env, item, tmap)
        pred = env.get_tool_calls()

        if not gold:
            s = 1.0 if not pred else 0.0
        else:
            _, _, f1, m = compute_call_f1(pred, gold, tools_map)
            s = f1 if (env.get_state_hash() == env_gt.get_state_hash() and m > 0) else 0.0
        scores.append(s)
        if s < 1.0:
            cat = infer_category(item)
            by_cat[cat] += 1
            damaged.append({"id": item["id"], "cat": cat, "score": round(s, 3), "gold": gold, "pred": pred})

    print(f"Oracle-through-agent score: {100 * sum(scores) / len(scores):.2f} / 100 on {len(scores)} samples")
    print(f"Damaged samples: {len(damaged)} | by category: {dict(by_cat)}")
    for d in damaged[: args.show]:
        print(json.dumps(d, ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
