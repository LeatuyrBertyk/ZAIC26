"""Create deterministic Stratified + Unseen-App Validation Splits from train.jsonl (1,545 -> 1,295 train / 250 val)."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE / "train+public_test" / "train+public_test"

sys.path.insert(0, str(DATA_ROOT / "train"))
from zaic_tools import ToolEnv  # noqa: E402

# 3 apps that exist ONLY in train (not in public_test) held out to simulate unseen zero-shot apps in public/private test
HELD_OUT_ZERO_SHOT_APPS = {"library", "robot_vacuum", "bike_rental"}
TARGET_VAL_SIZE = 250
SEED = 42


def classify_sample(item: dict) -> str:
    tcalls = item["label"]["tool_calls"]
    fstate = item["label"]["final_state"]
    names = [c["name"] for c in tcalls]
    if not tcalls:
        return "out_of_scope"
    if any(n == "sys.ask_missing_info" for n in names):
        return "ask_missing_info_direct" if len(tcalls) == 1 else "read_then_ask_missing_info"
    if fstate:
        return "direct_write" if len(tcalls) == 1 else "multi_step_write"
    return "read_only"


def used_apps(item: dict) -> set[str]:
    return {
        c["name"].split(".", 1)[0]
        for c in item["label"]["tool_calls"]
        if not c["name"].startswith("sys.")
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(SEED)

    train_path = DATA_ROOT / "train" / "train.jsonl"
    items = [json.loads(line) for line in open(train_path, encoding="utf-8") if line.strip()]

    unseen_val = []
    pool_by_cat: dict[str, list[dict]] = defaultdict(list)

    for item in items:
        cat = classify_sample(item)
        apps = used_apps(item)
        item_meta = {**item, "_category": cat}
        if apps & HELD_OUT_ZERO_SHOT_APPS:
            item_meta["_val_subset"] = "unseen_app"
            unseen_val.append(item_meta)
        else:
            pool_by_cat[cat].append(item_meta)

    remaining_needed = TARGET_VAL_SIZE - len(unseen_val)
    total_pool = sum(len(v) for v in pool_by_cat.values())

    seen_val = []
    train_split = []

    # Proportional stratified sampling across behavioral categories
    cats_sorted = sorted(pool_by_cat.keys())
    allocated = 0
    quotas = {}
    for i, cat in enumerate(cats_sorted):
        group = pool_by_cat[cat]
        rng.shuffle(group)
        if i < len(cats_sorted) - 1:
            q = round(remaining_needed * len(group) / total_pool)
        else:
            q = remaining_needed - allocated
        q = max(1, min(q, len(group)))
        quotas[cat] = q
        allocated += q

    for cat in cats_sorted:
        group = pool_by_cat[cat]
        q = quotas[cat]
        for x in group[:q]:
            x["_val_subset"] = "seen_app"
            seen_val.append(x)
        for x in group[q:]:
            x["_val_subset"] = "train"
            train_split.append(x)

    val_split = sorted(unseen_val + seen_val, key=lambda x: x["id"])
    train_split = sorted(train_split, key=lambda x: x["id"])

    out_dir = WORKSPACE / "outputs" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train_split.jsonl", "w", encoding="utf-8") as f:
        for x in train_split:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(out_dir / "val_split.jsonl", "w", encoding="utf-8") as f:
        for x in val_split:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # Build Oracle GT submission & Empty submission on val_split for local evaluator verification
    with (
        open(out_dir / "val_oracle_submission.jsonl", "w", encoding="utf-8") as f_gt,
        open(out_dir / "val_empty_submission.jsonl", "w", encoding="utf-8") as f_empty,
    ):
        for x in val_split:
            qid = x["id"]
            env_empty = ToolEnv("train", qid, root=DATA_ROOT)
            f_empty.write(
                json.dumps({"id": qid, "tool_calls": [], "state_hash": env_empty.get_state_hash()}, ensure_ascii=False)
                + "\n"
            )

            env_gt = ToolEnv("train", qid, root=DATA_ROOT)
            for c in x["label"]["tool_calls"]:
                env_gt.call(c["name"], c["arguments"])
            f_gt.write(
                json.dumps(
                    {"id": qid, "tool_calls": env_gt.get_tool_calls(), "state_hash": env_gt.get_state_hash()},
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta = {
        "seed": SEED,
        "total_original_train": len(items),
        "train_split_size": len(train_split),
        "val_split_size": len(val_split),
        "val_seen_app_size": len(seen_val),
        "val_unseen_app_size": len(unseen_val),
        "held_out_zero_shot_apps": sorted(HELD_OUT_ZERO_SHOT_APPS),
        "train_categories": dict(Counter(x["_category"] for x in train_split)),
        "val_categories": dict(Counter(x["_category"] for x in val_split)),
    }
    with open(out_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
