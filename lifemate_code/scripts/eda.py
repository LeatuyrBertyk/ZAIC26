"""Comprehensive Exploratory Data Analysis (EDA) & Ground-Truth Replay Check for ZAC 2026 LifeMate."""
from __future__ import annotations

import json
import sys
import wave
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE / "train+public_test" / "train+public_test"

sys.path.insert(0, str(DATA_ROOT / "train"))
from zaic_tools import ToolEnv, state_hash  # noqa: E402
from zaic_tools.validate import catalog_tools  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inspect_catalogs() -> dict:
    train_cat = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
    pub_cat = json.load(open(DATA_ROOT / "public_test" / "public_test_tools_catalog.json", encoding="utf-8"))

    train_apps = {a["id"]: a for a in train_cat["apps"]}
    pub_apps = {a["id"]: a for a in pub_cat["apps"]}

    train_tools = catalog_tools(train_cat)
    pub_tools = catalog_tools(pub_cat)

    only_train_apps = sorted(set(train_apps) - set(pub_apps))
    only_pub_apps = sorted(set(pub_apps) - set(train_apps))
    shared_apps = sorted(set(train_apps) & set(pub_apps))

    only_train_tools = sorted(set(train_tools) - set(pub_tools))
    only_pub_tools = sorted(set(pub_tools) - set(train_tools))

    # Inspect special schema extensions across all tools
    all_tools = {**train_tools, **pub_tools}
    xref_params = []
    fuzzy_params = []
    oneof_tools = []
    for tname, tspec in sorted(all_tools.items()):
        params = tspec.get("parameters") or {}
        if params.get("x-one-of"):
            oneof_tools.append((tname, params["x-one-of"]))
        for pname, pspec in (params.get("properties") or {}).items():
            if "x-ref" in pspec:
                xref_params.append((tname, pname, pspec["x-ref"]))
            if pspec.get("x-match") == "fuzzy":
                fuzzy_params.append((tname, pname))

    return {
        "train_num_apps": len(train_apps),
        "train_num_tools": len(train_tools),
        "pub_num_apps": len(pub_apps),
        "pub_num_tools": len(pub_tools),
        "shared_apps": shared_apps,
        "only_train_apps": only_train_apps,
        "only_pub_apps": only_pub_apps,
        "only_train_tools": only_train_tools,
        "only_pub_tools": only_pub_tools,
        "sys_tool_spec": all_tools.get("sys.ask_missing_info"),
        "xref_params": xref_params,
        "fuzzy_params": fuzzy_params,
        "oneof_tools": oneof_tools,
        "pub_unseen_app_details": [
            {
                "id": aid,
                "name_vi": pub_apps[aid].get("name_vi"),
                "description": pub_apps[aid].get("description"),
                "tools": [t["name"] for t in pub_apps[aid].get("tools", [])],
            }
            for aid in only_pub_apps
        ],
    }


def check_audio_stats(split: str, items: list[dict], max_check: int = 300) -> dict:
    durations = []
    sample_rates = Counter()
    missing = 0
    for item in items[:max_check]:
        wav_path = DATA_ROOT / split / item["audio"]
        if not wav_path.is_file():
            missing += 1
            continue
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            sample_rates[sr] += 1
            durations.append(round(frames / sr, 2))
    durations.sort()
    n = len(durations)
    return {
        "checked": len(items[:max_check]),
        "missing": missing,
        "sample_rates": dict(sample_rates),
        "min_sec": durations[0] if n else 0,
        "median_sec": durations[n // 2] if n else 0,
        "p95_sec": durations[int(n * 0.95)] if n else 0,
        "max_sec": durations[-1] if n else 0,
    }


def analyze_train(train_items: list[dict]) -> dict:
    cat_counts = Counter()
    call_len_counts = Counter()
    history_len_counts = Counter()
    app_usage = Counter()
    multi_app_count = 0
    replay_mismatches = []
    schema_errors_in_gt = 0

    for item in train_items:
        qid = item["id"]
        tcalls = item["label"]["tool_calls"]
        fstate = item["label"]["final_state"]
        call_len_counts[len(tcalls)] += 1
        history_len_counts[len(item.get("history") or [])] += 1

        names = [c["name"] for c in tcalls]
        apps_used = {n.split(".", 1)[0] for n in names if not n.startswith("sys.")}
        for a in apps_used:
            app_usage[a] += 1
        if len(apps_used) >= 2:
            multi_app_count += 1

        if not tcalls:
            cat_counts["out_of_scope_or_noop ([])"] += 1
        elif any(n == "sys.ask_missing_info" for n in names):
            if len(tcalls) == 1:
                cat_counts["ask_missing_info_direct"] += 1
            else:
                cat_counts["read_then_ask_missing_info"] += 1
        elif fstate:
            if len(tcalls) == 1:
                cat_counts["direct_write (1 call)"] += 1
            else:
                cat_counts["multi_step_write (>=2 calls)"] += 1
        else:
            cat_counts["read_only_or_conditional_noop"] += 1

        # Replay ground truth in ToolEnv to verify 100% consistency
        env = ToolEnv("train", qid, root=DATA_ROOT)
        before = env.get_state()
        for c in tcalls:
            res = env.call(c["name"], c["arguments"])
            if not res.get("ok"):
                schema_errors_in_gt += 1
        after = env.get_state()
        changed = {app: st for app, st in after.items() if st != before.get(app)}
        if state_hash(changed) != state_hash(fstate):
            replay_mismatches.append(qid)

    return {
        "total": len(train_items),
        "categories": dict(cat_counts),
        "call_length_dist": dict(sorted(call_len_counts.items())),
        "history_turns_dist": dict(sorted(history_len_counts.items())),
        "multi_app_samples": multi_app_count,
        "app_usage_top": dict(app_usage.most_common()),
        "gt_schema_errors": schema_errors_in_gt,
        "gt_replay_mismatches": replay_mismatches,
    }


def analyze_public_test(pub_items: list[dict], unseen_apps: set[str]) -> dict:
    history_len_counts = Counter()
    installed_counts = Counter()
    samples_with_unseen_apps = 0
    unseen_app_presence = Counter()

    for item in pub_items:
        history_len_counts[len(item.get("history") or [])] += 1
        inst = set((item.get("context") or {}).get("installed_apps") or {})
        installed_counts[len(inst)] += 1
        overlap = inst & unseen_apps
        if overlap:
            samples_with_unseen_apps += 1
            for a in overlap:
                unseen_app_presence[a] += 1

    return {
        "total": len(pub_items),
        "history_turns_dist": dict(sorted(history_len_counts.items())),
        "installed_apps_count_dist": dict(sorted(installed_counts.items())),
        "samples_with_unseen_apps": samples_with_unseen_apps,
        "unseen_app_presence": dict(unseen_app_presence),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    train_items = load_jsonl(DATA_ROOT / "train" / "train.jsonl")
    pub_items = load_jsonl(DATA_ROOT / "public_test" / "public_test.jsonl")

    cat_report = inspect_catalogs()
    train_report = analyze_train(train_items)
    pub_report = analyze_public_test(pub_items, set(cat_report["only_pub_apps"]))
    train_audio = check_audio_stats("train", train_items, max_check=1545)
    pub_audio = check_audio_stats("public_test", pub_items, max_check=634)

    summary = {
        "catalog_comparison": cat_report,
        "train_analysis": train_report,
        "public_test_analysis": pub_report,
        "train_audio": train_audio,
        "public_test_audio": pub_audio,
    }
    out_path = WORKSPACE / "outputs" / "eda_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
