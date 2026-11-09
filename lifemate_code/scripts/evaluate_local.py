"""Local Evaluator for ZAC 2026 LifeMate (Exact Official Competition Formula).

Replays a submission JSONL against ground-truth labels using `zaic_tools` and reports:
1. `official_score` (0..100): Exact BTC formula from the Evaluation tab:
   - If gold `tool_calls` is empty (`G == []`): `Score_i = 1.0` if `pred_calls == []` else `0.0`.
   - If gold `tool_calls` is non-empty:
     `Score_i = 0.0` if `replayed_hash != gt_hash` or `M == 0`, else `2 * P * R / (P + R)`
     where `P = M / (M + U)`, `R = M / |G|`, and fuzzy arguments require `token-F1 > 0.6` (after lowercasing & Unicode NFC).
2. `state_em`, `call_f1`, `call_prec`, `call_rec`, and `task_success`.
3. Breakdown by `seen_app` vs `unseen_app` and behavioral categories.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)

sys.path.insert(0, str(DATA_ROOT / "train"))
from zaic_tools import ToolEnv, schema_validate  # noqa: E402
from zaic_tools.utils import FREE_TEXT_KEYS, _canon_scalar  # noqa: E402
from zaic_tools.validate import catalog_tools  # noqa: E402


def _nfc_tokens(s: str) -> set[str]:
    """Tokenize after lowercasing and Unicode NFC (matching the official evaluation description)."""
    norm = unicodedata.normalize("NFC", str(s or "").lower())
    return set(re.findall(r"[^\W_]+", norm))


def fuzzy_text_match(pred_val: str, gt_val: str, threshold: float = 0.6) -> bool:
    if _canon_scalar(pred_val) == _canon_scalar(gt_val):
        return True
    t_pred = _nfc_tokens(pred_val)
    t_gt = _nfc_tokens(gt_val)
    if not t_gt and not t_pred:
        return True
    if not t_gt or not t_pred:
        return False
    inter = len(t_pred & t_gt)
    prec = inter / len(t_pred)
    rec = inter / len(t_gt)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return f1 > threshold


def args_match(tool_name: str, pred_args: dict, gt_args: dict, tool_spec: dict | None) -> bool:
    pred_args = {k: v for k, v in (pred_args or {}).items() if v not in (None, "", [], {})}
    gt_args = {k: v for k, v in (gt_args or {}).items() if v not in (None, "", [], {})}

    if tool_name == "sys.ask_missing_info":
        return pred_args.get("tool_name") == gt_args.get("tool_name") and set(pred_args.get("params") or []) == set(
            gt_args.get("params") or []
        )

    if set(pred_args.keys()) != set(gt_args.keys()):
        return False

    props = ((tool_spec or {}).get("parameters") or {}).get("properties") or {}
    for k, gt_v in gt_args.items():
        pred_v = pred_args[k]
        pspec = props.get(k) or {}
        is_fuzzy = pspec.get("x-match") == "fuzzy" or k in FREE_TEXT_KEYS
        if is_fuzzy and isinstance(gt_v, str) and isinstance(pred_v, str):
            if not fuzzy_text_match(pred_v, gt_v, threshold=0.6):
                return False
        elif isinstance(gt_v, list) and isinstance(pred_v, list):
            if len(gt_v) != len(pred_v):
                return False
            if all(not isinstance(x, (dict, list)) for x in gt_v + pred_v):
                if sorted(map(_canon_scalar, pred_v), key=str) != sorted(map(_canon_scalar, gt_v), key=str):
                    return False
            elif pred_v != gt_v:
                return False
        else:
            if _canon_scalar(pred_v) != _canon_scalar(gt_v):
                return False
    return True


def compute_call_f1(pred_calls: list[dict], gt_calls: list[dict], tools_map: dict) -> tuple[float, float, float, int]:
    if not gt_calls and not pred_calls:
        return 1.0, 1.0, 1.0, 0
    if not gt_calls or not pred_calls:
        return 0.0, 0.0, 0.0, 0

    matched_gt = set()
    tp = 0
    for pc in pred_calls:
        pname = pc.get("name", "")
        pargs = pc.get("arguments") or {}
        for j, gc in enumerate(gt_calls):
            if j in matched_gt:
                continue
            if pname == gc.get("name") and args_match(pname, pargs, gc.get("arguments") or {}, tools_map.get(pname)):
                matched_gt.add(j)
                tp += 1
                break

    prec = tp / len(pred_calls)
    rec = tp / len(gt_calls)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp


def infer_category(item: dict) -> str:
    if "_category" in item:
        return item["_category"]
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


def evaluate(
    gt_path: Path,
    sub_path: Path,
    split: str = "train",
    error_out: Path | None = None,
    only_present: bool = False,
) -> dict:
    gt_items = {json.loads(l)["id"]: json.loads(l) for l in open(gt_path, encoding="utf-8") if l.strip()}
    sub_rows = {json.loads(l)["id"]: json.loads(l) for l in open(sub_path, encoding="utf-8") if l.strip()}
    if only_present:
        gt_items = {k: v for k, v in gt_items.items() if k in sub_rows}

    cat_json = json.load(open(DATA_ROOT / split / f"{split}_tools_catalog.json", encoding="utf-8"))
    tools_map = catalog_tools(cat_json)

    missing_ids = sorted(set(gt_items) - set(sub_rows))
    extra_ids = sorted(set(sub_rows) - set(gt_items))

    metrics_by_group: dict[str, list[dict]] = defaultdict(list)
    errors = []
    total_schema_errors = 0
    hash_integrity_mismatches = 0

    for qid, item in sorted(gt_items.items()):
        cat = infer_category(item)
        subset = item.get("_val_subset", "all")
        gt_calls = item["label"]["tool_calls"]

        env_gt = ToolEnv(split, qid, root=DATA_ROOT)
        for c in gt_calls:
            env_gt.call(c["name"], c["arguments"])
        gt_hash = env_gt.get_state_hash()

        row = sub_rows.get(qid)
        if row is None:
            pred_calls = []
            replayed_hash = ""
            sub_hash = ""
        else:
            pred_calls = row.get("tool_calls") or []
            sub_hash = row.get("state_hash") or ""
            env_pred = ToolEnv(split, qid, root=DATA_ROOT)
            val_res = schema_validate(pred_calls, cat_json, env_pred.allowed)
            total_schema_errors += sum(1 for r in val_res if not r["ok"])
            for c in pred_calls:
                env_pred.call(c.get("name", ""), c.get("arguments") or {})
            replayed_hash = env_pred.get_state_hash()
            if sub_hash and sub_hash != replayed_hash:
                hash_integrity_mismatches += 1

        state_em = float(replayed_hash == gt_hash and row is not None)
        prec, rec, f1, matched_m = compute_call_f1(pred_calls, gt_calls, tools_map)

        # Official Score_i formula from BTC Evaluation tab:
        if not gt_calls:
            official_score_i = 1.0 if (pred_calls == [] and row is not None and state_em == 1.0) else 0.0
        else:
            if state_em != 1.0 or matched_m == 0:
                official_score_i = 0.0
            else:
                official_score_i = f1

        rec_entry = {
            "official_score": official_score_i,
            "state_em": state_em,
            "call_prec": prec,
            "call_rec": rec,
            "call_f1": f1,
        }
        metrics_by_group["OVERALL"].append(rec_entry)
        metrics_by_group[f"cat:{cat}"].append(rec_entry)
        if subset in ("seen_app", "unseen_app"):
            metrics_by_group[f"subset:{subset}"].append(rec_entry)

        if official_score_i < 1.0:
            errors.append(
                {
                    "id": qid,
                    "category": cat,
                    "subset": subset,
                    "audio_text": item.get("audio_text"),
                    "official_score": round(official_score_i, 4),
                    "state_em": state_em,
                    "call_f1": round(f1, 4),
                    "gt_calls": gt_calls,
                    "pred_calls": pred_calls,
                }
            )

    def summarize(entries: list[dict]) -> dict:
        n = len(entries)
        if n == 0:
            return {}
        return {
            "count": n,
            "official_score": round(100 * sum(e["official_score"] for e in entries) / n, 2),
            "state_em": round(100 * sum(e["state_em"] for e in entries) / n, 2),
            "call_f1": round(100 * sum(e["call_f1"] for e in entries) / n, 2),
            "call_prec": round(100 * sum(e["call_prec"] for e in entries) / n, 2),
            "call_rec": round(100 * sum(e["call_rec"] for e in entries) / n, 2),
        }

    if error_out:
        error_out.parent.mkdir(parents=True, exist_ok=True)
        with open(error_out, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return {
        "gt_file": str(gt_path),
        "submission_file": str(sub_path),
        "missing_ids_count": len(missing_ids),
        "extra_ids_count": len(extra_ids),
        "total_schema_errors": total_schema_errors,
        "hash_integrity_mismatches": hash_integrity_mismatches,
        "overall": summarize(metrics_by_group["OVERALL"]),
        "subsets": {
            k.split(":", 1)[1]: summarize(v)
            for k, v in sorted(metrics_by_group.items())
            if k.startswith("subset:")
        },
        "by_category": {
            k.split(":", 1)[1]: summarize(v)
            for k, v in sorted(metrics_by_group.items())
            if k.startswith("cat:")
        },
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Local Evaluator for ZAC 2026 LifeMate")
    ap.add_argument("--gt", default=str(WORKSPACE / "outputs" / "splits" / "val_split.jsonl"), help="Ground-truth JSONL")
    ap.add_argument("--sub", required=True, help="Predicted submission JSONL")
    ap.add_argument("--split", default="train", help="zaic_tools split name (default: train)")
    ap.add_argument("--error-out", default=str(WORKSPACE / "outputs" / "eval_errors.jsonl"), help="Save mismatched samples")
    ap.add_argument("--only-present", action="store_true", help="Evaluate only IDs present in --sub (useful with --limit)")
    args = ap.parse_args()

    report = evaluate(
        Path(args.gt),
        Path(args.sub),
        split=args.split,
        error_out=Path(args.error_out) if args.error_out else None,
        only_present=args.only_present,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
