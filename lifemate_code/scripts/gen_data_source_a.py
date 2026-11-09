"""Nguồn A: Tách toàn bộ các lượt hội thoại có `audio_text` + `tool_calls` nằm trong `history` của `train.jsonl`
thành các mẫu huấn luyện chuẩn độc lập (chuẩn BTC 100%, không tốn API, chạy 2 giây trên máy local).

Chạy lệnh:
    python -X utf8 scripts/gen_data_source_a.py --out outputs/sft/source_a_history.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)


def extract_history_samples(train_jsonl: Path, only_with_tools: bool = False) -> list[dict]:
    extracted: list[dict] = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            hist = item.get("history") or []
            # Duyệt các cặp (user_turn tại i, assistant_turn tại i+1) trong history
            for i in range(len(hist) - 1):
                u_turn = hist[i]
                a_turn = hist[i + 1]
                if u_turn.get("role") != "user" or a_turn.get("role") != "assistant":
                    continue
                u_text = (u_turn.get("audio_text") or "").strip()
                if not u_text:
                    continue

                tcalls = a_turn.get("tool_calls") or []
                tresults = a_turn.get("tool_results") or []
                if only_with_tools and not tcalls:
                    continue
                if len(tcalls) != len(tresults):
                    continue

                sub_sample = {
                    "id": f"{item['id']}_hist_{i}",
                    "audio": u_turn.get("audio") or f"audio/{item['id']}_h{i}.wav",
                    "audio_text": u_text,
                    "context": copy.deepcopy(item["context"]),
                    "history": copy.deepcopy(hist[:i]),
                    "label": {
                        "tool_calls": copy.deepcopy(tcalls),
                        "tool_results": copy.deepcopy(tresults),
                        "final_state": {},
                    },
                }
                extracted.append(sub_sample)
    return extracted


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Nguồn A: Khai thác các lượt hội thoại trong history của train.jsonl")
    ap.add_argument("--input", default=str(DATA_ROOT / "train" / "train.jsonl"))
    ap.add_argument("--out", default=str(WORKSPACE / "outputs" / "sft" / "source_a_history.jsonl"))
    ap.add_argument("--only-with-tools", action="store_true", help="Chỉ lấy các lượt có gọi ít nhất 1 tool")
    args = ap.parse_args()

    samples = extract_history_samples(Path(args.input), only_with_tools=args.only_with_tools)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for s in samples:
            out_f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with_tools = sum(1 for s in samples if s["label"]["tool_calls"])
    print(f"[Nguồn A] Đã tách thành công {len(samples)} mẫu từ history (trong đó {with_tools} mẫu có tool_calls) -> {out_path}")


if __name__ == "__main__":
    main()
