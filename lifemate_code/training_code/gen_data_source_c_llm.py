"""Nguồn C: Đa dạng hóa khẩu ngữ vùng miền (Bắc / Trung / Nam) bằng Gemini API (Free Tier) hoặc OpenAI API.

- Không cần cài thêm thư viện ngoài (dùng `urllib.request` chuẩn của Python).
- Gom theo Batch (mặc định 10 câu/lần gọi API) nên sinh 300-500 câu chỉ mất ~30-50 lượt gọi (~50k-80k tokens).
- Giữ nguyên 100% số lượng, giờ giấc, tên riêng và nhãn `tool_calls` / `tool_results` đã kiểm chứng qua Simulator ở Nguồn B!
- Hỗ trợ `--resume` lưu nối tiếp từng batch an toàn.

Cách chạy với Gemini API miễn phí (lấy key tại https://aistudio.google.com/apikey):
    $env:GEMINI_API_KEY="AIza..."
    python -X utf8 scripts/gen_data_source_c_llm.py --provider gemini --limit 300

Hoặc chạy với OpenAI API:
    $env:OPENAI_API_KEY="sk-..."
    python -X utf8 scripts/gen_data_source_c_llm.py --provider openai --limit 300
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent

SYSTEM_INSTRUCTION = """Bạn là chuyên gia tạo dữ liệu giọng nói tiếng Việt tự nhiên cho trợ lý ảo AI.
Nhiệm vụ: Viết lại danh sách các câu lệnh dưới đây sang văn nói đời thường đa dạng vùng miền (miền Bắc, miền Trung, miền Nam), có từ đệm tự nhiên ("ờ", "à", "trợ lý ơi", "coi giùm", "rứa em...", "lẹ nha").
QUY TẮC BẮT BUỘC:
1. PHẢI GIỮ NGUYÊN CHÍNH XÁC 100%: Mọi con số, số tiền, số lượng, giờ giấc, ngày tháng, mã cổ phiếu, tên người (kể cả tên tiếng Anh), tên sản phẩm, địa điểm và nội dung câu cần dịch trong dấu nháy!
2. Nếu câu gốc có ý đổi ý/hủy bỏ ("à mà thôi..."), câu viết lại CŨNG PHẢI giữ ý hủy bỏ đó.
3. Nếu câu gốc thiếu thông tin (không nói số lượng/số giờ/lý do), câu viết lại TUYỆT ĐỐI KHÔNG ĐƯỢC tự bịa thêm thông tin bị thiếu đó.
4. Mỗi câu đầu vào có dạng {"i": <số thứ tự>, "text": "<câu gốc>"}. Trả về ĐÚNG một JSON Array các object {"i": <giữ nguyên số i của câu gốc>, "text": "<câu viết lại của CHÍNH câu đó>"}, không giải thích gì thêm."""


def _indexed_payload(texts: list[str]) -> str:
    return json.dumps([{"i": i, "text": t} for i, t in enumerate(texts)], ensure_ascii=False, indent=1)


def _parse_indexed(out, n: int) -> list[str | None]:
    """Map the answer back BY INDEX (v4 zipped by position: the API re-ordered items and 295/300 rewrites ended up
    attached to the label of another sample). Missing / malformed entries -> None (sample skipped)."""
    if isinstance(out, dict):
        out = out.get("rewrites") or out.get("items") or next((v for v in out.values() if isinstance(v, list)), [])
    res: list[str | None] = [None] * n
    for obj in out if isinstance(out, list) else []:
        if isinstance(obj, dict) and isinstance(obj.get("i"), int) and 0 <= obj["i"] < n and isinstance(obj.get("text"), str):
            res[obj["i"]] = obj["text"].strip()
    return res


GEMINI_MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-3.8-flash",
]


def call_gemini_api(api_key: str, model: str, texts: list[str]) -> tuple[list[str | None], str]:
    """Call Gemini API, auto-falling back across candidate models if 404, 503, or timeout occurs."""
    models_to_try = [model] if model else []
    for cand in GEMINI_MODEL_CANDIDATES:
        if cand not in models_to_try:
            models_to_try.append(cand)
    last_err = None

    for candidate in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent?key={api_key}"
        prompt = f"{SYSTEM_INSTRUCTION}\n\nDanh sách {len(texts)} câu gốc (JSON Array):\n{_indexed_payload(texts)}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.8, "responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw_txt = data["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_indexed(json.loads(raw_txt), len(texts)), candidate
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            last_err = e
            if e.code in (404, 503):
                # 404 Not Found hoặc 503 High demand/Unavailable -> thử ngay model tiếp theo
                print(f"[Gemini {candidate} - HTTP {e.code}] Đang tự động chuyển sang model tiếp theo...")
                continue
            elif e.code == 429:
                print(f"[Cảnh báo HTTP 429 - Quota/Rate Limit] Đợi 20s... Chi tiết: {err_body[:150]}")
                time.sleep(20)
                raise
            else:
                print(f"[Lỗi HTTP {e.code}] {err_body[:200]}")
                raise
        except (TimeoutError, urllib.error.URLError) as e:
            print(f"[Gemini {candidate} - Timeout/Network] Đang tự động chuyển sang model tiếp theo... ({e})")
            last_err = e
            continue

    raise RuntimeError(f"Không thể kết nối tới các model Gemini {models_to_try}. Lỗi cuối: {last_err}")


_NUM_TOKEN_RE = re.compile(r"\d+(?:[.,:]\d+)*")


def _numbers(text: str) -> Counter:
    """Multiset of numbers in a sentence, separators removed ('500.000' == '500000', '7:30' -> '730')."""
    return Counter(re.sub(r"[.,:]", "", m) for m in _NUM_TOKEN_RE.findall(text or ""))


def rewrite_is_safe(orig: str, new: str) -> bool:
    """The label (tool_calls) is kept unchanged, so the rewrite must keep every number, every quoted span and every
    capitalised proper name of the original. Otherwise the training sample teaches a WRONG argument."""
    if not new or not new.strip():
        return False
    if _numbers(orig) != _numbers(new):
        return False
    for q in re.findall(r"[\"“'‘]([^\"”'’]{2,})[\"”'’]", orig):
        if q.lower() not in new.lower():
            return False
    names = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-ZÀ-Ỹ][\wÀ-ỹ]+)", orig)
    low_new = new.lower()
    return all(n.lower() in low_new for n in names)


def call_openai_api(api_key: str, model: str, texts: list[str], base_url: str) -> list[str | None]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    user_msg = (
        f"Viết lại đúng {len(texts)} câu sau và trả về JSON object dạng "
        f"{{\"rewrites\": [{{\"i\": 0, \"text\": \"...\"}}, ...]}}:\n{_indexed_payload(texts)}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    raw_txt = data["choices"][0]["message"]["content"]
    return _parse_indexed(json.loads(raw_txt), len(texts))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Nguồn C: Đa dạng hóa câu lệnh bằng Gemini / OpenAI API theo Batch")
    ap.add_argument("--input", default=str(WORKSPACE / "outputs" / "sft" / "source_b_simulator.jsonl"))
    ap.add_argument("--out", default=str(WORKSPACE / "outputs" / "sft" / "source_c_llm.jsonl"))
    ap.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    ap.add_argument("--model", default="", help="Mặc định: gemini-2.5-flash hoặc gpt-4o-mini")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, default=300, help="Số mẫu muốn viết lại")
    ap.add_argument("--sleep-sec", type=float, default=4.2, help="Nghỉ giữa các batch (phù hợp Free Tier 15 RPM của Gemini)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Chưa thấy file đầu vào {in_path}! Hãy chạy `scripts/gen_data_source_b.py` trước.")

    if args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Thiếu biến môi trường GEMINI_API_KEY! Chạy: $env:GEMINI_API_KEY='AIza...'")
        model = args.model or "gemini-3.5-flash-lite"
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Thiếu biến môi trường OPENAI_API_KEY! Chạy: $env:OPENAI_API_KEY='sk-...'")
        model = args.model or "gpt-4o-mini"

    src_rows = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()][: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if out_path.is_file():
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                done_ids.add(json.loads(line)["id"])

    pending = [r for r in src_rows if f"{r['id']}_llm" not in done_ids]
    print(f"[Nguồn C] Provider={args.provider} ({model}) | Đã xong={len(done_ids)} | Cần chạy={len(pending)}")

    rejected = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            orig_texts = [b["audio_text"] for b in batch]
            try:
                if args.provider == "gemini":
                    rewrites, active_model = call_gemini_api(api_key, model, orig_texts)
                    model = active_model
                else:
                    rewrites = call_openai_api(api_key, model, orig_texts, args.base_url)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                print(f"[Cảnh báo HTTP {e.code}] Chi tiết: {body[:200]}")
                time.sleep(15)
                continue
            except Exception as e:
                print(f"[Bỏ qua batch lỗi]: {e}")
                time.sleep(5)
                continue

            kept = 0
            for item, new_text in zip(batch, rewrites):
                if not rewrite_is_safe(item["audio_text"], new_text):
                    rejected += 1
                    continue  # numbers / names changed => label no longer true => drop
                cloned = copy.deepcopy(item)
                cloned["id"] = f"{item['id']}_llm"
                cloned["audio_text"] = new_text
                out_f.write(json.dumps(cloned, ensure_ascii=False) + "\n")
                kept += 1
            out_f.flush()
            print(
                f"  -> Batch {start // args.batch_size + 1}: giữ {kept}/{len(batch)} | tổng bị loại={rejected} | "
                f"VD: {rewrites[0]}"
            )
            if start + args.batch_size < len(pending):
                time.sleep(args.sleep_sec)

    print(f"[Nguồn C] Hoàn tất! File lưu tại: {out_path}")


if __name__ == "__main__":
    main()
