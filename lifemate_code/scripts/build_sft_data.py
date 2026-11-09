"""Build Multi-Turn Tool-Calling SFT Dataset v2.1 for Qwen3-4B-Instruct-2507.

Combines:
1. All ground-truth trajectories from `train.jsonl` (converted into exact step-by-step causal turns:
   User -> Assistant(call_1) -> Tool(result_1) -> Assistant(call_2) -> Tool(result_2) ... -> Assistant("Đã hoàn tất."))
2. Dual-ASR Transcript Augmentation (simulating `[Bản nghe ASR 1]` + `[Bản nghe ASR 2]` with realistic homophone noise)
3. Simulator-Verified Synthetic Trajectories (36 diverse scenarios x repeat factor) for the 11 unseen apps in
   `public_test_tools_catalog.json` (`ecommerce`, `event_tickets`, `grocery`, `habit_tracker`, `pantry`, `parking`,
   `public_transit`, `recipes`, `stocks`, `train_bus_tickets`, `translator`) + cross-app workflows (`notes`, `email`,
   `shopping_list`, `calendar`, `reminders`, `tasks`, `chat_ott`, `sms`) using purely synthetic states (100% rule-compliant).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "scripts"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

from lifemate.prompting import annotate_tool_result, build_messages, filter_installed_tools, to_openai_tools  # noqa: E402
from zaic_tools.executor import Simulator  # noqa: E402

ASR_NOISE_SUBS = [
    ("Minh", "mình"),
    ("Lan", "Lân"),
    ("Hương", "Hường"),
    ("Dũng", "Dụng"),
    ("sếp Dũng", "Sếp Nguyễn"),
    ("khám răng", "khám găng"),
    ("phòng khách", "phòng cắt"),
    ("phòng tiếp khách", "phòng tiết khách"),
    ("điều hòa", "điều hoà"),
    ("quạt đang chạy", "còn đang chạy"),
    ("muộn hơn một tiếng", "muộn tám một tiếng"),
    ("đến hạn", "tính hẹn"),
    ("trên 500 nghìn", "trừ 500 nghìn"),
    ("giờ bay ba tiếng", "giờ bấy báo tiện"),
    ("tập huấn", "tuần khoá quán"),
    ("Chợ Hôm", "Chợ Hàn"),
    ("logistics xanh", "Đô Tiếc Xanh"),
    ("Cẩm nang sơ cứu", "Cảnh Năng sơ cứu"),
    ("rưỡi", "ba mươi"),
    ("50 nghìn", "năm chục ngàn"),
    ("100 nghìn", "một trăm ngàn"),
    ("200 nghìn", "hai trăm ngàn"),
    ("chuyển khoản", "chuyển khoảng"),
    ("báo thức", "báo thực"),
    ("đặt báo thức", "đặt quán thức"),
    ("nhắc nhở", "nhắc nhờ"),
    ("tiền mạng", "tiền, mà"),
    ("đóng hộ", "đồng hồ"),
]


def make_noisy_dual_asr(clean_text: str, rng: random.Random, real_entry: dict[str, str] | None = None) -> dict[str, str]:
    """Use real Dual-ASR cache if provided (50% chance when available), else simulate realistic phonetic variations."""
    if real_entry and (real_entry.get("primary") or real_entry.get("secondary")) and rng.random() < 0.6:
        p = real_entry.get("primary") or clean_text
        s = real_entry.get("secondary") or clean_text
        return {"primary": p, "secondary": s, "base": clean_text}

    if rng.random() > 0.50:
        return {"primary": clean_text}

    noisy = clean_text
    applied = False
    shuffled = list(ASR_NOISE_SUBS)
    rng.shuffle(shuffled)
    for src, dst in shuffled:
        if src in noisy:
            noisy = noisy.replace(src, dst, 1)
            applied = True
            break
    if not applied:
        noisy = noisy.replace(",", "").replace(";", "")
    if rng.random() < 0.5:
        return {"primary": noisy, "secondary": clean_text}
    return {"primary": clean_text, "secondary": noisy}


def enrich_tool_result(name: str, result: dict) -> dict:
    """Same `_system_note` hints as the runtime agent (shared implementation => no train/inference mismatch)."""
    return annotate_tool_result(name, result)


def sample_to_sft_record(
    sample: dict,
    catalog: dict,
    rng: random.Random,
    asr_cache: dict[str, dict[str, str]] | None = None,
) -> dict:
    installed_apps = (sample.get("context") or {}).get("installed_apps") or {}
    filtered_tools = filter_installed_tools(catalog, installed_apps)
    openai_tools = to_openai_tools(filtered_tools)

    transcript_map = {}
    for h in sample.get("history") or []:
        if h.get("role") == "user" and h.get("audio"):
            h_rel = h["audio"]
            real_h = (asr_cache or {}).get(h_rel)
            transcript_map[h_rel] = make_noisy_dual_asr(h.get("audio_text") or "", rng, real_entry=real_h)
    final_rel = sample["audio"]
    transcript_map[final_rel] = make_noisy_dual_asr(
        sample.get("audio_text") or "", rng, real_entry=(asr_cache or {}).get(final_rel)
    )

    messages = build_messages(sample, transcript_map)

    tcalls = sample["label"]["tool_calls"]
    tresults = sample["label"]["tool_results"]

    for idx, (call, res) in enumerate(zip(tcalls, tresults), 1):
        cid = f"call_{idx}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "name": call["name"],
                "content": json.dumps(enrich_tool_result(call["name"], res), ensure_ascii=False),
            }
        )

    if not tcalls:
        messages.append({"role": "assistant", "content": "Yêu cầu không cần thực thi công cụ hoặc nằm ngoài phạm vi ứng dụng."})
    else:
        messages.append({"role": "assistant", "content": "Đã hoàn tất thực thi."})

    return {"id": sample["id"], "messages": messages, "tools": openai_tools}


def build_recovery_sft_records(
    train_items: list[dict],
    catalog: dict,
    rng: random.Random,
    max_records: int = 240,
) -> list[dict]:
    """Trick #2.2 (Recovery Training): Inject failed search / guardrail-blocked turns with `_supervise=False` (loss=0),
    followed by the supervised recovery action (`_supervise=True`, loss=1.0)."""
    recovery_records: list[dict] = []
    candidates = [it for it in train_items if len((it.get("label") or {}).get("tool_calls") or []) >= 2]
    rng.shuffle(candidates)

    for item in candidates:
        if len(recovery_records) >= max_records:
            break
        tcalls = item["label"]["tool_calls"]
        tresults = item["label"]["tool_results"]
        first_call = tcalls[0]
        first_name = first_call["name"]
        first_args = first_call.get("arguments") or {}

        base_rec = sample_to_sft_record(item, catalog, rng)
        user_indices = [i for i, m in enumerate(base_rec["messages"]) if m.get("role") == "user"]
        if not user_indices:
            continue
        last_u_idx = user_indices[-1]
        prefix_msgs = copy.deepcopy(base_rec["messages"][: last_u_idx + 1])
        gold_tail = copy.deepcopy(base_rec["messages"][last_u_idx + 1 :])

        # Case A: First call is contacts.search(query=...) -> inject a failed ASR-misheard search (loss=0), then recover with gold
        if first_name == "contacts.search" and isinstance(first_args.get("query"), str) and first_args["query"].strip():
            q_orig = first_args["query"].strip()
            q_bad = q_orig + " Nguyễn" if " " not in q_orig else q_orig.split()[0] + " Khác"
            bad_tc = {
                "role": "assistant",
                "_supervise": False,
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_err_0",
                        "type": "function",
                        "function": {"name": "contacts.search", "arguments": json.dumps({"query": q_bad}, ensure_ascii=False)},
                    }
                ],
            }
            bad_tr = {
                "role": "tool",
                "tool_call_id": "call_err_0",
                "name": "contacts.search",
                "content": json.dumps(
                    enrich_tool_result("contacts.search", {"ok": True, "contacts": []}), ensure_ascii=False
                ),
            }
            recovery_records.append(
                {
                    "id": f"{item['id']}_rec_search",
                    "messages": prefix_msgs + [bad_tc, bad_tr] + gold_tail,
                    "tools": base_rec["tools"],
                }
            )
            continue

        # Case B: First call is a [READ] and second call is a [WRITE] -> inject a blocked direct-write attempt (loss=0), then recover with [READ] -> [WRITE]
        second_call = tcalls[1]
        second_name = second_call["name"]
        second_args = second_call.get("arguments") or {}
        id_keys = [k for k in second_args if k.endswith("_id") and isinstance(second_args[k], str)]
        if first_name != second_name and id_keys and second_name != "sys.ask_missing_info":
            k_id = id_keys[0]
            ref_app = first_name.split(".", 1)[0]
            guessed_args = {**second_args, k_id: f"{k_id.split('_')[0]}_guessed"}
            bad_tc = {
                "role": "assistant",
                "_supervise": False,
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_err_0",
                        "type": "function",
                        "function": {"name": second_name, "arguments": json.dumps(guessed_args, ensure_ascii=False)},
                    }
                ],
            }
            bad_tr = {
                "role": "tool",
                "tool_call_id": "call_err_0",
                "name": second_name,
                "content": json.dumps(
                    {
                        "ok": False,
                        "xref_blocked": True,
                        "error": (
                            f"CHƯA THỰC THI: '{k_id}'='{guessed_args[k_id]}' chưa được tra cứu từ app '{ref_app}'. "
                            f"Hãy gọi tool [READ] của '{ref_app}' trước để lấy ID thật!"
                        ),
                    },
                    ensure_ascii=False,
                ),
            }
            recovery_records.append(
                {
                    "id": f"{item['id']}_rec_xref",
                    "messages": prefix_msgs + [bad_tc, bad_tr] + gold_tail,
                    "tools": base_rec["tools"],
                }
            )

    return recovery_records


def generate_synthetic_unseen_app_samples(pub_catalog: dict) -> list[dict]:
    """Construct 36 Simulator-verified multi-turn training scenarios covering all 11 unseen apps & cross-app workflows."""
    app_tools_map = {a["id"]: [t["name"] for t in a.get("tools", [])] for a in pub_catalog.get("apps", [])}

    base_state = {
        "ecommerce": {
            "products": [
                {"id": "ec_01", "name": "Tai nghe Bluetooth Sony", "price": 1250000, "seller": "Sony Official", "category": "điện tử", "in_stock": True},
                {"id": "ec_02", "name": "Bình giữ nhiệt Lock&Lock 500ml", "price": 320000, "seller": "Lock&Lock", "category": "gia dụng", "in_stock": True},
                {"id": "ec_03", "name": "Chuột không dây Logitech", "price": 450000, "seller": "Logitech", "category": "điện tử", "in_stock": True},
                {"id": "ec_04", "name": "Chuột không dây Razer", "price": 890000, "seller": "Razer", "category": "điện tử", "in_stock": True},
            ],
            "cart": [],
            "orders": [
                {"id": "ord_01", "items": [{"product_id": "ec_01", "quantity": 1}], "total": 1250000, "status": "placed"},
                {"id": "ord_02", "items": [{"product_id": "ec_02", "quantity": 1}], "total": 320000, "status": "delivered"},
            ],
            "returns": [],
        },
        "grocery": {
            "products": [
                {"id": "gr_01", "name": "Táo Envy Mỹ", "price": 120000, "unit": "kg", "category": "trái cây", "in_stock": True},
                {"id": "gr_02", "name": "Sữa tươi Vinamilk không đường", "price": 36000, "unit": "lốc", "category": "sữa", "in_stock": True},
                {"id": "gr_03", "name": "Thịt bò thăn", "price": 280000, "unit": "kg", "category": "thịt", "in_stock": True},
                {"id": "gr_04", "name": "Bánh phở tươi", "price": 25000, "unit": "kg", "category": "đồ khô", "in_stock": True},
            ],
            "cart": [],
            "orders": [],
        },
        "pantry": {
            "items": [
                {"id": "pt_01", "name": "Gạo ST25", "stock": 5, "unit": "kg"},
                {"id": "pt_02", "name": "Trứng gà", "stock": 10, "unit": "quả"},
                {"id": "pt_03", "name": "Bánh phở tươi", "stock": 1, "unit": "kg"},
                {"id": "pt_04", "name": "Thịt bò thăn", "stock": 0, "unit": "kg"},
            ]
        },
        "recipes": {
            "recipes": [
                {
                    "id": "rc_01",
                    "name": "Phở bò truyền thống",
                    "servings": 2,
                    "tags": ["món nước", "sáng"],
                    "ingredients": [
                        {"name": "Thịt bò thăn", "quantity": 1, "unit": "kg"},
                        {"name": "Bánh phở tươi", "quantity": 1, "unit": "kg"},
                    ],
                    "steps": ["Ninh xương", "Trần bánh phở"],
                },
                {
                    "id": "rc_02",
                    "name": "Cơm chiên trứng",
                    "servings": 2,
                    "tags": ["nhanh"],
                    "ingredients": [{"name": "Trứng gà", "quantity": 2, "unit": "quả"}],
                    "steps": ["Chiên cơm với trứng"],
                },
            ],
            "saved": [],
        },
        "event_tickets": {
            "events": [
                {"id": "ev_01", "name": "Đêm nhạc Hà Anh Tuấn", "category": "concert", "date": "2026-11-25", "time": "20:00", "venue": "Nhà hát Hòa Bình", "price": 1500000, "available": 10},
                {"id": "ev_02", "name": "Hội thảo AI 2026", "category": "tech", "date": "2026-11-22", "time": "09:00", "venue": "SECC Quận 7", "price": 500000, "available": 50},
            ],
            "bookings": [],
        },
        "habit_tracker": {
            "habits": [
                {"id": "hb_01", "name": "Đọc sách 30 phút", "streak": 4, "last_logged": "2026-11-15"},
                {"id": "hb_02", "name": "Chạy bộ buổi sáng", "streak": 2, "last_logged": "2026-11-15"},
            ],
            "logs": [],
        },
        "parking": {
            "lots": [
                {"id": "lot_01", "name": "Bãi xe Vincom Đồng Khởi", "address": "72 Lê Thánh Tôn, Quận 1", "area": "Quận 1", "price_per_hour": 35000, "available_spots": 12},
                {"id": "lot_02", "name": "Bãi xe Hồ Con Rùa", "address": "Công trường Quốc Tế, Quận 3", "area": "Quận 3", "price_per_hour": 25000, "available_spots": 8},
            ],
            "sessions": [
                {"id": "park_01", "lot_id": "lot_01", "from_time": "2026-11-16T08:00:00+07:00", "hours": 2, "end_time": "2026-11-16T10:00:00+07:00", "total": 70000, "status": "reserved"}
            ],
        },
        "public_transit": {
            "card_balance": 45000,
            "routes": [
                {"id": "bus_01", "name": "Tuyến 01 Bến Thành - Chợ Lớn", "type": "bus", "stops": ["Bến Thành", "Trần Hưng Đạo", "Chợ Lớn"], "fare": 7000, "duration_min": 30, "headway_min": 10},
                {"id": "metro_01", "name": "Metro Số 1 Bến Thành - Suối Tiên", "type": "metro", "stops": ["Bến Thành", "Thảo Điền", "Suối Tiên"], "fare": 15000, "duration_min": 28, "headway_min": 8},
            ],
            "departures": [
                {"stop": "Bến Thành", "route": "metro_01", "time": "10:15"},
                {"stop": "Bến Thành", "route": "bus_01", "time": "10:12"},
            ],
            "topups": [],
        },
        "stocks": {
            "holdings": [
                {"symbol": "FPT", "name": "FPT Corp", "quantity": 500, "avg_price": 125000, "price": 134000},
                {"symbol": "VNM", "name": "Vinamilk", "quantity": 300, "avg_price": 66000, "price": 64000},
            ],
            "quotes": [
                {"symbol": "FPT", "name": "FPT Corp", "price": 134000, "change_pct": 1.8},
                {"symbol": "VNM", "name": "Vinamilk", "price": 64000, "change_pct": -0.5},
                {"symbol": "VCB", "name": "Vietcombank", "price": 92000, "change_pct": 0.9},
            ],
            "orders": [],
            "alerts": [],
        },
        "train_bus_tickets": {
            "trips": [
                {"id": "tr_01", "type": "train", "operator": "Đường sắt VN SE1", "origin": "TP.HCM", "destination": "Nha Trang", "date": "2026-11-20", "depart": "07:00", "arrive": "15:00", "price": 480000, "seats_left": 20},
                {"id": "tr_02", "type": "bus", "operator": "Phương Trang", "origin": "TP.HCM", "destination": "Đà Lạt", "date": "2026-11-21", "depart": "22:00", "arrive": "05:00", "price": 300000, "seats_left": 15},
                {"id": "tr_03", "type": "bus", "operator": "Thành Bưởi", "origin": "TP.HCM", "destination": "Đà Lạt", "date": "2026-11-21", "depart": "23:00", "arrive": "06:00", "price": 310000, "seats_left": 12},
            ],
            "bookings": [
                {"id": "tk_01", "trip_id": "tr_01", "seats": 2, "total": 960000, "status": "confirmed"}
            ],
        },
        "translator": {
            "dictionary": [
                {"text": "Họp lúc 9 giờ sáng mai tại phòng 302", "to_lang": "en", "translation": "Meeting at 9 AM tomorrow in room 302"},
                {"text": "Cảm ơn đối tác đã hỗ trợ dự án", "to_lang": "en", "translation": "Thank you partner for supporting the project"},
                {"text": "Tôi bị dị ứng hải sản", "to_lang": "ja", "translation": "私はシーフードアレルギーがあります"},
            ]
        },
        "contacts": {
            "contacts": [
                {"id": "c_01", "name": "Sếp Hùng", "relation": "sếp", "phone": "0901122334", "email": "hung@company.vn"},
                {"id": "c_02", "name": "Vợ Mai", "relation": "vợ", "phone": "0909988776", "email": "mai@gmail.com"},
                {"id": "c_03", "name": "Michael Chen", "relation": "đối tác", "phone": "0908881122", "email": "michael.chen@global.com"},
            ]
        },
        "chat_ott": {
            "chats": [
                {"id": "chat_01", "name": "Vợ Mai", "unread": 0},
                {"id": "chat_02", "name": "Michael Chen", "unread": 1},
            ],
            "messages": [],
        },
        "email": {"emails": [], "drafts": [], "sent": []},
        "notes": {"notes": [{"id": "note_01", "title": "Sổ tay công việc", "content": "Theo dõi đơn hàng"}]},
        "tasks": {"tasks": [{"id": "task_01", "title": "Đọc sách mỗi ngày", "status": "todo"}]},
        "shopping_list": {"items": []},
        "sms": {"messages": []},
        "alarm": {"alarms": []},
        "reminders": {"reminders": []},
        "calendar": {"events": []},
    }

    scenarios = [
        # 1. ecommerce: search -> add_to_cart -> checkout (with address & cod)
        (
            "Tìm mua giúp mình 1 bình giữ nhiệt Lock&Lock trên sàn thương mại điện tử, giao về nhà và thanh toán COD nhé.",
            ["ecommerce", "contacts", "sms", "reminders"],
            [
                {"name": "ecommerce.search", "arguments": {"query": "bình giữ nhiệt Lock&Lock"}},
                {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "ec_02", "quantity": 1}},
                {"name": "ecommerce.checkout", "arguments": {"address": "77 Hoàng Văn Thụ, Phú Nhuận, TP.HCM", "payment": "cod"}},
            ],
        ),
        # 2. ecommerce: order_status -> cancel_order
        (
            "Kiểm tra đơn hàng đang đặt trên app mua sắm rồi hủy giúp tôi đơn tai nghe Sony nhé.",
            ["ecommerce", "banking", "sms"],
            [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_01"}},
            ],
        ),
        # 3. ecommerce: order_status -> request_return
        (
            "Đơn bình giữ nhiệt giao rồi nhưng bị móp hỏng, làm yêu cầu trả hàng giúp mình nha.",
            ["ecommerce", "sms", "reminders"],
            [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "ecommerce.request_return", "arguments": {"order_id": "ord_02", "reason": "damaged"}},
            ],
        ),
        # 4. ecommerce: order_status -> missing return reason -> sys.ask_missing_info
        (
            "Làm thủ tục trả lại đơn bình giữ nhiệt tuần trước giúp chị nhé.",
            ["ecommerce", "notes"],
            [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ecommerce.request_return", "params": ["reason"]}},
            ],
        ),
        # 5. ecommerce: ambiguous search -> sys.ask_missing_info
        (
            "Thêm giúp mình 1 con chuột không dây vào giỏ hàng nhé.",
            ["ecommerce", "sms"],
            [
                {"name": "ecommerce.search", "arguments": {"query": "chuột không dây"}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ecommerce.add_to_cart", "params": ["product_id"]}},
            ],
        ),
        # 6. grocery: search -> add_to_cart -> checkout({}) (no address mentioned -> empty checkout)
        (
            "Cho mình đặt 2 ký táo Envy Mỹ với 1 lốc sữa tươi Vinamilk rồi chốt đơn luôn nha.",
            ["grocery", "pantry", "recipes"],
            [
                {"name": "grocery.search", "arguments": {"query": "táo Envy"}},
                {"name": "grocery.add_to_cart", "arguments": {"product_id": "gr_01", "quantity": 2}},
                {"name": "grocery.search", "arguments": {"query": "sữa tươi Vinamilk"}},
                {"name": "grocery.add_to_cart", "arguments": {"product_id": "gr_02", "quantity": 1}},
                {"name": "grocery.checkout", "arguments": {}},
            ],
        ),
        # 7. grocery: search -> add_to_cart -> checkout (with home address)
        (
            "Đặt đi chợ online 1 ký thịt bò thăn giao về địa chỉ nhà giúp mình nhé.",
            ["grocery", "pantry", "sms"],
            [
                {"name": "grocery.search", "arguments": {"query": "thịt bò thăn"}},
                {"name": "grocery.add_to_cart", "arguments": {"product_id": "gr_03", "quantity": 1}},
                {"name": "grocery.checkout", "arguments": {"address": "77 Hoàng Văn Thụ, Phú Nhuận, TP.HCM"}},
            ],
        ),
        # 8. recipes + pantry + grocery multi-app workflow
        (
            "Xem công thức nấu phở bò truyền thống cho 2 người, kiểm tra tủ bếp xem nguyên liệu nào hết thì đặt mua trên app đi chợ luôn nhé.",
            ["recipes", "pantry", "grocery", "reminders"],
            [
                {"name": "recipes.search", "arguments": {"query": "phở bò", "servings": 2}},
                {"name": "pantry.list_items", "arguments": {}},
                {"name": "grocery.search", "arguments": {"query": "Thịt bò thăn"}},
                {"name": "grocery.add_to_cart", "arguments": {"product_id": "gr_03", "quantity": 1}},
                {"name": "grocery.checkout", "arguments": {}},
            ],
        ),
        # 9. recipes + pantry + shopping_list multi-app workflow
        (
            "Tìm công thức phở bò truyền thống, đối chiếu tủ bếp xem món nào hết thì thêm vào danh sách mua sắm giúp chị.",
            ["recipes", "pantry", "shopping_list", "notes"],
            [
                {"name": "recipes.search", "arguments": {"query": "phở bò"}},
                {"name": "pantry.list_items", "arguments": {}},
                {"name": "shopping_list.add", "arguments": {"name": "Thịt bò thăn", "quantity": 1}},
            ],
        ),
        # 10. recipes: search -> get -> save
        (
            "Tìm công thức món cơm chiên trứng, xem chi tiết các bước làm rồi lưu lại vào danh sách yêu thích nhé.",
            ["recipes", "pantry", "notes"],
            [
                {"name": "recipes.search", "arguments": {"query": "cơm chiên trứng"}},
                {"name": "recipes.get", "arguments": {"recipe_id": "rc_02"}},
                {"name": "recipes.save", "arguments": {"recipe_id": "rc_02"}},
            ],
        ),
        # 11. pantry: list_items -> consume
        (
            "Mình vừa lấy 4 quả trứng gà trong tủ bếp ra nấu ăn, cập nhật trừ tồn kho tủ bếp giúp mình.",
            ["pantry", "recipes", "grocery"],
            [
                {"name": "pantry.list_items", "arguments": {}},
                {"name": "pantry.consume", "arguments": {"item_id": "pt_02", "quantity": 4}},
            ],
        ),
        # 12. event_tickets: search -> book + sms
        (
            "Đặt giúp anh 2 vé đêm nhạc Hà Anh Tuấn rồi nhắn tin SMS báo cho vợ Mai là đã mua vé nhé.",
            ["event_tickets", "contacts", "sms", "calendar"],
            [
                {"name": "event_tickets.search", "arguments": {"query": "Hà Anh Tuấn"}},
                {"name": "event_tickets.book", "arguments": {"event_id": "ev_01", "quantity": 2}},
                {"name": "contacts.search", "arguments": {"query": "Vợ Mai"}},
                {"name": "sms.send", "arguments": {"contact_id": "c_02", "content": "Anh đã đặt 2 vé đêm nhạc Hà Anh Tuấn rồi nhé"}},
            ],
        ),
        # 13. event_tickets: search -> book + calendar.create_event
        (
            "Đặt 1 vé Hội thảo AI 2026 rồi thêm lịch sự kiện đó vào lịch cá nhân từ 9 giờ đến 12 giờ ngày hôm đó nhé.",
            ["event_tickets", "calendar", "reminders"],
            [
                {"name": "event_tickets.search", "arguments": {"query": "Hội thảo AI 2026"}},
                {"name": "event_tickets.book", "arguments": {"event_id": "ev_02", "quantity": 1}},
                {
                    "name": "calendar.create_event",
                    "arguments": {
                        "title": "Hội thảo AI 2026",
                        "start": "2026-11-22T09:00:00+07:00",
                        "end": "2026-11-22T12:00:00+07:00",
                        "location": "SECC Quận 7",
                    },
                },
            ],
        ),
        # 14. event_tickets: missing quantity -> ask_missing_info
        (
            "Tìm và đặt vé Hội thảo AI 2026 giúp mình nhé.",
            ["event_tickets", "calendar", "reminders"],
            [
                {"name": "event_tickets.search", "arguments": {"query": "Hội thảo AI 2026"}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "event_tickets.book", "params": ["quantity"]}},
            ],
        ),
        # 15. habit_tracker: list_habits -> log + tasks.complete
        (
            "Đánh dấu hoàn thành thói quen đọc sách 30 phút cho ngày hôm nay và gạch đầu việc đọc sách trong danh sách công việc luôn nhé.",
            ["habit_tracker", "tasks", "reminders"],
            [
                {"name": "habit_tracker.list_habits", "arguments": {}},
                {"name": "habit_tracker.log", "arguments": {"habit_id": "hb_01", "date": "2026-11-16"}},
                {"name": "tasks.list", "arguments": {}},
                {"name": "tasks.complete", "arguments": {"task_id": "task_01"}},
            ],
        ),
        # 16. parking: find -> reserve + reminders.create
        (
            "Tìm bãi đỗ xe gần Vincom Đồng Khởi, đặt chỗ từ 2 giờ chiều nay trong 3 tiếng và đặt nhắc nhở lấy xe lúc 4 giờ 45 chiều nay nhé.",
            ["parking", "maps", "reminders"],
            [
                {"name": "parking.find", "arguments": {"near": "Vincom Đồng Khởi"}},
                {"name": "parking.reserve", "arguments": {"lot_id": "lot_01", "from_time": "2026-11-16T14:00:00+07:00", "hours": 3}},
                {"name": "reminders.create", "arguments": {"content": "Lấy xe tại Vincom Đồng Khởi", "time": "2026-11-16T16:45:00+07:00"}},
            ],
        ),
        # 17. parking: list_sessions -> end_session
        (
            "Kết thúc phiên đỗ xe đang chạy ở bãi Vincom giúp tôi.",
            ["parking", "ewallet"],
            [
                {"name": "parking.list_sessions", "arguments": {}},
                {"name": "parking.end_session", "arguments": {"session_id": "park_01"}},
            ],
        ),
        # 18. parking: missing hours -> ask_missing_info
        (
            "Đặt chỗ đỗ xe ở bãi Hồ Con Rùa lúc 5 giờ chiều nay giúp mình.",
            ["parking", "maps"],
            [
                {"name": "parking.find", "arguments": {"near": "Hồ Con Rùa"}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "parking.reserve", "params": ["hours"]}},
            ],
        ),
        # 19. public_transit: find_routes + topup_card
        (
            "Tìm tuyến xe từ Bến Thành đi Suối Tiên và nạp thêm 100 nghìn vào thẻ giao thông công cộng nhé.",
            ["public_transit", "maps", "ewallet"],
            [
                {"name": "public_transit.find_routes", "arguments": {"origin": "Bến Thành", "destination": "Suối Tiên"}},
                {"name": "public_transit.topup_card", "arguments": {"amount": 100000}},
            ],
        ),
        # 20. public_transit: next_departures
        (
            "Xem giúp mình giờ tàu Metro số 1 sắp chạy tại trạm Bến Thành.",
            ["public_transit", "maps"],
            [
                {"name": "public_transit.next_departures", "arguments": {"stop": "Bến Thành", "route": "metro_01"}},
            ],
        ),
        # 21. stocks: get_quote -> place_order (buy)
        (
            "Kiểm tra giá cổ phiếu FPT hiện tại, nếu dưới 140 nghìn thì đặt lệnh mua 100 cổ phiếu theo giá thị trường.",
            ["stocks", "banking", "reminders"],
            [
                {"name": "stocks.get_quote", "arguments": {"symbol": "FPT"}},
                {"name": "stocks.place_order", "arguments": {"symbol": "FPT", "side": "buy", "quantity": 100}},
            ],
        ),
        # 22. stocks: get_quote -> conditional false (noop)
        (
            "Xem giá cổ phiếu VCB, nếu dưới 85 nghìn thì mua 200 cổ phiếu, còn cao hơn thì thôi.",
            ["stocks", "banking"],
            [
                {"name": "stocks.get_quote", "arguments": {"symbol": "VCB"}},
            ],
        ),
        # 23. stocks: get_portfolio -> sell + set_price_alert
        (
            "Xem danh mục cổ phiếu đang giữ, bán 100 cổ phiếu VNM và đặt cảnh báo khi FPT vượt 145 nghìn.",
            ["stocks", "banking", "reminders"],
            [
                {"name": "stocks.get_portfolio", "arguments": {}},
                {"name": "stocks.place_order", "arguments": {"symbol": "VNM", "side": "sell", "quantity": 100}},
                {"name": "stocks.set_price_alert", "arguments": {"symbol": "FPT", "price": 145000, "direction": "above"}},
            ],
        ),
        # 24. train_bus_tickets: search -> book + alarm
        (
            "Tìm vé tàu hỏa từ TP.HCM đi Nha Trang ngày 20/11 rồi đặt 2 ghế, đồng thời đặt báo thức 5 giờ 30 sáng hôm đó.",
            ["train_bus_tickets", "alarm", "calendar"],
            [
                {"name": "train_bus_tickets.search", "arguments": {"origin": "TP.HCM", "destination": "Nha Trang", "date": "2026-11-20", "type": "train"}},
                {"name": "train_bus_tickets.book", "arguments": {"trip_id": "tr_01", "seats": 2}},
                {"name": "alarm.create", "arguments": {"time": "2026-11-20T05:30:00+07:00"}},
            ],
        ),
        # 25. train_bus_tickets: ambiguous bus trips -> ask_missing_info
        (
            "Đặt giúp mình 1 vé xe khách từ TP.HCM đi Đà Lạt ngày 21/11 nhé.",
            ["train_bus_tickets", "calendar"],
            [
                {"name": "train_bus_tickets.search", "arguments": {"origin": "TP.HCM", "destination": "Đà Lạt", "date": "2026-11-21", "type": "bus"}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "train_bus_tickets.book", "params": ["trip_id"]}},
            ],
        ),
        # 26. train_bus_tickets: my_bookings -> cancel
        (
            "Kiểm tra vé tàu xe đã đặt và hủy giúp tôi vé đi Nha Trang nhé.",
            ["train_bus_tickets", "sms"],
            [
                {"name": "train_bus_tickets.my_bookings", "arguments": {}},
                {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tk_01"}},
            ],
        ),
        # 27. translator: translate -> sms.send
        (
            "Dịch câu 'Họp lúc 9 giờ sáng mai tại phòng 302' sang tiếng Anh rồi gửi SMS bản dịch đó cho Sếp Hùng.",
            ["translator", "contacts", "sms"],
            [
                {"name": "translator.translate", "arguments": {"text": "Họp lúc 9 giờ sáng mai tại phòng 302", "to_lang": "en"}},
                {"name": "contacts.search", "arguments": {"query": "Sếp Hùng"}},
                {"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "Meeting at 9 AM tomorrow in room 302"}},
            ],
        ),
        # 28. translator: translate -> chat_ott.send_message (foreign contact Michael Chen)
        (
            "Dịch câu 'Cảm ơn đối tác đã hỗ trợ dự án' sang tiếng Anh rồi nhắn qua chat cho Michael Chen nhé.",
            ["translator", "chat_ott", "contacts"],
            [
                {"name": "translator.translate", "arguments": {"text": "Cảm ơn đối tác đã hỗ trợ dự án", "to_lang": "en"}},
                {"name": "chat_ott.list_chats", "arguments": {}},
                {"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_02", "content": "Thank you partner for supporting the project"}},
            ],
        ),
        # 29. translator: translate -> notes.create
        (
            "Dịch câu 'Tôi bị dị ứng hải sản' sang tiếng Nhật rồi tạo một ghi chú mới tiêu đề 'Câu giao tiếp tiếng Nhật' lưu lại câu đó nhé.",
            ["translator", "notes"],
            [
                {"name": "translator.translate", "arguments": {"text": "Tôi bị dị ứng hải sản", "to_lang": "ja"}},
                {"name": "notes.create", "arguments": {"title": "Câu giao tiếp tiếng Nhật", "content": "私はシーフードアレルギーがあります"}},
            ],
        ),
        # 30. Self-correction / cancellation on unseen app ("À mà thôi...") -> []
        (
            "Đặt giúp anh 2 vé đêm nhạc Hà Anh Tuấn... à mà thôi khoan đặt, để anh hỏi lại vợ xem hôm đó rảnh không đã.",
            ["event_tickets", "calendar", "contacts"],
            [],
        ),
    ]

    synth_rng = random.Random(12345)
    all_pub_apps = [a for a in app_tools_map if a != "sys"]
    synth_samples = []
    for idx, (utterance, app_list, calls) in enumerate(scenarios, 1):
        padded_apps = list(app_list)
        cand_extra = [a for a in all_pub_apps if a not in padded_apps]
        padded_apps.extend(synth_rng.sample(cand_extra, min(5, len(cand_extra))))
        installed = {
            a: {"tools": app_tools_map[a], "state": copy.deepcopy(base_state.get(a, {}))}
            for a in padded_apps
            if a in app_tools_map
        }
        sample_sim = {
            "id": f"synth_unseen_{idx:04d}",
            "audio": f"audio/synth_{idx:04d}.wav",
            "audio_text": utterance,
            "context": {
                "current_time": "2026-11-16T10:00:00+07:00",
                "timezone": "Asia/Ho_Chi_Minh",
                "location": {"lat": 10.7769, "lng": 106.7009},
                "user_profile": {
                    "name": "Anh Thành",
                    "home_address": "77 Hoàng Văn Thụ, Phú Nhuận, TP.HCM",
                    "work_address": "Tòa nhà Blue Pearl, Quận 7, TP.HCM",
                    "city": "TP.HCM",
                },
                "installed_apps": {a: {"tools": app_tools_map[a]} for a in padded_apps if a in app_tools_map},
            },
            "history": [],
        }

        sim = Simulator(
            {
                "id": sample_sim["id"],
                "context": {
                    "current_time": sample_sim["context"]["current_time"],
                    "user_profile": sample_sim["context"]["user_profile"],
                    "installed_apps": installed,
                },
            },
            pub_catalog,
        )
        results = []
        for c in calls:
            r = sim.call(c["name"], c["arguments"])
            assert r.get("ok"), f"Synthetic scenario {idx} failed schema: {c} -> {r}"
            assert r.get("done") is not False, f"Synthetic scenario {idx} noop: {c} -> {r}"
            results.append(r)

        sample_sim["label"] = {"tool_calls": calls, "tool_results": results, "final_state": {}}
        synth_samples.append(sample_sim)

    return synth_samples


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Build Multi-Turn SFT Data for Qwen3-4B-Instruct")
    ap.add_argument("--input-jsonl", default=str(DATA_ROOT / "train" / "train.jsonl"), help="Train JSONL source")
    ap.add_argument("--out", default=str(WORKSPACE / "outputs" / "sft" / "sft_train_full.jsonl"), help="Output SFT JSONL")
    ap.add_argument("--synth-repeat", type=int, default=8, help="Repeat factor for unseen-app synthetic trajectories")
    ap.add_argument("--include-all-sources", action="store_true", help="Gộp thêm Nguồn A (history), Nguồn B (Simulator) và Nguồn C (LLM) nếu có")
    ap.add_argument("--exclude-val-jsonl", default=None, help="Optional holdout validation JSONL whose IDs (and history turns) are excluded from SFT")
    ap.add_argument("--asr-cache", default=None, help="Optional JSON cache of real ASR transcripts for training augmentation")
    ap.add_argument("--recovery-count", type=int, default=240, help="Number of Recovery Training samples (Trick #2.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_cat = json.load(open(DATA_ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))
    pub_cat = json.load(open(DATA_ROOT / "public_test" / "public_test_tools_catalog.json", encoding="utf-8"))

    asr_cache = {}
    if args.asr_cache and Path(args.asr_cache).is_file():
        asr_cache = json.load(open(args.asr_cache, encoding="utf-8"))
        print(f"[ASR Cache Augmentation] Loaded {len(asr_cache)} real ASR transcripts from {args.asr_cache}")

    excluded_ids: set[str] = set()
    if args.exclude_val_jsonl and Path(args.exclude_val_jsonl).is_file():
        excluded_ids = {
            json.loads(line)["id"] for line in open(args.exclude_val_jsonl, encoding="utf-8") if line.strip()
        }
        print(f"[Holdout Filter] Excluding {len(excluded_ids)} validation IDs (and their history turns) from training!")

    train_items = [
        json.loads(line)
        for line in open(args.input_jsonl, encoding="utf-8")
        if line.strip() and json.loads(line)["id"] not in excluded_ids
    ]
    records = [sample_to_sft_record(item, train_cat, rng, asr_cache=asr_cache) for item in train_items]

    if args.recovery_count > 0:
        rec_samples = build_recovery_sft_records(train_items, train_cat, rng, max_records=args.recovery_count)
        records.extend(rec_samples)
        print(f"[Recovery Training - Trick #2.2] Added {len(rec_samples)} recovery trajectories (loss=0 on error, loss=1 on recovery)")

    synth_items = generate_synthetic_unseen_app_samples(pub_cat)
    for rep in range(args.synth_repeat):
        for s_item in synth_items:
            rec = sample_to_sft_record(s_item, pub_cat, rng)
            rec["id"] = f"{s_item['id']}_r{rep+1}"
            records.append(rec)

    extra_counts = {}
    if args.include_all_sources:
        sft_dir = WORKSPACE / "outputs" / "sft"
        src_a = sft_dir / "source_a_history.jsonl"
        if src_a.is_file():
            a_items = [
                json.loads(l)
                for l in open(src_a, encoding="utf-8")
                if l.strip() and json.loads(l)["id"].split("_hist_")[0] not in excluded_ids
            ]
            for it in a_items:
                records.append(sample_to_sft_record(it, train_cat, rng))
            extra_counts["source_a_history"] = len(a_items)

        pub_tools_map = {a["id"]: [t["name"] for t in a.get("tools", [])] for a in pub_cat.get("apps", []) if a["id"] != "sys"}
        src_b_text = {}
        if (sft_dir / "source_b_simulator.jsonl").is_file():
            src_b_text = {
                json.loads(l)["id"]: json.loads(l)["audio_text"]
                for l in open(sft_dir / "source_b_simulator.jsonl", encoding="utf-8")
                if l.strip()
            }
        for fname in ("source_b_simulator.jsonl", "source_c_llm.jsonl"):
            fpath = sft_dir / fname
            if fpath.is_file():
                b_items = [json.loads(l) for l in open(fpath, encoding="utf-8") if l.strip()]
                if fname == "source_c_llm.jsonl":
                    # Guard against label/text mismatch: the rewrite must keep the numbers & names of its original.
                    from gen_data_source_c_llm import rewrite_is_safe

                    before = len(b_items)
                    b_items = [
                        it
                        for it in b_items
                        if it["id"].endswith("_llm")
                        and it["id"][:-4] in src_b_text
                        and rewrite_is_safe(src_b_text[it["id"][:-4]], it.get("audio_text") or "")
                    ]
                    print(f"[Source C filter] kept {len(b_items)}/{before} rewrites (others changed numbers/names or were misaligned)")
                for it in b_items:
                    inst = (it.get("context") or {}).get("installed_apps") or {}
                    if len(inst) < 7:
                        cands = [a for a in pub_tools_map if a not in inst]
                        for extra_a in rng.sample(cands, min(6, len(cands))):
                            inst[extra_a] = {"tools": pub_tools_map[extra_a]}
                    records.append(sample_to_sft_record(it, pub_cat, rng))
                extra_counts[fname.replace(".jsonl", "")] = len(b_items)

    rng.shuffle(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"Built {len(records)} multi-turn SFT records "
        f"(train={len(train_items)}, core_synth={len(synth_items) * args.synth_repeat}, extras={extra_counts}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
