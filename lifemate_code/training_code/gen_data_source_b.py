"""Nguồn B: Bộ sinh dữ liệu tự động bằng Python Grammar + Kiểm chứng trực tiếp qua `zaic_tools.Simulator`.

Sinh hàng trăm mẫu đa dạng (thay đổi ngẫu nhiên tên người Việt/Ngoại quốc, sản phẩm, mã cổ phiếu, tuyến xe,
công thức nấu ăn, giờ giấc, thành phố, xưng hô vùng miền) cho toàn bộ 11 ứng dụng mới của `public_test`.
100% mẫu sinh ra được chạy thử trên `Simulator` của BTC để lấy `tool_results` thật và đảm bảo 0 lỗi schema!

Chạy lệnh:
    python -X utf8 scripts/gen_data_source_b.py --num-samples 600 --out outputs/sft/source_b_simulator.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    WORKSPACE.parent / "train+public_test"
    if (WORKSPACE.parent / "train+public_test").exists()
    else WORKSPACE / "train+public_test" / "train+public_test"
)
sys.path.insert(0, str(DATA_ROOT / "public_test"))

from zaic_tools.executor import Simulator  # noqa: E402

PROFILES = [
    {"name": "Anh Thành", "city": "TP.HCM", "home_address": "77 Hoàng Văn Thụ, Phú Nhuận, TP.HCM", "work_address": "Tòa nhà Viettel, Quận 10, TP.HCM"},
    {"name": "Chị Lan", "city": "Hà Nội", "home_address": "25 Láng Hạ, Đống Đa, Hà Nội", "work_address": "Keangnam Landmark, Cầu Giấy, Hà Nội"},
    {"name": "Minh Khôi", "city": "Đà Nẵng", "home_address": "12 Nguyễn Văn Linh, Hải Châu, Đà Nẵng", "work_address": "Khu Công nghệ cao, Liên Chiểu, Đà Nẵng"},
    {"name": "Cô Hạnh", "city": "TP.HCM", "home_address": "108 Nguyễn Thị Minh Khai, Quận 3, TP.HCM", "work_address": "15 Lê Duẩn, Quận 1, TP.HCM"},
]

CONTACTS_POOL = [
    {"id": "c_101", "name": "Michael Chen", "relation": "đối tác", "phone": "0908112233", "email": "michael.chen@partner.com"},
    {"id": "c_102", "name": "Lucas Bennett", "relation": "khách hàng", "phone": "0908223344", "email": "lucas.b@client.org"},
    {"id": "c_103", "name": "Anna Schmidt", "relation": "đồng nghiệp", "phone": "0908334455", "email": "anna.schmidt@corp.eu"},
    {"id": "c_104", "name": "Mr. Tanaka", "relation": "sếp", "phone": "0908445566", "email": "tanaka@jp-tech.co"},
    {"id": "c_105", "name": "Vợ Mai", "relation": "vợ", "phone": "0909988776", "email": "mai.nguyen@gmail.com"},
    {"id": "c_106", "name": "Anh Tuấn", "relation": "anh trai", "phone": "0912345678", "email": "tuan.anh@gmail.com"},
]

GROCERY_ITEMS = [
    ("gro_01", "Táo Envy Mỹ", 120000, "kg", "trái cây"),
    ("gro_02", "Sữa tươi không đường", 36000, "lốc", "sữa"),
    ("gro_03", "Thịt bò thăn", 280000, "kg", "thịt"),
    ("gro_04", "Trứng gà ta", 42000, "hộp", "trứng"),
    ("gro_05", "Cá hồi phi lê", 350000, "khay", "hải sản"),
    ("gro_06", "Bông cải xanh", 35000, "bó", "rau củ"),
]

ECOMMERCE_ITEMS = [
    ("prod_01", "Bàn phím cơ không dây Keychron", 1850000, "Keychron VN", "điện tử"),
    ("prod_02", "Bình giữ nhiệt Lock&Lock 750ml", 390000, "Lock&Lock", "gia dụng"),
    ("prod_03", "Nồi chiên không dầu Philips", 2450000, "Philips Official", "gia dụng"),
    ("prod_04", "Ba lô chống nước Laptop 15 inch", 650000, "SimpleCarry", "thời trang"),
]

STOCKS_POOL = [
    ("FPT", "FPT Corp", 134000),
    ("VNM", "Vinamilk", 65000),
    ("VCB", "Vietcombank", 92000),
    ("HPG", "Hòa Phát", 28000),
    ("MWG", "Thế Giới Di Động", 59000),
]

TRANSLATE_PAIRS = [
    ("Họp lúc 9 giờ sáng mai tại phòng 302", "en", "tiếng Anh", "Meeting at 9 AM tomorrow in room 302"),
    ("Hạn chót nộp báo cáo quý là thứ Sáu này", "en", "tiếng Anh", "The deadline for the quarterly report is this Friday"),
    ("Tôi sẽ đến muộn mười lăm phút", "en", "tiếng Anh", "I will arrive fifteen minutes late"),
    ("Tôi bị dị ứng hải sản", "ja", "tiếng Nhật", "私はシーフードアレルギーがあります"),
    ("Cho tôi xem thực đơn", "ko", "tiếng Hàn", "메뉴 좀 보여주세요"),
    ("Cảm ơn đối tác đã hỗ trợ dự án", "fr", "tiếng Pháp", "Merci partenaire d'avoir soutenu le projet"),
]


def build_random_sample(idx: int, rng: random.Random, pub_catalog: dict, app_tools_map: dict[str, list[str]]) -> dict:
    prof = copy.deepcopy(rng.choice(PROFILES))
    base_dt = datetime(2026, 11, rng.randint(5, 25), rng.choice([8, 9, 10, 14, 16, 19]), 0, 0)
    cur_iso = base_dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    today_str = base_dt.strftime("%Y-%m-%d")
    tomorrow_dt = base_dt + timedelta(days=1)
    tomorrow_str = tomorrow_dt.strftime("%Y-%m-%d")

    pronoun = rng.choice(["mình", "anh", "chị", "em", "tôi", "cô"])
    helper = rng.choice(["giúp", "hộ", "giùm", "dùm"])
    ending = rng.choice(["nhé.", "nha.", "với nhé.", "nghe.", "."])

    template_type = rng.choice(
        [
            "grocery_no_addr",
            "grocery_home_addr",
            "recipes_pantry_grocery",
            "ecommerce_buy",
            "ecommerce_return_damaged",
            "ecommerce_return_ask_reason",
            "ecommerce_cancel",
            "stocks_cond_buy_true",
            "stocks_cond_buy_false",
            "stocks_portfolio_sell_alert",
            "stocks_unknown_noop",
            "translator_chat_foreign",
            "translator_email_reply_hist",
            "translator_note",
            "parking_reserve_reminder",
            "parking_end",
            "parking_ask_hours",
            "transit_route_topup",
            "event_book_calendar",
            "event_ask_qty",
            "event_confirm_from_hist",
            "train_bus_book_alarm",
            "train_bus_seats_filter",
            "train_bus_ask_trip",
            "habit_log",
            "pantry_consume",
            "pantry_followup_hist",
            "recipes_get_save",
            "transit_next_departures",
            "train_bus_cancel_booking",
            "email_send_direct",
            "email_unread_read_archive",
            "flights_checkin_or_cancel",
            "food_delivery_cancel",
            "smart_home_scene_or_switch",
            "shopping_list_remove",
            "self_cancel_noop",
        ]
    )

    history: list[dict] = []

    state: dict[str, dict] = {
        "contacts": {"contacts": copy.deepcopy(CONTACTS_POOL)},
        "chat_ott": {"chats": [{"id": f"chat_{c['id']}", "name": c["name"], "unread": 0} for c in CONTACTS_POOL], "messages": []},
        "sms": {"messages": []},
        "notes": {"notes": []},
        "reminders": {"reminders": []},
        "calendar": {"events": []},
        "alarm": {"alarms": []},
        "tasks": {"tasks": [{"id": "tsk_01", "title": "Hoàn thành mục tiêu ngày", "status": "todo"}]},
    }
    calls: list[dict] = []
    apps: list[str] = []
    utterance = ""

    if template_type in ("grocery_no_addr", "grocery_home_addr"):
        g1, g2 = rng.sample(GROCERY_ITEMS, 2)
        q1, q2 = rng.randint(1, 3), rng.randint(1, 2)
        state["grocery"] = {
            "products": [{"id": x[0], "name": x[1], "price": x[2], "unit": x[3], "category": x[4], "in_stock": True} for x in GROCERY_ITEMS],
            "cart": [],
            "orders": [],
        }
        apps = ["grocery", "pantry", "recipes", "notes"]
        calls = [
            {"name": "grocery.search", "arguments": {"query": g1[1]}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": g1[0], "quantity": q1}},
            {"name": "grocery.search", "arguments": {"query": g2[1]}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": g2[0], "quantity": q2}},
        ]
        if template_type == "grocery_no_addr":
            calls.append({"name": "grocery.checkout", "arguments": {}})
            utterance = f"Đặt đi chợ {helper} {pronoun} {q1} {g1[3]} {g1[1]} và {q2} {g2[3]} {g2[1]} rồi chốt đơn luôn {ending}"
        else:
            calls.append({"name": "grocery.checkout", "arguments": {"address": prof["home_address"]}})
            utterance = f"Mua {helper} {pronoun} {q1} {g1[3]} {g1[1]} với {q2} {g2[3]} {g2[1]} trên app đi chợ, giao về nhà {ending}"

    elif template_type == "recipes_pantry_grocery":
        missing_item = rng.choice(GROCERY_ITEMS[:4])
        servings = rng.choice([2, 4])
        state["recipes"] = {
            "recipes": [
                {
                    "id": "rcp_10",
                    "name": "Bò lúc lắc rau củ",
                    "servings": 2,
                    "tags": ["món chính"],
                    "ingredients": [{"name": missing_item[1], "quantity": 1, "unit": missing_item[3]}],
                    "steps": ["Áp chảo"],
                }
            ],
            "saved": [],
        }
        state["pantry"] = {"items": [{"id": "pan_01", "name": missing_item[1], "stock": 0, "unit": missing_item[3]}]}
        state["grocery"] = {
            "products": [{"id": x[0], "name": x[1], "price": x[2], "unit": x[3], "category": x[4], "in_stock": True} for x in GROCERY_ITEMS],
            "cart": [],
            "orders": [],
        }
        apps = ["recipes", "pantry", "grocery", "notes"]
        need_qty = servings // 2
        calls = [
            {"name": "recipes.search", "arguments": {"query": "Bò lúc lắc", "servings": servings}},
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "grocery.search", "arguments": {"query": missing_item[1]}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": missing_item[0], "quantity": need_qty}},
            {"name": "grocery.checkout", "arguments": {}},
        ]
        utterance = f"Xem công thức món Bò lúc lắc cho {servings} người, kiểm tra tủ bếp thấy thiếu nguyên liệu nào thì đặt mua đủ số lượng trên app đi chợ luôn {ending}"

    elif template_type == "ecommerce_buy":
        prod = rng.choice(ECOMMERCE_ITEMS)
        qty = rng.choice([1, 2])
        pay = rng.choice(["cod", "ewallet"])
        pay_vi = "tiền mặt COD" if pay == "cod" else "ví điện tử"
        state["ecommerce"] = {
            "products": [{"id": x[0], "name": x[1], "price": x[2], "seller": x[3], "category": x[4], "in_stock": True} for x in ECOMMERCE_ITEMS],
            "cart": [],
            "orders": [],
            "returns": [],
        }
        apps = ["ecommerce", "ewallet", "notes"]
        calls = [
            {"name": "ecommerce.search", "arguments": {"query": prod[1]}},
            {"name": "ecommerce.add_to_cart", "arguments": {"product_id": prod[0], "quantity": qty}},
            {"name": "ecommerce.checkout", "arguments": {"address": prof["home_address"], "payment": pay}},
        ]
        utterance = f"Tìm mua {helper} {pronoun} {qty} chiếc {prod[1]} giao về nhà, thanh toán bằng {pay_vi} {ending}"

    elif template_type in ("ecommerce_return_damaged", "ecommerce_return_ask_reason", "ecommerce_cancel"):
        prod = rng.choice(ECOMMERCE_ITEMS)
        status = "placed" if template_type == "ecommerce_cancel" else "delivered"
        state["ecommerce"] = {
            "products": [{"id": prod[0], "name": prod[1], "price": prod[2], "seller": prod[3], "category": prod[4], "in_stock": True}],
            "cart": [],
            "orders": [{"id": "ord_99", "items": [{"product_id": prod[0], "quantity": 1}], "total": prod[2], "status": status}],
            "returns": [],
        }
        apps = ["ecommerce", "sms", "notes"]
        if template_type == "ecommerce_cancel":
            calls = [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_99"}},
            ]
            utterance = f"Kiểm tra đơn hàng {prod[1]} vừa đặt rồi hủy đơn đó {helper} {pronoun} {ending}"
        elif template_type == "ecommerce_return_damaged":
            reason = rng.choice([("damaged", "bị nứt vỡ hỏng"), ("wrong_item", "giao sai mẫu")])
            calls = [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "ecommerce.request_return", "arguments": {"order_id": "ord_99", "reason": reason[0]}},
            ]
            utterance = f"Đơn {prod[1]} giao tới bị {reason[1]}, làm yêu cầu trả hàng {helper} {pronoun} {ending}"
        else:
            calls = [
                {"name": "ecommerce.order_status", "arguments": {}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ecommerce.request_return", "params": ["reason"]}},
            ]
            utterance = f"Kiểm tra đơn {prod[1]} rồi làm thủ tục trả hàng lại {helper} {pronoun} {ending}"

    elif template_type in ("stocks_cond_buy_true", "stocks_cond_buy_false", "stocks_portfolio_sell_alert"):
        sym, sname, sprice = rng.choice(STOCKS_POOL)
        qty = rng.choice([100, 200, 500])
        state["stocks"] = {
            "holdings": [{"symbol": sym, "name": sname, "quantity": 600, "avg_price": sprice - 3000, "price": sprice}],
            "quotes": [{"symbol": x[0], "name": x[1], "price": x[2], "change_pct": 1.2} for x in STOCKS_POOL],
            "orders": [],
            "alerts": [],
        }
        apps = ["stocks", "banking", "notes"]
        if template_type == "stocks_cond_buy_true":
            thresh = sprice + 5000
            calls = [
                {"name": "stocks.get_quote", "arguments": {"symbol": sym}},
                {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "buy", "quantity": qty}},
            ]
            utterance = f"Xem giá cổ phiếu {sym} hiện tại, nếu dưới {thresh // 1000} nghìn thì đặt mua {qty} cổ phiếu giá thị trường {ending}"
        elif template_type == "stocks_cond_buy_false":
            thresh = sprice - 8000
            calls = [{"name": "stocks.get_quote", "arguments": {"symbol": sym}}]
            utterance = f"Kiểm tra mã {sym} {helper} {pronoun}, nếu giảm xuống dưới {thresh // 1000} nghìn thì mua {qty} cổ phiếu, còn cao hơn thì thôi {ending}"
        else:
            alert_p = sprice + 10000
            calls = [
                {"name": "stocks.get_portfolio", "arguments": {}},
                {"name": "stocks.place_order", "arguments": {"symbol": sym, "side": "sell", "quantity": qty}},
                {"name": "stocks.set_price_alert", "arguments": {"symbol": sym, "price": alert_p, "direction": "above"}},
            ]
            utterance = f"Xem danh mục đầu tư, bán bớt {qty} cổ phiếu {sym} và đặt cảnh báo khi {sym} vượt {alert_p // 1000} nghìn {ending}"

    elif template_type in ("translator_chat_foreign", "translator_note"):
        src_txt, lang_code, lang_vi, trans_txt = rng.choice(TRANSLATE_PAIRS)
        foreign_contact = rng.choice(CONTACTS_POOL[:4])
        state["translator"] = {"dictionary": [{"text": src_txt, "to_lang": lang_code, "translation": trans_txt}]}
        if template_type == "translator_chat_foreign":
            apps = ["translator", "chat_ott", "contacts"]
            calls = [
                {"name": "translator.translate", "arguments": {"text": src_txt, "to_lang": lang_code}},
                {"name": "chat_ott.list_chats", "arguments": {}},
                {"name": "chat_ott.send_message", "arguments": {"chat_id": f"chat_{foreign_contact['id']}", "content": trans_txt}},
            ]
            utterance = f"Dịch câu '{src_txt}' sang {lang_vi} rồi gửi tin nhắn qua app chat cho {foreign_contact['name']} {ending}"
        else:
            apps = ["translator", "notes"]
            calls = [
                {"name": "translator.translate", "arguments": {"text": src_txt, "to_lang": lang_code}},
                {"name": "notes.create", "arguments": {"title": f"Bản dịch {lang_vi}", "content": trans_txt}},
            ]
            utterance = f"Dịch {helper} {pronoun} câu '{src_txt}' ra {lang_vi} rồi lưu vào ghi chú mới tên 'Bản dịch {lang_vi}' {ending}"

    elif template_type in ("parking_reserve_reminder", "parking_end", "parking_ask_hours"):
        lot_place = rng.choice([("lot_11", "Vincom Center", 30000), ("lot_12", "Hồ Gươm Plaza", 25000), ("lot_13", "Sân bay Tân Sơn Nhất", 40000)])
        state["parking"] = {
            "lots": [{"id": lot_place[0], "name": f"Bãi đỗ {lot_place[1]}", "address": lot_place[1], "area": "Trung tâm", "price_per_hour": lot_place[2], "available_spots": 10}],
            "sessions": [{"id": "park_88", "lot_id": lot_place[0], "from_time": cur_iso, "hours": 2, "end_time": cur_iso, "total": 60000, "status": "reserved"}],
        }
        apps = ["parking", "reminders", "maps"]
        if template_type == "parking_reserve_reminder":
            hrs = rng.choice([2, 3, 4])
            start_h = 15
            from_iso = f"{tomorrow_str}T{start_h:02d}:00:00+07:00"
            rem_iso = f"{tomorrow_str}T{start_h + hrs - 1:02d}:45:00+07:00"
            calls = [
                {"name": "parking.find", "arguments": {"near": lot_place[1]}},
                {"name": "parking.reserve", "arguments": {"lot_id": lot_place[0], "from_time": from_iso, "hours": hrs}},
                {"name": "reminders.create", "arguments": {"content": f"Lấy xe tại {lot_place[1]}", "time": rem_iso}},
            ]
            utterance = f"Tìm bãi đỗ xe gần {lot_place[1]}, đặt chỗ lúc {start_h} giờ chiều mai trong {hrs} tiếng và nhắc {pronoun} lấy xe trước khi hết giờ 15 phút {ending}"
        elif template_type == "parking_end":
            calls = [
                {"name": "parking.list_sessions", "arguments": {}},
                {"name": "parking.end_session", "arguments": {"session_id": "park_88"}},
            ]
            utterance = f"Kết thúc phiên gửi xe hiện tại ở {lot_place[1]} {helper} {pronoun} {ending}"
        else:
            calls = [
                {"name": "parking.find", "arguments": {"near": lot_place[1]}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "parking.reserve", "params": ["hours"]}},
            ]
            utterance = f"Đặt chỗ gửi xe ở gần {lot_place[1]} vào 9 giờ sáng mai {helper} {pronoun} {ending}"

    elif template_type == "transit_route_topup":
        amt = rng.choice([50000, 100000, 200000])
        state["public_transit"] = {
            "card_balance": 15000,
            "routes": [{"id": "metro_01", "name": "Metro Bến Thành - Suối Tiên", "type": "metro", "stops": ["Bến Thành", "Thảo Điền", "Suối Tiên"], "fare": 15000, "duration_min": 28, "headway_min": 8}],
            "departures": [],
            "topups": [],
        }
        apps = ["public_transit", "maps", "ewallet"]
        calls = [
            {"name": "public_transit.find_routes", "arguments": {"origin": "Bến Thành", "destination": "Suối Tiên"}},
            {"name": "public_transit.topup_card", "arguments": {"amount": amt}},
        ]
        utterance = f"Tra tuyến xe công cộng từ Bến Thành đi Suối Tiên rồi nạp thêm {amt // 1000} nghìn vào thẻ vé tháng {helper} {pronoun} {ending}"

    elif template_type in ("event_book_calendar", "event_ask_qty"):
        ev_name = rng.choice(["Đêm nhạc Acoustic Mùa Thu", "Triển lãm Công nghệ AI", "Lễ hội Ẩm thực Quốc tế"])
        state["event_tickets"] = {
            "events": [{"id": "ev_55", "name": ev_name, "category": "event", "date": tomorrow_str, "time": "19:00", "venue": "Nhà thi đấu Phú Thọ", "price": 500000, "available": 20}],
            "bookings": [],
        }
        apps = ["event_tickets", "calendar", "reminders"]
        if template_type == "event_book_calendar":
            qty = rng.choice([1, 2, 4])
            calls = [
                {"name": "event_tickets.search", "arguments": {"query": ev_name}},
                {"name": "event_tickets.book", "arguments": {"event_id": "ev_55", "quantity": qty}},
                {
                    "name": "calendar.create_event",
                    "arguments": {
                        "title": ev_name,
                        "start": f"{tomorrow_str}T19:00:00+07:00",
                        "end": f"{tomorrow_str}T21:00:00+07:00",
                        "location": "Nhà thi đấu Phú Thọ",
                    },
                },
            ]
            utterance = f"Đặt {qty} vé sự kiện {ev_name} rồi ghi vào lịch cá nhân từ 19 giờ đến 21 giờ ngày mai {ending}"
        else:
            calls = [
                {"name": "event_tickets.search", "arguments": {"query": ev_name}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "event_tickets.book", "params": ["quantity"]}},
            ]
            utterance = f"Đặt vé xem {ev_name} vào tối mai {helper} {pronoun} {ending}"

    elif template_type in ("train_bus_book_alarm", "train_bus_ask_trip"):
        dest = rng.choice(["Nha Trang", "Đà Lạt", "Phan Thiết", "Vinh"])
        if template_type == "train_bus_book_alarm":
            seats = rng.choice([1, 2, 3])
            state["train_bus_tickets"] = {
                "trips": [{"id": "trp_01", "type": "train", "operator": "SE2", "origin": prof["city"], "destination": dest, "date": tomorrow_str, "depart": "07:00", "arrive": "14:00", "price": 450000, "seats_left": 15}],
                "bookings": [],
            }
            apps = ["train_bus_tickets", "alarm", "calendar"]
            calls = [
                {"name": "train_bus_tickets.search", "arguments": {"origin": prof["city"], "destination": dest, "date": tomorrow_str, "type": "train"}},
                {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trp_01", "seats": seats}},
                {"name": "alarm.create", "arguments": {"time": f"{tomorrow_str}T05:30:00+07:00"}},
            ]
            utterance = f"Tìm và đặt {seats} vé tàu hỏa từ {prof['city']} đi {dest} vào ngày mai, kèm đặt báo thức 5 giờ 30 sáng mai nhé."
        else:
            state["train_bus_tickets"] = {
                "trips": [
                    {"id": "trp_11", "type": "bus", "operator": "Phương Trang", "origin": prof["city"], "destination": dest, "date": tomorrow_str, "depart": "08:00", "arrive": "14:00", "price": 280000, "seats_left": 10},
                    {"id": "trp_12", "type": "bus", "operator": "Thành Bưởi", "origin": prof["city"], "destination": dest, "date": tomorrow_str, "depart": "09:30", "arrive": "15:30", "price": 290000, "seats_left": 12},
                ],
                "bookings": [],
            }
            apps = ["train_bus_tickets", "calendar"]
            calls = [
                {"name": "train_bus_tickets.search", "arguments": {"origin": prof["city"], "destination": dest, "date": tomorrow_str, "type": "bus"}},
                {"name": "sys.ask_missing_info", "arguments": {"tool_name": "train_bus_tickets.book", "params": ["trip_id"]}},
            ]
            utterance = f"Đặt {helper} {pronoun} 1 vé xe khách từ {prof['city']} đi {dest} trong ngày mai {ending}"

    elif template_type == "habit_log":
        hname = rng.choice(["Thiền 15 phút", "Đọc sách 30 phút", "Tập chạy bộ 5km", "Học ngoại ngữ"])
        yesterday_str = (base_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        state["habit_tracker"] = {
            "habits": [{"id": "hb_77", "name": hname, "streak": 5, "last_logged": yesterday_str}],
            "logs": [],
        }
        apps = ["habit_tracker", "reminders", "tasks"]
        calls = [
            {"name": "habit_tracker.list_habits", "arguments": {}},
            {"name": "habit_tracker.log", "arguments": {"habit_id": "hb_77", "date": today_str}},
        ]
        utterance = f"Hôm nay {pronoun} đã hoàn thành '{hname}' rồi, ghi nhận vào sổ theo dõi thói quen {helper} {pronoun} {ending}"

    elif template_type == "pantry_consume":
        item = rng.choice([("pan_11", "Trứng gà", 12, "quả", 4), ("pan_12", "Gạo ST25", 10, "kg", 2), ("pan_13", "Sữa hạt", 6, "hộp", 2)])
        state["pantry"] = {"items": [{"id": item[0], "name": item[1], "stock": item[2], "unit": item[3]}]}
        apps = ["pantry", "grocery", "recipes"]
        calls = [
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "pantry.consume", "arguments": {"item_id": item[0], "quantity": item[4]}},
        ]
        utterance = f"Nãy nấu ăn {pronoun} vừa dùng hết {item[4]} {item[3]} {item[1]} trong tủ bếp, trừ tồn kho {helper} {pronoun} {ending}"

    elif template_type == "pantry_followup_hist":
        item = rng.choice([("pan_21", "mì tôm", 8, "gói", 3), ("pan_22", "sữa tươi", 6, "hộp", 2)])
        state["pantry"] = {"items": [{"id": item[0], "name": item[1], "stock": item[2], "unit": item[3]}]}
        apps = ["pantry", "recipes", "shopping_list", "grocery"]
        history = [
            {"role": "user", "audio": f"audio/src_b_h_{idx}.wav", "audio_text": f"Trừ bớt {item[1]} trong tủ bếp {helper} {pronoun} nhé."},
            {
                "role": "assistant",
                "text": f"Bạn muốn trừ bao nhiêu {item[3]} {item[1]}?",
                "tool_calls": [
                    {"name": "pantry.list_items", "arguments": {}},
                    {"name": "sys.ask_missing_info", "arguments": {"tool_name": "pantry.consume", "params": ["quantity"]}},
                ],
                "tool_results": [
                    {"ok": True, "items": [{"id": item[0], "name": item[1], "stock": item[2], "unit": item[3]}]},
                    {"ok": True, "waiting_for_user": True},
                ],
            },
        ]
        calls = [{"name": "pantry.consume", "arguments": {"item_id": item[0], "quantity": item[4]}}]
        utterance = f"{item[4]} {item[3]}, trừ {item[4]} {item[3]} thôi {ending}"

    elif template_type == "event_confirm_from_hist":
        ev_name = rng.choice(["Đêm nhạc Acoustic Mùa Thu", "Kịch Idecaf Cuối Tuần", "Hội thảo AI Việt Nam"])
        qty = rng.choice([2, 3])
        state["event_tickets"] = {
            "events": [{"id": "event_06", "name": ev_name, "category": "event", "date": tomorrow_str, "time": "19:30", "venue": "Nhà hát Thành Phố", "price": 300000, "available": 15}],
            "bookings": [],
        }
        apps = ["event_tickets", "calendar", "music", "reminders"]
        history = [
            {"role": "user", "audio": f"audio/src_b_h_{idx}.wav", "audio_text": f"Tìm {helper} {pronoun} vé {ev_name} xem còn chỗ không?"},
            {
                "role": "assistant",
                "text": f"Sự kiện {ev_name} còn 15 vé, giá 300.000đ/vé. Bạn muốn đặt mấy vé?",
                "tool_calls": [{"name": "event_tickets.search", "arguments": {"query": ev_name}}],
                "tool_results": [{"ok": True, "events": state["event_tickets"]["events"]}],
            },
        ]
        calls = [{"name": "event_tickets.book", "arguments": {"event_id": "event_06", "quantity": qty}}]
        utterance = f"Ờ đúng rồi, chốt đặt {qty} vé đó cho {pronoun} luôn đi {ending}"

    elif template_type == "translator_email_reply_hist":
        src_txt, lang_code, lang_vi, trans_txt = rng.choice(TRANSLATE_PAIRS[:3])
        state["translator"] = {"dictionary": [{"text": src_txt, "to_lang": lang_code, "translation": trans_txt}]}
        state["email"] = {
            "emails": [{"id": "mail_77", "from": "david.partner@global.test", "from_name": "David", "to": ["me@example.test"], "subject": "Project Schedule", "body": "When can we meet?", "unread": True}],
            "drafts": [],
            "sent": [],
        }
        apps = ["email", "translator", "calendar", "notes"]
        history = [
            {"role": "user", "audio": f"audio/src_b_h_{idx}.wav", "audio_text": "Kiểm tra email mới giúp anh."},
            {
                "role": "assistant",
                "text": "Có email mới từ David hỏi về lịch gặp.",
                "tool_calls": [{"name": "email.list", "arguments": {}}],
                "tool_results": [{"ok": True, "emails": state["email"]["emails"]}],
            },
        ]
        calls = [
            {"name": "translator.translate", "arguments": {"text": src_txt, "to_lang": lang_code}},
            {"name": "email.reply", "arguments": {"email_id": "mail_77", "body": trans_txt}},
        ]
        utterance = f"Trả lời mail đó {helper} {pronoun}, dịch sang {lang_vi} câu: {src_txt}"

    elif template_type == "train_bus_seats_filter":
        dest = rng.choice(["Nha Trang", "Đà Lạt", "Huế"])
        seats = rng.choice([2, 3])
        state["train_bus_tickets"] = {
            "trips": [
                {"id": "trip_full", "type": "train", "operator": "SE1", "origin": prof["city"], "destination": dest, "date": tomorrow_str, "depart": "06:30", "arrive": "13:30", "price": 420000, "seats_left": 1},
                {"id": "trip_ok", "type": "train", "operator": "SE3", "origin": prof["city"], "destination": dest, "date": tomorrow_str, "depart": "08:00", "arrive": "15:00", "price": 450000, "seats_left": 12},
            ],
            "bookings": [],
        }
        apps = ["train_bus_tickets", "calendar", "reminders"]
        calls = [
            {"name": "train_bus_tickets.search", "arguments": {"origin": prof["city"], "destination": dest, "date": tomorrow_str, "type": "train"}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_ok", "seats": seats}},
        ]
        utterance = f"Tìm vé tàu từ {prof['city']} đi {dest} ngày mai, chuyến nào còn đủ {seats} chỗ thì đặt luôn {seats} ghế {helper} {pronoun} {ending}"

    elif template_type == "stocks_unknown_noop":
        bad_sym = rng.choice(["ZCAN", "SUNR", "VIT"])
        state["stocks"] = {
            "holdings": [],
            "quotes": [{"symbol": x[0], "name": x[1], "price": x[2], "change_pct": 0.5} for x in STOCKS_POOL],
            "orders": [],
            "alerts": [],
        }
        apps = ["stocks", "banking", "notes"]
        calls = [{"name": "stocks.get_quote", "arguments": {"symbol": bad_sym}}]
        utterance = f"Kiểm tra mã cổ phiếu {bad_sym}, nếu có và dưới 80 nghìn thì mua 100 cổ phiếu {helper} {pronoun} {ending}"

    elif template_type == "recipes_get_save":
        rname = rng.choice(["Canh chua cá lóc", "Súp bí đỏ kem tươi", "Gà kho sả ớt"])
        state["recipes"] = {
            "recipes": [{"id": "rcp_55", "name": rname, "servings": 2, "tags": ["gia đình"], "ingredients": [{"name": "Gia vị", "quantity": 1, "unit": "gói"}], "steps": ["Sơ chế", "Nấu chín"]}],
            "saved": [],
        }
        apps = ["recipes", "pantry", "notes"]
        calls = [
            {"name": "recipes.search", "arguments": {"query": rname}},
            {"name": "recipes.get", "arguments": {"recipe_id": "rcp_55"}},
            {"name": "recipes.save", "arguments": {"recipe_id": "rcp_55"}},
        ]
        utterance = f"Tìm công thức món {rname}, mở xem chi tiết cách làm rồi lưu vào mục công thức yêu thích {helper} {pronoun} {ending}"

    elif template_type == "transit_next_departures":
        stop_name = rng.choice(["Bến Thành", "Thảo Điền", "Cầu Giấy", "Hồ Gươm"])
        state["public_transit"] = {
            "card_balance": 50000,
            "routes": [{"id": "metro_01", "name": "Metro Số 1", "type": "metro", "stops": [stop_name, "Ga Cuối"], "fare": 15000, "duration_min": 20, "headway_min": 8}],
            "departures": [{"stop": stop_name, "route": "metro_01", "time": "10:20"}],
            "topups": [],
        }
        apps = ["public_transit", "maps", "reminders"]
        calls = [{"name": "public_transit.next_departures", "arguments": {"stop": stop_name}}]
        utterance = f"Xem {helper} {pronoun} các chuyến xe/tàu sắp khởi hành tại trạm {stop_name} lúc mấy giờ {ending}"

    elif template_type == "train_bus_cancel_booking":
        dest = rng.choice(["Nha Trang", "Đà Lạt", "Huế"])
        state["train_bus_tickets"] = {
            "trips": [{"id": "trp_88", "type": "train", "operator": "SE1", "origin": prof["city"], "destination": dest, "date": tomorrow_str, "depart": "08:00", "arrive": "15:00", "price": 450000, "seats_left": 10}],
            "bookings": [{"id": "tk_88", "trip_id": "trp_88", "seats": 2, "total": 900000, "status": "confirmed"}],
        }
        apps = ["train_bus_tickets", "calendar", "sms"]
        calls = [
            {"name": "train_bus_tickets.my_bookings", "arguments": {}},
            {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tk_88"}},
        ]
        utterance = f"Kiểm tra vé tàu xe đã đặt và hủy {helper} {pronoun} vé đi {dest} ngày mai {ending}"

    elif template_type == "email_send_direct":
        c = rng.choice(CONTACTS_POOL[:4])
        subj = rng.choice(["Báo cáo tiến độ tuần", "Xác nhận lịch họp ngày mai", "Tài liệu thiết kế mới"])
        body_txt = f"Chào {c['name']}, mình gửi xác nhận thông tin về {subj.lower()} nhé."
        state["email"] = {"emails": [], "drafts": [], "sent": []}
        apps = ["email", "contacts", "notes"]
        calls = [
            {"name": "contacts.search", "arguments": {"query": c["name"]}},
            {"name": "email.send", "arguments": {"to": [c["email"]], "subject": subj, "body": body_txt}},
        ]
        utterance = f"Tìm email của {c['name']} trong danh bạ rồi gửi email mới tiêu đề '{subj}' với nội dung '{body_txt}' {ending}"

    elif template_type == "email_unread_read_archive":
        state["email"] = {
            "emails": [{"id": "mail_12", "from": "promo@shop.test", "from_name": "Shop KM", "to": ["me@example.test"], "subject": "Khuyến mãi cuối tuần", "body": "Giảm giá 50%", "unread": True}],
            "drafts": [],
            "sent": [],
        }
        apps = ["email", "notes", "reminders"]
        calls = [
            {"name": "email.list", "arguments": {"unread_only": True}},
            {"name": "email.read", "arguments": {"email_id": "mail_12"}},
            {"name": "email.archive", "arguments": {"email_id": "mail_12", "action": "archive"}},
        ]
        utterance = f"Kiểm tra các email chưa đọc, mở đọc nội dung thư khuyến mãi rồi lưu trữ gọn lại {helper} {pronoun} {ending}"

    elif template_type == "flights_checkin_or_cancel":
        do_checkin = rng.random() < 0.6
        state["flights"] = {
            "flights": [{"id": "fl_09", "origin": "SGN", "destination": "HAN", "date": tomorrow_str, "depart": "09:00", "arrive": "11:10", "airline": "VN Air", "price": 1600000}],
            "bookings": [{"id": "bk_09", "flight_id": "fl_09", "passengers": 1, "seat_class": "economy", "status": "confirmed"}],
        }
        apps = ["flights", "calendar", "alarm"]
        if do_checkin:
            calls = [
                {"name": "flights.my_bookings", "arguments": {}},
                {"name": "flights.check_in", "arguments": {"booking_id": "bk_09"}},
            ]
            utterance = f"Xem vé máy bay đi Hà Nội đã đặt rồi làm thủ tục check-in online {helper} {pronoun} luôn {ending}"
        else:
            calls = [
                {"name": "flights.my_bookings", "arguments": {}},
                {"name": "flights.cancel", "arguments": {"booking_id": "bk_09"}},
            ]
            utterance = f"Kiểm tra vé máy bay đã đặt và hủy {helper} {pronoun} chuyến bay đi Hà Nội ngày mai {ending}"

    elif template_type == "food_delivery_cancel":
        dish = rng.choice(["Bún bò Huế", "Cơm tấm sườn bì", "Phở gà ta"])
        state["food_delivery"] = {
            "restaurants": [{"id": "rest_01", "name": "Quán Ngon", "menu": [{"id": "m_01", "name": dish, "price": 55000}]}],
            "orders": [{"id": "ord_301", "restaurant_id": "rest_01", "items": [{"item_id": "m_01", "name": dish, "quantity": 1}], "total": 55000, "status": "placed"}],
        }
        apps = ["food_delivery", "ewallet", "notes"]
        calls = [
            {"name": "food_delivery.order_history", "arguments": {}},
            {"name": "food_delivery.cancel_order", "arguments": {"order_id": "ord_301"}},
        ]
        utterance = f"Kiểm tra đơn đặt đồ ăn rồi hủy {helper} {pronoun} đơn {dish} vừa đặt nhé, nhà nấu cơm rồi."

    elif template_type == "smart_home_scene_or_switch":
        use_scene = rng.random() < 0.5
        state["smart_home"] = {
            "devices": [
                {"id": "sw_heater", "name": "bình nóng lạnh", "type": "switch", "room": "phòng tắm", "floor": 1, "state": {"power": "off"}},
                {"id": "light_bed", "name": "đèn ngủ", "type": "light", "room": "phòng ngủ", "floor": 2, "state": {"power": "on", "brightness": 80}},
            ],
            "scenes": [{"id": "sc_sleep", "name": "Đi ngủ", "actions": [{"device_id": "light_bed", "state": {"power": "off"}}]}],
        }
        apps = ["smart_home", "alarm", "reminders"]
        if use_scene:
            calls = [
                {"name": "smart_home.get_status", "arguments": {}},
                {"name": "smart_home.scene_run", "arguments": {"scene_id": "sc_sleep"}},
            ]
            utterance = f"Kiểm tra nhà thông minh và bật ngữ cảnh 'Đi ngủ' {helper} {pronoun} {ending}"
        else:
            calls = [
                {"name": "smart_home.get_status", "arguments": {"room": "phòng tắm"}},
                {"name": "smart_home.switch_set", "arguments": {"device_id": "sw_heater", "power": "on"}},
            ]
            utterance = f"Kiểm tra thiết bị trong phòng tắm rồi bật công tắc bình nóng lạnh lên {helper} {pronoun} {ending}"

    elif template_type == "shopping_list_remove":
        iname = rng.choice(["Sữa đặc", "Nước mắm", "Khăn giấy"])
        state["shopping_list"] = {"items": [{"id": "sl_01", "name": iname, "quantity": 1, "checked": False}]}
        apps = ["shopping_list", "grocery", "pantry"]
        calls = [
            {"name": "shopping_list.get", "arguments": {}},
            {"name": "shopping_list.remove", "arguments": {"item_id": "sl_01"}},
        ]
        utterance = f"Xem danh sách mua sắm rồi xóa món {iname} khỏi danh sách {helper} {pronoun} vì mới mua rồi {ending}"

    else:  # self_cancel_noop
        apps = ["ecommerce", "grocery", "event_tickets", "calendar"]
        state["ecommerce"] = {"products": [], "cart": [], "orders": [], "returns": []}
        calls = []
        utterance = rng.choice(
            [
                f"Đặt mua {helper} {pronoun} cái bàn phím cơ... à mà thôi khoan mua, để cuối tháng có lương đặt sau nhé.",
                f"Tìm vé ca nhạc tối mai cho {pronoun}... ờ thôi hủy đi, tối mai {pronoun} bận họp mất rồi.",
                f"Hôm nay trời đẹp ghê ha, cảm ơn trợ lý nhiều nha!",
            ]
        )

    # Pad installed_apps to 8-12 realistic apps so prompt length & distractor density match `public_test`!
    all_pub_apps = [a for a in app_tools_map if a != "sys" and a not in apps]
    extra_n = rng.randint(5, 8)
    apps = apps + rng.sample(all_pub_apps, min(extra_n, len(all_pub_apps)))

    installed_sim = {
        a: {"tools": app_tools_map[a], "state": copy.deepcopy(state.get(a, {}))}
        for a in apps
        if a in app_tools_map
    }
    sim = Simulator(
        {
            "id": f"src_b_{idx:05d}",
            "context": {"current_time": cur_iso, "user_profile": prof, "installed_apps": installed_sim},
        },
        pub_catalog,
    )
    results = []
    for c in calls:
        r = sim.call(c["name"], c["arguments"])
        assert r.get("ok"), f"Template {template_type} schema fail: {c} -> {r}"
        if template_type != "stocks_unknown_noop":
            assert r.get("done") is not False, f"Template {template_type} noop fail: {c} -> {r}"
        results.append(r)

    return {
        "id": f"src_b_{idx:05d}",
        "audio": f"audio/src_b_{idx:05d}.wav",
        "audio_text": utterance,
        "context": {
            "current_time": cur_iso,
            "timezone": "Asia/Ho_Chi_Minh",
            "location": {"lat": 10.7769, "lng": 106.7009},
            "user_profile": prof,
            "installed_apps": {a: {"tools": app_tools_map[a]} for a in apps if a in app_tools_map},
        },
        "history": history,
        "label": {"tool_calls": calls, "tool_results": results, "final_state": {}},
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Nguồn B: Sinh dữ liệu tự động kiểm chứng qua Simulator")
    ap.add_argument("--num-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(WORKSPACE / "outputs" / "sft" / "source_b_simulator.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pub_catalog = json.load(open(DATA_ROOT / "public_test" / "public_test_tools_catalog.json", encoding="utf-8"))
    app_tools_map = {a["id"]: [t["name"] for t in a.get("tools", [])] for a in pub_catalog.get("apps", [])}

    samples = [build_random_sample(i, rng, pub_catalog, app_tools_map) for i in range(1, args.num_samples + 1)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Nguồn B] Đã sinh và kiểm chứng 100% qua Simulator: {len(samples)} mẫu -> {out_path}")


if __name__ == "__main__":
    main()
