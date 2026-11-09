"""Dynamic Tool Filtering, Weekday Table, Colloquial App Mapping & Multi-Turn Prompt Construction v7.0 for ZAC 2026 LifeMate."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta

from .asr import clean_asr_text

WEEKDAY_VI = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật",
}

# Explicit mapping from colloquial Vietnamese expressions in TRAIN to exact App IDs & key execution rules
APP_COLLOQUIAL_MAP: dict[str, str] = {
    "ewallet": '"bên ví", "lấy ví", "chuyển ví", "ví điện tử", "nạp tiền điện thoại", "quét QR quán" -> `ewallet` (`topup_mobile` dùng `phone`; `transfer` dùng `contact_id`; `pay_qr` dùng `merchant_id` từ `food_delivery.list_restaurants`)',
    "banking": '"bên ngân hàng", "chuyển khoản", "tài khoản thanh toán", "mở/gửi tiết kiệm" -> `banking` (`transfer` cần `from_account_id` từ `get_balance` + `beneficiary_id` từ `list_beneficiaries`)',
    "bill_payment": '"đóng tiền điện/nước/mạng/internet/truyền hình", "hóa đơn", "bật tự trả" -> `bill_payment` (`list_bills` lấy `bill_id` trước khi `pay`; kiểm tra `ewallet.get_balance` nếu có điều kiện "ví không đủ thì trừ ngân hàng")',
    "chat_ott": '"nhắn Zalo", "nhắn vào nhóm", "gửi vị trí qua chat" -> `chat_ott` (luôn gọi `chat_ott.list_chats` lấy `chat_id`, KHÔNG dùng `contact_id`)',
    "sms": '"nhắn SMS", "gửi tin nhắn thường" -> `sms` (dùng `contact_id` từ `contacts.search`)',
    "contacts": '"danh bạ", "sửa số/email liên hệ" -> `contacts` (`contacts.search` trước khi `update`/gửi; nếu tìm tên riêng ra rỗng do lỗi nghe ASR, hãy thử tìm theo chức danh/quan hệ hoặc `{}`)',
    "smart_home": '"đèn", "điều hòa/máy lạnh", "quạt", "rèm", "bình nóng lạnh" -> `smart_home` (gọi `smart_home.get_status({})` lấy đúng `device_id` trước khi `light_set`/`climate_set`/`fan_set`)',
    "robot_vacuum": '"con robot", "nó dọn", "hút bụi", "lau nhà" -> `robot_vacuum` (luôn gọi `robot_vacuum.get_status({})` trước khi gọi `robot_vacuum.start(rooms=[...])`)',
    "home_automation": '"cài hẹn giờ tự bật/tắt hằng ngày" -> `home_automation.create_rule` (cần `device_id` từ `smart_home.get_status`)',
    "home_security": '"camera", "cảnh báo an ninh/gara" -> `home_security` (`list_alerts` / `get_camera`)',
    "door_lock": '"khóa cửa", "mở khóa cửa/kho/gara" -> `door_lock` (`list_locks` lấy `lock_id` -> `set_lock`)',
    "appliances": '"máy rửa bát", "máy giặt", "máy sấy" -> `appliances` (gọi `appliances.list({})` không truyền filter -> `start_cycle`)',
    "household_chores": '"giao việc nhà", "phân công cất bát/dọn dẹp" -> `household_chores` (`list_chores({})` -> `assign`)',
    "calendar": '"lịch họp/cà phê/sự kiện", "dời lịch", "chặn lịch" -> `calendar` (`list_events` lấy `event_id` và giờ gốc trước khi `update_event`/`delete_event`)',
    "reminders": '"nhắc tôi/chị/anh...", "xoá nhắc nhở" -> `reminders` (`create` / `list` -> `delete`)',
    "tasks": '"thêm việc cần làm", "danh sách việc", "hạn (due)" -> `tasks` (`add` / `list` -> `complete`)',
    "alarm": '"đặt báo thức", "gọi dậy" -> `alarm` (`create` / `list` -> `update`/`delete`)',
    "video_meeting": '"phòng họp trực tuyến", "mở phòng họp online" -> `video_meeting.create`',
    "email": '"soạn thư nháp", "gửi mail", "trả lời mail" -> `email` (`create_draft` khi dặn "để nháp/đừng gửi vội"; `send` khi gửi luôn)',
    "shopping_list": '"danh sách mua sắm/đi chợ", "đánh dấu đã mua" -> `shopping_list` (`get` -> `check` / `add`)',
    "grocery": '"đặt đi chợ online", "mua rau/thịt/trái cây/sữa giao về" -> `grocery` (`search` -> `add_to_cart` -> `checkout`; chỉ truyền `address` nếu câu nói nêu rõ địa chỉ)',
    "ecommerce": '"mua trên sàn", "đặt hàng online", "trả hàng/hủy đơn" -> `ecommerce` (`search` -> `add_to_cart` -> `checkout`; `order_status` -> `cancel_order` [đơn placed] hoặc `request_return` [đơn delivered])',
    "pantry": '"tủ bếp", "còn nguyên liệu gì", "trừ tồn kho tủ bếp" -> `pantry` (`list_items` -> `consume`)',
    "recipes": '"công thức nấu", "cách làm món" -> `recipes` (`search` -> `get`/`save`)',
    "loyalty": '"đổi điểm", "xem điểm", "voucher/ưu đãi" -> `loyalty` (`get_points` -> `redeem`; `list_vouchers({})`)',
    "subscriptions": '"hủy gói", "gia hạn gói" -> `subscriptions` (`list({})` -> `cancel`)',
    "library": '"mượn sách", "giữ sách thư viện" -> `library` (`search` -> `reserve`)',
    "bike_rental": '"giữ/thuê xe đạp", "trạm xe đạp" -> `bike_rental` (`find_bikes` -> `reserve`)',
    "ride_hailing": '"đặt xe đón", "bắt xe 4 chỗ/xe máy" -> `ride_hailing.book`',
    "food_delivery": '"đặt đồ ăn", "quán ăn", "đơn đồ ăn" -> `food_delivery` (`list_restaurants` -> `get_menu` -> `place_order`)',
    "maps": '"chỉ đường", "tìm hiệu thuốc/địa điểm quanh...", "chạy xe mất bao lâu" -> `maps` (`get_route` / `search_place`)',
    "parking": '"bãi đỗ xe", "giữ chỗ đỗ xe", "kết thúc phiên đỗ" -> `parking` (`find` -> `reserve`; `list_sessions` -> `end_session`)',
    "public_transit": '"tuyến xe buýt/metro", "chuyến sắp chạy", "nạp thẻ xe" -> `public_transit` (`find_routes` / `next_departures` / `topup_card`)',
    "stocks": '"giá cổ phiếu", "danh mục", "mua/bán cổ phiếu", "cảnh báo giá" -> `stocks` (`get_quote` / `get_portfolio` -> `place_order` / `set_price_alert`)',
    "event_tickets": '"vé đêm nhạc/concert/hội thảo" -> `event_tickets` (`search` -> `book`)',
    "train_bus_tickets": '"vé tàu hỏa/xe khách" -> `train_bus_tickets` (`search` -> `book`; `my_bookings` -> `cancel`)',
    "habit_tracker": '"thói quen", "đánh dấu hoàn thành thói quen" -> `habit_tracker` (`list_habits` -> `log`)',
    "translator": '"dịch câu ... sang tiếng ..." -> `translator.translate` (gọi trước rồi lấy kết quả dịch gửi SMS/chat/email/note)',
}


def filter_installed_tools(catalog: dict, installed_apps: dict) -> list[dict]:
    """Keep strictly the tools allowed in `sample['context']['installed_apps']` plus `sys.ask_missing_info`."""
    all_tools = {}
    for app in catalog.get("apps", []):
        for t in app.get("tools", []):
            all_tools[t["name"]] = t

    ordered = []
    for app_info in (installed_apps or {}).values():
        for tname in app_info.get("tools", []):
            if tname in all_tools:
                ordered.append(all_tools[tname])
    if "sys.ask_missing_info" in all_tools:
        ordered.append(all_tools["sys.ask_missing_info"])
    return ordered


def to_openai_tools(filtered_tools: list[dict]) -> list[dict]:
    """Convert `zaic_tools` catalog specs to compact OpenAI/Qwen function-calling tool schemas,
    preserving `x-ref`, `x-one-of`, `format`, and `kind` hints concisely (~40% fewer prompt tokens)."""
    openai_tools = []
    for t in filtered_tools:
        params = copy.deepcopy(t.get("parameters") or {"type": "object", "properties": {}, "required": []})
        one_of = params.pop("x-one-of", None)

        desc_parts = [f"[{t.get('kind', 'tool').upper()}] {t.get('description', '')}".strip()]
        if one_of:
            groups_str = " | ".join("+".join(g) for g in one_of)
            desc_parts.append(f"(Cần >=1 nhóm: {groups_str})")
        if t.get("returns"):
            desc_parts.append(f"-> {t['returns']}")

        props = params.get("properties") or {}
        for _pname, pspec in props.items():
            xref = pspec.pop("x-ref", None)
            pspec.pop("x-match", None)
            pdesc = pspec.get("description") or ""
            notes = []
            if xref:
                notes.append(f"ref:{xref}")
            fmt = pspec.get("format")
            if fmt == "date":
                notes.append("YYYY-MM-DD")
            elif fmt == "date-time":
                notes.append("ISO+07:00")
            elif fmt == "time":
                notes.append("HH:MM")
            if notes:
                pspec["description"] = (pdesc + (" | " if pdesc else "") + "; ".join(notes)).strip()

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": " ".join(desc_parts),
                    "parameters": params,
                },
            }
        )
    return openai_tools


def annotate_tool_result(name: str, result: dict) -> dict:
    """Hints appended to a tool result shown to the LLM. Single source of truth for BOTH the SFT builder and the
    runtime agent (no train/inference mismatch)."""
    res = copy.deepcopy(result) if isinstance(result, dict) else result
    if not isinstance(res, dict):
        return res

    if name == "contacts.search" and isinstance(res.get("contacts"), list):
        n = len(res["contacts"])
        if n >= 2:
            res["_system_note"] = (
                f"CHÚ Ý: Có {n} người trùng tên! "
                "Nếu câu nói KHÔNG nêu rõ quan hệ (vợ/chồng/bố/mẹ/sếp/đồng nghiệp...) để phân biệt chính xác 1 người, "
                "BẮT BUỘC gọi sys.ask_missing_info(tool_name='<tool_đích>', params=['contact_id'])!"
            )
        elif n == 0:
            res["_system_note"] = (
                "CHÚ Ý: Không tìm thấy liên hệ. Nếu tên riêng có thể bị ASR nghe nhầm (VD 'Sếp Nguyễn' thay vì 'Sếp Dũng'), "
                "hãy thử gọi contacts.search(query='Sếp') hoặc contacts.search({}) để xem toàn bộ danh bạ; "
                "nếu vẫn không có và thiếu số điện thoại thì gọi sys.ask_missing_info với params=['contact_id']!"
            )
    elif name == "maps.get_route" and isinstance(res.get("duration_min"), (int, float)):
        dur = int(res["duration_min"])
        res["_time_note"] = (
            f"Thời gian di chuyển: {dur} phút. Nếu cần đặt nhắc nhở/báo thức xuất phát để có mặt lúc H:MM, "
            f"hãy lấy H:MM trừ chính xác {dur} phút (KHÔNG làm tròn phút, VD 16:00 trừ 27 phút = 15:33)."
        )
    return res


def format_temporal_anchor(current_time_iso: str) -> str:
    """Current time + an explicit Weekday Table for both THIS WEEK and NEXT WEEK (Trick #3.2: eliminates weekday math errors)."""
    try:
        dt = datetime.fromisoformat(current_time_iso)
        wd = WEEKDAY_VI[dt.weekday()]
        yesterday = dt - timedelta(days=1)
        tomorrow = dt + timedelta(days=1)
        day_after = dt + timedelta(days=2)

        # Monday of current week and Monday of next week
        this_monday = dt - timedelta(days=dt.weekday())
        next_monday = this_monday + timedelta(days=7)

        this_week_str = ", ".join(
            f"{WEEKDAY_VI[i]}={(this_monday + timedelta(days=i)).strftime('%Y-%m-%d')}" for i in range(7)
        )
        next_week_str = ", ".join(
            f"{WEEKDAY_VI[i]}={(next_monday + timedelta(days=i)).strftime('%Y-%m-%d')}" for i in range(7)
        )
        next_7_str = ", ".join(
            f"{WEEKDAY_VI[(dt + timedelta(days=i)).weekday()]} {(dt + timedelta(days=i)).strftime('%Y-%m-%d')}"
            for i in range(1, 8)
        )
        return (
            f"{current_time_iso} ({wd}, hôm qua={yesterday.strftime('%Y-%m-%d')}, "
            f"hôm nay={dt.strftime('%Y-%m-%d')}, "
            f"ngày mai={tomorrow.strftime('%Y-%m-%d')} [{WEEKDAY_VI[tomorrow.weekday()]}], "
            f"ngày kia/mốt={day_after.strftime('%Y-%m-%d')} [{WEEKDAY_VI[day_after.weekday()]}])\n"
            f"- Bảng ngày TUẦN NÀY: {this_week_str}\n"
            f"- Bảng ngày TUẦN SAU: {next_week_str}\n"
            f"- 7 ngày tới tính từ hôm nay: {next_7_str}"
        )
    except Exception:
        return str(current_time_iso)


def build_system_prompt(sample: dict) -> str:
    ctx = sample.get("context") or {}
    prof = ctx.get("user_profile") or {}
    loc = ctx.get("location") or {}
    installed = list((ctx.get("installed_apps") or {}).keys())
    time_str = format_temporal_anchor(ctx.get("current_time", ""))

    app_guides = [f"  • {APP_COLLOQUIAL_MAP[a]}" for a in installed if a in APP_COLLOQUIAL_MAP]
    app_guide_block = "\n".join(app_guides) if app_guides else "  • Chỉ dùng các app trong danh sách đã cài."

    return f"""Bạn là LifeMate — AI Agent thực thi hành động qua Tool Calling cho người dùng Việt Nam.
Hệ thống CHỈ chấm danh sách `tool_calls` (Precision + Recall) và `state_hash` cuối cùng, BỎ QUA lời thoại văn bản. Mọi lời gọi tool thừa (kể cả [READ] thừa hoặc lệnh lỗi) đều làm giảm Precision!

## BỐI CẢNH HIỆN TẠI & BẢNG NGÀY TRONG TUẦN
- Múi giờ: {ctx.get('timezone', 'Asia/Ho_Chi_Minh')} | Thời gian hiện tại: {time_str}
- Người dùng: {prof.get('name', '')} | Thành phố: {prof.get('city', '')}
- Địa chỉ nhà: {prof.get('home_address', '')} | Địa chỉ công ty: {prof.get('work_address', '')}
- Tọa độ hiện tại: lat={loc.get('lat')}, lng={loc.get('lng')}
- Ứng dụng đã cài: {', '.join(installed)}

## BẢNG ÁNH XẠ KHẨU NGỮ -> ỨNG DỤNG ĐÃ CÀI
{app_guide_block}

## QUY TẮC THỰC THI CỐT LÕI
1. **Đối chiếu Đa Giả thuyết ASR & Lịch sử (`history`)**: Kết hợp các `[Bản nghe ASR]` cùng ngữ cảnh `history` để khôi phục đúng tên người, số tiền, giờ giấc, mã cổ phiếu, địa điểm (kể cả giọng địa phương nói trại). Nếu ID thật (`contact_id`, `email_id`, `order_id`, `event_id`...) ĐÃ CÓ trong `history`, dùng trực tiếp không gọi lại `[READ]` thừa.
2. **Tra cứu ID thật trước khi `[WRITE]`**: Không tự bịa tham số `ref:...` (như `device_id`, `contact_id`, `chat_id`, `bill_id`, `product_id`, `trip_id`, `lot_id`, `account_id`, `beneficiary_id`). Khi gọi `robot_vacuum.start(rooms=[...])`, phải gọi `robot_vacuum.get_status({{}})` trước (trừ khi đã gọi trong `history`). Khi gọi các hàm liệt kê chung (`appliances.list`, `subscriptions.list`, `loyalty.list_vouchers`, `household_chores.list_chores`), ưu tiên truyền `{{}}` để xem toàn bộ danh sách.
3. **Giữ sát từ khóa gốc cho tham số văn bản (`title`, `content`, `subject`, `body`)**: Viết đầy đủ từ khóa chính mà người dùng đã nói (tên người nhận, tên việc/sự kiện, thời gian/trạng thái), không viết tắt cụt lủn hoặc tự bịa thêm tên riêng không được yêu cầu.
4. **Hoàn tất trọn vẹn chuỗi hành động & Tính toán chính xác**:
   - Mua hàng `grocery`/`ecommerce`: `search` -> `add_to_cart` -> `checkout` (chỉ truyền `address` nếu người dùng nêu địa chỉ/giao về nhà). Đơn `placed`/`shipping` dùng `cancel_order`; đơn `delivered` mới dùng `request_return`.
   - Khi có app `translator` và người dùng yêu cầu dịch hoặc nhắn tin/trả lời mail bằng ngoại ngữ: PHẢI gọi `translator.translate` trước rồi dùng kết quả dịch đó để gửi/lưu.
   - Khi dời lịch (`calendar.update_event`) hoặc đặt nhắc nhở trước giờ xuất phát/giờ chiếu: cộng/trừ chính xác từng phút từ kết quả tra cứu (`start`, `end`, `duration_min`), tuyệt đối không tự làm tròn phút.
5. **Quy tắc `sys.ask_missing_info`**: Khi kết quả tra cứu có >= 2 mục trùng (VD 2 liên hệ trùng tên, nhiều chuyến xe/bay cùng ngày mà không rõ tiêu chí chọn) hoặc thiếu tham số bắt buộc (`amount`, `time`, `quantity`, `hours`, `reason`, `contact_id`...), BẮT BUỘC gọi `sys.ask_missing_info(tool_name="<tool_đích>", params=[...])` (KHÔNG hỏi bằng văn bản). `tool_name` là tool đầu tiên trong chuỗi bị thiếu tham số bắt buộc.
6. **Điều kiện ("Nếu... thì...") & Đổi ý ("À khoan/thôi...") & Ngoài phạm vi**: Nếu điều kiện SAI thì dừng sau bước kiểm tra; nếu người nói tự hủy yêu cầu hoặc thiếu app cần thiết thì không gọi tool (`[]`)."""


def format_user_utterance(transcript_entry: dict[str, str]) -> str:
    """Format up to 3 ASR hypotheses (Fine-tuned Whisper-turbo, Fine-tuned PhoWhisper, and optional Base Whisper)."""
    primary = clean_asr_text(transcript_entry.get("primary"))
    secondary = clean_asr_text(transcript_entry.get("secondary"))
    base = clean_asr_text(transcript_entry.get("base"))

    seen_low: set[str] = set()
    lines: list[str] = []
    for idx, (label, txt) in enumerate(
        [("Bản nghe ASR 1", primary), ("Bản nghe ASR 2", secondary), ("Bản nghe ASR 3 (Base)", base)], 1
    ):
        if txt and txt.lower() not in seen_low:
            seen_low.add(txt.lower())
            lines.append((label, txt))

    if len(lines) <= 1:
        return lines[0][1] if lines else ""
    return "\n".join(f"[{label}]: {txt}" for label, txt in lines)


def build_messages(
    sample: dict,
    transcript_map: dict[str, dict[str, str]],
    max_history_turns: int = 12,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(sample)}]

    history = sample.get("history") or []
    if max_history_turns > 0 and len(history) > max_history_turns:
        history = history[-max_history_turns:]

    call_counter = 1
    for turn in history:
        role = turn.get("role")
        if role == "user":
            audio_rel = turn.get("audio", "")
            text = turn.get("audio_text")
            if not text and audio_rel in transcript_map:
                entry = transcript_map[audio_rel]
                text = format_user_utterance(entry)
            messages.append({"role": "user", "content": text or "[Người dùng nói trong lượt trước]"})
        elif role == "assistant":
            tcalls = turn.get("tool_calls") or []
            tresults = turn.get("tool_results") or []
            if tcalls:
                for c, r in zip(tcalls, tresults):
                    cid = f"hist_call_{call_counter}"
                    call_counter += 1
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": cid,
                                    "type": "function",
                                    "function": {
                                        "name": c["name"],
                                        "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": cid,
                            "name": c["name"],
                            "content": json.dumps(r, ensure_ascii=False),
                        }
                    )
                if turn.get("text"):
                    messages.append({"role": "assistant", "content": turn["text"]})
            else:
                messages.append({"role": "assistant", "content": turn.get("text") or ""})

    final_rel = sample["audio"]
    final_text = format_user_utterance(transcript_map.get(final_rel, {"primary": sample.get("audio_text") or ""}))
    messages.append({"role": "user", "content": final_text})
    return messages
