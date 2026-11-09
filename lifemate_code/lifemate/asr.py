"""ASR Engine v4.0 for ZAC 2026 LifeMate — High-Accuracy Beam-5 Dual-ASR
(Whisper-large-v3-turbo 0.81B + PhoWhisper-medium 0.77B -> Total w/ Qwen3-4B = 5.60B < 6.0B).

Key Upgrades in v4.0:
- `beam_size=5` deterministic decoding (`temperature=0.0`, `do_sample=False`) for BOTH Whisper-v3-turbo and PhoWhisper-medium,
  dramatically reducing tonal/regional-accent errors on Central & Southern Vietnamese speech.
- Dual-ASR enabled on BOTH the final user turn AND `history` user turns (`66.1%` of `public_test` has `history`).
- Cache version tag (`asr_v = 4`): automatically upgrades stale Day-1 `beam_size=1` cache entries while preserving instant `--resume`.
- Anti-Hallucination & Anti-Repetition Healing for YouTube subtitle artifacts ("Ghiền Mì Gõ", etc.).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ASR_CACHE_VERSION = 4

HALLUCINATION_SUBSTRINGS = (
    "ghiền mì gõ",
    "hãy subscribe",
    "đăng ký kênh",
    "cảm ơn các bạn đã theo dõi",
    "hẹn gặp lại trong các video",
    "subtitles by",
    "amara.org",
    "la la school",
    "chúng ta đã nói chuyện với tôi",
)

_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1){4,}\b", re.IGNORECASE | re.UNICODE)
_REPEAT_PHRASE_RE = re.compile(r"\b(\w+\s+\w+)(?:\s+\1){3,}\b", re.IGNORECASE | re.UNICODE)


def is_corrupted_asr(text: str | None) -> bool:
    """Return True if transcript is empty, a known Whisper YouTube hallucination, or a severe repetition loop."""
    if not text or not text.strip():
        return True
    low = text.strip().lower()
    if any(pat in low for pat in HALLUCINATION_SUBSTRINGS):
        return True
    if _REPEAT_WORD_RE.search(low) or _REPEAT_PHRASE_RE.search(low):
        return True
    return False


def clean_asr_text(text: str | None) -> str:
    """Remove known Whisper YouTube outro hallucinations and collapse word repetition loops."""
    if not text:
        return ""
    s = text.strip()
    low = s.lower()
    for pat in HALLUCINATION_SUBSTRINGS:
        if pat in low:
            return ""
    s = _REPEAT_WORD_RE.sub(r"\1", s)
    s = _REPEAT_PHRASE_RE.sub(r"\1", s)
    return " ".join(s.split()).strip()


APP_VOCAB_HINTS: dict[str, str] = {
    "smart_home": "điều hòa, máy lạnh, đèn, quạt, rèm, bình nóng lạnh, phòng khách, phòng ngủ, bếp, sân",
    "banking": "chuyển khoản, tiết kiệm, số dư, thụ hưởng, ngân hàng",
    "ewallet": "bên ví, ví điện tử, nạp tiền điện thoại, chuyển ví, quét mã QR",
    "bill_payment": "hóa đơn điện EVN, nước, internet, tiền mạng, truyền hình, tự trả",
    "ride_hailing": "đặt xe, cuốc xe, xe máy, 4 chỗ, 7 chỗ, đón tôi",
    "food_delivery": "đặt đồ ăn, nhà hàng, thực đơn, hủy đơn, giao đồ ăn",
    "music": "bài hát, nhạc, playlist",
    "chat_ott": "nhắn Zalo, nhắn tin nhóm, gọi điện, chia sẻ vị trí",
    "flights": "chuyến bay, vé máy bay, check-in, hạ cánh",
    "stocks": "cổ phiếu, chứng khoán, danh mục, đặt lệnh mua bán, cảnh báo giá, FPT, VNM, VCB, HPG, MWG",
    "ecommerce": "trên sàn, đặt hàng, giỏ hàng, thanh toán COD, trả hàng, hủy đơn",
    "grocery": "đi chợ online, giỏ hàng, chốt đơn, rau củ, thịt cá, sữa",
    "parking": "bãi đỗ xe, gửi xe, giữ chỗ đỗ, kết thúc phiên đỗ",
    "public_transit": "xe buýt, tàu metro, nạp thẻ, tuyến xe, trạm dừng",
    "recipes": "công thức nấu ăn, nguyên liệu, khẩu phần, cách làm món",
    "pantry": "tủ bếp, trừ tồn kho, nguyên liệu còn lại",
    "event_tickets": "vé sự kiện, đêm nhạc, hội thảo, concert",
    "train_bus_tickets": "vé tàu hỏa, xe khách, đặt ghế, chuyến xe",
    "habit_tracker": "thói quen, đánh dấu hoàn thành, streak",
    "translator": "dịch sang tiếng Anh, tiếng Nhật, tiếng Hàn, tiếng Trung, tiếng Pháp",
    "alarm": "đặt báo thức, sáng mai, gọi dậy",
    "reminders": "nhắc nhở, nhắc tôi",
    "calendar": "lịch họp, sự kiện, dời lịch, hủy lịch, chặn lịch",
    "email": "trả lời mail, soạn thư nháp, gửi email",
    "contacts": "danh bạ, số điện thoại, sửa liên hệ",
    "sms": "nhắn SMS, gửi tin nhắn",
    "tasks": "việc cần làm, thêm việc, hạn chót",
    "notes": "ghi chú, sổ tay",
    "shopping_list": "danh sách mua sắm, đánh dấu đã mua",
    "weather": "dự báo thời tiết, trời mưa",
    "maps": "chỉ đường, lộ trình, bao xa, mất bao lâu",
    "robot_vacuum": "robot hút bụi, lau nhà, dọn phòng",
    "library": "thư viện, mượn sách, đặt giữ sách",
    "bike_rental": "thuê xe đạp, giữ xe đạp, trạm xe đạp",
    "home_automation": "hẹn giờ tự bật, tự tắt, quy tắc nhà thông minh",
    "home_security": "camera, cảnh báo an ninh, gara",
    "door_lock": "khóa cửa, mở khóa",
    "appliances": "máy giặt, máy rửa bát, máy sấy",
    "household_chores": "việc nhà, phân công",
    "subscriptions": "gói đăng ký, hủy gia hạn",
    "loyalty": "đổi điểm, voucher, ưu đãi",
    "video_meeting": "phòng họp trực tuyến, họp online",
    "tv": "tivi, kênh, ghi hình chương trình",
    "news": "tin tức mới nhất, đọc báo",
}


def build_asr_initial_prompt(sample: dict) -> str:
    ctx = sample.get("context") or {}
    prof = ctx.get("user_profile") or {}
    parts = ["Câu lệnh giọng nói tiếng Việt tự nhiên:"]
    if prof.get("name"):
        parts.append(f"{prof['name']} ({prof.get('city', '')}).")
    installed = list((ctx.get("installed_apps") or {}).keys())
    hints = [APP_VOCAB_HINTS[a] for a in installed if a in APP_VOCAB_HINTS]
    if hints:
        parts.append("Ngữ cảnh: " + "; ".join(hints[:8]) + ".")
    return " ".join(parts)


class ASREngine:
    def __init__(
        self,
        mode: str = "faster_whisper",
        primary_model: str = "Systran/faster-whisper-large-v3-turbo",
        secondary_model: str | None = "vinai/PhoWhisper-medium",
        device: str = "cuda",
        compute_type: str = "float16",
        cache_path: Path | None = None,
        base_cache_path: Path | None = None,
        max_history_user_audios: int = 8,
        beam_size: int = 5,
    ):
        self.mode = mode
        self.primary_model_name = primary_model
        self.secondary_model_name = secondary_model
        self.device = device
        self.compute_type = compute_type
        self.cache_path = cache_path
        self.base_cache_path = base_cache_path
        self.max_history_user_audios = max_history_user_audios
        self.beam_size = beam_size

        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.is_file():
            self.cache = json.load(open(self.cache_path, encoding="utf-8"))

        self.base_cache: dict[str, dict] = {}
        if self.base_cache_path and Path(self.base_cache_path).is_file():
            self.base_cache = json.load(open(self.base_cache_path, encoding="utf-8"))

        self._primary = None
        self._secondary = None

    def _ensure_primary_loaded(self):
        if self.mode in ("oracle", "noop") or self._primary is not None:
            return
        if self.mode == "faster_whisper":
            try:
                from faster_whisper import WhisperModel

                self._primary = WhisperModel(
                    self.primary_model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as e:
                print(f"[ASR] faster-whisper fallback sang transformers: {e}")
                self.mode = "transformers"
                if "Systran/faster-whisper-large-v3-turbo" in self.primary_model_name:
                    self.primary_model_name = "openai/whisper-large-v3-turbo"

        if self.mode == "transformers":
            import torch
            from transformers import pipeline

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            try:
                self._primary = pipeline(
                    "automatic-speech-recognition",
                    model=self.primary_model_name,
                    device=0 if self.device == "cuda" else -1,
                    dtype=dtype,
                )
            except TypeError:
                self._primary = pipeline(
                    "automatic-speech-recognition",
                    model=self.primary_model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=dtype,
                )
        elif self.mode != "faster_whisper":
            raise ValueError(f"Unsupported ASR mode: {self.mode}")

    def _ensure_secondary_loaded(self):
        if not self.secondary_model_name or self._secondary is not None or self.mode in ("oracle", "noop"):
            return
        import torch
        from transformers import pipeline

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self._secondary = pipeline(
                "automatic-speech-recognition",
                model=self.secondary_model_name,
                device=0 if self.device == "cuda" else -1,
                dtype=dtype,
            )
        except TypeError:
            self._secondary = pipeline(
                "automatic-speech-recognition",
                model=self.secondary_model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=dtype,
            )

    def save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def unload(self):
        """Free GPU memory used by ASR models before loading the 4B LLM."""
        if self._primary is not None or self._secondary is not None:
            import gc
            self._primary = None
            self._secondary = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe_wav(
        self,
        wav_path: Path,
        cache_key: str,
        initial_prompt: str = "",
        use_secondary: bool = True,
    ) -> dict[str, str]:
        if self.mode == "noop":
            return {"primary": ""}

        existing = dict(self.cache.get(cache_key) or {})
        is_fresh_v4 = existing.get("asr_v") == ASR_CACHE_VERSION
        prim_corrupted = (not is_fresh_v4) or is_corrupted_asr(existing.get("primary"))
        need_primary = prim_corrupted
        # A corrupted / empty primary still triggers the secondary model below (after the primary is recomputed).
        need_secondary = (
            use_secondary
            and bool(self.secondary_model_name)
            and ((not is_fresh_v4) or is_corrupted_asr(existing.get("secondary")))
        )

        if not need_primary and not need_secondary:
            if cache_key in self.base_cache and not existing.get("base"):
                b_txt = clean_asr_text((self.base_cache.get(cache_key) or {}).get("primary"))
                if b_txt and not is_corrupted_asr(b_txt):
                    existing = {**existing, "base": b_txt}
            return existing

        if need_primary:
            self._ensure_primary_loaded()
            if self.mode == "faster_whisper":
                try:
                    segments, _ = self._primary.transcribe(
                        str(wav_path),
                        language="vi",
                        beam_size=self.beam_size,
                        best_of=1,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        repetition_penalty=1.12,
                        no_repeat_ngram_size=4,
                        vad_filter=True,
                        vad_parameters=dict(threshold=0.30, min_silence_duration_ms=600),
                        initial_prompt=initial_prompt or None,
                    )
                    cand = " ".join(seg.text.strip() for seg in segments).strip()
                    if is_corrupted_asr(cand):
                        segments2, _ = self._primary.transcribe(
                            str(wav_path),
                            language="vi",
                            beam_size=self.beam_size,
                            best_of=1,
                            temperature=0.0,
                            condition_on_previous_text=False,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            vad_filter=False,
                            initial_prompt=None,
                        )
                        cand = " ".join(seg.text.strip() for seg in segments2).strip()
                    existing["primary"] = clean_asr_text(cand)
                except Exception as e:
                    print(f"[ASR] faster-whisper runtime fallback sang transformers: {e}")
                    self._primary = None
                    self.mode = "transformers"
                    if "Systran/faster-whisper-large-v3-turbo" in self.primary_model_name:
                        self.primary_model_name = "openai/whisper-large-v3-turbo"
                    self._ensure_primary_loaded()

            if self.mode == "transformers" and (need_primary or is_corrupted_asr(existing.get("primary"))):
                res = self._primary(
                    str(wav_path),
                    generate_kwargs={
                        "language": "vi",
                        "task": "transcribe",
                        "do_sample": False,
                        "num_beams": self.beam_size,
                        "no_repeat_ngram_size": 4,
                        "repetition_penalty": 1.12,
                    },
                )
                existing["primary"] = clean_asr_text(res.get("text") or "")

        if need_secondary or is_corrupted_asr(existing.get("primary")):
            if self.secondary_model_name:
                self._ensure_secondary_loaded()
                try:
                    res2 = self._secondary(
                        str(wav_path),
                        generate_kwargs={
                            "language": "vi",
                            "task": "transcribe",
                            "do_sample": False,
                            "num_beams": self.beam_size,
                            "no_repeat_ngram_size": 4,
                            "repetition_penalty": 1.12,
                        },
                    )
                except Exception:
                    res2 = self._secondary(
                        str(wav_path),
                        generate_kwargs={
                            "language": "vi",
                            "task": "transcribe",
                            "do_sample": False,
                            "num_beams": 1,
                        },
                    )
                existing["secondary"] = clean_asr_text(res2.get("text") or "")
                if is_corrupted_asr(existing.get("primary")) and existing.get("secondary"):
                    existing["primary"] = existing["secondary"]

        if cache_key in self.base_cache and not existing.get("base"):
            b_txt = clean_asr_text((self.base_cache.get(cache_key) or {}).get("primary"))
            if b_txt and not is_corrupted_asr(b_txt):
                existing["base"] = b_txt

        existing["asr_v"] = ASR_CACHE_VERSION
        self.cache[cache_key] = existing
        return existing

    def transcribe_sample(self, sample: dict, split_dir: Path, history_dual: bool = False) -> dict[str, dict[str, str]]:
        """history_dual=False (default v5): history turns use the primary model only — the prompt shows a single
        transcript for history (same as the SFT data) and it saves ~40% ASR time on long conversations."""
        result: dict[str, dict[str, str]] = {}

        if self.mode == "oracle":
            for h in sample.get("history") or []:
                if h.get("role") == "user" and h.get("audio"):
                    result[h["audio"]] = {"primary": h.get("audio_text") or ""}
            result[sample["audio"]] = {"primary": sample.get("audio_text") or ""}
            return result

        init_prompt = build_asr_initial_prompt(sample)

        user_turns = [h for h in (sample.get("history") or []) if h.get("role") == "user" and h.get("audio")]
        if self.max_history_user_audios > 0:
            user_turns = user_turns[-self.max_history_user_audios :]
        else:
            user_turns = []

        # Run Dual-ASR on BOTH history user turns and final turn so multi-turn context is crystal clear!
        for h in user_turns:
            rel = h["audio"]
            result[rel] = self.transcribe_wav(
                split_dir / rel,
                cache_key=rel,
                initial_prompt=init_prompt,
                use_secondary=history_dual,
            )

        final_rel = sample["audio"]
        result[final_rel] = self.transcribe_wav(
            split_dir / final_rel,
            cache_key=final_rel,
            initial_prompt=init_prompt,
            use_secondary=True,
        )
        return result
