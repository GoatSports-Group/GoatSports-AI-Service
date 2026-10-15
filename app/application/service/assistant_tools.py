"""Công cụ chỉ đọc của trợ lý GOAT AI: LLM chọn công cụ, dịch vụ này kiểm tham số, đọc dữ liệu thật và trả
về (1) bản tóm tắt gọn cho LLM và (2) các thẻ để client hiện, bấm vào mở đúng trang.

Không công cụ nào đặt, hủy hay thanh toán. "Đơn của tôi" chỉ dùng danh tính do gateway gắn, không bao giờ dùng
mã người dùng do LLM điền.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

from app.application.dto.response.models import ChatbotCard
from app.application.port.client import CallerIdentity, PlatformDataClient, PlatformDataError, ToolCall

logger = logging.getLogger(__name__)

SPORTS = ["FOOTBALL", "BADMINTON", "TENNIS", "PICKLEBALL", "BASKETBALL", "VOLLEYBALL"]
SPORT_LABEL = {
    "FOOTBALL": "Bóng đá", "BADMINTON": "Cầu lông", "TENNIS": "Tennis",
    "PICKLEBALL": "Pickleball", "BASKETBALL": "Bóng rổ", "VOLLEYBALL": "Bóng chuyền",
}
TOURNAMENT_STATUS = {"dang_mo": "REGISTRATION_OPEN", "sap_mo": "PUBLISHED", "dang_dien_ra": "IN_PROGRESS"}
BOOKING_FILTER = {"sap_toi": "UPCOMING", "tat_ca": "ALL", "cho_hoan_tien": "REFUND_PENDING"}
BOOKING_STATUS_LABEL = {
    "PENDING_PAYMENT": "Chờ thanh toán", "CONFIRMED": "Đã xác nhận", "CHECKED_IN": "Đã nhận sân",
    "COMPLETED": "Hoàn thành", "CANCELLED": "Đã hủy", "REFUND_PENDING": "Chờ hoàn cọc", "REFUNDED": "Đã hoàn tiền",
    "EXPIRED": "Hết hạn", "NO_SHOW": "Không đến",
}
CANCELLATION_LABEL = {
    "PENDING": "chờ chủ sân duyệt", "APPROVED": "đã chấp thuận", "REJECTED": "bị từ chối",
    "REFUND_AWAITING_CLAIM": "chờ bạn bấm Nhận hoàn tiền", "REFUND_AWAITING_BANK_ACCOUNT": "chờ bạn liên kết ngân hàng",
    "REFUND_PROCESSING": "đang chuyển tiền hoàn", "REFUND_MANUAL_REVIEW": "đang được kiểm tra thủ công",
    "REFUNDED": "đã hoàn tiền", "REFUND_FAILED": "chuyển tiền hoàn lỗi, cần bấm Thử hoàn tiền lại",
}
TOURNAMENT_STATUS_LABEL = {
    "PUBLISHED": "Sắp mở đăng ký", "REGISTRATION_OPEN": "Đang mở đăng ký", "REGISTRATION_CLOSED": "Đã đóng đăng ký",
    "IN_PROGRESS": "Đang diễn ra", "COMPLETED": "Đã kết thúc", "CANCELLED": "Đã hủy",
}
_MAX_ROWS = 5
_MAX_DAYS_AHEAD = 60


def _function(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


_SPORT_PARAM = {"type": "string", "enum": SPORTS, "description": "Môn thể thao"}

TOOL_SPECS: list[dict] = [
    _function(
        "tim_san",
        "Tìm cơ sở sân thể thao đang hoạt động trên GoatSports theo tên, môn, khu vực (quận/huyện) và giá tối đa mỗi giờ.",
        {
            "tu_khoa": {"type": "string", "description": "Tên hoặc một phần tên cơ sở"},
            "mon": _SPORT_PARAM,
            "khu_vuc": {"type": "string", "description": "Quận/huyện, ví dụ 'Gò Vấp'"},
            "gia_toi_da": {"type": "number", "description": "Giá tối đa mỗi giờ (đồng)"},
        },
    ),
    _function(
        "gio_trong",
        "Xem các khung giờ còn trống của một cơ sở sân trong một ngày.",
        {
            "ten_san": {"type": "string", "description": "Tên cơ sở sân"},
            "ngay": {"type": "string", "description": "Ngày chơi, dạng YYYY-MM-DD"},
            "mon": _SPORT_PARAM,
        },
        ["ten_san", "ngay"],
    ),
    _function(
        "giai_dau",
        "Tìm giải đấu trên GoatSports theo môn và trạng thái (đang mở đăng ký, sắp mở, đang diễn ra).",
        {
            "mon": _SPORT_PARAM,
            "trang_thai": {"type": "string", "enum": list(TOURNAMENT_STATUS), "description": "Mặc định dang_mo"},
            "tu_khoa": {"type": "string", "description": "Tên hoặc một phần tên giải"},
        },
    ),
    _function(
        "don_cua_toi",
        "Xem đơn đặt sân của chính người đang hỏi: trạng thái đơn, tiền cọc, yêu cầu hủy và tiền hoàn.",
        {"loai": {"type": "string", "enum": list(BOOKING_FILTER), "description": "Mặc định sap_toi"}},
    ),
]


class AssistantTools:
    def __init__(self, data: PlatformDataClient) -> None:
        self.data = data

    def execute(
        self, call: ToolCall, identity: CallerIdentity | None, today: date
    ) -> tuple[dict[str, Any], list[ChatbotCard]]:
        """Kết quả gọn cho LLM và thẻ cho client. Lỗi tham số hay lỗi dịch vụ trả về dạng {"loi": ...} để LLM nói lại."""
        try:
            args = json.loads(call.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return {"loi": "Tham số không hợp lệ."}, []
        handler = {
            "tim_san": self._venues,
            "gio_trong": self._slots,
            "giai_dau": self._tournaments,
            "don_cua_toi": self._bookings,
        }.get(call.name)
        if handler is None:
            return {"loi": f"Không có công cụ {call.name}."}, []
        try:
            return handler(args, identity, today)
        except PlatformDataError:
            return {"loi": "Chưa lấy được dữ liệu từ hệ thống lúc này, hãy thử lại sau."}, []
        except Exception:  # noqa: BLE001 - một công cụ lỗi không được làm hỏng cả câu trả lời
            logger.exception("Assistant tool %s failed", call.name)
            return {"loi": "Chưa lấy được dữ liệu từ hệ thống lúc này, hãy thử lại sau."}, []

    # ---- tim_san ----

    def _venues(self, args, _identity, _today):
        venues = self.data.search_venues(
            keyword=_text(args.get("tu_khoa")),
            sport_type=_sport(args.get("mon")),
            district=_text(args.get("khu_vuc")),
            max_price=_positive(args.get("gia_toi_da")),
            limit=_MAX_ROWS,
        )
        rows, cards = [], []
        for venue in venues[:_MAX_ROWS]:
            price = _price_range(venue.get("minPrice"), venue.get("maxPrice"))
            rating = venue.get("averageRating")
            place = ", ".join(str(part) for part in (venue.get("address"), venue.get("district")) if part)
            rows.append({
                "ten": venue.get("name"), "dia_chi": place, "gia": price,
                "danh_gia": f"{rating:.1f}/5" if rating else "chưa có",
                "mon": [SPORT_LABEL.get(sport, sport) for sport in venue.get("sportTypes") or []],
            })
            cards.append(ChatbotCard(
                kind="VENUE", title=str(venue.get("name") or "Sân thể thao"), subtitle=place or None,
                details=[part for part in (price, f"★ {rating:.1f}" if rating else None) if part],
                link=f"/venues/{venue.get('venueId')}",
            ))
        return {"so_co_so": len(rows), "co_so": rows}, cards

    # ---- gio_trong ----

    def _slots(self, args, _identity, today):
        name = _text(args.get("ten_san"))
        if not name:
            return {"loi": "Cần tên cơ sở sân."}, []
        try:
            play_date = date.fromisoformat(str(args.get("ngay")))
        except ValueError:
            return {"loi": "Ngày không hợp lệ, cần dạng YYYY-MM-DD."}, []
        if play_date < today or play_date > today + timedelta(days=_MAX_DAYS_AHEAD):
            return {"loi": f"Chỉ xem được lịch từ hôm nay tới {_MAX_DAYS_AHEAD} ngày tới."}, []
        found = self.data.free_slots(name, play_date, _sport(args.get("mon")))
        if not found:
            return {"loi": f"Không tìm thấy cơ sở nào tên giống '{name}'."}, []
        courts = [
            {"san": court["court"], "gio_trong": [_slot_label(slot) for slot in court["slots"]]}
            for court in found["courts"]
        ]
        free = [court for court in courts if court["gio_trong"]]
        day = play_date.strftime("%d/%m/%Y")
        card = ChatbotCard(
            kind="SLOTS",
            title=f"{found['venue_name']} · {day}",
            subtitle=f"{len(free)}/{len(courts)} sân còn giờ trống" if courts else "Chưa có sân phù hợp",
            details=[f"{court['san']}: {', '.join(court['gio_trong'][:6])}" for court in free[:4]],
            link=f"/venues/{found['venue_id']}",
        )
        return {"co_so": found["venue_name"], "ngay": play_date.isoformat(), "san": courts}, [card]

    # ---- giai_dau ----

    def _tournaments(self, args, _identity, _today):
        status = TOURNAMENT_STATUS.get(str(args.get("trang_thai") or "dang_mo"), "REGISTRATION_OPEN")
        tournaments = self.data.search_tournaments(
            sport_type=_sport(args.get("mon")), status=status, keyword=_text(args.get("tu_khoa")), limit=_MAX_ROWS
        )
        rows, cards = [], []
        for item in tournaments[:_MAX_ROWS]:
            fee = item.get("entryFee") or 0
            seats = f"{item.get('currentParticipants', 0)}/{item.get('maxParticipants') or '?'} suất"
            dates = f"{_dmy(item.get('startDate'))}–{_dmy(item.get('endDate'))}"
            fee_label = "Miễn phí" if not fee else f"{fee:,.0f}đ".replace(",", ".")
            rows.append({
                "ten": item.get("name"), "mon": SPORT_LABEL.get(item.get("sportType"), item.get("sportType")),
                "hinh_thuc": item.get("playFormatLabel"), "thi_dau": dates, "le_phi": fee_label, "suat": seats,
                "han_dang_ky": _dmy(item.get("registrationCloseDate")),
                "trang_thai": TOURNAMENT_STATUS_LABEL.get(item.get("status"), item.get("status")),
            })
            cards.append(ChatbotCard(
                kind="TOURNAMENT", title=str(item.get("name") or "Giải đấu"),
                subtitle=" · ".join(part for part in (SPORT_LABEL.get(item.get("sportType")), item.get("playFormatLabel")) if part) or None,
                details=[f"Thi đấu {dates}", f"{fee_label} · {seats}", f"Hạn đăng ký {_dmy(item.get('registrationCloseDate'))}"],
                link=f"/tournaments/{item.get('tournamentId')}",
            ))
        return {"so_giai": len(rows), "giai": rows}, cards

    # ---- don_cua_toi ----

    def _bookings(self, args, identity, _today):
        if identity is None:
            return {"loi": "Người hỏi chưa đăng nhập. Hãy mời họ đăng nhập để xem đơn đặt sân của mình."}, []
        status = BOOKING_FILTER.get(str(args.get("loai") or "sap_toi"), "UPCOMING")
        bookings = self.data.my_bookings(identity, status, _MAX_ROWS)
        rows, cards = [], []
        for booking in bookings[:_MAX_ROWS]:
            when = f"{_dmy(booking.get('playDate'))} {str(booking.get('startTime') or '')[:5]}–{str(booking.get('endTime') or '')[:5]}"
            label = BOOKING_STATUS_LABEL.get(booking.get("status"), booking.get("status"))
            cancellation = booking.get("cancellation") or {}
            refund = None
            if cancellation:
                refund = f"Yêu cầu hủy {CANCELLATION_LABEL.get(cancellation.get('status'), cancellation.get('status'))}"
                if cancellation.get("refundAmount"):
                    refund += f", hoàn {_money(cancellation['refundAmount'])}"
            rows.append({
                "ma": booking.get("bookingCode"), "co_so": booking.get("venueName"), "san": booking.get("courtName"),
                "gio": when, "trang_thai": label, "da_coc": _money(booking.get("depositAmount")),
                "tra_tai_san": _money(booking.get("remainingAmount")), "huy_va_hoan_tien": refund,
            })
            cards.append(ChatbotCard(
                kind="BOOKING", title=f"{booking.get('venueName')} · {booking.get('courtName')}",
                subtitle=f"{label} · {when}",
                details=[part for part in (f"Mã {booking.get('bookingCode')}", refund) if part],
                link=f"/booking/detail/{booking.get('bookingId')}",
            ))
        return {"so_don": len(rows), "don": rows}, cards


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text[:100] or None


def _sport(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in SPORTS else None


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _money(value: Any) -> str | None:
    try:
        return f"{float(value):,.0f}đ".replace(",", ".")
    except (TypeError, ValueError):
        return None


def _price_range(low: Any, high: Any) -> str | None:
    if low is None and high is None:
        return None
    if low is not None and high is not None and float(low) != float(high):
        return f"{_money(low)}–{_money(high)}/giờ"
    return f"{_money(low if low is not None else high)}/giờ"


def _dmy(value: Any) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%d/%m")
    except ValueError:
        return "?"


def _slot_label(slot: dict) -> str:
    return f"{str(slot['start'])[:5]}–{str(slot['end'])[:5]}"
