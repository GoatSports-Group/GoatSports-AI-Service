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
from app.application.port.client import (
    CallerIdentity,
    ExternalKnowledgeClient,
    PlatformDataClient,
    PlatformDataError,
    ToolCall,
)

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
CLUB_ROLE_LABEL = {"OWNER": "Chủ CLB", "ADMIN": "Quản lý", "MEMBER": "Thành viên"}
VENUE_SORT = {"gia_thap": "price", "danh_gia_cao": "rating"}
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
        "Tìm cơ sở sân thể thao đang hoạt động trên GoatSports theo tên, môn, khu vực (tỉnh/thành, quận, phường) và giá tối đa mỗi "
        "giờ. Trả về tổng số cơ sở khớp (tong_so) và 5 cơ sở đầu theo thứ tự đã chọn. Gọi không tham số để biết "
        "GoatSports có bao nhiêu sân.",
        {
            "tu_khoa": {"type": "string", "description": "Tên hoặc một phần tên cơ sở"},
            "mon": _SPORT_PARAM,
            "khu_vuc": {"type": "string",
                        "description": "Tỉnh/thành, quận/huyện hoặc phường, ví dụ 'Khánh Hòa', 'Nha Trang', 'Gò Vấp'"},
            "gia_toi_da": {"type": "number", "description": "Giá tối đa mỗi giờ (đồng)"},
            "sap_xep": {"type": "string", "enum": list(VENUE_SORT),
                        "description": "gia_thap: rẻ nhất trước; danh_gia_cao: đánh giá cao nhất trước"},
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
    _function(
        "clb_cua_toi",
        "Xem các câu lạc bộ người đang hỏi đang tham gia (vai trò, môn, số thành viên) và hoạt động CLB sắp tới.",
        {},
    ),
    _function(
        "tim_clb",
        "Tìm câu lạc bộ trên GoatSports theo môn, thành phố / tỉnh hoặc tên. Trả về tổng số CLB khớp và 5 CLB đầu.",
        {
            "mon": _SPORT_PARAM,
            "thanh_pho": {"type": "string", "description": "Thành phố / tỉnh, ví dụ 'Hồ Chí Minh'"},
            "tu_khoa": {"type": "string", "description": "Tên hoặc một phần tên CLB"},
        },
    ),
]


EXTERNAL_TOOL_SPEC = _function(
    "tra_cuu_ngoai",
    "Tra kiến thức thể thao CHUNG trên Wikipedia tiếng Việt: luật chơi, kỹ thuật, lịch sử môn, giải đấu và cầu thủ thế "
    "giới, địa danh. Chỉ dùng khi tài liệu và các công cụ GoatSports không có câu trả lời. KHÔNG dùng cho giá, chính "
    "sách, tính năng hay dữ liệu của GoatSports.",
    {
        "chu_de": {"type": "string", "description": "Chủ đề tiếng Việt, ngắn gọn, ví dụ 'luật việt vị bóng đá', 'futsal'"},
        "chu_de_tieng_anh": {"type": "string", "description": "Cùng chủ đề bằng tiếng Anh, ví dụ 'five-a-side football': "
                             "bài tiếng Việt thường thiếu, khi đó dùng bài tiếng Anh và trả lời bằng tiếng Việt"},
    },
    ["chu_de"],
)


class AssistantTools:
    def __init__(self, data: PlatformDataClient, external: ExternalKnowledgeClient | None = None) -> None:
        self.data = data
        self.external = external

    @property
    def specs(self) -> list[dict]:
        """Công cụ đưa cho LLM; tra nguồn ngoài chỉ có khi đã bật."""
        return TOOL_SPECS + ([EXTERNAL_TOOL_SPEC] if self.external else [])

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
            "clb_cua_toi": self._my_clubs,
            "tim_clb": self._clubs,
            "tra_cuu_ngoai": self._external if self.external else None,
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
        venues, total = self.data.search_venues(
            keyword=_text(args.get("tu_khoa")),
            sport_type=_sport(args.get("mon")),
            district=_text(args.get("khu_vuc")),
            max_price=_positive(args.get("gia_toi_da")),
            sort=VENUE_SORT.get(str(args.get("sap_xep") or "")),
            limit=_MAX_ROWS,
        )
        rows, cards = [], []
        for venue in venues[:_MAX_ROWS]:
            price = _price_range(venue.get("minPrice"), venue.get("maxPrice"))
            rating = _rating(venue.get("averageRating"), venue.get("totalReviews"))
            place = ", ".join(str(part) for part in (venue.get("address"), venue.get("district"), venue.get("city")) if part)
            rows.append({
                "ten": venue.get("name"), "dia_chi": place, "gia": price, "danh_gia": rating,
                "mon": [SPORT_LABEL.get(sport, sport) for sport in venue.get("sportTypes") or []],
            })
            cards.append(ChatbotCard(
                kind="VENUE", title=str(venue.get("name") or "Sân thể thao"), subtitle=place or None,
                details=[part for part in (price, rating) if part],
                link=f"/venues/{venue.get('venueId')}",
            ))
        return {"tong_so": total, "hien": len(rows), "co_so": rows}, cards

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


    # ---- tra_cuu_ngoai ----

    def _external(self, args, _identity, _today):
        topic = _text(args.get("chu_de"))
        if not topic:
            return {"loi": "Cần chủ đề để tra."}, []
        passages = self.external.lookup(topic, 2)
        english = _text(args.get("chu_de_tieng_anh"))
        if english:
            passages += self.external.lookup(english, 1, "en")
        passages = list({item.url: item for item in passages}.values())
        if not passages:
            return {"loi": f"Wikipedia không có bài phù hợp với '{topic}'."}, []
        rows = [{"tieu_de": item.title, "tom_tat": item.summary, "link": item.url} for item in passages]
        cards = [ChatbotCard(kind="WEB", title=item.title, subtitle=item.source,
                             details=[item.summary[:140].rsplit(" ", 1)[0] + "…" if len(item.summary) > 140 else item.summary],
                             link=item.url) for item in passages]
        return {"nguon": "Wikipedia", "luu_y": "Kiến thức chung từ nguồn ngoài, không phải thông tin của GoatSports. "
                "Bài tiếng Anh thì dịch ý sang tiếng Việt.",
                "bai": rows}, cards

    # ---- clb_cua_toi / tim_clb ----

    def _my_clubs(self, _args, identity, _today):
        if identity is None:
            return {"loi": "Người hỏi chưa đăng nhập. Hãy mời họ đăng nhập để xem câu lạc bộ của mình."}, []
        memberships, activities = self.data.my_clubs(identity, _MAX_ROWS)
        names = {}
        rows, cards = [], []
        for membership in memberships:
            club = membership.get("club") or {}
            names[club.get("clubId")] = club.get("name")
            role = CLUB_ROLE_LABEL.get(membership.get("role"), membership.get("role"))
            rows.append({"ten": club.get("name"), "mon": SPORT_LABEL.get(club.get("sportType"), club.get("sportType")),
                         "vai_tro": role, "thanh_pho": club.get("city"), "thanh_vien": club.get("memberCount")})
            cards.append(_club_card(club, role))
        upcoming = [{"clb": names.get(item.get("clubId")), "hoat_dong": item.get("title"),
                     "luc": _when(item.get("startAt"))} for item in activities[:_MAX_ROWS]]
        return {"so_clb": len(rows), "clb": rows[:_MAX_ROWS * 2], "hoat_dong_sap_toi": upcoming}, cards[:_MAX_ROWS]

    def _clubs(self, args, _identity, _today):
        clubs, total = self.data.search_clubs(
            sport_type=_sport(args.get("mon")), city=_text(args.get("thanh_pho")),
            keyword=_text(args.get("tu_khoa")), limit=_MAX_ROWS,
        )
        rows = [{"ten": club.get("name"), "mon": SPORT_LABEL.get(club.get("sportType"), club.get("sportType")),
                 "thanh_pho": club.get("city"), "thanh_vien": club.get("memberCount"),
                 "ti_le_thang": f"{round((club.get('winRate') or 0) * 100)}%" if club.get("matchCount") else None}
                for club in clubs[:_MAX_ROWS]]
        return {"tong_so": total, "hien": len(rows), "clb": rows}, [_club_card(club) for club in clubs[:_MAX_ROWS]]


def _club_card(club: dict, role: str | None = None) -> ChatbotCard:
    sport = SPORT_LABEL.get(club.get("sportType"), club.get("sportType"))
    return ChatbotCard(
        kind="CLUB", title=str(club.get("name") or "Câu lạc bộ"),
        subtitle=" · ".join(str(part) for part in (sport, club.get("city")) if part) or None,
        details=[part for part in (role, f"{club.get('memberCount')} thành viên" if club.get("memberCount") else None) if part],
        link=f"/clubs/{club.get('clubId')}",
    )


def _rating(average: Any, count: Any) -> str:
    """Kèm số lượt: 5.0 từ 1 đánh giá khác hẳn 4.5 từ 40 đánh giá."""
    try:
        reviews = int(count or 0)
        return f"{float(average):.1f}/5 ({reviews} đánh giá)" if reviews and average else "chưa có đánh giá"
    except (TypeError, ValueError):
        return "chưa có đánh giá"


def _when(value: Any) -> str:
    text = str(value or "")
    try:
        return f"{date.fromisoformat(text[:10]).strftime('%d/%m')} {text[11:16]}".strip()
    except ValueError:
        return "?"


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
