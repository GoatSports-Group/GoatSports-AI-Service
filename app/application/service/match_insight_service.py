"""Phân tích kèo bằng LLM (Ollama): vì sao hai bên hợp nhau, điều cần lưu ý, nhận xét người khác từng để lại.

Chạy nền sau khi ghép. LLM chạy CPU mất khoảng 1–2 phút, nên kèo hiện ngay và phần phân tích đến sau. Mỗi lúc
chỉ một yêu cầu tới LLM để không làm nghẽn máy.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from uuid import UUID

from app.application.port.client import LanguageModelClient
from app.application.port.persistence import MatchmakingSessionRepository
from app.domain.entity.models import ClubMatchProfile, MatchmakingSession, MatchParticipantProfile, PlayerProfile

logger = logging.getLogger(__name__)

PENDING, READY, FAILED, DISABLED = "PENDING", "READY", "FAILED", "DISABLED"
# Kèo cũ hơn mức này không tự đi phân tích (tránh dồn LLM cho cả lịch sử khi mới bật tính năng).
_FRESH_FOR = timedelta(hours=6)
_TERMINAL = {"REJECTED", "EXPIRED", "CANCELLED"}

_SPORT = {
    "BADMINTON": "cầu lông", "PICKLEBALL": "pickleball", "TENNIS": "tennis",
    "FOOTBALL": "bóng đá", "BASKETBALL": "bóng rổ", "VOLLEYBALL": "bóng chuyền",
}
_SKILL = {"BEGINNER": "mới chơi", "INTERMEDIATE": "trung bình", "ADVANCED": "khá", "PRO": "nâng cao"}
_STYLE = {"BALANCED": "cân bằng", "FAIR_PLAY": "ưu tiên fair-play", "COMPETITIVE": "cạnh tranh"}

SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích kèo thể thao của GoatSports. Viết bằng tiếng Việt, đúng 3 câu ngắn (tổng dưới 70 chữ), "
    "giọng thân thiện, xưng hô 'hai bên'. Chỉ dùng dữ kiện được cung cấp, không bịa số liệu hay tên. Không dùng markdown, tiêu đề hay "
    "gạch đầu dòng. Nêu vì sao hai bên hợp nhau, một điều cần lưu ý nếu có (ví dụ chênh ELO, nhận xét fair-play), "
    "và một gợi ý ngắn về chuẩn bị trước trận (khởi động, đến sớm, thống nhất luật và số set qua trò chuyện). "
    "Tuyệt đối không nhắc tới cá cược hay tiền thắng thua, không đoán ai sẽ thắng."
)

# Lưới an toàn sau prompt: model nhỏ đôi khi vẫn lỡ lời, câu nào dính các từ này bị bỏ.
_BANNED = ("cược", "cá độ", "cá cược", "kèo nhà cái", "bet")
# Trên CPU, thời gian viết tỷ lệ với số token: 3 câu ngắn đủ cho phần phân tích.
_MAX_TOKENS = 160


class MatchInsightService:
    def __init__(
        self,
        session_repository: MatchmakingSessionRepository,
        language_model_client: LanguageModelClient,
        enabled: bool = True,
    ) -> None:
        self.session_repository = session_repository
        self.language_model_client = language_model_client
        self.enabled = enabled
        self._llm_lock = threading.Lock()
        self._in_flight: set[UUID] = set()
        self._guard = threading.Lock()

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.language_model_client.model_name)

    def claim(self, session: MatchmakingSession) -> bool:
        """Đánh dấu phiên cần phân tích; trả True nếu người gọi nên chạy `explain` ở nền."""
        if session.ai_insight_status in {READY, FAILED, DISABLED}:
            return False
        if session.status.value in _TERMINAL or not self._fresh(session):
            return False
        if not self.available:
            session.ai_insight_status = DISABLED
            self.session_repository.update_insight(session.session_id, None, DISABLED, None)
            return False
        with self._guard:
            if session.session_id in self._in_flight:
                session.ai_insight_status = PENDING
                return False
            self._in_flight.add(session.session_id)
        if session.ai_insight_status != PENDING:
            session.ai_insight_status = PENDING
            self.session_repository.update_insight(session.session_id, None, PENDING, None)
        return True

    def explain(self, session_id: UUID) -> None:
        try:
            with self._llm_lock:
                session = self.session_repository.find_by_id(session_id)
                if session is None or session.ai_insight_status == READY:
                    return
                text = self.language_model_client.complete(
                    SYSTEM_PROMPT, self._prompt(session), max_tokens=_MAX_TOKENS
                )
                cleaned = _clean(text)
                if cleaned:
                    self.session_repository.update_insight(
                        session_id, cleaned, READY, self.language_model_client.model_name
                    )
                else:
                    self.session_repository.update_insight(session_id, None, FAILED, None)
        except Exception:  # noqa: BLE001 - phân tích là phần phụ, không được làm hỏng luồng ghép
            logger.warning("Match insight failed for session %s", session_id, exc_info=True)
            self.session_repository.update_insight(session_id, None, FAILED, None)
        finally:
            with self._guard:
                self._in_flight.discard(session_id)

    def _prompt(self, session: MatchmakingSession) -> str:
        first, second = session.participants
        lines = [
            f"Môn: {_SPORT.get(session.sport_type.value, session.sport_type.value)}"
            + (f", hình thức {first.play_format}" if first.play_format else "")
            + f". Ngày {session.play_date.strftime('%d/%m/%Y')}, khung rảnh chung "
            f"{session.start_time.strftime('%H:%M')}–{session.end_time.strftime('%H:%M')}. "
            f"Khoảng cách giữa hai bên {session.distance_km:.1f} km, chênh {session.elo_difference} ELO.",
            self._side("Bên thứ nhất", first),
            self._side("Bên thứ hai", second),
        ]
        breakdown = session.score_breakdown
        if breakdown:
            lines.append(
                f"Điểm phù hợp (0–100): ELO {breakdown.elo_score:.0f}, khung giờ {breakdown.schedule_score:.0f}, "
                f"khoảng cách {breakdown.distance_score:.0f}, trình độ {breakdown.skill_score:.0f}, "
                f"phong cách {breakdown.play_style_score:.0f}, kinh nghiệm {breakdown.experience_score:.0f}; "
                f"tổng {breakdown.total_score:.0f}."
            )
            if breakdown.success_probability is not None:
                lines.append(f"Mô hình học máy dự đoán {breakdown.success_probability:.0f}% kèo diễn ra suôn sẻ.")
        return "\n".join(lines)

    def _side(self, title: str, side: MatchParticipantProfile) -> str:
        if isinstance(side, ClubMatchProfile):
            text = f"{title}: CLB {side.name}, ELO CLB {side.elo_rating}, {side.match_count} trận giao hữu."
        else:
            text = f"{title}: {side.name}, ELO {side.elo_rating}"
            if isinstance(side, PlayerProfile):
                text += (
                    f", trình độ {_SKILL.get(side.skill_level.value, side.skill_level.value)}, "
                    f"phong cách {_STYLE.get(side.play_style, side.play_style)}, {side.match_count} trận, "
                    f"tỷ lệ thắng {round(side.win_rate * 100)}%"
                )
            if len(side.members) > 1:
                text += f", đội {len(side.members)} người"
            text += "."
        reviews = self._reviews_about(side)
        if reviews:
            text += " Nhận xét từ đối thủ cũ: " + "; ".join(reviews) + "."
        return text

    def _reviews_about(self, side: MatchParticipantProfile) -> list[str]:
        reviews: list[str] = []
        for past in self.session_repository.find_by_participant(side.participant_id, 20):
            for item in past.feedback:
                if item.reviewee_id != side.participant_id:
                    continue
                note = f"hài lòng {item.rating}/5, fair-play {item.fair_play_rating}/5"
                if item.comment:
                    note += f" ('{item.comment[:160]}')"
                reviews.append(note)
                if len(reviews) >= 5:
                    return reviews
        return reviews

    @staticmethod
    def _fresh(session: MatchmakingSession) -> bool:
        matched = session.matched_at if session.matched_at.tzinfo else session.matched_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - matched <= _FRESH_FOR


def _clean(text: str | None) -> str | None:
    """Bỏ phần suy nghĩ (<think>…</think>) và định dạng thừa mà một số model vẫn trả về."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"[*#`]+", "", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1 and not sentences[-1].rstrip().endswith((".", "!", "?")):
        sentences = sentences[:-1]  # câu cuối bị cắt dở vì chạm giới hạn token
    text = " ".join(item for item in sentences if not any(word in item.lower() for word in _BANNED)).strip()
    return text[:2000] or None
