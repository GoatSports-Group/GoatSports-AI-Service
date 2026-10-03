"""Thẻ "khoe" trận đấu gửi sang Cộng đồng (social-service).

ai-service là nơi duy nhất biết ai đã chơi trận nào và kết quả ra sao, nên thẻ được dựng ở đây từ dữ liệu phiên,
không nhận từ client. social-service chỉ lưu và báo cho người được gắn thẻ.
"""
from uuid import UUID

from app.domain.entity.models import ClubMatchProfile, MatchmakingSession, MatchParticipantProfile
from app.domain.enums import MatchSessionStatus

SPORT_LABEL = {
    "FOOTBALL": "Bóng đá",
    "BADMINTON": "Cầu lông",
    "TENNIS": "Tennis",
    "PICKLEBALL": "Pickleball",
    "BASKETBALL": "Bóng rổ",
    "VOLLEYBALL": "Bóng chuyền",
}

FORMAT_LABEL = {
    "BADMINTON_SINGLES": "Đánh đơn",
    "BADMINTON_DOUBLES": "Đánh đôi",
    "TENNIS_SINGLES": "Đánh đơn",
    "TENNIS_DOUBLES": "Đánh đôi",
    "PICKLEBALL_SINGLES": "Đánh đơn",
    "PICKLEBALL_DOUBLES": "Đánh đôi",
    "FOOTBALL_5": "Sân 5",
    "FOOTBALL_7": "Sân 7",
    "FOOTBALL_11": "Sân 11",
    "BASKETBALL_3X3": "3x3",
    "BASKETBALL_5X5": "5x5",
    "VOLLEYBALL_6": "6 người",
}

WEEKDAY = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

# Kèo đã chốt (hai bên nhận lời) hoặc đã đấu xong. Kèo bị từ chối, hết hạn, hủy hay đang tranh chấp kết quả thì
# không có gì để khoe.
SHAREABLE = {
    MatchSessionStatus.ACCEPTED,
    MatchSessionStatus.VENUE_SELECTED,
    MatchSessionStatus.BOOKING_PENDING,
    MatchSessionStatus.CONFIRMED,
    MatchSessionStatus.CHECKED_IN,
    MatchSessionStatus.RESULT_PENDING,
    MatchSessionStatus.COMPLETED,
}


def side_user_ids(side: MatchParticipantProfile) -> list[UUID]:
    """Người thật của một bên: bên CLB là quản lý và các thành viên ra sân (participant_id của CLB không phải người)."""
    if isinstance(side, ClubMatchProfile):
        return list(dict.fromkeys([*side.manager_ids, *(member.user_id for member in side.members)]))
    return side.user_ids()


def build_match_highlight(
    session: MatchmakingSession,
    side: MatchParticipantProfile,
    user_id: UUID,
    title: str,
    content: str | None,
) -> dict:
    """Body cho POST /internal/posts/highlights, nhìn từ phía người chia sẻ (bên mình đứng trước)."""
    is_first = side.participant_id == session.participants[0].participant_id
    opponent = session.participants[1 if is_first else 0]
    sport = session.sport_type.value
    play_format = side.play_format or opponent.play_format
    headline = SPORT_LABEL.get(sport, sport)
    if play_format in FORMAT_LABEL:
        headline = f"{headline} · {FORMAT_LABEL[play_format]}"

    subtitle = (
        f"{WEEKDAY[session.play_date.weekday()]}, {session.play_date.strftime('%d/%m/%Y')}"
        f" · {session.start_time.strftime('%H:%M')}–{session.end_time.strftime('%H:%M')}"
    )
    venue = _venue_name(session)
    if venue:
        subtitle = f"{subtitle} · {venue}"

    result = session.result if session.status == MatchSessionStatus.COMPLETED else None
    mine = theirs = None
    if result is not None:
        mine = result.participant_one_score if is_first else result.participant_two_score
        theirs = result.participant_two_score if is_first else result.participant_one_score
        outcome = "Hòa" if result.winner_id is None else ("Thắng" if result.winner_id == side.participant_id else "Thua")
        result_line = f"{outcome} {mine}–{theirs}"
    elif session.status == MatchSessionStatus.RESULT_PENDING:
        result_line = "Chờ xác nhận kết quả"
    else:
        result_line = "Sắp ra sân"

    stats = [{"label": "Độ phù hợp", "value": f"{round(session.compatibility_score)}%"}]
    elo_change = _elo_change(session, side, user_id)
    if elo_change is not None:
        stats.append({"label": "ELO", "value": f"+{elo_change}" if elo_change >= 0 else f"−{abs(elo_change)}"})

    winner_id = result.winner_id if result is not None else None
    tagged = [
        member
        for participant in (side, opponent)
        for member in side_user_ids(participant)
        if member != user_id
    ]
    return {
        "authorId": str(user_id),
        "sport": sport,
        "content": content,
        "kind": "MATCH",
        "refId": str(session.session_id),
        "title": title.strip(),
        "headline": headline,
        "subtitle": subtitle,
        "result": result_line,
        "sides": [
            {"name": side.name, "score": mine, "winner": winner_id is not None and winner_id == side.participant_id},
            {"name": opponent.name, "score": theirs,
             "winner": winner_id is not None and winner_id == opponent.participant_id},
        ],
        "stats": stats,
        "tagged": [str(member) for member in dict.fromkeys(tagged)],
    }


def _venue_name(session: MatchmakingSession) -> str | None:
    proposal = session.proposal
    if proposal is None or proposal.venue_id is None:
        return None
    return next((option.venue_name for option in proposal.venue_options if option.venue_id == proposal.venue_id), None)


def _elo_change(session: MatchmakingSession, side: MatchParticipantProfile, user_id: UUID) -> int | None:
    """ELO người chia sẻ được cộng / trừ sau trận (chỉ khi kết quả đã chốt và ELO đã ghi nhận)."""
    result = session.result
    if session.status != MatchSessionStatus.COMPLETED or result is None or not result.elo_applied:
        return None
    member = next((item for item in side.members if item.user_id == user_id), None)
    before, key = (member.elo_rating, str(member.user_id)) if member else (side.elo_rating, str(side.participant_id))
    after = result.elo_updates.get(key)
    return None if after is None else after - before
