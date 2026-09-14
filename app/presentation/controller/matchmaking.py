from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Header, Query, status

from app.application.dto.request.models import (
    MatchAcceptanceRequest,
    MatchBookingRegistrationRequest,
    MatchmakingQueueRequest,
    MatchProposalUpdateRequest,
    MatchResultRequest,
    MatchVenueSelectionRequest,
    OpponentFeedbackRequest,
)
from app.application.dto.response.models import (
    ActionResponse,
    MatchCandidate,
    QueueJoinResponse,
    QueueStatusResponse,
)
from app.domain.entity.models import MatchmakingSession
from app.domain.enums import QueueStatus
from app.infrastructure.config.container import matchmaking_service

router = APIRouter(prefix="/matchmaking", tags=["AI Matchmaking"])


@router.post(
    "/queue",
    response_model=QueueJoinResponse,
    status_code=status.HTTP_200_OK,
    summary="Tham gia hàng chờ ghép kèo",
)
def join_queue(
    request: MatchmakingQueueRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> QueueJoinResponse:
    session = matchmaking_service.join_queue(x_user_id, request)
    if session:
        background_tasks.add_task(matchmaking_service.notify_match_found, session, x_user_id)
        return QueueJoinResponse(
            status=QueueStatus.MATCHED,
            message="Đã tìm thấy đối thủ phù hợp.",
            session=session,
        )
    return QueueJoinResponse(
        status=QueueStatus.QUEUED,
        message="Đang chờ một đối thủ phù hợp tham gia.",
    )


@router.get(
    "/status",
    response_model=QueueStatusResponse,
    summary="Kiểm tra trạng thái ghép kèo của tôi",
)
def check_my_status(
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> QueueStatusResponse:
    matchmaking_status, queue_size, session = matchmaking_service.get_status(x_user_id)
    return QueueStatusResponse(
        status=matchmaking_status,
        queue_size=queue_size,
        session=session,
    )


@router.get(
    "/candidates",
    response_model=list[MatchCandidate],
    summary="Xem các đối thủ phù hợp đang ở trong hàng chờ",
)
def get_candidates(
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
) -> list[MatchCandidate]:
    return matchmaking_service.get_candidates(x_user_id, limit)


@router.post(
    "/candidates/{candidate_participant_id}/select",
    response_model=MatchmakingSession,
    summary="Chọn một đối thủ trong danh sách AI xếp hạng",
)
def select_candidate(
    candidate_participant_id: UUID,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    session = matchmaking_service.select_candidate(x_user_id, candidate_participant_id)
    background_tasks.add_task(matchmaking_service.notify_match_found, session, x_user_id)
    return session


@router.delete(
    "/queue",
    response_model=ActionResponse,
    summary="Rời hàng chờ ghép kèo",
)
def leave_my_queue(
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> ActionResponse:
    removed = matchmaking_service.leave_queue(x_user_id)
    return ActionResponse(
        success=removed,
        message="Đã rời hàng chờ." if removed else "Bạn không ở trong hàng chờ.",
    )


@router.get(
    "/sessions",
    response_model=list[MatchmakingSession],
    summary="Xem lịch sử ghép kèo của tôi",
)
def get_my_matchmaking_history(
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[MatchmakingSession]:
    return matchmaking_service.get_history(x_user_id, limit)


@router.get(
    "/sessions/{session_id}",
    response_model=MatchmakingSession,
    summary="Xem phiên ghép kèo",
)
def get_session(
    session_id: UUID,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    return matchmaking_service.get_session(session_id, x_user_id)


@router.post(
    "/sessions/{session_id}/acceptances",
    response_model=MatchmakingSession,
    summary="Chấp nhận hoặc từ chối kèo",
)
def decide_match(
    session_id: UUID,
    request: MatchAcceptanceRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    session = matchmaking_service.decide(session_id, x_user_id, request.decision)
    if session.status.value == "ACCEPTED":
        background_tasks.add_task(matchmaking_service.enrich_session_venue, session.session_id)
    background_tasks.add_task(
        matchmaking_service.notify_decision,
        session,
        x_user_id,
        request.decision,
    )
    return session


@router.post(
    "/sessions/{session_id}/venue",
    response_model=MatchmakingSession,
    summary="Chọn một trong ba sân AI đề xuất",
)
def select_venue(
    session_id: UUID,
    request: MatchVenueSelectionRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    session = matchmaking_service.select_venue(session_id, x_user_id, request)
    background_tasks.add_task(matchmaking_service.notify_lifecycle, session, x_user_id)
    return session


@router.post(
    "/sessions/{session_id}/venues/refresh",
    response_model=MatchmakingSession,
    summary="Kiểm tra lại các sân còn lịch trống",
)
def refresh_venues(
    session_id: UUID,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    return matchmaking_service.refresh_venue_options(session_id, x_user_id)


@router.delete(
    "/sessions/{session_id}",
    response_model=MatchmakingSession,
    summary="Hủy phiên ghép kèo trước khi tạo booking",
)
def cancel_session(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    session = matchmaking_service.cancel_session(session_id, x_user_id)
    background_tasks.add_task(matchmaking_service.notify_lifecycle, session, x_user_id)
    return session


@router.post(
    "/sessions/{session_id}/booking",
    response_model=MatchmakingSession,
    summary="Gắn booking với phiên ghép trận",
)
def register_booking(
    session_id: UUID,
    request: MatchBookingRegistrationRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    session = matchmaking_service.register_booking(session_id, x_user_id, request)
    background_tasks.add_task(matchmaking_service.notify_lifecycle, session, x_user_id)
    return session


@router.post(
    "/sessions/{session_id}/result",
    response_model=MatchmakingSession,
    summary="Nhập hoặc xác nhận kết quả trận đấu",
)
def submit_result(
    session_id: UUID,
    request: MatchResultRequest,
    background_tasks: BackgroundTasks,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    session = matchmaking_service.submit_result(session_id, x_user_id, request)
    background_tasks.add_task(matchmaking_service.notify_result, session, x_user_id)
    return session


@router.post(
    "/sessions/{session_id}/feedback",
    response_model=MatchmakingSession,
    summary="Đánh giá đối thủ sau trận đấu",
)
def submit_feedback(
    session_id: UUID,
    request: OpponentFeedbackRequest,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    return matchmaking_service.submit_feedback(session_id, x_user_id, request)


@router.patch(
    "/sessions/{session_id}/proposal",
    response_model=MatchmakingSession,
    summary="Cập nhật đề xuất và booking của kèo",
)
def update_proposal(
    session_id: UUID,
    request: MatchProposalUpdateRequest,
    x_user_id: Annotated[UUID, Header(alias="X-User-Id")],
) -> MatchmakingSession:
    return matchmaking_service.update_proposal(session_id, x_user_id, request)
