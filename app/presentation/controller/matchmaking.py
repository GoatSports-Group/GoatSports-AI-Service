from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Header, status

from app.application.dto.request.models import (
    MatchAcceptanceRequest,
    MatchmakingQueueRequest,
    MatchProposalUpdateRequest,
)
from app.application.dto.response.models import (
    ActionResponse,
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
        background_tasks.add_task(matchmaking_service.enrich_session_venue, session.session_id)
        background_tasks.add_task(matchmaking_service.notify_match_found, session)
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
    background_tasks.add_task(
        matchmaking_service.notify_decision,
        session,
        x_user_id,
        request.decision,
    )
    return session


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
