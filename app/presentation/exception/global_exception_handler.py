import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.application.exception import (
    ConflictException,
    DownstreamServiceException,
    ForbiddenException,
    InsufficientForecastDataException,
    ResourceNotFoundException,
    ServiceException,
)

logger = logging.getLogger(__name__)


def _error(status_code: int, message: Any, error: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"statusCode": status_code, "message": message, "error": error},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ResourceNotFoundException)
    async def handle_not_found(_: Request, exception: ResourceNotFoundException) -> JSONResponse:
        return _error(status.HTTP_404_NOT_FOUND, str(exception), "Không tìm thấy dữ liệu")

    @app.exception_handler(ForbiddenException)
    async def handle_forbidden(_: Request, exception: ForbiddenException) -> JSONResponse:
        return _error(status.HTTP_403_FORBIDDEN, str(exception), "Không có quyền truy cập")

    @app.exception_handler(ConflictException)
    async def handle_conflict(_: Request, exception: ConflictException) -> JSONResponse:
        return _error(status.HTTP_409_CONFLICT, str(exception), "Xung đột dữ liệu")

    @app.exception_handler(InsufficientForecastDataException)
    async def handle_insufficient_data(
        _: Request, exception: InsufficientForecastDataException
    ) -> JSONResponse:
        return _error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            str(exception),
            "Dữ liệu dự báo chưa đầy đủ",
        )

    @app.exception_handler(DownstreamServiceException)
    async def handle_downstream(_: Request, exception: DownstreamServiceException) -> JSONResponse:
        return _error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(exception),
            "Dịch vụ liên quan không khả dụng",
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exception: RequestValidationError) -> JSONResponse:
        messages = [error.get("msg", "Dữ liệu không hợp lệ") for error in exception.errors()]
        return _error(
            status.HTTP_400_BAD_REQUEST,
            messages[0] if len(messages) == 1 else messages,
            "Dữ liệu không hợp lệ",
        )

    @app.exception_handler(ServiceException)
    async def handle_service_exception(_: Request, exception: ServiceException) -> JSONResponse:
        return _error(status.HTTP_400_BAD_REQUEST, str(exception), "Yêu cầu không hợp lệ")

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exception: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception while processing %s %s",
            request.method,
            request.url.path,
            exc_info=exception,
        )
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
            "Có lỗi xảy ra",
        )
