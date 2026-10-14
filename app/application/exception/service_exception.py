class ServiceException(Exception):
    """Base class for expected application errors."""


class ResourceNotFoundException(ServiceException):
    pass


class ForbiddenException(ServiceException):
    pass


class ConflictException(ServiceException):
    pass


class DownstreamServiceException(ServiceException):
    pass


class InsufficientForecastDataException(ServiceException):
    pass


class CapabilityUnavailableException(ServiceException):
    pass


class UnreadableModelOutputException(ServiceException):
    """LLM trả lời nhưng không đúng định dạng: lỗi của riêng mục đó, bên gọi không nên thử lại mãi."""
