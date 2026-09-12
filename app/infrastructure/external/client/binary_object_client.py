from urllib.parse import urlparse

import httpx

from app.application.exception import DownstreamServiceException
from app.infrastructure.config.settings import Settings


class HttpBinaryObjectClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def download(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise DownstreamServiceException("Đường dẫn tài liệu xác minh không hợp lệ.")
        hostname = (parsed.hostname or "").lower()
        if not any(
            hostname == suffix.lstrip(".") or hostname.endswith(suffix)
            for suffix in self.settings.IDENTITY_ALLOWED_OBJECT_HOST_SUFFIXES
        ):
            raise DownstreamServiceException("Nguồn tài liệu xác minh không được phép.")
        try:
            with httpx.Client(
                timeout=self.settings.IDENTITY_DOWNLOAD_TIMEOUT_SECONDS,
                follow_redirects=False,
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not content_type.startswith("image/"):
                        raise DownstreamServiceException("Tài liệu xác minh phải là hình ảnh.")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > self.settings.IDENTITY_MAX_IMAGE_BYTES:
                            raise DownstreamServiceException("Ảnh xác minh vượt quá dung lượng cho phép.")
                        chunks.append(chunk)
                    content = b"".join(chunks)
                    if not content:
                        raise DownstreamServiceException("Không đọc được ảnh xác minh.")
                    return content
        except httpx.HTTPError as exception:
            raise DownstreamServiceException("Không thể tải tài liệu xác minh từ Storage Service.") from exception
