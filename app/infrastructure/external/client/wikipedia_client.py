"""Kiến thức thể thao chung từ Wikipedia tiếng Việt (REST API công khai, không cần khóa).

Chỉ để bù chỗ tài liệu hệ thống không có (luật chơi, kỹ thuật, giải thế giới, địa danh); trợ lý không dùng nó cho
giá, chính sách hay dữ liệu GoatSports. Nội dung lấy về là dữ liệu không tin cậy: chỉ giữ tiêu đề, tóm tắt thuần chữ,
đường dẫn; cắt ngắn trước khi đưa cho LLM.
"""

from __future__ import annotations

import logging
import threading
import time
from urllib.parse import quote

import httpx

from app.application.port.client import ExternalPassage, PlatformDataError

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 6.0
_CACHE_SECONDS = 24 * 3600
_CACHE_ENTRIES = 500
_SUMMARY_CHARS = 900


_SOURCE = {"vi": "Wikipedia tiếng Việt", "en": "Wikipedia tiếng Anh"}


class WikipediaClient:
    def __init__(self, base_url: str, user_agent: str) -> None:
        # base_url là bản tiếng Việt (https://vi.wikipedia.org); bản khác đổi mã ngôn ngữ ở tên miền.
        self.base_url = base_url
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self._cache: dict[tuple[str, int, str], tuple[float, list[ExternalPassage]]] = {}
        self._lock = threading.Lock()

    def lookup(self, query: str, limit: int, language: str = "vi") -> list[ExternalPassage]:
        language = language if language in _SOURCE else "vi"
        key = (" ".join(query.lower().split()), limit, language)
        with self._lock:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < _CACHE_SECONDS:
                return hit[1]
        passages = self._fetch(query, limit, language)
        with self._lock:
            # ponytail: cache trong bộ nhớ từng tiến trình, xóa cả khi đầy; nhiều bản sao thì chuyển sang Redis.
            if len(self._cache) >= _CACHE_ENTRIES:
                self._cache.clear()
            self._cache[key] = (time.monotonic(), passages)
        return passages

    def _fetch(self, query: str, limit: int, language: str) -> list[ExternalPassage]:
        site = self.base_url.replace("://vi.", f"://{language}.")
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS, headers=self.headers) as client:
                found = client.get(f"{site}/w/rest.php/v1/search/page", params={"q": query, "limit": limit})
                found.raise_for_status()
                passages = []
                for page in (found.json().get("pages") or [])[:limit]:
                    summary = client.get(f"{site}/api/rest_v1/page/summary/{quote(page['key'], safe='')}")
                    if summary.status_code != 200:
                        continue
                    body = summary.json()
                    # Trang định hướng chỉ là danh sách liên kết, không có nội dung để trả lời.
                    if body.get("type") == "disambiguation" or not body.get("extract"):
                        continue
                    passages.append(ExternalPassage(
                        title=str(body.get("title") or page.get("title") or ""),
                        summary=str(body["extract"])[:_SUMMARY_CHARS],
                        url=str(((body.get("content_urls") or {}).get("desktop") or {}).get("page")
                                or f"{site}/wiki/{quote(page['key'])}"),
                        source=_SOURCE[language],
                    ))
                return passages
        except (httpx.HTTPError, ValueError, KeyError) as exception:
            logger.warning("Wikipedia lookup failed: %s", exception)
            raise PlatformDataError(str(exception)) from exception
