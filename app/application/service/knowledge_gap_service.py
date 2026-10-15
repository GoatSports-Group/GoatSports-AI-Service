"""Vòng bổ sung tri thức cho trợ lý: ghi nhận câu chưa trả lời được → admin trả lời → thành tài liệu FAQ ngay.

Không "huấn luyện lại" model: RAG học bằng cách thêm tài liệu. Câu trả lời của admin được nạp như một tài liệu
(`faq/<gapId>`), nhúng vector, và câu hỏi giống thế lần sau tìm thấy nó. Thư mục knowledge/*.md vẫn là nguồn chính;
FAQ là phần bổ sung, đồng bộ lúc khởi động không xóa chúng.

Nguồn ghi nhận: không có tài liệu lẫn công cụ nào dùng được, LLM trả lời "chưa có thông tin", LLM không dùng được,
người dùng bấm "Chưa hữu ích".
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from uuid import UUID

import numpy as np

from app.application.dto.request.models import KnowledgeDocumentRequest
from app.application.exception import ResourceNotFoundException
from app.application.port.persistence.repositories import AiDocumentRepository
from app.application.service.rag_chatbot_service import RagChatbotService
from app.domain.entity.models import KnowledgeDocument, KnowledgeGap
from app.domain.enums import KnowledgeGapStatus

logger = logging.getLogger(__name__)

FAQ_SOURCE_PREFIX = "faq/"
# e5 cho câu khác ý vẫn tới 0,93 ("đổi số điện thoại" / "đổi mật khẩu"): chỉ gộp câu gần như trùng. Gộp nhầm thì
# admin trả lời sai câu, còn để hai dòng giống nhau thì chỉ tốn thêm một lượt đọc.
SAME_QUESTION_SIMILARITY = 0.95
_MAX_SAMPLES = 5
_MAX_BAD_ANSWERS = 3

REASON_LABELS = {
    "no_documents_no_tools": "Không có tài liệu hay dữ liệu phù hợp",
    "refused": "Trợ lý trả lời chưa có thông tin",
    "llm_unavailable": "LLM không dùng được",
    "not_helpful": "Người dùng báo chưa hữu ích",
}


class KnowledgeGapService:
    def __init__(self, repository: AiDocumentRepository, knowledge: RagChatbotService, clock=None) -> None:
        self.repository = repository
        self.knowledge = knowledge
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()

    # ---- ghi nhận ----

    def record(self, question: str, reason: str, bad_answer: str | None = None) -> KnowledgeGap:
        question = " ".join(question.split())[:2000]
        vector = np.asarray(self.knowledge.embedder.embed_query(question), dtype=np.float32)
        now = self.clock()
        # Một khóa cho cả "tìm nhóm rồi cộng": hai câu giống nhau tới cùng lúc không tạo hai nhóm.
        with self._lock:
            gap = self._same_open_gap(question, vector)
            if gap is None:
                gap = KnowledgeGap(question=question, samples=[question], embedding=vector.tolist(),
                                   first_asked_at=now, last_asked_at=now, count=0)
            gap.count += 1
            gap.last_asked_at = now
            gap.reasons[reason] = gap.reasons.get(reason, 0) + 1
            if question not in gap.samples and len(gap.samples) < _MAX_SAMPLES:
                gap.samples.append(question)
            if bad_answer and len(gap.bad_answers) < _MAX_BAD_ANSWERS:
                gap.bad_answers.append(bad_answer[:1500])
            return self.repository.save_knowledge_gap(gap)

    def _same_open_gap(self, question: str, vector: np.ndarray) -> KnowledgeGap | None:
        # ponytail: so với mọi nhóm đang mở mỗi lần ghi; vài nghìn nhóm mở trở lên thì dùng $vectorSearch.
        best, best_score = None, 0.0
        for gap in self.repository.find_knowledge_gaps(KnowledgeGapStatus.OPEN)[0]:
            if question.lower() in (sample.lower() for sample in gap.samples):
                return gap
            if len(gap.embedding) != len(vector):
                continue
            score = float(np.dot(vector, np.asarray(gap.embedding, dtype=np.float32)))
            if score > best_score:
                best, best_score = gap, score
        return best if best_score >= SAME_QUESTION_SIMILARITY else None

    def import_legacy(self) -> int:
        """Bản ghi kiểu cũ (chatbot_unanswered, mỗi lần hỏi một dòng) chuyển sang nhóm, chạy lúc khởi động."""
        rows = self.repository.pop_legacy_unanswered()
        for row in rows:
            if row.get("question"):
                self.record(str(row["question"]), str(row.get("reason") or "no_documents_no_tools"))
        return len(rows)

    # ---- admin ----

    def list(self, status: KnowledgeGapStatus | None, page: int, size: int) -> tuple[list[KnowledgeGap], int]:
        return self.repository.find_knowledge_gaps(status, page * size, size)

    def answer(self, gap_id: UUID, question: str, answer: str, admin: str) -> tuple[KnowledgeGap, bool]:
        """Nạp câu trả lời thành tài liệu FAQ và thử hỏi lại ngay: True khi trợ lý đã tìm thấy câu trả lời này."""
        gap = self._require(gap_id)
        question = " ".join(question.split())
        variants = [sample for sample in gap.samples if sample != question]
        content = f"# {question}\n\n{answer.strip()}"
        if variants:
            content += "\n\nCách hỏi khác: " + "; ".join(variants)
        document = self.knowledge.ingest(KnowledgeDocumentRequest(
            title=question, content=content, source_url=f"{FAQ_SOURCE_PREFIX}{gap.gap_id}",
        )).document
        gap.status = KnowledgeGapStatus.ANSWERED
        gap.answer = answer.strip()
        gap.resolved_by = admin
        gap.resolved_at = self.clock()
        gap.knowledge_document_id = document.document_id
        self.repository.save_knowledge_gap(gap)
        answerable = all(
            any(item.document.document_id == document.document_id for item in self.knowledge.retrieve(text))
            for text in [question, *gap.samples]
        )
        return gap, answerable

    def dismiss(self, gap_id: UUID, admin: str) -> KnowledgeGap:
        """Câu không cần tài liệu (lạc đề, hoặc cần tính năng / công cụ mới thay vì một câu trả lời)."""
        gap = self._require(gap_id)
        gap.status = KnowledgeGapStatus.DISMISSED
        gap.resolved_by = admin
        gap.resolved_at = self.clock()
        return self.repository.save_knowledge_gap(gap)

    def faqs(self) -> list[tuple[KnowledgeDocument, KnowledgeGap | None]]:
        documents = [doc for doc in self.repository.find_knowledge_documents().values()
                     if (doc.source_url or "").startswith(FAQ_SOURCE_PREFIX)]
        documents.sort(key=lambda doc: doc.updated_at, reverse=True)
        return [(doc, self.repository.find_knowledge_gap_by_document(doc.document_id)) for doc in documents]

    def delete_faq(self, document_id: UUID) -> None:
        """Gỡ một câu trả lời sai: tài liệu biến mất khỏi tìm kiếm, câu hỏi quay lại hàng chờ."""
        document = self.repository.find_knowledge_documents().get(document_id)
        if document is None or not (document.source_url or "").startswith(FAQ_SOURCE_PREFIX):
            raise ResourceNotFoundException("Không tìm thấy câu trả lời bổ sung này.")
        self.repository.delete_knowledge_document(document_id)
        gap = self.repository.find_knowledge_gap_by_document(document_id)
        if gap is not None:
            gap.status = KnowledgeGapStatus.OPEN
            gap.knowledge_document_id = None
            self.repository.save_knowledge_gap(gap)

    def _require(self, gap_id: UUID) -> KnowledgeGap:
        gap = self.repository.find_knowledge_gap(gap_id)
        if gap is None:
            raise ResourceNotFoundException("Không tìm thấy câu hỏi này.")
        return gap
