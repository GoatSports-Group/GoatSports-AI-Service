import tempfile
import unittest
from pathlib import Path

from app.application.service.knowledge_gap_service import KnowledgeGapService
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.service_config import RagConfig
from app.domain.enums import KnowledgeGapStatus
from scripts.eval_rag import InMemoryKnowledgeRepository
from tests.test_rag_chatbot import KeywordEmbedder


class InMemoryGapRepository(InMemoryKnowledgeRepository):
    def __init__(self):
        super().__init__()
        self.gaps = {}
        self.legacy = []

    def save_knowledge_gap(self, gap):
        self.gaps[gap.gap_id] = gap.model_copy(deep=True)
        return gap

    def find_knowledge_gap(self, gap_id):
        gap = self.gaps.get(gap_id)
        return gap.model_copy(deep=True) if gap else None

    def find_knowledge_gaps(self, status, skip=0, limit=20_000):
        rows = sorted((g for g in self.gaps.values() if status is None or g.status == status),
                      key=lambda g: (-g.count, -g.last_asked_at.timestamp()))
        return [g.model_copy(deep=True) for g in rows[skip:skip + limit]], len(rows)

    def find_knowledge_gap_by_document(self, document_id):
        return next((g.model_copy(deep=True) for g in self.gaps.values() if g.knowledge_document_id == document_id), None)

    def pop_legacy_unanswered(self):
        rows, self.legacy = self.legacy, []
        return rows


class KnowledgeGapServiceTest(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryGapRepository()
        self.knowledge = RagChatbotService(self.repository, KeywordEmbedder(), RagConfig(max_context_chunks=3))
        self.service = KnowledgeGapService(self.repository, self.knowledge)

    def test_the_same_question_is_counted_not_listed_twice(self):
        self.service.record("Giải nào đang mở?", "refused")
        self.service.record("giải  nào đang mở?", "not_helpful", "Mình không biết")
        self.service.record("Hủy sân có mất cọc không?", "no_documents_no_tools")

        gaps, total = self.service.list(KnowledgeGapStatus.OPEN, 0, 20)
        self.assertEqual(2, total)
        top = gaps[0]
        self.assertEqual(2, top.count)
        self.assertEqual({"refused": 1, "not_helpful": 1}, top.reasons)
        self.assertEqual(["Mình không biết"], top.bad_answers)

    def test_an_admin_answer_becomes_searchable_knowledge_immediately(self):
        gap = self.service.record("Giải nào đang mở?", "refused")

        answered, answerable = self.service.answer(gap.gap_id, "Giải nào đang mở đăng ký?",
                                                   "Xem ở trang Giải đấu, tab Đang mở đăng ký.", "admin@goat.test")

        self.assertTrue(answerable)
        self.assertEqual(KnowledgeGapStatus.ANSWERED, answered.status)
        retrieved = self.knowledge.retrieve("giải đấu mở khi nào")
        self.assertEqual("faq/" + str(gap.gap_id), retrieved[0].document.source_url)
        self.assertEqual(0, self.service.list(KnowledgeGapStatus.OPEN, 0, 20)[1])

    def test_directory_sync_keeps_admin_answers(self):
        gap = self.service.record("Giải nào đang mở?", "refused")
        self.service.answer(gap.gap_id, "Giải nào đang mở?", "Xem ở trang Giải đấu.", "admin")
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "huy.md").write_text("# Hủy sân\n\nHủy trong 60 phút được hoàn 100%.", encoding="utf-8")
            report = self.knowledge.sync_directory(Path(directory))
        self.assertEqual([], report.removed)
        self.assertEqual(1, len(self.service.faqs()))

    def test_removing_a_wrong_answer_puts_the_question_back_in_the_queue(self):
        gap = self.service.record("Giải nào đang mở?", "refused")
        answered, _ = self.service.answer(gap.gap_id, "Giải nào đang mở?", "Câu trả lời sai.", "admin")
        document_id = self.service.faqs()[0][0].document_id

        self.service.delete_faq(document_id)

        self.assertEqual([], self.service.faqs())
        self.assertEqual(KnowledgeGapStatus.OPEN, self.repository.find_knowledge_gap(gap.gap_id).status)
        self.assertEqual([], self.knowledge.retrieve("giải đấu mở khi nào"))

    def test_dismissed_and_legacy_questions(self):
        gap = self.service.record("Ai vô địch World Cup?", "refused")
        self.assertEqual(KnowledgeGapStatus.DISMISSED, self.service.dismiss(gap.gap_id, "admin").status)
        self.repository.legacy = [{"question": "Hủy sân thế nào?", "reason": "llm_unavailable"}]
        self.assertEqual(1, self.service.import_legacy())
        self.assertEqual(1, self.service.list(KnowledgeGapStatus.OPEN, 0, 20)[1])


if __name__ == "__main__":
    unittest.main()
