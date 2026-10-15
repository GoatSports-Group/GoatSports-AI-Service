import tempfile
import unittest
from pathlib import Path

import json

from app.application.dto.request.models import KnowledgeDocumentRequest
from app.application.port.client import ExternalPassage
from app.application.service.knowledge_gap_service import AUTO_LEARNER, KnowledgeGapService
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


class ScriptedDrafts:
    """LLM giả: từ khóa Wikipedia theo `topic`, rồi lần lượt các bản nháp JSON; None = LLM không dùng được."""

    def __init__(self, *replies, topic="NONE"):
        self.replies = list(replies)
        self.topic = topic
        self.prompts = []

    def complete(self, system, prompt, max_tokens=None):
        if max_tokens == 80:
            return self.topic
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else None


class FakeWikipedia:
    def __init__(self):
        self.queries = []

    def lookup(self, query, limit, language="vi"):
        self.queries.append(query if language == "vi" else f"{language}:{query}")
        if query == "futsal":
            return [ExternalPassage("Futsal", "Futsal là bóng đá trong nhà 5 người…", "https://vi.wikipedia.org/wiki/Futsal")]
        if language == "en":
            return [ExternalPassage("Five-a-side football", "Five-a-side is…", "https://en.wikipedia.org/wiki/5",
                                    "Wikipedia tiếng Anh")]
        return [ExternalPassage("Luật việt vị (bóng đá)", "Luật việt vị là luật XI…", "https://vi.wikipedia.org/wiki/x")]


def draft(basis, answer="", note=""):
    return "```json\n" + json.dumps({"basis": basis, "answer": answer, "note": note}, ensure_ascii=False) + "\n```"


class KnowledgeGapDraftTest(unittest.TestCase):
    def service(self, *replies, auto_learn=True, topic="NONE"):
        repository = InMemoryGapRepository()
        knowledge = RagChatbotService(repository, KeywordEmbedder(), RagConfig(max_context_chunks=3))
        knowledge.ingest(KnowledgeDocumentRequest(
            title="Giải đấu", content="# Giải đấu\n\nĐăng ký giải ở trang Giải đấu, lệ phí đóng trong 24 giờ.",
            source_url="knowledge/giai.md"))
        llm = ScriptedDrafts(*replies, topic=topic)
        self.wikipedia = FakeWikipedia()
        return KnowledgeGapService(repository, knowledge, llm, self.wikipedia, auto_learn=auto_learn), llm

    def test_a_system_based_draft_is_learned_automatically(self):
        service, llm = self.service(draft("SYSTEM", "Đăng ký ở trang Giải đấu; lệ phí đóng trong 24 giờ."))
        gap = service.record("giải đóng lệ phí khi nào", "refused")

        learned = service.draft(gap.gap_id)

        # Câu về GoatSports: không tra nguồn ngoài.
        self.assertEqual([], self.wikipedia.queries)

        self.assertIn("<he_thong>", llm.prompts[0])
        self.assertIn("Giải đấu", llm.prompts[0])
        self.assertEqual("ANSWERED", learned.status.value)
        self.assertTrue(learned.auto_learned)
        self.assertEqual(AUTO_LEARNER, learned.resolved_by)
        self.assertEqual(1, len(service.faqs()))

    def test_an_external_draft_waits_for_an_admin(self):
        service, llm = self.service(draft("EXTERNAL", "Theo Wikipedia, việt vị là…"), topic="luật việt vị bóng đá")
        gap = service.record("giải thích luật việt vị", "refused")

        drafted = service.draft(gap.gap_id)

        # Tìm Wikipedia bằng từ khóa LLM viết, không bằng nguyên câu hỏi.
        self.assertEqual(["luật việt vị bóng đá"], self.wikipedia.queries)
        self.assertIn("<ben_ngoai>\n[Luật việt vị (bóng đá)]", llm.prompts[0])

        self.assertEqual("OPEN", drafted.status.value)
        self.assertEqual("EXTERNAL", drafted.draft_basis)
        self.assertEqual(["Luật việt vị (bóng đá) (Wikipedia tiếng Việt) — https://vi.wikipedia.org/wiki/x"],
                         drafted.draft_sources)
        self.assertEqual([], service.faqs())

    def test_synonyms_and_english_wikipedia_are_searched_and_model_drafts_wait_for_review(self):
        service, llm = self.service(draft("MODEL", "Bóng đá 5 người: mỗi đội 5 cầu thủ…", "Kiểm lại thời gian thi đấu"),
                                    topic="luật bóng đá 5 người\n- futsal\nen: five-a-side football")
        gap = service.record("Luật về bóng đá 5 vs 5 như thế nào?", "refused")

        drafted = service.draft(gap.gap_id)

        self.assertEqual(["luật bóng đá 5 người", "futsal", "en:five-a-side football"], self.wikipedia.queries)
        self.assertIn("[Futsal]", llm.prompts[0])
        self.assertIn("[Five-a-side football]", llm.prompts[0])
        # Hiểu biết của AI (không có nguồn) không bao giờ tự nạp.
        self.assertEqual("OPEN", drafted.status.value)
        self.assertEqual("MODEL", drafted.draft_basis)
        self.assertEqual([], service.faqs())

    def test_live_data_questions_are_never_written_down_as_knowledge(self):
        service, _ = self.service(draft("NONE", "Có 7 sân", "Cần dữ liệu thật: số sân thay đổi"))
        gap = service.record("Khánh Hòa có bao nhiêu sân?", "no_documents_no_tools")

        drafted = service.draft(gap.gap_id)

        self.assertIsNone(drafted.draft_answer)
        self.assertEqual("Cần dữ liệu thật: số sân thay đổi", drafted.draft_note)
        self.assertEqual([], service.faqs())

    def test_auto_learning_can_be_switched_off(self):
        service, _ = self.service(draft("SYSTEM", "Đăng ký ở trang Giải đấu."), auto_learn=False)
        gap = service.record("giải đóng lệ phí khi nào", "refused")
        self.assertEqual("OPEN", service.draft(gap.gap_id).status.value)

    def test_background_drafting_stops_when_the_llm_is_down_and_ignores_garbage(self):
        service, _ = self.service("không phải JSON")
        first = service.record("giải đóng lệ phí khi nào", "refused")
        service.record("hủy sân thế nào", "refused")

        self.assertEqual(0, service.draft_pending(5))
        self.assertIsNone(service.repository.find_knowledge_gap(first.gap_id).drafted_at)


if __name__ == "__main__":
    unittest.main()
