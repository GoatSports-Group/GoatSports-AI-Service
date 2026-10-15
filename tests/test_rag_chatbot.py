import tempfile
import unittest
from pathlib import Path

from app.application.dto.request.models import KnowledgeDocumentRequest
from app.application.service.rag_chatbot_service import RagChatbotService, split_markdown
from app.application.service.service_config import RagConfig
from scripts.eval_rag import InMemoryKnowledgeRepository

_TOPICS = ("hủy", "check", "giải")


class KeywordEmbedder:
    """Vector theo chủ đề từ khóa: đủ để kiểm ngưỡng liên quan mà không cần model thật."""

    model_name = "keyword-test-v1"
    relevance_floor = 0.5
    quote_floor = 0.5

    def _vector(self, text):
        lowered = text.lower()
        vector = [1.0 if topic in lowered else 0.0 for topic in _TOPICS]
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else [0.0] * len(_TOPICS)

    def embed_query(self, text):
        return self._vector(text)

    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]


POLICY = """# Hủy sân

## Trong 60 phút
Hủy trong 60 phút được hoàn 100%.

## Sau 60 phút
### Theo cơ sở
Chủ sân duyệt yêu cầu hủy.
"""


def service():
    return RagChatbotService(InMemoryKnowledgeRepository(), KeywordEmbedder(), RagConfig(max_context_chunks=3))


class MarkdownSplitTest(unittest.TestCase):
    def test_sections_keep_their_heading_path_without_the_title(self):
        self.assertEqual(
            [("Trong 60 phút", "Hủy trong 60 phút được hoàn 100%."),
             ("Sau 60 phút › Theo cơ sở", "Chủ sân duyệt yêu cầu hủy.")],
            split_markdown(POLICY),
        )


class RagChatbotServiceTest(unittest.TestCase):
    def test_reingesting_overwrites_the_same_chunks_instead_of_adding_new_ones(self):
        rag = service()
        request = KnowledgeDocumentRequest(title="Hủy sân", content=POLICY, source_url="knowledge/huy.md")
        rag.ingest(request)
        first = set(rag.repository.chunks)
        rag.ingest(request)

        self.assertEqual(first, set(rag.repository.chunks))
        self.assertTrue(all(chunk.embedding for chunk in rag.repository.chunks.values()))

    def test_sync_indexes_new_files_skips_unchanged_ones_and_removes_deleted_ones(self):
        rag = service()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "huy.md").write_text(POLICY, encoding="utf-8")
            (folder / "giai.md").write_text("# Giải đấu\n\n## Lệ phí\nĐóng lệ phí giải trong 24 giờ.", encoding="utf-8")

            first = rag.sync_directory(folder)
            second = rag.sync_directory(folder)
            (folder / "giai.md").unlink()
            third = rag.sync_directory(folder)

        self.assertEqual(["giai.md", "huy.md"], first.indexed)
        self.assertEqual(["giai.md", "huy.md"], second.unchanged)
        self.assertEqual(["knowledge/giai.md"], third.removed)
        self.assertEqual({"Hủy sân"}, {doc.title for doc in rag.repository.documents.values()})

    def test_only_chunks_above_the_relevance_floor_are_retrieved(self):
        rag = service()
        rag.ingest(KnowledgeDocumentRequest(title="Hủy sân", content=POLICY, source_url="knowledge/huy.md"))

        relevant = rag.retrieve("Hủy sân có được hoàn tiền không?")

        self.assertIn("Trong 60 phút", {item.chunk.heading for item in relevant})
        self.assertEqual([], rag.retrieve("Giá bitcoin hôm nay?"))


if __name__ == "__main__":
    unittest.main()
