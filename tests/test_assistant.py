import json
import unittest
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from app.application.dto.request.models import ChatbotQuery, KnowledgeDocumentRequest
from app.application.port.client import ChatTurn, PlatformDataError, ToolCall
from app.application.service.assistant_service import (
    RATE_LIMITED_ANSWER,
    UNAVAILABLE_ANSWER,
    AssistantService,
    identity_from_headers,
)
from app.application.service.assistant_tools import AssistantTools
from app.application.service.rag_chatbot_service import RagChatbotService
from app.application.service.service_config import RagConfig
from scripts.eval_rag import (
    EVAL_USER,
    FakePlatformData,
    InMemoryConversationStore,
    InMemoryKnowledgeRepository,
)
from tests.test_rag_chatbot import POLICY, KeywordEmbedder

TODAY = date(2026, 10, 15)


class ScriptedLlm:
    """Trả lần lượt các lượt đã soạn; ghi lại mọi tin nhắn được gửi tới."""

    model_name = "scripted"

    def __init__(self, *turns, rewrite="câu đã viết lại"):
        self.turns = list(turns)
        self.rewrite = rewrite
        self.chats, self.completions = [], []

    def chat(self, messages, tools, max_tokens):
        self.chats.append((list(messages), tools))
        return self.turns.pop(0) if self.turns else None

    def complete(self, system, prompt, max_tokens=None):
        self.completions.append(prompt)
        return self.rewrite


def call(name, **args):
    return ChatTurn(content=None, tool_calls=[ToolCall(f"c-{name}", name, json.dumps(args))])


def assistant(llm, data=None, store=None):
    knowledge = RagChatbotService(InMemoryKnowledgeRepository(), KeywordEmbedder(), RagConfig(max_context_chunks=3))
    knowledge.ingest(KnowledgeDocumentRequest(title="Hủy sân", content=POLICY, source_url="knowledge/huy.md"))
    return AssistantService(
        knowledge, AssistantTools(data or FakePlatformData()), llm, store or InMemoryConversationStore(),
        knowledge.repository, clock=lambda: datetime(2026, 10, 15, 9, tzinfo=timezone(timedelta(hours=7))),
    )


class AssistantServiceTest(unittest.TestCase):
    def test_tool_result_goes_back_to_the_llm_and_comes_out_as_cards(self):
        llm = ScriptedLlm(call("giai_dau", mon="PICKLEBALL"), ChatTurn("Có 1 giải đang mở."))

        response = assistant(llm).answer(ChatbotQuery(message="Giải pickleball đang mở?"), None)

        self.assertEqual("Có 1 giải đang mở.", response.answer)
        self.assertTrue(response.grounded)
        self.assertEqual(["TOURNAMENT"], [card.kind for card in response.cards])
        self.assertEqual("/tournaments/t-1", response.cards[0].link)
        tool_message = llm.chats[1][0][-1]
        self.assertEqual("tool", tool_message["role"])
        self.assertIn("Goat Pickleball Open 2026", tool_message["content"])

    def test_my_bookings_needs_a_gateway_identity_and_never_uses_ids_from_the_llm(self):
        data = FakePlatformData()
        llm = ScriptedLlm(call("don_cua_toi", user_id=str(uuid4())), ChatTurn("Hãy đăng nhập."))
        assistant(llm, data).answer(ChatbotQuery(message="Đơn của tôi?"), None)
        self.assertEqual([], data.calls)
        self.assertIn("chưa đăng nhập", llm.chats[1][0][-1]["content"])

        llm = ScriptedLlm(call("don_cua_toi", loai="tat_ca"), ChatTurn("Bạn có 1 đơn."))
        response = assistant(llm, data).answer(ChatbotQuery(message="Đơn của tôi?"), EVAL_USER)
        self.assertEqual([("my_bookings", EVAL_USER.user_id, "ALL")], data.calls)
        self.assertIn("chờ chủ sân duyệt", response.cards[0].details[1])

    def test_follow_up_question_is_rewritten_with_history_before_retrieval(self):
        store = InMemoryConversationStore()
        conversation = uuid4()
        llm = ScriptedLlm(ChatTurn("Trả lời 1"), ChatTurn("Trả lời 2"), rewrite="Hủy sân sau 60 phút thì sao?")
        bot = assistant(llm, store=store)

        bot.answer(ChatbotQuery(message="Hủy sân trong 60 phút?", conversation_id=conversation), EVAL_USER)
        second = bot.answer(ChatbotQuery(message="Còn sau đó?", conversation_id=conversation), EVAL_USER)

        self.assertEqual(1, len(llm.completions))  # chỉ câu thứ hai mới cần viết lại
        self.assertIn("Hủy sân trong 60 phút?", llm.completions[0])
        self.assertTrue(second.sources)  # câu viết lại có "hủy" nên tìm được tài liệu
        self.assertIn({"role": "assistant", "content": "Trả lời 1"}, llm.chats[1][0])
        self.assertEqual(4, len(store.history[f"{EVAL_USER.user_id}:{conversation}"]))

    def test_rate_limited_question_never_reaches_the_llm(self):
        llm = ScriptedLlm(ChatTurn("x"))
        response = assistant(llm, store=InMemoryConversationStore(limit_reached=True)).answer(
            ChatbotQuery(message="Hủy sân?"), EVAL_USER
        )
        self.assertEqual(RATE_LIMITED_ANSWER, response.answer)
        self.assertEqual([], llm.chats)

    def test_llm_outage_falls_back_and_is_logged_as_unanswered(self):
        bot = assistant(ScriptedLlm())
        with_docs = bot.answer(ChatbotQuery(message="hủy trong 60 phút"), None)
        without_docs = bot.answer(ChatbotQuery(message="giá bitcoin"), None)

        self.assertEqual("extractive-fallback", with_docs.model)
        self.assertIn("[Nguồn 1]", with_docs.answer)
        self.assertEqual(UNAVAILABLE_ANSWER, without_docs.answer)
        self.assertEqual(2, len(bot.repository.unanswered))

    def test_without_an_llm_only_confident_passages_are_quoted(self):
        bot = assistant(ScriptedLlm())
        # Đoạn qua ngưỡng liên quan nhưng chưa đủ chắc để trích nguyên văn: nói chưa trả lời được, không trích bừa.
        bot.knowledge.embedder.quote_floor = 1.01
        response = bot.answer(ChatbotQuery(message="hủy trong 60 phút"), None)

        self.assertEqual(UNAVAILABLE_ANSWER, response.answer)
        self.assertTrue(response.sources)

    def test_tool_rounds_are_capped(self):
        llm = ScriptedLlm(*[call("tim_san", mon="BADMINTON")] * 10)
        response = assistant(llm).answer(ChatbotQuery(message="tìm sân"), None)

        self.assertEqual(4, len(llm.chats))  # 3 vòng công cụ + 1 lượt bắt buộc trả lời (không kèm công cụ)
        self.assertIsNone(llm.chats[-1][1])
        self.assertEqual(UNAVAILABLE_ANSWER, response.answer)


class AssistantToolsTest(unittest.TestCase):
    def setUp(self):
        self.data = FakePlatformData()
        self.tools = AssistantTools(self.data)

    def run_tool(self, name, arguments, identity=None):
        return self.tools.execute(ToolCall("c", name, arguments), identity, TODAY)

    def test_free_slots_only_for_today_onward_and_list_free_courts(self):
        result, cards = self.run_tool("gio_trong", json.dumps({"ten_san": "Goat Arena", "ngay": "2026-10-16"}))
        self.assertEqual(["18:00–19:00", "20:00–21:00"], result["san"][0]["gio_trong"])
        self.assertEqual("1/2 sân còn giờ trống", cards[0].subtitle)

        past, _ = self.run_tool("gio_trong", json.dumps({"ten_san": "Goat Arena", "ngay": "2026-10-01"}))
        self.assertIn("loi", past)
        missing, _ = self.run_tool("gio_trong", json.dumps({"ten_san": "Không có", "ngay": "2026-10-16"}))
        self.assertIn("Không tìm thấy", missing["loi"])

    def test_bad_arguments_and_unknown_tools_become_errors_for_the_llm(self):
        self.assertIn("loi", self.run_tool("tim_san", "{khong phai json")[0])
        self.assertIn("loi", self.run_tool("dat_san", "{}")[0])
        result, cards = self.run_tool("tim_san", json.dumps({"mon": "CO_TUONG", "gia_toi_da": "re"}))
        self.assertEqual(("search_venues", None, None, None, None), self.data.calls[-1])
        self.assertEqual(2, len(cards))

    def test_service_outage_is_reported_not_guessed(self):
        class Down(FakePlatformData):
            def search_tournaments(self, *args, **kwargs):
                raise PlatformDataError("timeout")

        result, cards = AssistantTools(Down()).execute(ToolCall("c", "giai_dau", "{}"), None, TODAY)
        self.assertIn("Chưa lấy được dữ liệu", result["loi"])
        self.assertEqual([], cards)

    def test_identity_from_headers_rejects_garbage(self):
        self.assertIsNone(identity_from_headers("not-a-uuid", None, None, None, None))
        self.assertIsNone(identity_from_headers(None, "a@b.c", None, None, None))
        self.assertEqual(EVAL_USER.user_id, identity_from_headers(str(EVAL_USER.user_id), "", "", "", "").user_id)


if __name__ == "__main__":
    unittest.main()
