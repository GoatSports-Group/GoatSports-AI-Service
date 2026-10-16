import json
import unittest
from datetime import date, time

from app.application.dto.request.models import ChatbotQuery
from app.application.port.client import ChatTurn, ToolCall
from app.application.service.assistant_tools import AssistantTools, training_args_from_text
from app.application.service.training_plan import Window, build_plan
from scripts.eval_rag import EVAL_USER, FakePlatformData
from tests.test_assistant import ScriptedLlm, assistant, call

TODAY = date(2026, 10, 15)
EVENINGS = [Window(day, time(18), time(21)) for day in range(7)]


class BuildPlanTest(unittest.TestCase):
    def test_sessions_follow_skill_and_goal_and_are_spread_through_the_week(self):
        plan = build_plan("BADMINTON", "INTERMEDIATE", "ky_thuat", EVENINGS, recent_sessions=12)
        self.assertEqual(3, plan.sessions_per_week)
        days = [session.day for session in plan.sessions]
        # Ba buổi trên bảy ngày: không có hai buổi liền nhau.
        gaps = [(days[(index + 1) % 3] - days[index]) % 7 for index in range(3)]
        self.assertGreaterEqual(min(gaps), 2)
        self.assertEqual(4, build_plan("BADMINTON", "INTERMEDIATE", "thi_dau", EVENINGS, 12).sessions_per_week)
        self.assertEqual(2, build_plan("BADMINTON", "INTERMEDIATE", "duy_tri", EVENINGS, 12).sessions_per_week)

    def test_each_session_fits_its_window_and_has_warm_up_main_and_cool_down(self):
        windows = [Window(1, time(18, 30), time(19, 30)), Window(4, time(19), time(22))]
        plan = build_plan("FOOTBALL", "ADVANCED", "ky_thuat", windows, recent_sessions=16)
        self.assertEqual(2, plan.sessions_per_week)  # chỉ có hai ngày rảnh
        self.assertTrue(any("chỉ đủ 2 ngày" in note for note in plan.notes))
        short = plan.sessions[0]
        self.assertEqual((time(18, 30), time(19, 30)), (short.start, short.end))
        self.assertEqual(["Khởi động", "Phần chính", "Thả lỏng"], [block.name for block in short.blocks])
        self.assertEqual(60, sum(block.minutes for block in short.blocks))
        self.assertEqual(90, sum(block.minutes for block in plan.sessions[1].blocks))

    def test_low_recent_activity_ramps_up_and_week_four_is_a_deload(self):
        plan = build_plan("TENNIS", "ADVANCED", "ky_thuat", EVENINGS, recent_sessions=2)
        self.assertEqual([1, 2, 3, 4], [week.sessions for week in plan.weeks])
        self.assertEqual([100, 110, 120, 80], [week.load for week in plan.weeks])
        steady = build_plan("TENNIS", "ADVANCED", "ky_thuat", EVENINGS, recent_sessions=14)
        self.assertEqual([4, 4, 4, 4], [week.sessions for week in steady.weeks])

    def test_no_history_on_the_platform_starts_light_instead_of_from_zero(self):
        plan = build_plan("TENNIS", "PRO", "ky_thuat", EVENINGS, recent_sessions=0)
        self.assertEqual([2, 3, 4, 5], [week.sessions for week in plan.weeks])
        self.assertTrue(any(note.startswith("Chưa thấy buổi chơi nào") for note in plan.notes))

    def test_two_hard_sessions_are_not_put_on_back_to_back_days(self):
        windows = [Window(day, time(18), time(21)) for day in (1, 2, 3, 5)]  # T3, T4, T5, T7
        plan = build_plan("BADMINTON", "INTERMEDIATE", "thi_dau", windows, recent_sessions=16)
        by_day = {session.day: session.intensity for session in plan.sessions}
        for day in (1, 2):
            self.assertFalse(by_day[day] == "cao" and by_day[day + 1] == "cao", by_day)

    def test_drills_respect_the_skill_level(self):
        beginner = build_plan("BADMINTON", "BEGINNER", "ky_thuat", EVENINGS, recent_sessions=8)
        text = " ".join(block.content for session in beginner.sessions for block in session.blocks)
        self.assertNotIn("smash", text)  # bài đập cầu dành cho từ trình trung bình
        self.assertIn("clear", text)

    def test_without_availability_it_uses_common_hours_and_says_so(self):
        plan = build_plan("PICKLEBALL", "BEGINNER", "the_luc", [], recent_sessions=8)
        self.assertTrue(plan.notes[0].startswith("Bạn chưa khai giờ rảnh"))
        self.assertTrue(all(session.start in (time(18), time(7)) for session in plan.sessions))

    def test_requested_sessions_are_honoured_within_bounds(self):
        self.assertEqual(5, build_plan("BASKETBALL", "BEGINNER", "ky_thuat", EVENINGS, 20, requested_sessions=5)
                         .sessions_per_week)
        self.assertEqual(6, build_plan("BASKETBALL", "BEGINNER", "ky_thuat", EVENINGS, 30, requested_sessions=9)
                         .sessions_per_week)


class TrainingToolTest(unittest.TestCase):
    def run_tool(self, arguments, identity=EVAL_USER, data=None):
        data = data or FakePlatformData()
        result, cards = AssistantTools(data).execute(ToolCall("c", "lich_tap", json.dumps(arguments)), identity, TODAY)
        return result, cards, data

    def test_plan_uses_the_real_profile_and_recent_bookings(self):
        result, cards, data = self.run_tool({})
        self.assertIn(("my_sport_profiles", EVAL_USER.user_id), data.calls)
        self.assertIn(("my_bookings", EVAL_USER.user_id, "ALL"), data.calls)
        self.assertEqual("Cầu lông", result["mon"])  # môn chính trong hồ sơ
        self.assertEqual("Trung bình", result["trinh_do"])
        self.assertEqual(3, result["so_buoi_moi_tuan"])
        # Giờ rảnh thật: thứ 3, 4, 5 tối và sáng thứ 7; thứ 5 chỉ có 90 phút.
        self.assertEqual({"Thứ Ba", "Thứ Năm", "Thứ Bảy"}, {item["ngay"] for item in result["buoi"]})
        self.assertEqual(["PLAN", "SESSION", "SESSION", "SESSION"], [card.kind for card in cards])
        self.assertEqual("/venues?sportType=BADMINTON&city=H%E1%BB%93+Ch%C3%AD+Minh", cards[1].link)

    def test_unknown_sport_without_profile_asks_back(self):
        result, cards, _ = self.run_tool({}, identity=None)
        self.assertIn("loi", result)
        self.assertEqual([], cards)

    def test_guest_gets_a_generic_plan_for_the_named_sport(self):
        result, cards, data = self.run_tool({"mon": "FOOTBALL", "muc_tieu": "the_luc"}, identity=None)
        self.assertEqual([], data.calls)
        self.assertEqual("Bóng đá", result["mon"])
        self.assertTrue(result["luu_y"][0].startswith("Bạn chưa đăng nhập"))

    def test_question_text_is_parsed_without_an_llm(self):
        self.assertEqual({"mon": "TENNIS", "muc_tieu": "thi_dau", "so_buoi_moi_tuan": 4, "so_tuan": 6},
                         training_args_from_text("Giáo án quần vợt 4 buổi mỗi tuần trong 6 tuần để thi đấu"))
        self.assertEqual({"mon": "BADMINTON"}, training_args_from_text("lich tap cau long giai doan dau"))


class TrainingAssistantTest(unittest.TestCase):
    def test_llm_calls_the_training_tool_and_cards_reach_the_client(self):
        llm = ScriptedLlm(call("lich_tap", muc_tieu="ky_thuat"), ChatTurn("Mình lập 3 buổi/tuần cho bạn."))
        response = assistant(llm).answer(ChatbotQuery(message="Lập lịch tập cầu lông giúp mình"), EVAL_USER)
        self.assertEqual("Mình lập 3 buổi/tuần cho bạn.", response.answer)
        self.assertEqual("PLAN", response.cards[0].kind)
        self.assertIn("lich_tap", [tool["function"]["name"] for tool in llm.chats[0][1]])

    def test_without_an_llm_a_training_request_still_gets_a_plan(self):
        bot = assistant(ScriptedLlm())
        response = bot.answer(ChatbotQuery(message="Lên lịch tập cầu lông 3 buổi một tuần"), EVAL_USER)
        self.assertEqual("training-planner", response.model)
        self.assertIn("3 buổi/tuần", response.answer)
        self.assertEqual("PLAN", response.cards[0].kind)
        self.assertEqual([], bot.gaps.unanswered)  # đã trả lời được, không vào hàng chờ admin


if __name__ == "__main__":
    unittest.main()
