"""Lập lịch tập theo tuần cho một người chơi: thuần, tất định, không gọi mạng.

Dựa trên hồ sơ thể thao thật (môn, trình độ, giờ rảnh) và số buổi đã chơi 4 tuần qua:
- Số buổi/tuần theo trình độ và mục tiêu; người tự chọn thì theo người chọn (1–6).
- Ngày tập lấy từ giờ rảnh đã khai, chọn sao cho các buổi cách đều nhau nhất trong tuần (tránh hai buổi liền).
- Ít chơi gần đây thì tăng dần số buổi qua từng tuần (không nhảy vọt khối lượng, dễ chấn thương).
- Chu kỳ 4 tuần: nền → +10% → +20% → tuần giảm tải; dài hơn thì lặp lại.
- Mỗi buổi: khởi động, phần chính theo trọng tâm (bài tập theo môn và trình độ), thả lỏng.

Không dùng LLM nên lịch luôn giống nhau với cùng dữ liệu và vẫn lập được khi LLM không gọi được.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from itertools import combinations

SKILLS = ("BEGINNER", "INTERMEDIATE", "ADVANCED", "PRO")
SKILL_LABEL = {"BEGINNER": "Mới chơi", "INTERMEDIATE": "Trung bình", "ADVANCED": "Khá", "PRO": "Chuyên nghiệp"}
GOALS = {
    "ky_thuat": "Nâng kỹ thuật",
    "the_luc": "Tăng thể lực",
    "thi_dau": "Chuẩn bị thi đấu",
    "duy_tri": "Duy trì, chơi vui khỏe",
}
WEEKDAY = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")
DAY_INDEX = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3, "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}

_BASE_SESSIONS = {"BEGINNER": 2, "INTERMEDIATE": 3, "ADVANCED": 4, "PRO": 5}
_GOAL_SESSIONS = {"ky_thuat": 0, "the_luc": 0, "thi_dau": 1, "duy_tri": -1}
_MINUTES = {"BEGINNER": 60, "INTERMEDIATE": 75, "ADVANCED": 90, "PRO": 90}
_MIN_SESSION_MINUTES = 45
# Chưa khai giờ rảnh: tối ngày thường và sáng cuối tuần, giờ phổ biến nhất của người chơi phong trào.
_DEFAULT_WINDOWS = tuple(
    (day, time(7, 0), time(10, 0)) if day >= 5 else (day, time(18, 0), time(21, 0)) for day in range(7)
)
_LOADS = (100, 110, 120, 80)
_WEEK_NOTES = (
    "Làm quen nhịp tập, ưu tiên đúng kỹ thuật hơn tốc độ",
    "Tăng khoảng 10% khối lượng: thêm hiệp hoặc thêm lượt mỗi bài",
    "Tuần nặng nhất: giữ cường độ, nghỉ đủ giữa các hiệp",
    "Tuần giảm tải: bớt khoảng 20%, đánh giá lại tiến bộ bằng một buổi đấu tập",
)

_FOCUS_BY_GOAL = {
    "ky_thuat": ("Kỹ thuật", "Chiến thuật & phối hợp", "Kỹ thuật dưới áp lực", "Thể lực bổ trợ", "Đấu tập", "Kỹ thuật"),
    "the_luc": ("Thể lực nền", "Kỹ thuật", "Sức bền tốc độ", "Đấu tập", "Sức mạnh & phòng chấn thương", "Thể lực nền"),
    "thi_dau": ("Chiến thuật & phối hợp", "Đấu tập", "Kỹ thuật dưới áp lực", "Thể lực bổ trợ", "Đấu tập",
                "Hồi phục chủ động"),
    "duy_tri": ("Đấu tập", "Kỹ thuật", "Thể lực bổ trợ", "Đấu tập", "Kỹ thuật", "Đấu tập"),
}
_FOCUS = {  # trọng tâm -> (nhóm bài, cường độ)
    "Kỹ thuật": ("ky_thuat", "vừa"),
    "Kỹ thuật dưới áp lực": ("ky_thuat", "cao"),
    "Chiến thuật & phối hợp": ("chien_thuat", "vừa"),
    "Đấu tập": ("dau_tap", "cao"),
    "Thể lực nền": ("the_luc", "vừa"),
    "Thể lực bổ trợ": ("the_luc", "vừa"),
    "Sức bền tốc độ": ("the_luc", "cao"),
    "Sức mạnh & phòng chấn thương": ("suc_manh", "vừa"),
    "Hồi phục chủ động": ("hoi_phuc", "nhẹ"),
}

# (bài tập, trình độ tối thiểu: 0 mới chơi … 3 chuyên nghiệp)
_COMMON = {
    "the_luc": (
        ("Chạy nhẹ xen đi bộ 2 × 8 phút, nghỉ 2 phút", 0),
        ("Chạy biến tốc 6 × (1 phút nhanh / 1 phút chậm)", 1),
        ("Chạy nước rút 10 × 20 m, nghỉ 40 giây", 2),
    ),
    "suc_manh": (
        ("Squat, lunge, plank: 3 vòng × 40 giây mỗi bài", 0),
        ("Bài cổ chân, gối (đứng một chân, nhón gót) 3 × 12 lần mỗi bên", 0),
        ("Bật nhảy plyometric 3 × 8 lần, đáp đất êm", 2),
    ),
    "hoi_phuc": (
        ("Đạp xe hoặc chạy rất nhẹ 20 phút", 0),
        ("Giãn cơ, lăn foam roller toàn thân 15 phút", 0),
    ),
}
_COOL_DOWN = "Giãn cơ tĩnh (đùi trước, đùi sau, bắp chân, hông, vai) và uống nước"

_SPORTS: dict[str, dict] = {
    "FOOTBALL": {
        "warmup": "Chạy nhẹ quanh sân, khởi động động (xoay hông, đá lăng chân), chuyền ngắn theo cặp",
        "ky_thuat": (
            ("Rondo 4 vs 1 / 5 vs 2, chuyền 1–2 chạm", 0),
            ("Dẫn bóng qua cọc bằng cả hai chân, đổi hướng", 0),
            ("Khống chế bước một rồi sút cầu môn", 1),
            ("Bật tường 1–2 và dứt điểm trong 10 giây", 2),
        ),
        "chien_thuat": (
            ("Giữ cự ly đội hình khi phòng ngự, di chuyển theo bóng", 0),
            ("Mất bóng là pressing ngay trong 5 giây", 1),
            ("Tình huống cố định: phạt góc, đá phạt có phân công", 1),
            ("Tấn công biên 3 vs 2 và căng ngang", 2),
        ),
        "dau_tap": (
            ("Đá trận nhỏ 5 vs 5 sân thu hẹp, hiệp 8 phút", 0),
            ("Đá trận điều kiện: tối đa 2 chạm", 1),
        ),
        "the_luc": (("Chạy con thoi 5–10–5 m, 6 lượt", 1),),
    },
    "BADMINTON": {
        "warmup": "Nhảy dây nhẹ, xoay khớp vai và cổ tay, đánh cầu cao sâu nhẹ nhàng",
        "ky_thuat": (
            ("Đánh cầu cao sâu (clear) cuối sân 3 × 20 quả", 0),
            ("Bỏ nhỏ sát lưới và đẩy cầu, 3 × 15 quả mỗi bên", 0),
            ("Đập cầu (smash) từ cầu nâng, chú ý xoay hông", 1),
            ("Phòng thủ đỡ đập liên tục 3 × 1 phút", 2),
        ),
        "chien_thuat": (
            ("Đánh đôi: tấn công đứng trước–sau, phòng thủ đứng ngang", 0),
            ("Giao cầu và trả giao cầu có chủ đích", 1),
            ("Đánh đơn: kéo đối thủ ra 4 góc rồi kết thúc", 1),
        ),
        "dau_tap": (
            ("Đánh set 21 điểm với bạn cùng trình", 0),
            ("Đánh tính điểm từ 15–15 để luyện điểm quyết định", 1),
        ),
        "the_luc": (("Di chuyển 6 điểm không cầu (shadow) 6 × 30 giây", 0),),
    },
    "PICKLEBALL": {
        "warmup": "Xoay khớp, dink nhẹ ở khu bếp (kitchen) 3 phút",
        "ky_thuat": (
            ("Dink chéo sân liên tục 3 × 2 phút", 0),
            ("Volley chặn bóng ở vạch kitchen, giữ vợt trước ngực", 0),
            ("Cú thứ ba thả bóng (third shot drop) từ cuối sân vào kitchen", 1),
            ("Reset bóng khi bị tấn công ở giữa sân", 2),
        ),
        "chien_thuat": (
            ("Đánh đôi: cùng tiến lên kitchen sau cú thứ ba", 0),
            ("Chọn thời điểm dứt điểm bóng cao (put-away)", 1),
        ),
        "dau_tap": (("Đánh ván 11 điểm, đổi cặp sau mỗi ván", 0),),
        "the_luc": (("Bước ngang nhanh dọc vạch kitchen 6 × 20 giây", 0),),
    },
    "TENNIS": {
        "warmup": "Chạy nhẹ, xoay vai, mini-tennis ở vạch giao bóng 5 phút",
        "ky_thuat": (
            ("Thuận tay, trái tay đường chéo 3 × 20 bóng", 0),
            ("Giao bóng: tung bóng ổn định, 30 quả vào ô", 0),
            ("Lên lưới volley sau bóng tiếp cận", 1),
            ("Trả giao bóng tấn công", 2),
        ),
        "chien_thuat": (
            ("Giữ bóng sâu, qua lưới cao và an toàn", 0),
            ("Rally chéo sân rồi đổi hướng dọc dây", 1),
            ("Giao bóng + cú đánh thứ ba theo kịch bản", 2),
        ),
        "dau_tap": (
            ("Đánh tie-break 7 điểm", 0),
            ("Đánh điểm bắt đầu từ 30–30", 1),
        ),
        "the_luc": (("Split step + di chuyển ngang 6 × 30 giây", 0),),
    },
    "BASKETBALL": {
        "warmup": "Chạy nhẹ, dẫn bóng tại chỗ hai tay, ném gần rổ",
        "ky_thuat": (
            ("Ném rổ cự ly gần–trung bình, 5 vị trí × 10 quả", 0),
            ("Dẫn bóng đổi tay, qua chân, sau lưng", 0),
            ("Lên rổ hai bước cả hai tay", 0),
            ("Dẫn bóng rồi ném dừng (pull-up)", 2),
        ),
        "chien_thuat": (
            ("Chạy chỗ không bóng, cắt vào rổ", 0),
            ("Pick and roll 2 vs 2", 1),
            ("Phòng thủ khu vực 2-3, xoay người", 1),
        ),
        "dau_tap": (("Đấu 3x3 nửa sân, 10 phút mỗi trận", 0),),
        "the_luc": (("Chạy suicide (biên – vạch ném phạt – giữa sân) 4 lượt", 1),),
    },
    "VOLLEYBALL": {
        "warmup": "Chạy nhẹ, xoay vai, chuyền bước một theo cặp",
        "ky_thuat": (
            ("Đệm bóng bước một chính xác về vị trí chuyền hai", 0),
            ("Chuyền hai cao tay cho tay đập", 0),
            ("Giao bóng trên tay 30 quả", 1),
            ("Đập bóng theo nhịp chuyền nhanh", 2),
        ),
        "chien_thuat": (
            ("Xoay vị trí và phối hợp chuyền–đập cơ bản", 0),
            ("Đội hình đỡ phát bóng 3–4 người", 1),
            ("Chắn bóng tập thể và yểm hộ sau chắn", 2),
        ),
        "dau_tap": (("Đấu set 25 điểm", 0),),
        "the_luc": (("Bật nhảy chắn liên tục 4 × 8 lần", 1),),
    },
}


@dataclass(frozen=True)
class Window:
    day: int  # 0 = thứ Hai … 6 = Chủ nhật
    start: time
    end: time

    @property
    def minutes(self) -> int:
        return (self.end.hour * 60 + self.end.minute) - (self.start.hour * 60 + self.start.minute)


@dataclass(frozen=True)
class Block:
    name: str
    minutes: int
    content: str


@dataclass(frozen=True)
class Session:
    day: int
    start: time
    end: time
    focus: str
    intensity: str
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Week:
    index: int
    sessions: int
    load: int
    note: str


@dataclass(frozen=True)
class TrainingPlan:
    sport: str
    skill: str
    goal: str
    sessions_per_week: int
    sessions: tuple[Session, ...]
    weeks: tuple[Week, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


def build_plan(
    sport: str,
    skill: str,
    goal: str,
    windows: list[Window],
    recent_sessions: int = 0,
    requested_sessions: int | None = None,
    weeks: int = 4,
) -> TrainingPlan:
    """recent_sessions: số buổi đã chơi 4 tuần qua (đơn đặt sân đã hoàn tất)."""
    skill = skill if skill in SKILLS else "BEGINNER"
    goal = goal if goal in GOALS else "ky_thuat"
    sport = sport if sport in _SPORTS else "BADMINTON"
    weeks = max(1, min(8, weeks))
    notes: list[str] = []

    target = (max(1, min(6, requested_sessions)) if requested_sessions
              else max(2, min(5, _BASE_SESSIONS[skill] + _GOAL_SESSIONS[goal])))
    usable = _usable_windows(windows)
    if not usable:
        usable = [Window(day, start, end) for day, start, end in _DEFAULT_WINDOWS]
        notes.append("Bạn chưa khai giờ rảnh nên lịch dùng tối ngày thường và sáng cuối tuần; "
                     "cập nhật giờ rảnh ở hồ sơ thể thao để lịch khớp hơn.")
    if len(usable) < target:
        notes.append(f"Giờ rảnh hiện chỉ đủ {len(usable)} ngày/tuần (cần ít nhất {_MIN_SESSION_MINUTES} phút mỗi buổi); "
                     f"thêm giờ rảnh để tập đủ {target} buổi.")
        target = len(usable)

    chosen = _spread_days(usable, target)
    minutes = _MINUTES[skill] - (15 if goal == "duy_tri" else 0)
    sessions = []
    for index, (window, focus) in enumerate(zip(chosen, _assign_focuses(chosen, _FOCUS_BY_GOAL[goal][:len(chosen)]))):
        length = min(minutes, window.minutes)
        start_minutes = window.start.hour * 60 + window.start.minute
        end = time((start_minutes + length) // 60, (start_minutes + length) % 60)
        sessions.append(Session(window.day, window.start, end, focus, _FOCUS[focus][1],
                                _blocks(sport, skill, focus, length, index)))

    # Tăng dần: đang chơi ít hơn mục tiêu thì tuần 1 chỉ hơn mức hiện tại một buổi, mỗi tuần thêm một buổi.
    # Không có buổi nào trên GoatSports chưa chắc là không chơi (có thể chơi nơi khác): chỉ bắt đầu nhẹ.
    current = round(recent_sessions / 4)
    if recent_sessions == 0:
        start_sessions = min(target, 2)
        reason = "Chưa thấy buổi chơi nào trên GoatSports 4 tuần qua"
    else:
        start_sessions = target if current >= target - 1 else current + 1
        reason = f"4 tuần qua bạn chơi khoảng {max(current, 1)} buổi/tuần"
    if start_sessions < target:
        notes.append(f"{reason} nên tuần 1 tập {start_sessions} buổi rồi tăng dần, tránh tăng khối lượng đột ngột dễ "
                     "chấn thương. Tuần ít buổi hơn thì bỏ các buổi cuối danh sách.")
    plan_weeks = tuple(
        Week(index + 1, min(target, start_sessions + index), _LOADS[index % 4], _WEEK_NOTES[index % 4])
        for index in range(weeks)
    )
    notes.append("Ngủ 7–8 tiếng, uống nước trước, trong và sau buổi tập; đau nhói hay sưng khớp thì dừng và nghỉ.")
    if goal == "thi_dau":
        notes.append("Tuần trước ngày thi đấu: giữ cường độ nhưng giảm khoảng một nửa khối lượng.")
    return TrainingPlan(sport, skill, goal, target, tuple(sessions), plan_weeks, tuple(notes))


def _usable_windows(windows: list[Window]) -> list[Window]:
    """Mỗi ngày một khung (khung dài nhất), đủ dài cho một buổi."""
    best: dict[int, Window] = {}
    for window in windows:
        if window.minutes >= _MIN_SESSION_MINUTES and (window.day not in best or window.minutes > best[window.day].minutes):
            best[window.day] = window
    return [best[day] for day in sorted(best)]


def _spread_days(windows: list[Window], count: int) -> list[Window]:
    """Chọn `count` ngày sao cho khoảng nghỉ ngắn nhất giữa hai buổi (tính vòng tuần) là dài nhất."""
    if count >= len(windows):
        return list(windows)

    def score(option: tuple[Window, ...]):
        days = [window.day for window in option]
        gaps = [(days[(index + 1) % len(days)] - days[index]) % 7 or 7 for index in range(len(days))]
        return min(gaps), sum(window.minutes for window in option), [-day for day in days]

    return list(max(combinations(windows, count), key=score))


def _assign_focuses(windows: list[Window], focuses: tuple[str, ...]) -> list[str]:
    """Giữ thứ tự ưu tiên của mục tiêu, nhưng không xếp hai buổi cường độ cao vào hai ngày liền nhau khi còn buổi nhẹ
    hơn để đổi chỗ."""
    remaining = list(focuses)
    assigned: list[str] = []
    for index, window in enumerate(windows):
        follows_hard = (index > 0 and window.day - windows[index - 1].day == 1
                        and _FOCUS[assigned[-1]][1] == "cao")
        pick = next((focus for focus in remaining if _FOCUS[focus][1] != "cao"), remaining[0]) if follows_hard \
            else remaining[0]
        remaining.remove(pick)
        assigned.append(pick)
    return assigned


def _blocks(sport: str, skill: str, focus: str, minutes: int, index: int) -> tuple[Block, ...]:
    level = SKILLS.index(skill)
    warm = 15 if minutes >= 75 else 10
    cool = 10 if minutes >= 60 else 5
    main = minutes - warm - cool
    group = _FOCUS[focus][0]
    library = _SPORTS[sport]
    if group == "hoi_phuc":
        return (Block("Hồi phục", minutes - cool, "; ".join(item for item, _ in _COMMON["hoi_phuc"])),
                Block("Thả lỏng", cool, _COOL_DOWN))
    pool = [item for item, minimum in (*library.get(group, ()), *_COMMON.get(group, ())) if minimum <= level]
    # Hai bài mỗi buổi, xoay vòng theo thứ tự buổi để các buổi cùng trọng tâm không lặp y hệt.
    drills = [pool[(index + offset) % len(pool)] for offset in range(min(2, len(pool)))]
    if focus == "Kỹ thuật dưới áp lực":
        drills = [f"{drill} (giới hạn thời gian hoặc có người gây áp lực)" for drill in drills]
    if group in ("ky_thuat", "chien_thuat") and focus != "Kỹ thuật dưới áp lực" and level >= 1:
        drills.append(library["dau_tap"][0][0] + " để áp dụng ngay")
    return (
        Block("Khởi động", warm, library["warmup"]),
        Block("Phần chính", main, "; ".join(dict.fromkeys(drills))),
        Block("Thả lỏng", cool, _COOL_DOWN),
    )


def hhmm(value: time) -> str:
    return value.strftime("%H:%M")
