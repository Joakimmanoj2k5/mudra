"""Canonical gesture catalog for seeding lessons and DB records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class GestureSpec:
    code: str
    display_name: str
    lesson_type: str
    gesture_mode: str
    category: str
    requires_two_hands: bool = False
    level: int = 1


# ── Level metadata for UI ──
LEVEL_INFO: Dict[int, Dict[str, str]] = {
    1: {"title": "Letters", "emoji": "🔤", "accent": "#22c55e", "bg": "#14532d",
        "desc": "Master the 26 ISL finger-spelling alphabets"},
    2: {"title": "Core Words", "emoji": "🔢", "accent": "#3b82f6", "bg": "#1e3a5f",
        "desc": "Numbers, yes/no, and the first essential signs"},
    3: {"title": "Daily Words", "emoji": "💬", "accent": "#f59e0b", "bg": "#713f12",
        "desc": "Greetings, family, food, and everyday communication"},
    4: {"title": "Actions + Places", "emoji": "🏫", "accent": "#ec4899", "bg": "#701a3e",
        "desc": "Travel, school, emergency, and action signs"},
    5: {"title": "Emotions + Phrases", "emoji": "😊", "accent": "#a78bfa", "bg": "#3b0764",
        "desc": "Feelings, reactions, and longer phrases"},
}


ALPHABETS: List[GestureSpec] = [
    GestureSpec(code=f"ALPHABET_{ch}", display_name=ch, lesson_type="alphabet",
               gesture_mode="static", category="alphabets", level=1)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
]

_WORD_VIDEO_STEM_OVERRIDES = {
    "WORD_OK": "okay",
    "WORD_I_AM_FINE": "fine",
    "WORD_ONE": "1",
    "WORD_TWO": "2",
    "WORD_THREE": "3",
    "WORD_FOUR": "4",
    "WORD_FIVE": "5",
    "WORD_SIX": "6",
    "WORD_SEVEN": "7",
    "WORD_EIGHT": "8",
    "WORD_NINE": "9",
    "WORD_TEN": "10",
}


def _slug(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def _available_video_stems() -> set[str]:
    video_root = Path("isl_videos")
    if not video_root.exists():
        return set()
    return {path.stem.lower() for path in video_root.glob("*.mp4")}


def _word_video_stem(spec: GestureSpec) -> str:
    return _WORD_VIDEO_STEM_OVERRIDES.get(spec.code, _slug(spec.display_name))


ALL_WORD_SPECS: List[GestureSpec] = [
    # ── Level 2: Numbers & Basics ──
    GestureSpec("WORD_ONE", "One", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_TWO", "Two", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_THREE", "Three", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_FOUR", "Four", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_FIVE", "Five", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_SIX", "Six", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_SEVEN", "Seven", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_EIGHT", "Eight", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_NINE", "Nine", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_TEN", "Ten", "word", "static", "numbers_time", level=2),
    GestureSpec("WORD_YES", "Yes", "word", "static", "common", level=2),
    GestureSpec("WORD_NO", "No", "word", "static", "common", level=2),
    GestureSpec("WORD_OK", "Okay", "word", "static", "common", level=2),
    GestureSpec("WORD_STOP", "Stop", "word", "static", "common", level=2),
    GestureSpec("WORD_WAIT", "Wait", "word", "static", "common", level=2),
    # ── Level 3: Everyday Words (greetings, family, daily) ──
    GestureSpec("WORD_NAMASTE", "Namaste", "word", "static", "greetings", True, level=3),
    GestureSpec("WORD_HELLO", "Hello", "word", "dynamic", "greetings", level=3),
    GestureSpec("WORD_GOOD_MORNING", "Good Morning", "word", "dynamic", "greetings", level=3),
    GestureSpec("WORD_GOOD_NIGHT", "Good Night", "word", "dynamic", "greetings", level=3),
    GestureSpec("WORD_THANK_YOU", "Thank You", "word", "dynamic", "greetings", level=3),
    GestureSpec("WORD_SORRY", "Sorry", "word", "static", "greetings", level=3),
    GestureSpec("WORD_PLEASE", "Please", "word", "dynamic", "greetings", level=3),
    GestureSpec("WORD_WELCOME", "Welcome", "word", "dynamic", "greetings", level=3),
    GestureSpec("WORD_MOTHER", "Mother", "word", "static", "family", level=3),
    GestureSpec("WORD_FATHER", "Father", "word", "static", "family", level=3),
    GestureSpec("WORD_BROTHER", "Brother", "word", "static", "family", level=3),
    GestureSpec("WORD_SISTER", "Sister", "word", "static", "family", level=3),
    GestureSpec("WORD_FRIEND", "Friend", "word", "dynamic", "family", level=3),
    GestureSpec("WORD_HOME", "Home", "word", "static", "family", level=3),
    GestureSpec("WORD_WATER", "Water", "word", "static", "daily", level=3),
    GestureSpec("WORD_FOOD", "Food", "word", "dynamic", "daily", level=3),
    GestureSpec("WORD_SLEEP", "Sleep", "word", "dynamic", "daily", level=3),
    GestureSpec("WORD_TIME", "Time", "word", "static", "numbers_time", level=3),
    GestureSpec("WORD_DAY", "Day", "word", "static", "numbers_time", level=3),
    GestureSpec("WORD_TODAY", "Today", "word", "dynamic", "numbers_time", level=3),
    GestureSpec("WORD_TOMORROW", "Tomorrow", "word", "dynamic", "numbers_time", level=3),
    GestureSpec("WORD_YESTERDAY", "Yesterday", "word", "dynamic", "numbers_time", level=3),
    # ── Level 4: Actions & Places ──
    GestureSpec("WORD_HOW_ARE_YOU", "How Are You", "word", "dynamic", "greetings", level=4),
    GestureSpec("WORD_I_AM_FINE", "I Am Fine", "word", "dynamic", "greetings", level=4),
    GestureSpec("WORD_HELP", "Help", "word", "dynamic", "common", level=4),
    GestureSpec("WORD_GO", "Go", "word", "dynamic", "common", level=4),
    GestureSpec("WORD_COME", "Come", "word", "dynamic", "common", level=4),
    GestureSpec("WORD_FINISH", "Finish", "word", "dynamic", "common", level=4),
    GestureSpec("WORD_START", "Start", "word", "dynamic", "common", level=4),
    GestureSpec("WORD_LEARN", "Learn", "word", "dynamic", "education", level=4),
    GestureSpec("WORD_STUDY", "Study", "word", "dynamic", "education", level=4),
    GestureSpec("WORD_TEACHER", "Teacher", "word", "dynamic", "education", level=4),
    GestureSpec("WORD_STUDENT", "Student", "word", "static", "education", level=4),
    GestureSpec("WORD_BOOK", "Book", "word", "static", "education", level=4),
    GestureSpec("WORD_PEN", "Pen", "word", "static", "education", level=4),
    GestureSpec("WORD_NOTEBOOK", "Notebook", "word", "static", "education", level=4),
    GestureSpec("WORD_EXAM", "Exam", "word", "dynamic", "education", level=4),
    GestureSpec("WORD_PASS", "Pass", "word", "dynamic", "education", level=4),
    GestureSpec("WORD_FAIL", "Fail", "word", "dynamic", "education", level=4),
    GestureSpec("WORD_FAMILY", "Family", "word", "dynamic", "family", level=4),
    GestureSpec("WORD_BABY", "Baby", "word", "dynamic", "family", level=4),
    GestureSpec("WORD_MARRIAGE", "Marriage", "word", "dynamic", "family", level=4),
    GestureSpec("WORD_CHILD", "Child", "word", "dynamic", "family", level=4),
    GestureSpec("WORD_RICE", "Rice", "word", "static", "daily", level=4),
    GestureSpec("WORD_MILK", "Milk", "word", "dynamic", "daily", level=4),
    GestureSpec("WORD_TEA", "Tea", "word", "static", "daily", level=4),
    GestureSpec("WORD_COFFEE", "Coffee", "word", "dynamic", "daily", level=4),
    GestureSpec("WORD_WAKE_UP", "Wake Up", "word", "dynamic", "daily", level=4),
    GestureSpec("WORD_BATH", "Bath", "word", "dynamic", "daily", level=4),
    GestureSpec("WORD_WORK", "Work", "word", "dynamic", "daily", level=4),
    GestureSpec("WORD_SHOP", "Shop", "word", "dynamic", "daily", level=4),
    GestureSpec("WORD_MONEY", "Money", "word", "static", "daily", level=4),
    GestureSpec("WORD_NIGHT", "Night", "word", "static", "numbers_time", level=4),
    GestureSpec("WORD_BUS", "Bus", "word", "dynamic", "travel", level=4),
    GestureSpec("WORD_TRAIN", "Train", "word", "dynamic", "travel", level=4),
    GestureSpec("WORD_CAR", "Car", "word", "static", "travel", level=4),
    GestureSpec("WORD_BIKE", "Bike", "word", "dynamic", "travel", level=4),
    GestureSpec("WORD_ROAD", "Road", "word", "dynamic", "travel", level=4),
    GestureSpec("WORD_LEFT", "Left", "word", "static", "travel", level=4),
    GestureSpec("WORD_RIGHT", "Right", "word", "static", "travel", level=4),
    GestureSpec("WORD_STRAIGHT", "Straight", "word", "dynamic", "travel", level=4),
    GestureSpec("WORD_NEAR", "Near", "word", "static", "travel", level=4),
    GestureSpec("WORD_FAR", "Far", "word", "static", "travel", level=4),
    GestureSpec("WORD_HOSPITAL", "Hospital", "word", "dynamic", "emergency", level=4),
    GestureSpec("WORD_DOCTOR", "Doctor", "word", "dynamic", "emergency", level=4),
    GestureSpec("WORD_MEDICINE", "Medicine", "word", "static", "emergency", level=4),
    GestureSpec("WORD_PAIN", "Pain", "word", "dynamic", "emergency", level=4),
    GestureSpec("WORD_BLOOD", "Blood", "word", "static", "emergency", level=4),
    GestureSpec("WORD_EMERGENCY", "Emergency", "word", "dynamic", "emergency", level=4),
    GestureSpec("WORD_POLICE", "Police", "word", "dynamic", "emergency", level=4),
    GestureSpec("WORD_FIRE", "Fire", "word", "dynamic", "emergency", level=4),
    GestureSpec("WORD_DANGER", "Danger", "word", "static", "emergency", level=4),
    GestureSpec("WORD_CALL", "Call", "word", "dynamic", "emergency", level=4),
    # ── Level 5: Emotions & Advanced ──
    GestureSpec("WORD_LOVE", "Love", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_HAPPY", "Happy", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_SAD", "Sad", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_ANGRY", "Angry", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_SCARED", "Scared", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_TIRED", "Tired", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_COLD", "Cold", "word", "static", "emotion", level=5),
    GestureSpec("WORD_HOT", "Hot", "word", "static", "emotion", level=5),
    GestureSpec("WORD_GREAT", "Great", "word", "dynamic", "emotion", level=5),
    GestureSpec("WORD_BAD", "Bad", "word", "dynamic", "emotion", level=5),
]

_AVAILABLE_VIDEO_STEMS = _available_video_stems()
WORD_SPECS: List[GestureSpec] = [
    spec for spec in ALL_WORD_SPECS
    if _word_video_stem(spec).lower() in _AVAILABLE_VIDEO_STEMS
]


def all_gestures() -> List[GestureSpec]:
    return ALPHABETS + WORD_SPECS
