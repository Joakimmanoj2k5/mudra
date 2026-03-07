"""Canonical gesture catalog for seeding lessons and DB records."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class GestureSpec:
    code: str
    display_name: str
    lesson_type: str
    gesture_mode: str
    category: str
    requires_two_hands: bool = False


ALPHABETS: List[GestureSpec] = [
    GestureSpec(code=f"ALPHABET_{ch}", display_name=ch, lesson_type="alphabet", gesture_mode="static", category="alphabets")
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
    GestureSpec("WORD_NAMASTE", "Namaste", "word", "static", "greetings", True),
    GestureSpec("WORD_HELLO", "Hello", "word", "dynamic", "greetings"),
    GestureSpec("WORD_GOOD_MORNING", "Good Morning", "word", "dynamic", "greetings"),
    GestureSpec("WORD_GOOD_NIGHT", "Good Night", "word", "dynamic", "greetings"),
    GestureSpec("WORD_HOW_ARE_YOU", "How Are You", "word", "dynamic", "greetings"),
    GestureSpec("WORD_I_AM_FINE", "I Am Fine", "word", "dynamic", "greetings"),
    GestureSpec("WORD_THANK_YOU", "Thank You", "word", "dynamic", "greetings"),
    GestureSpec("WORD_SORRY", "Sorry", "word", "static", "greetings"),
    GestureSpec("WORD_PLEASE", "Please", "word", "dynamic", "greetings"),
    GestureSpec("WORD_WELCOME", "Welcome", "word", "dynamic", "greetings"),
    GestureSpec("WORD_YES", "Yes", "word", "static", "common"),
    GestureSpec("WORD_NO", "No", "word", "static", "common"),
    GestureSpec("WORD_OK", "Okay", "word", "static", "common"),
    GestureSpec("WORD_HELP", "Help", "word", "dynamic", "common"),
    GestureSpec("WORD_STOP", "Stop", "word", "static", "common"),
    GestureSpec("WORD_GO", "Go", "word", "dynamic", "common"),
    GestureSpec("WORD_COME", "Come", "word", "dynamic", "common"),
    GestureSpec("WORD_WAIT", "Wait", "word", "static", "common"),
    GestureSpec("WORD_FINISH", "Finish", "word", "dynamic", "common"),
    GestureSpec("WORD_START", "Start", "word", "dynamic", "common"),
    GestureSpec("WORD_LEARN", "Learn", "word", "dynamic", "education"),
    GestureSpec("WORD_STUDY", "Study", "word", "dynamic", "education"),
    GestureSpec("WORD_TEACHER", "Teacher", "word", "dynamic", "education"),
    GestureSpec("WORD_STUDENT", "Student", "word", "static", "education"),
    GestureSpec("WORD_BOOK", "Book", "word", "static", "education"),
    GestureSpec("WORD_PEN", "Pen", "word", "static", "education"),
    GestureSpec("WORD_NOTEBOOK", "Notebook", "word", "static", "education"),
    GestureSpec("WORD_EXAM", "Exam", "word", "dynamic", "education"),
    GestureSpec("WORD_PASS", "Pass", "word", "dynamic", "education"),
    GestureSpec("WORD_FAIL", "Fail", "word", "dynamic", "education"),
    GestureSpec("WORD_FAMILY", "Family", "word", "dynamic", "family"),
    GestureSpec("WORD_MOTHER", "Mother", "word", "static", "family"),
    GestureSpec("WORD_FATHER", "Father", "word", "static", "family"),
    GestureSpec("WORD_BROTHER", "Brother", "word", "static", "family"),
    GestureSpec("WORD_SISTER", "Sister", "word", "static", "family"),
    GestureSpec("WORD_BABY", "Baby", "word", "dynamic", "family"),
    GestureSpec("WORD_FRIEND", "Friend", "word", "dynamic", "family"),
    GestureSpec("WORD_HOME", "Home", "word", "static", "family"),
    GestureSpec("WORD_MARRIAGE", "Marriage", "word", "dynamic", "family"),
    GestureSpec("WORD_CHILD", "Child", "word", "dynamic", "family"),
    GestureSpec("WORD_WATER", "Water", "word", "static", "daily"),
    GestureSpec("WORD_FOOD", "Food", "word", "dynamic", "daily"),
    GestureSpec("WORD_RICE", "Rice", "word", "static", "daily"),
    GestureSpec("WORD_MILK", "Milk", "word", "dynamic", "daily"),
    GestureSpec("WORD_TEA", "Tea", "word", "static", "daily"),
    GestureSpec("WORD_COFFEE", "Coffee", "word", "dynamic", "daily"),
    GestureSpec("WORD_SLEEP", "Sleep", "word", "dynamic", "daily"),
    GestureSpec("WORD_WAKE_UP", "Wake Up", "word", "dynamic", "daily"),
    GestureSpec("WORD_BATH", "Bath", "word", "dynamic", "daily"),
    GestureSpec("WORD_WORK", "Work", "word", "dynamic", "daily"),
    GestureSpec("WORD_SHOP", "Shop", "word", "dynamic", "daily"),
    GestureSpec("WORD_MONEY", "Money", "word", "static", "daily"),
    GestureSpec("WORD_BUS", "Bus", "word", "dynamic", "travel"),
    GestureSpec("WORD_TRAIN", "Train", "word", "dynamic", "travel"),
    GestureSpec("WORD_CAR", "Car", "word", "static", "travel"),
    GestureSpec("WORD_BIKE", "Bike", "word", "dynamic", "travel"),
    GestureSpec("WORD_ROAD", "Road", "word", "dynamic", "travel"),
    GestureSpec("WORD_LEFT", "Left", "word", "static", "travel"),
    GestureSpec("WORD_RIGHT", "Right", "word", "static", "travel"),
    GestureSpec("WORD_STRAIGHT", "Straight", "word", "dynamic", "travel"),
    GestureSpec("WORD_NEAR", "Near", "word", "static", "travel"),
    GestureSpec("WORD_FAR", "Far", "word", "static", "travel"),
    GestureSpec("WORD_HOSPITAL", "Hospital", "word", "dynamic", "emergency"),
    GestureSpec("WORD_DOCTOR", "Doctor", "word", "dynamic", "emergency"),
    GestureSpec("WORD_MEDICINE", "Medicine", "word", "static", "emergency"),
    GestureSpec("WORD_PAIN", "Pain", "word", "dynamic", "emergency"),
    GestureSpec("WORD_BLOOD", "Blood", "word", "static", "emergency"),
    GestureSpec("WORD_EMERGENCY", "Emergency", "word", "dynamic", "emergency"),
    GestureSpec("WORD_POLICE", "Police", "word", "dynamic", "emergency"),
    GestureSpec("WORD_FIRE", "Fire", "word", "dynamic", "emergency"),
    GestureSpec("WORD_DANGER", "Danger", "word", "static", "emergency"),
    GestureSpec("WORD_CALL", "Call", "word", "dynamic", "emergency"),
    GestureSpec("WORD_TIME", "Time", "word", "static", "numbers_time"),
    GestureSpec("WORD_DAY", "Day", "word", "static", "numbers_time"),
    GestureSpec("WORD_NIGHT", "Night", "word", "static", "numbers_time"),
    GestureSpec("WORD_TODAY", "Today", "word", "dynamic", "numbers_time"),
    GestureSpec("WORD_TOMORROW", "Tomorrow", "word", "dynamic", "numbers_time"),
    GestureSpec("WORD_YESTERDAY", "Yesterday", "word", "dynamic", "numbers_time"),
    GestureSpec("WORD_ONE", "One", "word", "static", "numbers_time"),
    GestureSpec("WORD_TWO", "Two", "word", "static", "numbers_time"),
    GestureSpec("WORD_THREE", "Three", "word", "static", "numbers_time"),
    GestureSpec("WORD_FOUR", "Four", "word", "static", "numbers_time"),
    GestureSpec("WORD_FIVE", "Five", "word", "static", "numbers_time"),
    GestureSpec("WORD_SIX", "Six", "word", "static", "numbers_time"),
    GestureSpec("WORD_SEVEN", "Seven", "word", "static", "numbers_time"),
    GestureSpec("WORD_EIGHT", "Eight", "word", "static", "numbers_time"),
    GestureSpec("WORD_NINE", "Nine", "word", "static", "numbers_time"),
    GestureSpec("WORD_TEN", "Ten", "word", "static", "numbers_time"),
    GestureSpec("WORD_LOVE", "Love", "word", "dynamic", "emotion"),
    GestureSpec("WORD_HAPPY", "Happy", "word", "dynamic", "emotion"),
    GestureSpec("WORD_SAD", "Sad", "word", "dynamic", "emotion"),
    GestureSpec("WORD_ANGRY", "Angry", "word", "dynamic", "emotion"),
    GestureSpec("WORD_SCARED", "Scared", "word", "dynamic", "emotion"),
    GestureSpec("WORD_TIRED", "Tired", "word", "dynamic", "emotion"),
    GestureSpec("WORD_COLD", "Cold", "word", "static", "emotion"),
    GestureSpec("WORD_HOT", "Hot", "word", "static", "emotion"),
    GestureSpec("WORD_GREAT", "Great", "word", "dynamic", "emotion"),
    GestureSpec("WORD_BAD", "Bad", "word", "dynamic", "emotion"),
]

_AVAILABLE_VIDEO_STEMS = _available_video_stems()
WORD_SPECS: List[GestureSpec] = [
    spec for spec in ALL_WORD_SPECS
    if _word_video_stem(spec).lower() in _AVAILABLE_VIDEO_STEMS
]


def all_gestures() -> List[GestureSpec]:
    return ALPHABETS + WORD_SPECS
