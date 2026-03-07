"""Organize ISL videos from isl_videos/ into data/raw/<ClassName>/ for training.

Maps the 71 label_map classes to matching video files using:
- Case-insensitive matching (A -> a.mp4)
- Space-to-underscore (Good Morning -> good_morning.mp4)
- Number words to digits (One -> 1.mp4)
- Synonym expansion for broader coverage

Videos are COPIED (not moved) so isl_videos/ stays intact.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Number word -> digit mapping for video filenames
NUMBER_MAP = {
    "One": "1", "Two": "2", "Three": "3", "Four": "4", "Five": "5",
    "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9",
}

# Synonyms: label_map class -> list of alternative video filenames to also include
SYNONYMS = {
    "Scared": ["afraid", "fear", "nervous"],
    "Bad": ["wrong", "incorrect"],
    "Pain": ["ache", "hurt"],
    "Happy": ["smile", "proud"],
    "Stop": ["halt", "pause", "time_out"],
    "Please": ["request"],
    "Call": ["invite"],
    "Learn": ["study", "comprehend", "understand"],
    "Help": ["helpless"],
    "Hello": ["hi"],
    "Sorry": ["excuse_me"],
    "Food": ["eat", "hungry"],
    "Sleep": ["tired", "exhausted", "fatigued"],
    "Friend": ["meet"],
    "Come": ["follow", "here"],
    "Go": ["out", "outside"],
    "No": ["disagree", "dissent", "never"],
    "Yes": ["agree", "concur", "consent", "fine", "ok"],
    "Okay": ["ok", "fine"],
    "Work": ["office"],
    "Student": ["school", "college"],
    "Teacher": ["explain"],
    "Welcome": ["nice_to_meet_you"],
    "Tired": ["exhausted", "fatigued"],
    "Angry": ["frustrated", "stressed", "pressured"],
    "Wait": ["stay", "pause"],
    "Thank You": ["thank_you"],
    "Good Morning": ["good_morning"],
    "Good Night": ["good_night"],
    "Today": ["now", "right_now"],
    "Tomorrow": ["see_you_tomorrow"],
    "Yesterday": ["before"],
}


def normalize_to_filename(label: str) -> str:
    """Convert a label_map key to expected video filename (without extension)."""
    return label.strip().lower().replace(" ", "_")


def main():
    label_map_path = PROJECT_ROOT / "models" / "registry" / "label_map.json"
    video_dir = PROJECT_ROOT / "isl_videos"
    output_dir = PROJECT_ROOT / "data" / "raw"

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    available_videos = {p.stem.lower(): p for p in video_dir.glob("*.mp4")}

    matched = 0
    unmatched = []
    total_copied = 0

    for class_name in sorted(label_map.keys()):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        copied_for_class = 0

        # Primary match: class name -> video filename
        primary_key = normalize_to_filename(class_name)

        # Check number words
        if class_name in NUMBER_MAP:
            primary_key = NUMBER_MAP[class_name]

        if primary_key in available_videos:
            src = available_videos[primary_key]
            dst = class_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied_for_class += 1

        # Synonym matches
        synonym_keys = SYNONYMS.get(class_name, [])
        for syn in synonym_keys:
            syn_lower = syn.lower()
            if syn_lower in available_videos:
                src = available_videos[syn_lower]
                dst = class_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied_for_class += 1

        if copied_for_class > 0:
            matched += 1
            total_copied += copied_for_class
            print(f"  ✅ {class_name:20s} → {copied_for_class} video(s)")
        else:
            unmatched.append(class_name)
            print(f"  ❌ {class_name:20s} → NO VIDEO FOUND")

    print(f"\n{'='*50}")
    print(f"Matched classes:   {matched}/{len(label_map)}")
    print(f"Unmatched classes:  {len(unmatched)}/{len(label_map)}")
    print(f"Total videos copied: {total_copied}")

    if unmatched:
        print(f"\nClasses without videos: {', '.join(unmatched)}")
        print("These classes will have zero training samples.")

    # List unused videos (not mapped to any class)
    used_videos = set()
    for class_name in label_map:
        key = normalize_to_filename(class_name)
        if class_name in NUMBER_MAP:
            key = NUMBER_MAP[class_name]
        used_videos.add(key)
        for syn in SYNONYMS.get(class_name, []):
            used_videos.add(syn.lower())

    unused = sorted(set(available_videos.keys()) - used_videos)
    if unused:
        print(f"\nUnused videos ({len(unused)}): {', '.join(unused)}")
        print("Consider mapping these to existing classes for more data.")


if __name__ == "__main__":
    main()
