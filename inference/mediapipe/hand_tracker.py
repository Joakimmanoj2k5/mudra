from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _MP_AVAILABLE = True
except Exception:
    mp = None
    _MP_AVAILABLE = False

# Hand connections for drawing (21 landmarks)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


class HandTracker:
    def __init__(self, max_num_hands: int = 2, det_conf: float = 0.6, track_conf: float = 0.6):
        self.available = _MP_AVAILABLE
        self._landmarker = None
        if self.available:
            model_path = self._find_model()
            if model_path:
                try:
                    options = vision.HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=max_num_hands,
                        min_hand_detection_confidence=det_conf,
                        min_tracking_confidence=track_conf,
                    )
                    self._landmarker = vision.HandLandmarker.create_from_options(options)
                except Exception as e:
                    print(f"[HandTracker] Failed to create landmarker: {e}")
                    self._landmarker = None
            else:
                print("[HandTracker] hand_landmarker.task model not found")

    @staticmethod
    def _find_model() -> str | None:
        candidates = [
            "models/mediapipe/hand_landmarker.task",
            "hand_landmarker.task",
        ]
        for p in candidates:
            if Path(p).exists():
                return str(Path(p))
        return None

    def extract(self, frame_bgr) -> Dict[str, object]:
        if not self.available or self._landmarker is None:
            return {"status": "mediapipe_unavailable", "hands": []}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception:
            return {"status": "mediapipe_error", "hands": []}

        if not result.hand_landmarks:
            return {"status": "no_hand", "hands": []}

        hands = []
        for idx, landmarks in enumerate(result.hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32,
            )
            label = "Unknown"
            score = 0.0
            if idx < len(result.handedness):
                cls = result.handedness[idx][0]
                label = cls.category_name
                score = float(cls.score)
            hands.append({"coords": coords, "label": label, "score": score})

        hands.sort(key=lambda h: 0 if h["label"].lower() == "left" else 1)
        return {"status": "ok", "hands": hands}

    def draw(self, frame_bgr, extraction: Dict[str, object]) -> None:
        if extraction.get("status") != "ok":
            return
        h, w = frame_bgr.shape[:2]
        for hand in extraction["hands"]:
            coords = hand["coords"]
            # Draw connections
            for start, end in _HAND_CONNECTIONS:
                x1, y1 = int(coords[start][0] * w), int(coords[start][1] * h)
                x2, y2 = int(coords[end][0] * w), int(coords[end][1] * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw landmarks
            for lm in coords:
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(frame_bgr, (cx, cy), 4, (255, 0, 0), -1)
