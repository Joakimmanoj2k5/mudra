from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np


class PredictionSmoother:
    def __init__(self, alpha: float = 0.6, confirm_frames: int = 3, vote_window: int = 8):
        self.alpha = alpha
        self.confirm_frames = confirm_frames
        self.vote_window = vote_window
        self.ema_probs = None
        self.recent_labels: Deque[int] = deque(maxlen=vote_window)
        self.last_stable = -1
        self.candidate = -1
        self.candidate_count = 0

    def update(self, probs: np.ndarray) -> Tuple[int, float]:
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.alpha * probs + (1 - self.alpha) * self.ema_probs

        label = int(np.argmax(self.ema_probs))
        confidence = float(self.ema_probs[label])
        self.recent_labels.append(label)

        if label == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = label
            self.candidate_count = 1

        if self.candidate_count >= self.confirm_frames:
            self.last_stable = self.candidate

        if self.last_stable >= 0 and len(self.recent_labels) >= self.vote_window // 2:
            majority = max(set(self.recent_labels), key=list(self.recent_labels).count)
            self.last_stable = majority

        stable = self.last_stable if self.last_stable >= 0 else label
        return stable, confidence
