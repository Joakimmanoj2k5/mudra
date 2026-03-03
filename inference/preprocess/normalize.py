from __future__ import annotations

from typing import Dict, List

import numpy as np


def _normalize_hand(coords: np.ndarray) -> np.ndarray:
    if coords.shape != (21, 3):
        return np.zeros((21, 3), dtype=np.float32)
    wrist = coords[0]
    centered = coords - wrist
    scale = np.linalg.norm(coords[9] - coords[0])
    if scale < 1e-6:
        scale = 1.0
    centered /= scale

    # Canonical rotation using wrist->index_mcp axis (landmark 5)
    ref = centered[5][:2]
    angle = np.arctan2(ref[1], ref[0])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    centered_xy = centered[:, :2] @ rot.T
    centered[:, :2] = centered_xy
    return centered


def build_feature_vector(extraction: Dict[str, object], include_engineered: bool = True) -> np.ndarray:
    hands = extraction.get("hands", []) if extraction else []
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    for hand in hands:
        normalized = _normalize_hand(hand["coords"])
        if hand["label"].lower() == "left":
            left = normalized
        else:
            right = normalized

    base = np.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)
    if not include_engineered:
        return base

    angles = []
    joint_triplets = [(0, 5, 8), (0, 9, 12), (0, 13, 16), (0, 17, 20), (1, 2, 4)]
    for hand in [left, right]:
        for a, b, c in joint_triplets:
            v1 = hand[a] - hand[b]
            v2 = hand[c] - hand[b]
            d = (np.linalg.norm(v1) * np.linalg.norm(v2))
            if d < 1e-6:
                angles.append(0.0)
            else:
                cos = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
                angles.append(float(np.arccos(cos)))

    return np.concatenate([base, np.array(angles, dtype=np.float32)], axis=0)


class FeatureNormalizer:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std
