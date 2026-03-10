from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np

from inference.mediapipe.hand_tracker import HandTracker
from inference.preprocess.normalize import FeatureNormalizer, build_feature_vector
from inference.smoothing.temporal import PredictionSmoother

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


class StaticMLP(nn.Module if nn else object):
    def __init__(self, input_dim: int, output_dim: int):
        if not nn:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicBiGRU(nn.Module if nn else object):
    def __init__(self, input_dim: int, output_dim: int):
        if not nn:
            return
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, output_dim))

    def forward(self, x):
        y, _ = self.gru(x)
        pooled = y.mean(dim=1)
        return self.head(pooled)


class GesturePredictor:
    def __init__(self, config: Dict[str, object]):
        self.config = config
        self.tracker = HandTracker()
        self.smoother = PredictionSmoother(
            alpha=float(config.get("inference", {}).get("smoothing_alpha", 0.5)),
            confirm_frames=int(config.get("inference", {}).get("confirmation_frames", 4)),
        )
        self.sequence = deque(maxlen=30)
        self.frame_count = 0
        self.dynamic_stride = 4
        # Thresholds tuned for real ISL video-trained model
        self.static_threshold = float(config.get("inference", {}).get("static_threshold", 0.30))
        self.dynamic_threshold = float(config.get("inference", {}).get("dynamic_threshold", 0.25))
        self.static_hold_seconds = 0.6
        self.dynamic_confirm_frames = 3

        self.label_map = self._load_label_map(config.get("model", {}).get("label_map_path", "models/registry/label_map.json"))
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        self.class_modes = self._load_class_modes("models/registry/class_modes.json")
        self.dynamic_class_indices = {
            idx for name, idx in self.label_map.items() if self.class_modes.get(name, "static") == "dynamic"
        }

        self.normalizer = FeatureNormalizer()
        self._load_norm(config.get("model", {}).get("norm_stats_path", "models/registry/norm_stats.json"))
        self.static_model = self._load_static_model(config.get("model", {}).get("static_model_path", ""))
        self.dynamic_normalizer = FeatureNormalizer()
        self._load_dynamic_norm("models/registry/dynamic_norm_stats.json")
        self.dynamic_model = self._load_dynamic_model(config.get("model", {}).get("dynamic_model_path", ""))
        self.last_dynamic_probs = None
        self._static_candidate_idx = -1
        self._static_candidate_since = 0.0
        self._dynamic_streak_idx = -1
        self._dynamic_streak_count = 0

    def _load_label_map(self, path: str) -> Dict[str, int]:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        from utils.common.gesture_catalog import all_gestures

        labels = [g.display_name for g in all_gestures()]
        mapping = {name: i for i, name in enumerate(sorted(set(labels)))}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        return mapping

    def _load_norm(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.normalizer.mean = np.array(data["mean"], dtype=np.float32)
        self.normalizer.std = np.array(data["std"], dtype=np.float32)

    def _load_dynamic_norm(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.dynamic_normalizer.mean = np.array(data["mean"], dtype=np.float32)
        self.dynamic_normalizer.std = np.array(data["std"], dtype=np.float32)

    def _load_class_modes(self, path: str) -> Dict[str, str]:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        from utils.common.gesture_catalog import all_gestures

        modes = {g.display_name: g.gesture_mode for g in all_gestures()}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(modes, indent=2), encoding="utf-8")
        return modes

    def _load_static_model(self, path: str):
        if torch is None:
            return None
        input_dim = 136
        output_dim = max(self.label_map.values()) + 1 if self.label_map else 1
        model = StaticMLP(input_dim=input_dim, output_dim=output_dim)
        p = Path(path)
        if p.exists():
            state = torch.load(p, map_location="cpu")
            model.load_state_dict(state)
        model.eval()
        return model

    def _load_dynamic_model(self, path: str):
        if torch is None:
            return None
        output_dim = max(self.label_map.values()) + 1 if self.label_map else 1
        model = DynamicBiGRU(input_dim=136, output_dim=output_dim)
        p = Path(path)
        if p.exists():
            state = torch.load(p, map_location="cpu")
            model.load_state_dict(state)
        model.eval()
        return model

    def _rule_based_probs(self, feature: np.ndarray) -> np.ndarray:
        probs = np.zeros(len(self.label_map), dtype=np.float32)
        if len(probs) == 0:
            return np.array([1.0], dtype=np.float32)
        idx = int(abs(np.sum(feature) * 1000)) % len(probs)
        probs[idx] = 0.75
        probs += 0.25 / len(probs)
        return probs

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        ex = np.exp(logits - np.max(logits))
        return ex / np.sum(ex)

    def _mask_probs(self, probs: np.ndarray, keep_indices: set[int]) -> np.ndarray:
        masked = np.zeros_like(probs)
        if not keep_indices:
            return probs
        for idx in keep_indices:
            if 0 <= idx < len(masked):
                masked[idx] = probs[idx]
        total = masked.sum()
        if total <= 1e-8:
            return probs
        return masked / total

    def _predict_dynamic_probs(self) -> np.ndarray | None:
        if self.dynamic_model is None or torch is None or len(self.sequence) < self.sequence.maxlen:
            return self.last_dynamic_probs
        if self.frame_count % self.dynamic_stride != 0 and self.last_dynamic_probs is not None:
            return self.last_dynamic_probs

        seq = np.stack(self.sequence, axis=0)
        if self.dynamic_normalizer.mean is not None:
            seq = (seq - self.dynamic_normalizer.mean) / self.dynamic_normalizer.std
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            logits = self.dynamic_model(x).squeeze(0).numpy()
        self.last_dynamic_probs = self._softmax(logits)
        return self.last_dynamic_probs

    def predict(self, frame_bgr, mode: str = "practice", target_mode: str = "static") -> Dict[str, object]:
        start = time.time()
        self.frame_count += 1
        extraction = self.tracker.extract(frame_bgr)
        if extraction["status"] != "ok":
            if extraction["status"] == "no_hand" and len(self.sequence) > 0:
                self.sequence.clear()
            return {
                "status": extraction["status"],
                "label": "NO_HAND",
                "confidence": 0.0,
                "model_used": "dynamic" if target_mode == "dynamic" else "static",
                "latency_ms": int((time.time() - start) * 1000),
                "extraction": extraction,
                "stable": False,
            }

        feature = build_feature_vector(extraction)
        if self.normalizer.mean is not None:
            feature = self.normalizer.transform(feature)
        self.sequence.append(feature)

        if self.static_model is not None and torch is not None:
            with torch.no_grad():
                x = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
                logits = self.static_model(x).squeeze(0).numpy()
                static_probs = self._softmax(logits)
        else:
            static_probs = self._rule_based_probs(feature)

        if target_mode == "dynamic":
            if self.dynamic_model is None or torch is None:
                return {
                    "status": "dynamic_model_unavailable",
                    "label": "DYNAMIC_MODEL_UNAVAILABLE",
                    "confidence": 0.0,
                    "model_used": "dynamic",
                    "latency_ms": int((time.time() - start) * 1000),
                    "extraction": extraction,
                    "stable": False,
                }

            dynamic_probs = self._predict_dynamic_probs()
            if dynamic_probs is None:
                return {
                    "status": "warming_up",
                    "label": "WARMING_UP_SEQUENCE",
                    "confidence": 0.0,
                    "model_used": "dynamic",
                    "latency_ms": int((time.time() - start) * 1000),
                    "extraction": extraction,
                    "stable": False,
                }
            probs = self._mask_probs(dynamic_probs, self.dynamic_class_indices)
            model_used = "dynamic"
            threshold = self.dynamic_threshold
        elif target_mode == "static":
            probs = static_probs
            model_used = "static"
            threshold = self.static_threshold
        else:
            probs = static_probs
            model_used = "static"
            threshold = self.static_threshold

        pred_idx, smoothed_conf = self.smoother.update(probs.astype(np.float32))
        label = self.idx_to_label.get(pred_idx, "UNKNOWN")

        now = time.time()
        stable = False
        if target_mode == "static":
            if pred_idx != self._static_candidate_idx:
                self._static_candidate_idx = pred_idx
                self._static_candidate_since = now
            stable = (now - self._static_candidate_since) >= self.static_hold_seconds and smoothed_conf >= threshold
        else:
            if pred_idx == self._dynamic_streak_idx and smoothed_conf >= threshold:
                self._dynamic_streak_count += 1
            else:
                self._dynamic_streak_idx = pred_idx
                self._dynamic_streak_count = 1 if smoothed_conf >= threshold else 0
            stable = self._dynamic_streak_count >= self.dynamic_confirm_frames

        status = "ok" if stable else "uncertain"

        return {
            "status": status,
            "label": label,
            "confidence": smoothed_conf,
            "model_used": model_used,
            "latency_ms": int((time.time() - start) * 1000),
            "extraction": extraction,
            "stable": stable,
        }
