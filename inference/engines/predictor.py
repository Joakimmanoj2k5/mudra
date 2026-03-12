from __future__ import annotations

import json
import pickle
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

try:
    import cv2 as _cv2_keras
except Exception:
    _cv2_keras = None

try:
    import os as _os_keras
    _os_keras.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    _os_keras.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    import tf_keras as _tf_keras
except Exception:
    _tf_keras = None


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
        self.static_threshold = float(config.get("inference", {}).get("static_threshold", 0.15))
        self.dynamic_threshold = float(config.get("inference", {}).get("dynamic_threshold", 0.25))
        self.static_hold_seconds = 0.3
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
        self._load_centroids()
        self._load_letter_rf()
        self.dynamic_normalizer = FeatureNormalizer()
        self._load_dynamic_norm("models/registry/dynamic_norm_stats.json")
        self.dynamic_model = self._load_dynamic_model(config.get("model", {}).get("dynamic_model_path", ""))
        self.last_dynamic_probs = None
        self._static_candidate_idx = -1
        self._static_candidate_since = 0.0
        self._dynamic_streak_idx = -1
        self._dynamic_streak_count = 0

        # Keras image model (Teachable Machine)
        self.keras_model = None
        self.keras_labels: list[str] = []
        self._keras_label_to_main_idx: dict[int, int] = {}
        self._load_keras_model()

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

    def _load_keras_model(self) -> None:
        """Load Teachable Machine Keras H5 model + labels.txt."""
        model_path = Path("converted_keras/keras_model.h5")
        label_path = Path("converted_keras/labels.txt")
        if _tf_keras is None or not model_path.exists() or not label_path.exists():
            return
        try:
            self.keras_model = _tf_keras.models.load_model(str(model_path), compile=False)
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            self.keras_labels = []
            for line in lines:
                parts = line.strip().split(maxsplit=1)
                self.keras_labels.append(parts[1] if len(parts) > 1 else parts[0])
            # Map each keras label index → main label_map index (case-insensitive match)
            label_map_lower = {k.lower(): v for k, v in self.label_map.items()}
            for ki, kname in enumerate(self.keras_labels):
                main_idx = label_map_lower.get(kname.lower().strip())
                if main_idx is not None:
                    self._keras_label_to_main_idx[ki] = main_idx
        except Exception:
            self.keras_model = None

    def _keras_probs(self, frame_bgr) -> np.ndarray | None:
        """Run Keras image model on a camera frame, return 71-class probs."""
        if self.keras_model is None or _cv2_keras is None:
            return None
        try:
            img = _cv2_keras.resize(frame_bgr, (224, 224))
            img = _cv2_keras.cvtColor(img, _cv2_keras.COLOR_BGR2RGB)
            arr = (img.astype(np.float32) / 127.5) - 1.0
            preds = self.keras_model.predict(arr[np.newaxis, ...], verbose=0)[0]
            n_cls = len(self.label_map)
            full_probs = np.zeros(n_cls, dtype=np.float32)
            for ki, main_idx in self._keras_label_to_main_idx.items():
                if ki < len(preds) and main_idx < n_cls:
                    full_probs[main_idx] = preds[ki]
            total = full_probs.sum()
            if total > 1e-8:
                full_probs /= total
            return full_probs
        except Exception:
            return None

    def _load_class_modes(self, path: str) -> Dict[str, str]:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        from utils.common.gesture_catalog import all_gestures

        modes = {g.display_name: g.gesture_mode for g in all_gestures()}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(modes, indent=2), encoding="utf-8")
        return modes

    @staticmethod
    def _swap_hand_slots(feature: np.ndarray) -> np.ndarray:
        """Swap left/right hand slots in a 136-dim feature vector."""
        swapped = feature.copy()
        swapped[:63] = feature[63:126]
        swapped[63:126] = feature[:63]
        swapped[126:131] = feature[131:136]
        swapped[131:136] = feature[126:131]
        return swapped

    @staticmethod
    def _get_active_hand_coords(feature: np.ndarray) -> np.ndarray:
        """Get 21x3 coords of the dominant hand from a 136-dim raw feature."""
        left = feature[:63].reshape(21, 3)
        right = feature[63:126].reshape(21, 3)
        return left if np.abs(left).sum() > np.abs(right).sum() else right

    @staticmethod
    def _hand_shape_features(coords: np.ndarray) -> np.ndarray:
        """Extract 36-dim geometric hand shape descriptor from 21x3 landmarks."""
        c = coords
        feats = []
        wrist = c[0]
        finger_ids = [
            (4, 3, 2, 1), (8, 7, 6, 5), (12, 11, 10, 9),
            (16, 15, 14, 13), (20, 19, 18, 17),
        ]
        # Finger extension ratios (5)
        for tip, dip, pip, mcp in finger_ids:
            td = np.linalg.norm(c[tip] - wrist)
            pd = np.linalg.norm(c[pip] - wrist)
            feats.append(td / pd if pd > 1e-6 else 1.0)
        # Finger curl angles at PIP (5)
        for tip, dip, pip, mcp in finger_ids:
            v1 = c[mcp] - c[pip]
            v2 = c[tip] - c[pip]
            d = np.linalg.norm(v1) * np.linalg.norm(v2)
            if d > 1e-6:
                feats.append(float(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1))))
            else:
                feats.append(0.0)
        # Fingertip positions relative to palm center (10)
        palm = c[9]
        for tip, dip, pip, mcp in finger_ids:
            diff = c[tip] - palm
            feats.append(diff[0])
            feats.append(diff[1])
        # Inter-finger spread (4)
        tips = [4, 8, 12, 16, 20]
        for i in range(4):
            v1 = c[tips[i]] - palm
            v2 = c[tips[i + 1]] - palm
            d = np.linalg.norm(v1) * np.linalg.norm(v2)
            if d > 1e-6:
                feats.append(float(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1))))
            else:
                feats.append(0.0)
        # Thumb-to-finger distances (4)
        for tid in [8, 12, 16, 20]:
            feats.append(np.linalg.norm(c[4] - c[tid]))
        # Fingertip y-coords (5)
        for tip, dip, pip, mcp in finger_ids:
            feats.append(c[tip][1])
        # Thumb-finger contact (3)
        for tid in [8, 12, 16]:
            feats.append(np.linalg.norm(c[4] - c[tid]))
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _extract_active_hand(feature: np.ndarray) -> np.ndarray:
        """Extract 68-dim active hand features (63 coords + 5 angles)."""
        left = feature[:63]
        right = feature[63:126]
        if np.abs(left).sum() > np.abs(right).sum():
            return np.concatenate([left, feature[126:131]])
        return np.concatenate([right, feature[131:136]])

    def _load_letter_rf(self) -> None:
        """Load the Random Forest letter classifier."""
        self.letter_rf = None
        self.letter_scaler = None
        self.letter_idx_list = None
        p = Path("models/registry/letter_rf_model.pkl")
        if not p.exists():
            return
        with open(p, "rb") as f:
            d = pickle.load(f)
        self.letter_rf = d["model"]
        self.letter_scaler = d["scaler"]
        self.letter_idx_list = d["letter_idx_list"]

    def _letter_rf_probs(self, raw_feature: np.ndarray) -> np.ndarray:
        """Get 71-class probabilities using the RF letter classifier.
        Non-letter classes get zero probability."""
        n_cls = len(self.label_map)
        if self.letter_rf is None:
            return np.ones(n_cls, dtype=np.float32) / n_cls
        coords = self._get_active_hand_coords(raw_feature)
        shape = self._hand_shape_features(coords)
        active = self._extract_active_hand(raw_feature)
        combined = np.concatenate([shape, active])
        scaled = self.letter_scaler.transform(combined.reshape(1, -1))
        rf_probs = self.letter_rf.predict_proba(scaled)[0]
        # Map compact letter probs back to 71-class space
        full_probs = np.zeros(n_cls, dtype=np.float32)
        for compact_idx, orig_idx in enumerate(self.letter_idx_list):
            if orig_idx < n_cls:
                full_probs[orig_idx] = rf_probs[compact_idx]
        return full_probs

    def _load_centroids(self) -> None:
        """Load pre-computed class centroids and exemplars for similarity matching."""
        self.centroids = None
        self.exemplar_matrix = None
        self.exemplar_labels = None
        p = Path("models/registry/static_centroids.npz")
        if not p.exists():
            return
        data = np.load(p)
        self.centroids = data["centroids"]  # (n_cls, feat_dim), L2-normalized

        ep = Path("models/registry/static_exemplars.npz")
        if not ep.exists():
            return
        edata = np.load(ep)
        feats_list = []
        labels_list = []
        for key in edata.files:
            cls_idx = int(key.split("_")[1])
            for feat in edata[key]:
                norm = np.linalg.norm(feat)
                if norm > 1e-8:
                    feats_list.append(feat / norm)
                    labels_list.append(cls_idx)
        if feats_list:
            self.exemplar_matrix = np.stack(feats_list, axis=0)
            self.exemplar_labels = np.array(labels_list, dtype=np.int32)

    def _centroid_probs(self, feature: np.ndarray) -> np.ndarray:
        """Class probabilities via cosine similarity to centroids + KNN exemplars.
        Tries both original and hand-swapped feature, takes the best per class."""
        n_cls = self.centroids.shape[0]
        norm = np.linalg.norm(feature)
        if norm < 1e-8:
            return np.ones(n_cls, dtype=np.float32) / n_cls

        # Try both the original and hand-swapped feature
        swapped = self._swap_hand_slots(feature)
        swap_norm = np.linalg.norm(swapped)

        best_probs = np.zeros(n_cls, dtype=np.float32)
        for feat, fnorm in [(feature, norm), (swapped, swap_norm)]:
            if fnorm < 1e-8:
                continue
            feat_unit = feat / fnorm

            centroid_sims = self.centroids @ feat_unit
            centroid_probs = self._softmax(centroid_sims / 0.07)

            if self.exemplar_matrix is not None:
                all_sims = self.exemplar_matrix @ feat_unit
                k = min(15, len(all_sims))
                top_k_idx = np.argpartition(-all_sims, k)[:k]
                knn_votes = np.zeros(n_cls, dtype=np.float32)
                for idx in top_k_idx:
                    sim = max(all_sims[idx], 0.0)
                    knn_votes[self.exemplar_labels[idx]] += sim * sim
                total = knn_votes.sum()
                if total > 1e-8:
                    knn_probs = knn_votes / total
                else:
                    knn_probs = centroid_probs
                probs = 0.7 * knn_probs + 0.3 * centroid_probs
            else:
                probs = centroid_probs

            # Element-wise max: for each class, take the better probability
            best_probs = np.maximum(best_probs, probs)

        total = best_probs.sum()
        if total > 1e-8:
            best_probs /= total
        return best_probs

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
        raw_feature = feature.copy()  # Keep un-normalized copy for shape features
        if self.normalizer.mean is not None:
            feature = self.normalizer.transform(feature)
        self.sequence.append(feature)

        if self.centroids is not None:
            centroid_p = self._centroid_probs(feature)
            if self.static_model is not None and torch is not None:
                with torch.no_grad():
                    x_orig = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
                    swapped_feat = self._swap_hand_slots(feature)
                    x_swap = torch.tensor(swapped_feat, dtype=torch.float32).unsqueeze(0)
                    logits_orig = self.static_model(x_orig).squeeze(0).numpy()
                    logits_swap = self.static_model(x_swap).squeeze(0).numpy()
                    mlp_orig = self._softmax(logits_orig)
                    mlp_swap = self._softmax(logits_swap)
                    mlp_p = np.maximum(mlp_orig, mlp_swap)
                    mlp_p /= mlp_p.sum()
                static_probs = 0.70 * centroid_p + 0.30 * mlp_p
            else:
                static_probs = centroid_p
        elif self.static_model is not None and torch is not None:
            with torch.no_grad():
                x_orig = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
                swapped_feat = self._swap_hand_slots(feature)
                x_swap = torch.tensor(swapped_feat, dtype=torch.float32).unsqueeze(0)
                logits_orig = self.static_model(x_orig).squeeze(0).numpy()
                logits_swap = self.static_model(x_swap).squeeze(0).numpy()
                mlp_orig = self._softmax(logits_orig)
                mlp_swap = self._softmax(logits_swap)
                static_probs = np.maximum(mlp_orig, mlp_swap)
                static_probs /= static_probs.sum()
        else:
            static_probs = self._rule_based_probs(feature)

        # Blend with RF letter classifier for much better letter recognition
        if self.letter_rf is not None:
            rf_probs = self._letter_rf_probs(raw_feature)
            # RF dominates for letters; existing system handles non-letter words
            static_probs = 0.65 * rf_probs + 0.35 * static_probs

        # Blend with Keras image model if available
        keras_p = self._keras_probs(frame_bgr)
        if keras_p is not None:
            static_probs = 0.50 * keras_p + 0.50 * static_probs

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
