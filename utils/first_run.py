"""First-run setup for MUDRA.

Generates initial (untrained) model weights, downloads MediaPipe model,
and generates reference images if they don't already exist.
Called automatically on app startup.
"""

from __future__ import annotations

import json
from pathlib import Path


def ensure_mediapipe_model() -> None:
    """Download the MediaPipe hand landmarker model if it doesn't exist."""
    model_path = Path("models/mediapipe/hand_landmarker.task")
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print("[FirstRun] Downloading MediaPipe hand landmarker model...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"[FirstRun] Downloaded hand_landmarker.task ({model_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"[FirstRun] Could not download MediaPipe model: {e}")


def ensure_models() -> None:
    """Create initial model weight files if they don't exist."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return

    label_map_path = Path("models/registry/label_map.json")
    if not label_map_path.exists():
        return

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    output_dim = max(label_map.values()) + 1
    input_dim = 136

    static_path = Path("models/static/static_mlp_v001.pt")
    if not static_path.exists():
        static_path.parent.mkdir(parents=True, exist_ok=True)

        class StaticMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
                    nn.Linear(128, output_dim),
                )
            def forward(self, x):
                return self.net(x)

        torch.save(StaticMLP().state_dict(), str(static_path))

    dynamic_path = Path("models/dynamic/dynamic_bigru_v001.pt")
    if not dynamic_path.exists():
        dynamic_path.parent.mkdir(parents=True, exist_ok=True)

        class DynamicBiGRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(input_dim, 128, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
                self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, output_dim))
            def forward(self, x):
                y, _ = self.gru(x)
                return self.head(y.mean(dim=1))

        torch.save(DynamicBiGRU().state_dict(), str(dynamic_path))


def ensure_reference_images() -> None:
    """Generate reference card images if the cache is empty."""
    cache = Path("data/assets/gestures/image_cache")
    if cache.exists() and len(list(cache.glob("*.png"))) >= 100:
        return  # Already generated
    try:
        from scripts.generate_reference_images import generate_all_reference_images
        generate_all_reference_images()
    except Exception as e:
        print(f"[FirstRun] Could not generate reference images: {e}")


def run_first_time_setup() -> None:
    """Run all first-time setup tasks."""
    ensure_mediapipe_model()
    ensure_models()
    ensure_reference_images()
