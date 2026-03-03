"""Runtime environment capability checks for MUDRA."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def _check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _check_camera(index: int = 0) -> bool:
    try:
        import cv2

        cap = cv2.VideoCapture(index)
        ok = bool(cap and cap.isOpened())
        if ok:
            ret, _ = cap.read()
            ok = bool(ret)
        cap.release()
        return ok
    except Exception:
        return False


def check_environment(config: Dict[str, object]) -> Dict[str, bool]:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    static_path = Path(str(model_cfg.get("static_model_path", "models/static/static_mlp_v001.pt")))
    dynamic_path = Path(str(model_cfg.get("dynamic_model_path", "models/dynamic/dynamic_bigru_v001.pt")))

    mediapipe_ok = _check_import("mediapipe")
    _ = _check_import("cv2")
    torch_ok = _check_import("torch")
    camera_ok = _check_camera()

    return {
        "mediapipe": mediapipe_ok,
        "torch": torch_ok,
        "camera": camera_ok,
        "static_model_loaded": torch_ok and static_path.exists(),
        "dynamic_model_loaded": torch_ok and dynamic_path.exists(),
    }
