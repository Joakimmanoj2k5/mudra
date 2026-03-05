from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class CameraFrame:
    frame: any
    ts: float
    fps: float
    should_process: bool


class CameraService:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        max_process_fps: float = 18.0,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.max_process_fps = max_process_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_ts = 0.0
        self.last_process_ts = 0.0

    def open(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True

        # Try preferred index first, then common alternates.
        index_candidates = [self.camera_index] + [idx for idx in (0, 1, 2) if idx != self.camera_index]
        backend_candidates = [None]
        for name in ("CAP_AVFOUNDATION", "CAP_DSHOW", "CAP_MSMF", "CAP_V4L2", "CAP_ANY"):
            if hasattr(cv2, name):
                backend_candidates.append(getattr(cv2, name))

        for idx in index_candidates:
            for backend in backend_candidates:
                try:
                    cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
                    if not cap or not cap.isOpened():
                        if cap:
                            cap.release()
                        continue
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    # Warm-up read — some cameras need a few frames
                    for _ in range(3):
                        cap.grab()
                    self.cap = cap
                    self.camera_index = idx
                    self.last_ts = time.time()
                    return True
                except Exception:
                    continue
        self.cap = None
        return False

    def read(self) -> Optional[CameraFrame]:
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return None
        try:
            ok, frame = self.cap.read()
        except Exception:
            self.release()
            return None
        if not ok or frame is None:
            self.release()
            return None
        now = time.time()
        dt = max(now - self.last_ts, 1e-6)
        self.last_ts = now
        fps = 1.0 / dt
        should_process = (now - self.last_process_ts) >= (1.0 / max(self.max_process_fps, 1.0))
        if should_process:
            self.last_process_ts = now
        return CameraFrame(frame=frame, ts=now, fps=fps, should_process=should_process)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
