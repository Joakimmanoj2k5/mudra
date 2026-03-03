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
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.last_ts = time.time()
        return bool(self.cap and self.cap.isOpened())

    def read(self) -> Optional[CameraFrame]:
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return None
        ok, frame = self.cap.read()
        if not ok:
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
