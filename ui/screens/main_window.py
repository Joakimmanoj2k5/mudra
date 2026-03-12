from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QSize, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from database.db import DatabaseManager
from inference.camera.camera_worker import CameraService
from inference.engines.predictor import GesturePredictor
from inference.overlay.draw import draw_overlay
from ui.state.session import SessionState
from utils.common.gesture_catalog import LEVEL_INFO, all_gestures
from utils.common.security import create_access_token, verify_password
from utils.environment_check import check_environment
from utils.gesture_media_mapper import get_gesture_reference, get_media_path
from utils.io.config_loader import load_config, save_config


class ReferenceVideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.media_path: Optional[str] = None
        self._media_changed = False

    def set_media(self, media_path: Optional[str]):
        self.media_path = media_path
        self._media_changed = True

    def run(self):
        while self._run_flag:
            if not self.media_path:
                if self._media_changed:
                    self.status_signal.emit("Reference video not available")
                    self._media_changed = False
                time.sleep(0.5)
                continue

            cap = cv2.VideoCapture(self.media_path)
            if not cap.isOpened():
                self.status_signal.emit("Could not open reference video")
                self._media_changed = False
                time.sleep(1.0)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1:
                fps = 24
            sleep_s = 1.0 / min(max(fps, 10.0), 30.0)

            self.status_signal.emit("▶ Playing reference")
            current_path = self.media_path
            while self._run_flag and self.media_path == current_path:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_signal.emit(qimg.copy())
                time.sleep(sleep_s)
            cap.release()

    def stop(self):
        self._run_flag = False
        self.wait(1000)


class InferenceThread(QThread):
    frame_signal = pyqtSignal(QImage)
    result_signal = pyqtSignal(dict)

    def __init__(self, predictor: GesturePredictor, env_status: Dict[str, bool]):
        super().__init__()
        self._run_flag = True
        self.predictor = predictor
        self.camera = CameraService(max_process_fps=18.0)
        self.target_name = ""
        self.target_mode = "static"
        self.env_status = env_status
        self.last_result: Dict[str, object] = {
            "status": "uncertain",
            "label": "-",
            "confidence": 0.0,
            "model_used": "-",
            "latency_ms": 0,
            "stable": False,
        }

    def run(self):
        fail_count = 0
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            if self.camera.open():
                break
            retry_count += 1
            time.sleep(0.5)
        else:
            self.result_signal.emit({
                "status": "camera_error",
                "label": "CAMERA_ERROR",
                "confidence": 0.0,
                "latency_ms": 0,
                "fps": 0.0,
                "model_used": "-",
                "stable": False,
                "perf_warning": "Camera could not be opened. Check permissions and connections.",
            })
            return

        while self._run_flag:
            try:
                packet = self.camera.read()
            except Exception:
                packet = None

            if packet is None:
                fail_count += 1
                if fail_count > 30:
                    self.result_signal.emit({
                        "status": "camera_error",
                        "label": "CAMERA_DISCONNECTED",
                        "confidence": 0.0,
                        "latency_ms": 0,
                        "fps": 0.0,
                        "model_used": "-",
                        "stable": False,
                        "perf_warning": "",
                    })
                    break
                time.sleep(0.05)
                continue

            fail_count = 0
            frame = packet.frame
            if packet.should_process:
                self.last_result = self.predictor.predict(frame, target_mode=self.target_mode)

            display_result = dict(self.last_result)
            display_result["env"] = dict(self.env_status)
            display_result["perf_warning"] = "Low Performance Warning" if packet.fps < 15 else ""

            # Draw skeleton on original frame (landmark coords match original space),
            # then flip for mirror display, then draw text overlays so they read normally.
            self.predictor.tracker.draw(frame, self.last_result.get("extraction", {}))
            display_frame = cv2.flip(frame, 1)
            draw_overlay(display_frame, display_result, packet.fps, target=self.target_name)

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_signal.emit(qimg.copy())

            emit_result = {
                "status": display_result.get("status"),
                "label": display_result.get("label"),
                "confidence": float(display_result.get("confidence", 0.0)),
                "latency_ms": int(display_result.get("latency_ms", 0)),
                "fps": float(packet.fps),
                "model_used": display_result.get("model_used", "-"),
                "stable": bool(display_result.get("stable", False)),
                "perf_warning": display_result.get("perf_warning", ""),
            }
            self.result_signal.emit(emit_result)

        self.camera.release()

    def set_target(self, target_name: str) -> None:
        self.target_name = target_name

    def set_target_mode(self, target_mode: str) -> None:
        self.target_mode = target_mode if target_mode in {"static", "dynamic"} else "static"

    def stop(self):
        self._run_flag = False
        self.wait(1500)


class MudraMainWindow(QMainWindow):
    IDX_LOGIN = 0
    IDX_DASH = 1
    IDX_STUDY = 2
    IDX_PRACTICE = 3
    IDX_QUIZ = 4
    IDX_ANALYTICS = 5
    IDX_ADMIN = 6

    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.db = DatabaseManager(self.config.get("database", {}).get("sqlite_path", "database/mudra.db"))
        self.db.seed_core_data()
        self.session = SessionState()
        self.env_status = check_environment(self.config)
        self.predictor = GesturePredictor(self.config)

        self.gesture_rows: List[Dict[str, object]] = []
        self.level_groups: Dict[int, List[Dict[str, object]]] = {lvl: [] for lvl in LEVEL_INFO}
        self.gesture_level_by_id: Dict[str, int] = {}
        self.selected_gesture: Optional[Dict[str, object]] = None
        self.current_result: Optional[Dict[str, object]] = None
        self.quiz_queue: List[Dict[str, object]] = []
        self.quiz_index = 0
        self.quiz_score = 0
        self._current_study_level = min(LEVEL_INFO) if LEVEL_INFO else 1
        self._current_practice_level = min(LEVEL_INFO) if LEVEL_INFO else 1
        self._current_study_rows: List[Dict[str, object]] = []
        self._current_practice_rows: List[Dict[str, object]] = []

        self._study_gesture_id: Optional[str] = None
        self._study_started_at: Optional[float] = None

        self.setWindowTitle(self.config.get("window_title", "MUDRA"))
        self.resize(1280, 840)
        self.setMinimumSize(980, 680)
        import platform
        self.sys_font = "Segoe UI" if platform.system() == "Windows" else "Helvetica Neue"
        self.setStyleSheet(self._theme())

        self.inference_thread: Optional[InferenceThread] = None
        self.study_ref_thread: Optional[ReferenceVideoThread] = None
        self.practice_ref_thread: Optional[ReferenceVideoThread] = None
        self._last_qimage: Optional[QImage] = None
        self._last_study_ref: Optional[QImage] = None
        self._last_practice_ref: Optional[QImage] = None

        self._build_ui()
        self._start_reference_threads()
        self._apply_environment_status()

    def _theme(self) -> str:
        return """
            QMainWindow {
                background-color: #05070a;
                color: #e5e7eb;
            }
            QWidget {
                font-family: "Helvetica Neue";
                color: #e5e7eb;
            }
            QLabel { color: #e5e7eb; font-size: 13px; }

            #Sidebar {
                background-color: #0b0e14;
                border: 1px solid #1f2937;
                border-radius: 18px;
            }
            #BrandTitle {
                color: #22c55e;
                font-size: 30px;
                font-weight: 700;
            }
            #BrandSub {
                color: #94a3b8;
                font-size: 12px;
            }
            QPushButton#NavButton {
                border-radius: 12px;
                padding: 11px 12px;
                font-weight: 600;
                color: #cbd5e1;
                background-color: transparent;
                border: 1px solid transparent;
                text-align: left;
            }
            QPushButton#NavButton:hover {
                background-color: #121822;
                border-color: #263241;
                color: #f8fafc;
            }
            QPushButton#NavButton:checked {
                background-color: #12351d;
                border-color: #22c55e;
                color: #dcfce7;
            }
            QPushButton#LogoutButton {
                border-radius: 12px;
                padding: 11px 12px;
                color: #fecaca;
                background-color: #3f1d1d;
                border: 1px solid #ef4444;
                text-align: left;
                font-weight: 700;
            }
            QPushButton#LogoutButton:hover {
                background-color: #5c2323;
                border-color: #ef4444;
                color: #ffe4e6;
            }

            #CenterWrap {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #060a10,
                    stop: 0.55 #0a0f17,
                    stop: 1 #101726
                );
                border: 1px solid #1f2937;
                border-radius: 18px;
            }
            #FeedbackPanel {
                background-color: #0a0f17;
                border: 1px solid #1f2937;
                border-radius: 18px;
            }
            #InfoCard {
                background-color: #121826;
                border: 1px solid #2a3446;
                border-radius: 14px;
            }
            #SectionTitle {
                color: #f8fafc;
                font-size: 15px;
                font-weight: 700;
            }
            #SectionMeta {
                color: #94a3b8;
                font-size: 12px;
            }
            #HeroCard {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0d1b2c,
                    stop: 0.55 #10233b,
                    stop: 1 #153429
                );
                border: 1px solid #21455f;
                border-radius: 24px;
            }
            #MetricCard {
                background-color: rgba(9, 13, 24, 0.92);
                border: 1px solid #243041;
                border-radius: 16px;
            }
            QLabel#MetricLabel {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#MetricValue {
                color: #f8fafc;
                font-size: 23px;
                font-weight: 700;
            }
            QPushButton#DashboardLevelCard {
                border-radius: 20px;
                padding: 18px;
                text-align: left;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton#DashboardLevelCard:hover {
                border-color: #f8fafc;
            }
            #StatusPill {
                background-color: #0b0f18;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 6px 10px;
                font-weight: 700;
            }
            QPushButton#LevelSelector {
                border-radius: 18px;
                padding: 14px 16px;
                text-align: left;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton#QuickAction {
                border-radius: 18px;
                padding: 16px 18px;
                text-align: left;
                font-weight: 700;
                font-size: 14px;
            }
            #LoginPage {
                background: qradialgradient(
                    cx: 0.18, cy: 0.22, radius: 1.05,
                    fx: 0.18, fy: 0.22,
                    stop: 0 rgba(24, 119, 242, 0.18),
                    stop: 0.42 rgba(10, 18, 32, 0.10),
                    stop: 1 #04070d
                );
            }
            #LoginShell {
                background: rgba(5, 11, 21, 0.92);
                border: 1px solid #1c2c43;
                border-radius: 28px;
            }
            #LoginHero {
                background: transparent;
                border: none;
            }
            #LoginFormCard {
                background: rgba(10, 18, 32, 0.96);
                border: 1px solid #20324a;
                border-radius: 22px;
            }
            QLabel#LoginEyebrow {
                color: #22c55e;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
                background: transparent;
                border: none;
            }
            QLabel#LoginTitle {
                color: #f8fafc;
                font-size: 34px;
                font-weight: 700;
                background: transparent;
                border: none;
            }
            QLabel#LoginSubtitle {
                color: #9fb7d1;
                font-size: 14px;
                line-height: 1.5;
                background: transparent;
                border: none;
            }
            QLabel#LoginFeature {
                color: #dbeafe;
                font-size: 13px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#LoginHint {
                color: #7dd3fc;
                font-size: 12px;
                background: transparent;
                border: none;
            }
            QLabel#LoginFieldLabel {
                color: #cbd5e1;
                font-size: 12px;
                font-weight: 700;
                background: transparent;
                border: none;
            }
            QLineEdit#LoginInput {
                border-radius: 14px;
                border: 1px solid #2b3d57;
                padding: 12px 14px;
                background-color: #091221;
                color: #f8fafc;
                font-size: 14px;
            }
            QLineEdit#LoginInput:focus {
                border-color: #22c55e;
                background-color: #0b1627;
            }
            QPushButton#LoginPrimary {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #16a34a, stop:1 #15803d);
                border-radius: 14px;
                color: #dcfce7;
                border: none;
                font-size: 15px;
                font-weight: 700;
                padding: 12px;
            }
            QPushButton#LoginPrimary:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22c55e, stop:1 #16a34a);
            }
            QPushButton#LoginSecondary {
                background: #0a1526;
                color: #cbd5e1;
                border: 1px solid #2b3d57;
                border-radius: 14px;
                font-size: 13px;
                font-weight: 600;
                padding: 11px;
            }
            QPushButton#LoginSecondary:hover {
                border-color: #7dd3fc;
                color: #f8fafc;
                background: #0e1a2d;
            }

            QPushButton {
                border-radius: 10px;
                padding: 10px;
                font-weight: 600;
                color: #e2e8f0;
                background-color: #1a2334;
                border: 1px solid #334155;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #263348; }
            QPushButton#AccentButton {
                background-color: #16a34a;
                border-color: #22c55e;
                color: #dcfce7;
            }
            QPushButton#AccentButton:hover {
                background-color: #15803d;
                border-color: #4ade80;
            }
            QPushButton#GhostButton {
                background-color: #111827;
                border: 1px solid #334155;
                color: #cbd5e1;
            }
            QPushButton#GhostButton:hover {
                background-color: #1e293b;
                color: #f8fafc;
            }

            QLineEdit {
                border-radius: 8px;
                border: 1px solid #334155;
                padding: 8px;
                background-color: #0b1220;
                color: #f8fafc;
            }
            QTextEdit, QComboBox {
                border-radius: 8px;
                border: 1px solid #334155;
                background-color: #0b1220;
                color: #f8fafc;
                padding: 6px;
            }
            QListWidget, QTableWidget {
                background-color: #0b1220;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 10px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background-color: #14532d;
                color: #dcfce7;
            }
            QListWidget::item:hover {
                background-color: #1e293b;
            }
            QListWidget#GestureList::item {
                padding: 10px 12px;
                border-radius: 10px;
                margin: 2px 0;
            }
            QHeaderView::section {
                background-color: #111827;
                color: #cbd5e1;
                border: 1px solid #334155;
                padding: 4px;
            }
        """

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        self.setCentralWidget(central)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self._sidebar_expanded = True
        self._sidebar_full_width = 220
        self._sidebar_collapsed_width = 54
        self.sidebar.setFixedWidth(self._sidebar_full_width)
        sbl = QVBoxLayout(self.sidebar)
        sbl.setContentsMargins(12, 14, 12, 14)
        sbl.setSpacing(8)

        # Collapse / expand toggle
        self.sidebar_toggle = QPushButton("\u276E")
        self.sidebar_toggle.setFixedSize(30, 30)
        self.sidebar_toggle.setCursor(Qt.PointingHandCursor)
        self.sidebar_toggle.setStyleSheet(
            "QPushButton { background:#1a2332; color:#94a3b8; border:1px solid #2a3446; "
            "border-radius:8px; font-size:14px; font-weight:700; } "
            "QPushButton:hover { background:#243041; color:#f8fafc; }"
        )
        self.sidebar_toggle.clicked.connect(self._toggle_sidebar)
        toggle_row = QHBoxLayout()
        toggle_row.addStretch()
        toggle_row.addWidget(self.sidebar_toggle)
        sbl.addLayout(toggle_row)

        self.logo_label = QLabel("MUDRA")
        self.logo_label.setObjectName("BrandTitle")
        self.logo_label.setFont(QFont(self.sys_font, 28, QFont.Bold))
        self.logo_sub = QLabel("Interactive ISL Learning")
        self.logo_sub.setObjectName("BrandSub")
        sbl.addWidget(self.logo_label)
        sbl.addWidget(self.logo_sub)
        sbl.addSpacing(8)

        nav = [
            ("Dashboard", self.IDX_DASH, QStyle.SP_DesktopIcon),
            ("Study", self.IDX_STUDY, QStyle.SP_FileDialogContentsView),
            ("Practice", self.IDX_PRACTICE, QStyle.SP_MediaPlay),
            ("Quiz", self.IDX_QUIZ, QStyle.SP_FileDialogDetailedView),
            ("Analytics", self.IDX_ANALYTICS, QStyle.SP_DriveHDIcon),
            ("Admin", self.IDX_ADMIN, QStyle.SP_FileDialogInfoView),
        ]
        self.nav_buttons = {}
        self._nav_index_map: Dict[int, QPushButton] = {}
        for title, idx, icon_role in nav:
            b = QPushButton(title)
            b.setObjectName("NavButton")
            b.setCheckable(True)
            b.setIcon(self.style().standardIcon(icon_role))
            b.setIconSize(QSize(18, 18))
            b.clicked.connect(lambda _=False, i=idx: self.navigate_to(i))
            sbl.addWidget(b)
            self.nav_buttons[title] = b
            self._nav_index_map[idx] = b

        self.btn_logout = QPushButton("Logout")
        self.btn_logout.setObjectName("LogoutButton")
        self.btn_logout.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.btn_logout.clicked.connect(self.logout)
        sbl.addStretch()
        sbl.addWidget(self.btn_logout)

        # Keep references for collapse / expand
        self._sidebar_nav_buttons_list = list(self.nav_buttons.values())

        content_wrap = QFrame()
        content_wrap.setObjectName("CenterWrap")
        content = QVBoxLayout(content_wrap)
        content.setContentsMargins(16, 16, 16, 16)
        content.setSpacing(12)
        self.status_bar_widget = self._build_env_header()

        self.stack = QStackedWidget()
        self.stack.currentChanged.connect(self._on_stack_changed)
        self.stack.addWidget(self._build_login_page())
        self.stack.addWidget(self._build_dashboard_page())
        self.stack.addWidget(self._build_study_page())
        self.stack.addWidget(self._build_practice_page())
        self.stack.addWidget(self._build_quiz_page())
        self.stack.addWidget(self._build_analytics_page())
        self.stack.addWidget(self._build_admin_page())
        content.addWidget(self.stack)

        self.right_panel = self._build_feedback_panel()
        root.addWidget(self.sidebar)
        root.addWidget(content_wrap, 1)
        self.sidebar.setVisible(False)
        self.right_panel.setVisible(False)
        self.status_bar_widget.setVisible(False)

    def _build_env_header(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("InfoCard")
        l = QHBoxLayout(bar)
        l.setContentsMargins(12, 8, 12, 8)
        l.setSpacing(10)
        self.env_title = QLabel("Environment Health")
        self.env_title.setObjectName("SectionTitle")
        self.env_mp = QLabel("MediaPipe")
        self.env_torch = QLabel("Torch")
        self.env_cam = QLabel("Camera")
        self.env_static = QLabel("StaticModel")
        self.env_dynamic = QLabel("DynamicModel")
        btn_refresh_env = QPushButton("Refresh")
        btn_refresh_env.setObjectName("GhostButton")
        btn_refresh_env.setFixedWidth(100)
        btn_refresh_env.clicked.connect(self.refresh_environment_status)

        l.addWidget(self.env_title)
        l.addSpacing(12)
        for x in [self.env_mp, self.env_torch, self.env_cam, self.env_static, self.env_dynamic]:
            l.addWidget(x)
        l.addStretch()
        l.addWidget(btn_refresh_env)
        return bar

    def _build_feedback_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("FeedbackPanel")
        panel.setFixedWidth(300)
        l = QVBoxLayout(panel)
        l.setContentsMargins(12, 12, 12, 12)
        l.setSpacing(10)

        title = QLabel("Live Feedback")
        title.setObjectName("SectionTitle")
        subtitle = QLabel("Session and recognition status")
        subtitle.setObjectName("SectionMeta")
        l.addWidget(title)
        l.addWidget(subtitle)

        card_session = QFrame()
        card_session.setObjectName("InfoCard")
        cs = QVBoxLayout(card_session)
        cs.setContentsMargins(10, 10, 10, 10)
        cs.setSpacing(6)
        session_head = QLabel("Session")
        session_head.setObjectName("SectionTitle")
        self.feedback_user = QLabel("User: -")
        self.feedback_mode = QLabel("Mode: Login")
        self.feedback_target = QLabel("Target: -")
        cs.addWidget(session_head)
        cs.addWidget(self.feedback_user)
        cs.addWidget(self.feedback_mode)
        cs.addWidget(self.feedback_target)
        l.addWidget(card_session)

        card_infer = QFrame()
        card_infer.setObjectName("InfoCard")
        ci = QVBoxLayout(card_infer)
        ci.setContentsMargins(10, 10, 10, 10)
        ci.setSpacing(6)
        infer_head = QLabel("Recognition")
        infer_head.setObjectName("SectionTitle")
        self.feedback_prediction = QLabel("Prediction: -")
        self.feedback_conf = QLabel("Confidence: 0.00")
        self.feedback_model = QLabel("Model: -")
        self.feedback_fps = QLabel("FPS: 0.0")
        self.feedback_status = QLabel("Status: idle")
        self.feedback_status.setObjectName("StatusPill")
        ci.addWidget(infer_head)
        ci.addWidget(self.feedback_prediction)
        ci.addWidget(self.feedback_conf)
        ci.addWidget(self.feedback_model)
        ci.addWidget(self.feedback_fps)
        ci.addWidget(self.feedback_status)
        l.addWidget(card_infer)

        card_ref = QFrame()
        card_ref.setObjectName("InfoCard")
        cr = QVBoxLayout(card_ref)
        cr.setContentsMargins(10, 10, 10, 10)
        cr.setSpacing(6)
        ref_head = QLabel("Reference")
        ref_head.setObjectName("SectionTitle")
        self.feedback_ref_status = QLabel("Reference: waiting")
        self.feedback_ref_source = QLabel("Source: isl_videos (local)")
        cr.addWidget(ref_head)
        cr.addWidget(self.feedback_ref_status)
        cr.addWidget(self.feedback_ref_source)
        l.addWidget(card_ref)
        l.addStretch()
        return panel

    def _build_login_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("LoginPage")
        l = QVBoxLayout(page)
        l.setContentsMargins(52, 44, 52, 44)
        l.setSpacing(0)
        l.addStretch(1)

        shell = QFrame()
        shell.setObjectName("LoginShell")
        shell.setMaximumWidth(1040)
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(34, 34, 34, 34)
        shell_layout.setSpacing(28)

        hero = QFrame()
        hero.setObjectName("LoginHero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(8, 8, 8, 8)
        hero_layout.setSpacing(14)

        eyebrow = QLabel("MUDRA")
        eyebrow.setObjectName("LoginEyebrow")
        title = QLabel("Interactive ISL learning, arranged by clear study levels.")
        title.setObjectName("LoginTitle")
        title.setWordWrap(True)
        subtitle = QLabel(
            "Learn letters first, move into words next, and practice with live recognition "
            "using the same structured path across the app."
        )
        subtitle.setObjectName("LoginSubtitle")
        subtitle.setWordWrap(True)

        hero_layout.addWidget(eyebrow)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addSpacing(8)

        for feature_text in [
            "Level 1 starts with letters and finger spelling.",
            "Practice targets follow the same level structure.",
            "All reference media stays local inside the app.",
        ]:
            feature = QLabel(feature_text)
            feature.setObjectName("LoginFeature")
            feature.setWordWrap(True)
            hero_layout.addWidget(feature)
        hero_layout.addStretch()

        form = QFrame()
        form.setObjectName("LoginFormCard")
        form.setMaximumWidth(420)
        fl = QVBoxLayout(form)
        fl.setContentsMargins(30, 30, 30, 30)
        fl.setSpacing(12)

        form_title = QLabel("Sign in")
        form_title.setObjectName("SectionTitle")
        form_subtitle = QLabel("Use your MUDRA account to continue learning.")
        form_subtitle.setObjectName("LoginSubtitle")
        form_subtitle.setWordWrap(True)
        fl.addWidget(form_title)
        fl.addWidget(form_subtitle)
        fl.addSpacing(6)

        email_label = QLabel("Email")
        email_label.setObjectName("LoginFieldLabel")
        self.login_email = QLineEdit()
        self.login_email.setObjectName("LoginInput")
        self.login_email.setMinimumHeight(48)
        self.login_email.setPlaceholderText("demo@mudra.local")

        password_label = QLabel("Password")
        password_label.setObjectName("LoginFieldLabel")
        self.login_password = QLineEdit()
        self.login_password.setObjectName("LoginInput")
        self.login_password.setMinimumHeight(48)
        self.login_password.setPlaceholderText("Enter password")
        self.login_password.setEchoMode(QLineEdit.Password)

        fl.addWidget(email_label)
        fl.addWidget(self.login_email)
        fl.addWidget(password_label)
        fl.addWidget(self.login_password)

        hint = QLabel("Demo access: demo@mudra.local / demo123")
        hint.setObjectName("LoginHint")
        hint.setWordWrap(True)
        fl.addWidget(hint)
        fl.addSpacing(4)

        btn_login = QPushButton("Sign In")
        btn_login.setObjectName("LoginPrimary")
        btn_login.setMinimumHeight(50)
        btn_login.setCursor(Qt.PointingHandCursor)
        btn_login.clicked.connect(self.handle_login)
        fl.addWidget(btn_login)

        btn_demo = QPushButton("Fill Demo Credentials")
        btn_demo.setObjectName("LoginSecondary")
        btn_demo.setMinimumHeight(46)
        btn_demo.setCursor(Qt.PointingHandCursor)
        btn_demo.clicked.connect(
            lambda: (self.login_email.setText("demo@mudra.local"),
                     self.login_password.setText("demo123"))
        )
        fl.addWidget(btn_demo)
        fl.addStretch()

        shell_layout.addWidget(hero, 5)
        shell_layout.addWidget(form, 4)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(shell)
        row.addStretch()
        l.addLayout(row)
        l.addStretch(1)
        return page

    def _build_dashboard_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        page = QWidget()
        l = QVBoxLayout(page)
        l.setContentsMargins(8, 8, 8, 8)
        l.setSpacing(14)

        welcome_card = QFrame()
        welcome_card.setObjectName("HeroCard")
        wc = QHBoxLayout(welcome_card)
        wc.setContentsMargins(26, 24, 26, 24)
        wc.setSpacing(20)

        hero_copy = QVBoxLayout()
        hero_copy.setSpacing(10)
        hero_eyebrow = QLabel("LEARNING DASHBOARD")
        hero_eyebrow.setStyleSheet("color:#7dd3fc; font-size:12px; font-weight:700; letter-spacing:1px;")
        self.welcome = QLabel("Welcome")
        self.welcome.setFont(QFont(self.sys_font, 30, QFont.Bold))
        self.welcome.setStyleSheet("color:#f8fafc; background:transparent;")
        self.lesson_summary = QLabel("Structured study levels and live practice are ready.")
        self.lesson_summary.setStyleSheet("color:#dbeafe; font-size:16px; font-weight:600; background:transparent;")
        self.progress_summary = QLabel("Begin with letters, move into core words, then practice live with the camera.")
        self.progress_summary.setWordWrap(True)
        self.progress_summary.setStyleSheet("color:#9fb7d1; font-size:13px; line-height:1.5; background:transparent;")
        hero_copy.addWidget(hero_eyebrow)
        hero_copy.addWidget(self.welcome)
        hero_copy.addWidget(self.lesson_summary)
        hero_copy.addWidget(self.progress_summary)

        hero_actions = QHBoxLayout()
        hero_actions.setSpacing(10)
        btn_continue = QPushButton("Open Study Path")
        btn_continue.setObjectName("AccentButton")
        btn_continue.setMinimumHeight(44)
        btn_continue.clicked.connect(lambda: self._open_study_level(min(LEVEL_INFO) if LEVEL_INFO else 1))
        btn_practice = QPushButton("Practice Live")
        btn_practice.setObjectName("GhostButton")
        btn_practice.setMinimumHeight(44)
        btn_practice.clicked.connect(lambda: self.navigate_to(self.IDX_PRACTICE))
        btn_quiz = QPushButton("Quick Quiz")
        btn_quiz.setObjectName("GhostButton")
        btn_quiz.setMinimumHeight(44)
        btn_quiz.clicked.connect(lambda: (self.navigate_to(self.IDX_QUIZ), self.start_quiz()))
        hero_actions.addWidget(btn_continue)
        hero_actions.addWidget(btn_practice)
        hero_actions.addWidget(btn_quiz)
        hero_actions.addStretch()
        hero_copy.addSpacing(6)
        hero_copy.addLayout(hero_actions)
        hero_copy.addStretch()
        wc.addLayout(hero_copy, 5)

        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(10)
        stats_grid.setVerticalSpacing(10)
        self.metric_levels = QLabel("0")
        self.metric_levels.setObjectName("MetricValue")
        self.metric_gestures = QLabel("0")
        self.metric_gestures.setObjectName("MetricValue")
        self.metric_focus = QLabel("Level 1")
        self.metric_focus.setObjectName("MetricValue")
        self.metric_mode = QLabel("Live + Local")
        self.metric_mode.setObjectName("MetricValue")
        for i, (label_text, value_widget) in enumerate([
            ("Study Levels", self.metric_levels),
            ("Gestures Ready", self.metric_gestures),
            ("Suggested Start", self.metric_focus),
            ("Practice Mode", self.metric_mode),
        ]):
            card = QFrame()
            card.setObjectName("MetricCard")
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(14, 12, 14, 12)
            card_l.setSpacing(4)
            title_lbl = QLabel(label_text)
            title_lbl.setObjectName("MetricLabel")
            card_l.addWidget(title_lbl)
            card_l.addWidget(value_widget)
            stats_grid.addWidget(card, i // 2, i % 2)
        wc.addLayout(stats_grid, 4)
        l.addWidget(welcome_card)

        body = QHBoxLayout()
        body.setSpacing(12)

        roadmap_card = QFrame()
        roadmap_card.setObjectName("InfoCard")
        rc = QVBoxLayout(roadmap_card)
        rc.setContentsMargins(18, 18, 18, 18)
        rc.setSpacing(14)
        roadmap_title = QLabel("Learning Path")
        roadmap_title.setObjectName("SectionTitle")
        roadmap_copy = QLabel("Move through the catalog in order so letters, words, and longer signs feel connected instead of random.")
        roadmap_copy.setObjectName("SectionMeta")
        roadmap_copy.setWordWrap(True)
        rc.addWidget(roadmap_title)
        rc.addWidget(roadmap_copy)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.dashboard_level_buttons: Dict[int, QPushButton] = {}
        for i, lvl in enumerate(sorted(LEVEL_INFO)):
            info = LEVEL_INFO[lvl]
            c = QPushButton()
            c.setObjectName("DashboardLevelCard")
            c.setCursor(Qt.PointingHandCursor)
            c.setMinimumHeight(122)
            c.setStyleSheet(
                f"QPushButton#DashboardLevelCard {{ background: qlineargradient("
                f"x1:0, y1:0, x2:1, y2:1, stop:0 {info['bg']}, stop:1 #0f172a); "
                f"border: 1px solid {info['accent']}; color:#f8fafc; }}"
                f"QPushButton#DashboardLevelCard:hover {{ border: 1px solid #f8fafc; }}"
            )
            c.clicked.connect(lambda _, level=lvl: self._open_study_level(level))
            c.setText(
                f"{info['emoji']}  Level {lvl} · {info['title']}\n"
                f"0 gestures\n"
                f"{info['desc']}"
            )
            self.dashboard_level_buttons[lvl] = c
            grid.addWidget(c, i // 2, i % 2)
        rc.addLayout(grid)
        body.addWidget(roadmap_card, 7)

        focus_card = QFrame()
        focus_card.setObjectName("InfoCard")
        fc = QVBoxLayout(focus_card)
        fc.setContentsMargins(18, 18, 18, 18)
        fc.setSpacing(12)
        focus_title = QLabel("Next Step")
        focus_title.setObjectName("SectionTitle")
        self.dashboard_focus_title = QLabel("Open Level 1")
        self.dashboard_focus_title.setFont(QFont(self.sys_font, 22, QFont.Bold))
        self.dashboard_focus_title.setStyleSheet("color:#f8fafc;")
        self.dashboard_focus_copy = QLabel("Start with letters, then move into live practice once the hand shapes feel natural.")
        self.dashboard_focus_copy.setWordWrap(True)
        self.dashboard_focus_copy.setStyleSheet("color:#cbd5e1; font-size:14px; line-height:1.5;")
        self.dashboard_focus_meta = QLabel("Local references and camera practice stay synced to the same target path.")
        self.dashboard_focus_meta.setWordWrap(True)
        self.dashboard_focus_meta.setObjectName("StatusPill")
        focus_note = QLabel("Fast actions")
        focus_note.setStyleSheet("color:#94a3b8; font-size:12px; font-weight:700; padding-top:4px;")
        fc.addWidget(focus_title)
        fc.addWidget(self.dashboard_focus_title)
        fc.addWidget(self.dashboard_focus_copy)
        fc.addWidget(self.dashboard_focus_meta)
        fc.addWidget(focus_note)

        for title, subtitle, accent, bg, callback in [
            ("Live Practice", "Open reference video and camera side by side.", "#06b6d4", "#123e52", lambda: self.navigate_to(self.IDX_PRACTICE)),
            ("Quick Quiz", "Run a short recognition round from trained signs.", "#f59e0b", "#5f3b12", lambda: (self.navigate_to(self.IDX_QUIZ), self.start_quiz())),
            ("Analytics", "Review accuracy, attempts, and weak targets.", "#8b5cf6", "#312e81", lambda: self.navigate_to(self.IDX_ANALYTICS)),
        ]:
            button = QPushButton(f"{title}\n{subtitle}")
            button.setObjectName("QuickAction")
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(78)
            button.setStyleSheet(
                f"QPushButton#QuickAction {{ background: qlineargradient("
                f"x1:0, y1:0, x2:1, y2:1, stop:0 {bg}, stop:1 #101726); "
                f"border: 1px solid {accent}; color: #f8fafc; }}"
                f"QPushButton#QuickAction:hover {{ border: 1px solid #f8fafc; }}"
            )
            button.clicked.connect(callback)
            fc.addWidget(button)
        fc.addStretch()
        body.addWidget(focus_card, 4)

        l.addLayout(body)
        l.addStretch()

        scroll.setWidget(page)
        return scroll

    def _build_study_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(12)

        hdr_card = QFrame()
        hdr_card.setObjectName("HeroCard")
        hdr_l = QHBoxLayout(hdr_card)
        hdr_l.setContentsMargins(22, 18, 22, 18)
        hdr_l.setSpacing(16)
        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(6)
        study_title = QLabel("Study by Level")
        study_title.setFont(QFont(self.sys_font, 24, QFont.Bold))
        study_title.setStyleSheet("color:#f8fafc;")
        study_sub = QLabel("Level 1 starts with letters. Level 2 moves into core words and each next level increases complexity.")
        study_sub.setWordWrap(True)
        study_sub.setStyleSheet("color:#cbd5e1; font-size:13px;")
        title_wrap.addWidget(study_title)
        title_wrap.addWidget(study_sub)
        hdr_l.addLayout(title_wrap, 3)
        self.study_header_meta = QLabel("Pick a level to open its gesture set.")
        self.study_header_meta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.study_header_meta.setStyleSheet("color:#dbeafe; font-size:13px; font-weight:600;")
        hdr_l.addWidget(self.study_header_meta, 2)
        outer.addWidget(hdr_card)

        content_split = QHBoxLayout()
        content_split.setSpacing(12)

        levels_card = QFrame()
        levels_card.setObjectName("InfoCard")
        levels_card.setFixedWidth(290)
        levels_l = QVBoxLayout(levels_card)
        levels_l.setContentsMargins(16, 16, 16, 16)
        levels_l.setSpacing(10)
        levels_title = QLabel("Learning Levels")
        levels_title.setObjectName("SectionTitle")
        levels_copy = QLabel("Follow the sequence instead of jumping into a flat list.")
        levels_copy.setObjectName("SectionMeta")
        levels_copy.setWordWrap(True)
        levels_l.addWidget(levels_title)
        levels_l.addWidget(levels_copy)
        self.study_level_buttons: Dict[int, QPushButton] = {}
        for lvl in sorted(LEVEL_INFO):
            button = QPushButton()
            button.setObjectName("LevelSelector")
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(94)
            button.clicked.connect(lambda _, level=lvl: self._open_study_level(level))
            self.study_level_buttons[lvl] = button
            levels_l.addWidget(button)
        levels_l.addStretch()

        gesture_card = QFrame()
        gesture_card.setObjectName("InfoCard")
        gesture_l = QVBoxLayout(gesture_card)
        gesture_l.setContentsMargins(16, 16, 16, 16)
        gesture_l.setSpacing(10)
        self.study_level_title = QLabel("Level 1")
        self.study_level_title.setFont(QFont(self.sys_font, 18, QFont.Bold))
        self.study_level_title.setStyleSheet("color:#f8fafc;")
        self.study_level_caption = QLabel("Choose a level to load its gestures.")
        self.study_level_caption.setObjectName("SectionMeta")
        self.study_level_caption.setWordWrap(True)
        self.study_level_count = QLabel("0 gestures")
        self.study_level_count.setStyleSheet("color:#22c55e; font-weight:700;")
        self.study_gesture_list = QListWidget()
        self.study_gesture_list.setObjectName("GestureList")
        self.study_gesture_list.itemSelectionChanged.connect(self.on_select_study_gesture)
        gesture_l.addWidget(self.study_level_title)
        gesture_l.addWidget(self.study_level_caption)
        gesture_l.addWidget(self.study_level_count)
        gesture_l.addWidget(self.study_gesture_list, 1)
        study_legend = QLabel("Static signs are held shapes. Dynamic signs require movement.")
        study_legend.setObjectName("SectionMeta")
        study_legend.setWordWrap(True)
        gesture_l.addWidget(study_legend)

        detail_card = QFrame()
        detail_card.setObjectName("InfoCard")
        detail_l = QVBoxLayout(detail_card)
        detail_l.setContentsMargins(16, 16, 16, 16)
        detail_l.setSpacing(10)

        detail_header = QLabel("Gesture Details")
        detail_header.setFont(QFont(self.sys_font, 16, QFont.Bold))
        detail_header.setStyleSheet("color:#f8fafc;")
        detail_l.addWidget(detail_header)

        self.study_ref_label = QLabel("Select a gesture to see its reference")
        self.study_ref_label.setAlignment(Qt.AlignCenter)
        self.study_ref_label.setMinimumSize(400, 320)
        self.study_ref_label.setWordWrap(True)
        self.study_ref_label.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #05070a, stop:1 #0d1117);"
            "border: 1px solid #334155; border-radius: 16px; padding: 18px; color:#94a3b8; font-size:14px;"
        )
        self.study_ref_status = QLabel("Reference not loaded")
        self.study_ref_status.setObjectName("SectionMeta")
        detail_l.addWidget(self.study_ref_label, 1)
        detail_l.addWidget(self.study_ref_status)

        self.study_name = QLabel("Select a gesture")
        self.study_name.setFont(QFont(self.sys_font, 18, QFont.Bold))
        self.study_name.setStyleSheet("color:#f8fafc;")
        self.study_type = QLabel("Type: -")
        self.study_type.setStyleSheet("color:#94a3b8;")
        self.study_diff = QLabel("Difficulty: -")
        self.study_diff.setStyleSheet("color:#94a3b8;")
        self.study_desc = QLabel("Choose a gesture from the level list to load its explanation, tips, and reference media.")
        self.study_desc.setWordWrap(True)
        self.study_desc.setStyleSheet("color:#cbd5e1; font-size:13px; line-height:1.5;")
        detail_l.addWidget(self.study_name)
        detail_l.addWidget(self.study_type)
        detail_l.addWidget(self.study_diff)
        detail_l.addWidget(self.study_desc)

        self.btn_start_practice_from_study = QPushButton("\u2728  Start Practice")
        self.btn_start_practice_from_study.setObjectName("AccentButton")
        self.btn_start_practice_from_study.setFont(QFont(self.sys_font, 13, QFont.Bold))
        self.btn_start_practice_from_study.clicked.connect(self.start_practice_from_study)
        detail_l.addWidget(self.btn_start_practice_from_study)
        detail_l.addStretch()

        content_split.addWidget(levels_card, 3)
        content_split.addWidget(gesture_card, 4)
        content_split.addWidget(detail_card, 5)
        outer.addLayout(content_split, 1)
        return page

    def _build_practice_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        target_card = QFrame()
        target_card.setObjectName("InfoCard")
        target_l = QHBoxLayout(target_card)
        target_l.setContentsMargins(18, 18, 18, 18)
        target_l.setSpacing(18)

        target_picker = QVBoxLayout()
        target_picker.setSpacing(8)
        target_title = QLabel("Target Gesture")
        target_title.setObjectName("SectionTitle")
        target_copy = QLabel("Pick the sign you want to practice, then compare the reference video with your live camera.")
        target_copy.setObjectName("SectionMeta")
        target_copy.setWordWrap(True)
        self.practice_target_combo = QComboBox()
        self.practice_target_combo.setMinimumHeight(44)
        self.practice_target_combo.currentIndexChanged.connect(self.on_select_practice_gesture)
        self.practice_target_meta = QLabel("No target selected")
        self.practice_target_meta.setStyleSheet("color:#cbd5e1; font-size:13px; font-weight:600;")
        target_picker.addWidget(target_title)
        target_picker.addWidget(target_copy)
        target_picker.addWidget(self.practice_target_combo)
        target_picker.addWidget(self.practice_target_meta)
        target_l.addLayout(target_picker, 5)

        session_status = QVBoxLayout()
        session_status.setSpacing(8)
        self.practice_target_stat = QLabel("Target: -")
        self.practice_target_stat.setFont(QFont(self.sys_font, 22, QFont.Bold))
        self.practice_target_stat.setStyleSheet("color:#f8fafc;")
        self.live_prediction = QLabel("Model idle. Choose a gesture and start the camera when you're ready.")
        self.live_prediction.setWordWrap(True)
        self.live_prediction.setStyleSheet("color:#94a3b8; font-size:13px;")
        self.practice_feedback = QLabel("Select a target gesture to prepare the session.")
        self.practice_feedback.setObjectName("StatusPill")
        self.practice_feedback.setWordWrap(True)
        session_status.addWidget(self.practice_target_stat)
        session_status.addWidget(self.live_prediction)
        session_status.addWidget(self.practice_feedback)
        session_status.addStretch()
        target_l.addLayout(session_status, 4)
        layout.addWidget(target_card)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(12)

        left_card = QFrame()
        left_card.setObjectName("InfoCard")
        left = QVBoxLayout(left_card)
        left.setContentsMargins(14, 14, 14, 14)
        left.setSpacing(10)
        ref_header = QLabel("Reference Video")
        ref_header.setFont(QFont(self.sys_font, 15, QFont.Bold))
        ref_header.setStyleSheet("color:#10b981;")
        ref_header.setAlignment(Qt.AlignCenter)
        left.addWidget(ref_header)
        self.practice_ref_label = QLabel("Reference not available")
        self.practice_ref_label.setAlignment(Qt.AlignCenter)
        self.practice_ref_label.setMinimumSize(420, 470)
        self.practice_ref_label.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.practice_ref_status = QLabel("Reference not available")
        self.practice_ref_status.setAlignment(Qt.AlignCenter)
        self.practice_ref_status.setObjectName("SectionMeta")
        left.addWidget(self.practice_ref_label, 1)
        left.addWidget(self.practice_ref_status)

        right_card = QFrame()
        right_card.setObjectName("InfoCard")
        right = QVBoxLayout(right_card)
        right.setContentsMargins(14, 14, 14, 14)
        right.setSpacing(10)
        cam_header = QLabel("Live Camera")
        cam_header.setFont(QFont(self.sys_font, 15, QFont.Bold))
        cam_header.setStyleSheet("color:#1cb0f6;")
        cam_header.setAlignment(Qt.AlignCenter)
        right.addWidget(cam_header)
        self.camera_view = QLabel("Camera preview")
        self.camera_view.setMinimumSize(420, 470)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:14px;")
        right.addWidget(self.camera_view, 1)

        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(12)
        stats_grid.setVerticalSpacing(6)
        self.practice_pred_stat = QLabel("Prediction: -")
        self.practice_conf_stat = QLabel("Confidence: 0.00")
        self.practice_model_stat = QLabel("Model: -")
        self.practice_fps_stat = QLabel("FPS: 0.0")
        stats_grid.addWidget(self.practice_pred_stat, 0, 0)
        stats_grid.addWidget(self.practice_conf_stat, 0, 1)
        stats_grid.addWidget(self.practice_model_stat, 1, 0)
        stats_grid.addWidget(self.practice_fps_stat, 1, 1)
        right.addLayout(stats_grid)

        self.btn_start_camera = QPushButton("Start Live Practice")
        self.btn_start_camera.setObjectName("AccentButton")
        self.btn_stop_camera = QPushButton("Stop Camera")
        self.btn_stop_camera.setObjectName("GhostButton")
        self.btn_mark_attempt = QPushButton("Record Current Attempt")
        self.btn_mark_attempt.setObjectName("GhostButton")

        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_mark_attempt.clicked.connect(self.record_current_attempt)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_start_camera)
        btn_row.addWidget(self.btn_stop_camera)
        right.addLayout(btn_row)
        right.addWidget(self.btn_mark_attempt)

        preview_row.addWidget(left_card, 1)
        preview_row.addWidget(right_card, 1)
        layout.addLayout(preview_row, 1)
        return page

    def _build_quiz_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(10)

        # ---- Top bar: controls ----
        top_card = QFrame()
        top_card.setObjectName("InfoCard")
        top_l = QHBoxLayout(top_card)
        top_l.setContentsMargins(14, 10, 14, 10)
        top_l.setSpacing(14)
        self.quiz_progress = QLabel("Question: 0/10")
        self.quiz_progress.setStyleSheet("color:#94a3b8; font-size:13px;")
        self.quiz_state = QLabel("Score: 0/0")
        self.quiz_state.setFont(QFont(self.sys_font, 14, QFont.Bold))
        self.quiz_state.setStyleSheet("color:#facc15;")
        b1 = QPushButton("Start Quiz (10 Questions)")
        b1.setObjectName("AccentButton")
        b2 = QPushButton("Submit Answer")
        b2.setObjectName("GhostButton")
        b1.clicked.connect(self.start_quiz)
        b2.clicked.connect(self.submit_quiz_answer)
        top_l.addWidget(self.quiz_progress)
        top_l.addWidget(self.quiz_state)
        top_l.addStretch()
        top_l.addWidget(b1)
        top_l.addWidget(b2)
        outer.addWidget(top_card)

        # ---- Prominent question card ----
        q_card = QFrame()
        q_card.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #1a1040, stop:1 #0f172a);"
            "border:1px solid #6366f1; border-radius:18px;"
        )
        q_inner = QVBoxLayout(q_card)
        q_inner.setContentsMargins(24, 20, 24, 20)
        q_inner.setSpacing(6)
        q_label = QLabel("\U0001F4AC  Show the sign for:")
        q_label.setStyleSheet("color:#a5b4fc; font-size:14px; background:transparent;")
        self.quiz_target = QLabel("Press Start Quiz")
        self.quiz_target.setFont(QFont(self.sys_font, 26, QFont.Bold))
        self.quiz_target.setStyleSheet("color:#f8fafc; background:transparent;")
        self.quiz_target.setAlignment(Qt.AlignCenter)
        self.quiz_target.setWordWrap(True)
        q_inner.addWidget(q_label, 0, Qt.AlignCenter)
        q_inner.addWidget(self.quiz_target, 0, Qt.AlignCenter)
        outer.addWidget(q_card)

        # ---- Camera + hidden reference side-by-side ----
        body = QHBoxLayout()
        body.setSpacing(10)

        cam_card = QFrame()
        cam_card.setObjectName("InfoCard")
        cam_l = QVBoxLayout(cam_card)
        cam_l.setContentsMargins(10, 10, 10, 10)
        cam_l.setSpacing(8)
        qcam_header = QLabel("Your Camera")
        qcam_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        qcam_header.setStyleSheet("color:#1cb0f6;")
        qcam_header.setAlignment(Qt.AlignCenter)
        cam_l.addWidget(qcam_header)
        self.quiz_camera_view = QLabel("Camera will start with the quiz")
        self.quiz_camera_view.setAlignment(Qt.AlignCenter)
        self.quiz_camera_view.setMinimumSize(380, 320)
        self.quiz_camera_view.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.quiz_pred_label = QLabel("")
        self.quiz_pred_label.setAlignment(Qt.AlignCenter)
        self.quiz_pred_label.setStyleSheet("color:#94a3b8; font-size:13px;")
        self.quiz_feedback = QLabel("Press 'Start Quiz' to begin")
        self.quiz_feedback.setObjectName("StatusPill")
        cam_l.addWidget(self.quiz_camera_view, 1)
        cam_l.addWidget(self.quiz_pred_label)
        cam_l.addWidget(self.quiz_feedback)
        body.addWidget(cam_card, 3)

        # Reference card — hidden by default, revealed on button click
        self.quiz_ref_card = QFrame()
        self.quiz_ref_card.setObjectName("InfoCard")
        ref_l = QVBoxLayout(self.quiz_ref_card)
        ref_l.setContentsMargins(10, 10, 10, 10)
        ref_l.setSpacing(8)
        qref_header = QLabel("Answer Reference")
        qref_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        qref_header.setStyleSheet("color:#10b981;")
        qref_header.setAlignment(Qt.AlignCenter)
        ref_l.addWidget(qref_header)
        self.quiz_ref_label = QLabel("")
        self.quiz_ref_label.setAlignment(Qt.AlignCenter)
        self.quiz_ref_label.setMinimumSize(320, 280)
        self.quiz_ref_label.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.quiz_ref_status = QLabel("")
        self.quiz_ref_status.setAlignment(Qt.AlignCenter)
        self.quiz_ref_status.setObjectName("SectionMeta")
        ref_l.addWidget(self.quiz_ref_label, 1)
        ref_l.addWidget(self.quiz_ref_status)
        self.quiz_ref_card.setVisible(False)
        body.addWidget(self.quiz_ref_card, 2)

        outer.addLayout(body, 1)

        # Reveal answer button
        self.btn_reveal_answer = QPushButton("\U0001F441  Reveal Answer")
        self.btn_reveal_answer.setCursor(Qt.PointingHandCursor)
        self.btn_reveal_answer.setStyleSheet(
            "QPushButton { background:#1e293b; color:#a5b4fc; border:1px solid #4f46e5; "
            "border-radius:12px; padding:10px 20px; font-size:14px; font-weight:600; } "
            "QPushButton:hover { background:#312e81; color:#e0e7ff; }"
        )
        self.btn_reveal_answer.clicked.connect(self._toggle_quiz_reveal)
        outer.addWidget(self.btn_reveal_answer, 0, Qt.AlignCenter)
        return page

    def _toggle_quiz_reveal(self):
        visible = self.quiz_ref_card.isVisible()
        self.quiz_ref_card.setVisible(not visible)
        if visible:
            self.btn_reveal_answer.setText("\U0001F441  Reveal Answer")
        else:
            self.btn_reveal_answer.setText("\U0001F648  Hide Answer")

    def _build_analytics_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(10)

        # ---- Summary cards row ----
        summary_card = QFrame()
        summary_card.setObjectName("InfoCard")
        sc_l = QVBoxLayout(summary_card)
        sc_l.setContentsMargins(16, 12, 16, 12)
        sc_l.setSpacing(6)
        analytics_header = QLabel("Performance Analytics")
        analytics_header.setFont(QFont(self.sys_font, 16, QFont.Bold))
        analytics_header.setStyleSheet("color:#f8fafc;")
        sc_l.addWidget(analytics_header)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        self.stat_total = QLabel("0")
        self.stat_accuracy = QLabel("0.00")
        self.stat_avg_conf = QLabel("0.00")
        self.stat_avg_latency = QLabel("0.0ms")
        for stat_label, stat_widget, color in [
            ("Total Attempts", self.stat_total, "#22c55e"),
            ("Accuracy", self.stat_accuracy, "#1cb0f6"),
            ("Avg Confidence", self.stat_avg_conf, "#facc15"),
            ("Avg Latency", self.stat_avg_latency, "#a78bfa"),
        ]:
            mini_card = QFrame()
            mini_card.setStyleSheet(
                f"background:#0e1525; border:1px solid #2a3446; border-radius:12px; padding:10px;"
            )
            mc = QVBoxLayout(mini_card)
            mc.setContentsMargins(12, 8, 12, 8)
            mc.setSpacing(2)
            title_lbl = QLabel(stat_label)
            title_lbl.setStyleSheet("color:#94a3b8; font-size:11px; font-weight:600;")
            stat_widget.setFont(QFont(self.sys_font, 20, QFont.Bold))
            stat_widget.setStyleSheet(f"color:{color}; font-size:22px;")
            mc.addWidget(title_lbl)
            mc.addWidget(stat_widget)
            stats_row.addWidget(mini_card)
        sc_l.addLayout(stats_row)
        self.analytics_summary = QLabel("")  # kept for compatibility
        self.analytics_summary.setVisible(False)
        sc_l.addWidget(self.analytics_summary)
        outer.addWidget(summary_card)

        # ---- Progress summary ----
        progress_card = QFrame()
        progress_card.setObjectName("InfoCard")
        pc_l = QVBoxLayout(progress_card)
        pc_l.setContentsMargins(14, 10, 14, 10)
        self.progress_detail = QLabel("Complete some practice sessions to see progress breakdown.")
        self.progress_detail.setWordWrap(True)
        self.progress_detail.setStyleSheet("color:#cbd5e1; font-size:13px;")
        pc_l.addWidget(self.progress_detail)
        outer.addWidget(progress_card)

        # ---- Attempts table ----
        table_card = QFrame()
        table_card.setObjectName("InfoCard")
        tc = QVBoxLayout(table_card)
        tc.setContentsMargins(10, 10, 10, 10)
        tc.setSpacing(6)
        table_header = QLabel("Recent Attempts")
        table_header.setFont(QFont(self.sys_font, 13, QFont.Bold))
        table_header.setStyleSheet("color:#f8fafc;")
        tc.addWidget(table_header)
        self.analytics_table = QTableWidget(0, 6)
        self.analytics_table.setHorizontalHeaderLabels(["Time", "Target", "Predicted", "Conf", "Correct", "Mode"])
        self.analytics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.analytics_table.setAlternatingRowColors(True)
        self.analytics_table.setStyleSheet(
            "QTableWidget { alternate-background-color: #0e1525; }"
        )
        tc.addWidget(self.analytics_table)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_refresh = QPushButton("Refresh Analytics")
        btn_refresh.setObjectName("AccentButton")
        btn_conf = QPushButton("Load Confusion Matrix")
        btn_conf.setObjectName("GhostButton")
        btn_refresh.clicked.connect(self.load_analytics)
        btn_conf.clicked.connect(self.load_confusion_matrix_view)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_conf)
        btn_row.addStretch()
        tc.addLayout(btn_row)
        outer.addWidget(table_card, 1)

        # ---- Confusion Matrix ----
        cm_card = QFrame()
        cm_card.setObjectName("InfoCard")
        cm_l = QVBoxLayout(cm_card)
        cm_l.setContentsMargins(10, 10, 10, 10)
        cm_l.setSpacing(6)
        cm_header = QLabel("Confusion Matrix")
        cm_header.setFont(QFont(self.sys_font, 13, QFont.Bold))
        cm_header.setStyleSheet("color:#f8fafc;")
        self.confusion_note = QLabel("Record attempts to see the confusion matrix.")
        self.confusion_note.setStyleSheet("color:#94a3b8; font-size:12px;")
        self.confusion_table = QTableWidget(0, 0)
        self.confusion_table.setMaximumHeight(360)
        self.confusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.confusion_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        cm_l.addWidget(cm_header)
        cm_l.addWidget(self.confusion_note)
        cm_l.addWidget(self.confusion_table)
        outer.addWidget(cm_card)
        return page

    def _build_admin_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(10)

        # ---- Header ----
        header_card = QFrame()
        header_card.setObjectName("InfoCard")
        hc = QHBoxLayout(header_card)
        hc.setContentsMargins(16, 12, 16, 12)
        self.admin_label = QLabel("Admin Panel")
        self.admin_label.setFont(QFont(self.sys_font, 18, QFont.Bold))
        self.admin_label.setStyleSheet("color:#f8fafc;")
        admin_sub = QLabel("Manage models, database, and system configuration")
        admin_sub.setStyleSheet("color:#94a3b8; font-size:12px;")
        hc.addWidget(self.admin_label)
        hc.addStretch()
        hc.addWidget(admin_sub)
        outer.addWidget(header_card)

        # ---- Quick Actions ----
        actions_card = QFrame()
        actions_card.setObjectName("InfoCard")
        ac = QVBoxLayout(actions_card)
        ac.setContentsMargins(14, 12, 14, 12)
        ac.setSpacing(8)
        actions_title = QLabel("Quick Actions")
        actions_title.setFont(QFont(self.sys_font, 14, QFont.Bold))
        actions_title.setStyleSheet("color:#facc15;")
        ac.addWidget(actions_title)
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        btn_seed = QPushButton("Reseed Core Data")
        btn_seed.setObjectName("GhostButton")
        btn_seed.setToolTip("Re-populate gesture catalog and demo user from seed data")
        btn_models = QPushButton("Refresh Model Registry")
        btn_models.setObjectName("GhostButton")
        btn_models.setToolTip("Reload model version list from database")
        btn_activate = QPushButton("Activate Selected Model")
        btn_activate.setObjectName("AccentButton")
        btn_activate.setToolTip("Set the selected model row as active")
        btn_reload = QPushButton("Reload Predictor")
        btn_reload.setObjectName("AccentButton")
        btn_reload.setToolTip("Re-initialize predictor from active model paths")
        btn_seed.clicked.connect(self.reseed)
        btn_models.clicked.connect(self.load_model_versions)
        btn_activate.clicked.connect(self.activate_selected_model)
        btn_reload.clicked.connect(self.reload_predictor_from_registry)
        btn_row1.addWidget(btn_seed)
        btn_row1.addWidget(btn_models)
        btn_row1.addWidget(btn_activate)
        btn_row1.addWidget(btn_reload)
        ac.addLayout(btn_row1)
        outer.addWidget(actions_card)

        # ---- Model Registry Table ----
        table_card = QFrame()
        table_card.setObjectName("InfoCard")
        tbl = QVBoxLayout(table_card)
        tbl.setContentsMargins(14, 12, 14, 12)
        tbl.setSpacing(6)
        tbl_title = QLabel("Model Versions")
        tbl_title.setFont(QFont(self.sys_font, 14, QFont.Bold))
        tbl_title.setStyleSheet("color:#1cb0f6;")
        tbl.addWidget(tbl_title)
        self.model_table = QTableWidget(0, 6)
        self.model_table.setHorizontalHeaderLabels(["Model", "Version", "Framework", "Artifact", "Active", "Trained At"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setStyleSheet("QTableWidget { alternate-background-color: #0e1525; }")
        self.model_table.setSelectionBehavior(self.model_table.SelectRows)
        tbl.addWidget(self.model_table)
        outer.addWidget(table_card)

        # ---- Register New Model ----
        reg_card = QFrame()
        reg_card.setObjectName("InfoCard")
        reg_layout = QVBoxLayout(reg_card)
        reg_layout.setContentsMargins(14, 12, 14, 12)
        reg_layout.setSpacing(8)
        reg_title = QLabel("Register New Model Version")
        reg_title.setFont(QFont(self.sys_font, 14, QFont.Bold))
        reg_title.setStyleSheet("color:#a78bfa;")
        reg_layout.addWidget(reg_title)

        form_grid = QGridLayout()
        form_grid.setSpacing(8)
        self.reg_model_name = QComboBox()
        self.reg_model_name.addItems(["static_mlp", "dynamic_bigru"])
        self.reg_framework = QLineEdit("pytorch")
        self.reg_version_tag = QLineEdit("")
        self.reg_version_tag.setPlaceholderText("Auto-generated if empty")
        self.reg_artifact_path = QLineEdit()
        self.reg_artifact_path.setPlaceholderText("Path to model .pt file")
        self.reg_label_map_path = QLineEdit("models/registry/label_map.json")
        self.reg_norm_stats_path = QLineEdit("models/registry/norm_stats.json")

        form_fields = [
            ("Model Name", self.reg_model_name, None),
            ("Framework", self.reg_framework, None),
            ("Version Tag", self.reg_version_tag, None),
            ("Artifact Path", self.reg_artifact_path, "Browse Artifact"),
            ("Label Map Path", self.reg_label_map_path, "Browse Label Map"),
            ("Norm Stats Path", self.reg_norm_stats_path, "Browse Norm Stats"),
        ]
        for row_idx, (label_text, widget, browse_text) in enumerate(form_fields):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color:#cbd5e1; font-size:12px; font-weight:600;")
            form_grid.addWidget(lbl, row_idx, 0)
            form_grid.addWidget(widget, row_idx, 1)
            if browse_text and isinstance(widget, QLineEdit):
                b = QPushButton(browse_text)
                b.setObjectName("GhostButton")
                b.setFixedWidth(130)
                b.clicked.connect(lambda _, le=widget: self._pick_file_into(le))
                form_grid.addWidget(b, row_idx, 2)
        reg_layout.addLayout(form_grid)

        self.reg_activate = QCheckBox("Activate immediately")
        self.reg_activate.setChecked(True)
        self.reg_activate.setStyleSheet("color:#cbd5e1; font-size:13px;")
        reg_layout.addWidget(self.reg_activate)

        metrics_lbl = QLabel("Metrics JSON")
        metrics_lbl.setStyleSheet("color:#cbd5e1; font-size:12px; font-weight:600;")
        reg_layout.addWidget(metrics_lbl)
        self.reg_metrics = QTextEdit('{\"accuracy\":0.0,\"precision\":0.0,\"recall\":0.0,\"f1\":0.0}')
        self.reg_metrics.setFixedHeight(70)
        reg_layout.addWidget(self.reg_metrics)

        reg_btns = QHBoxLayout()
        reg_btns.setSpacing(8)
        b_prefill = QPushButton("Prefill From Active")
        b_prefill.setObjectName("GhostButton")
        b_register = QPushButton("Register Model")
        b_register.setObjectName("AccentButton")
        b_rollback = QPushButton("Rollback Version")
        b_rollback.setStyleSheet(
            "background-color:#3f1d1d; border:1px solid #ef4444; color:#fecaca; "
            "border-radius:10px; padding:10px; font-weight:600;"
        )
        b_prefill.clicked.connect(self.prefill_model_paths)
        b_register.clicked.connect(self.register_model_version_from_ui)
        b_rollback.clicked.connect(self.rollback_model_family_from_ui)
        reg_btns.addWidget(b_prefill)
        reg_btns.addWidget(b_register)
        reg_btns.addWidget(b_rollback)
        reg_btns.addStretch()
        reg_layout.addLayout(reg_btns)
        outer.addWidget(reg_card)
        outer.addStretch()
        return page

    def _start_reference_threads(self):
        self.study_ref_thread = ReferenceVideoThread()
        self.study_ref_thread.frame_signal.connect(self._update_study_ref_frame)
        self.study_ref_thread.status_signal.connect(lambda msg: self._sync_reference_status(msg, "study"))
        self.study_ref_thread.start()

        self.practice_ref_thread = ReferenceVideoThread()
        self.practice_ref_thread.frame_signal.connect(self._update_practice_ref_frame)
        self.practice_ref_thread.status_signal.connect(lambda msg: self._sync_reference_status(msg, "practice"))
        self.practice_ref_thread.start()

    def _on_stack_changed(self, idx: int):
        if idx != self.IDX_STUDY:
            self._flush_study_timer()
        if idx == self.IDX_LOGIN:
            for button in self._nav_index_map.values():
                button.setChecked(False)
            if hasattr(self, "feedback_mode"):
                self.feedback_mode.setText("Mode: Login")
            return
        if hasattr(self, "feedback_mode"):
            self.feedback_mode.setText(f"Mode: {self._mode_name(idx)}")
        self._set_active_nav(idx)
        if idx == self.IDX_ANALYTICS and self.session.is_authenticated():
            self.load_analytics()

    def navigate_to(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def _set_active_nav(self, idx: int) -> None:
        for nav_idx, button in self._nav_index_map.items():
            button.setChecked(nav_idx == idx)

    def _mode_name(self, idx: int) -> str:
        return {
            self.IDX_LOGIN: "Login",
            self.IDX_DASH: "Dashboard",
            self.IDX_STUDY: "Study",
            self.IDX_PRACTICE: "Practice",
            self.IDX_QUIZ: "Quiz",
            self.IDX_ANALYTICS: "Analytics",
            self.IDX_ADMIN: "Admin",
        }.get(idx, "Unknown")

    def _sync_reference_status(self, message: str, source: str) -> None:
        if source == "study":
            self.study_ref_status.setText(message)
        elif source == "quiz":
            if hasattr(self, 'quiz_ref_status'):
                self.quiz_ref_status.setText(message)
        else:
            self.practice_ref_status.setText(message)
        self.feedback_ref_status.setText(f"Reference: {message}")

    def _set_feedback_status(self, text: str, color: str) -> None:
        self.feedback_status.setText(f"Status: {text}")
        self.feedback_status.setStyleSheet(
            f"background-color:#0b0f18; border:1px solid {color}; border-radius:12px; "
            f"padding:6px 10px; color:{color}; font-weight:700;"
        )

    def _set_practice_feedback(self, text: str, color: str) -> None:
        self.practice_feedback.setText(text)
        self.practice_feedback.setStyleSheet(
            f"background-color:#0b0f18; border:1px solid {color}; border-radius:12px; "
            f"padding:6px 10px; color:{color}; font-weight:700;"
        )

    def _apply_environment_status(self):
        self._set_indicator(self.env_mp, "MediaPipe", self.env_status.get("mediapipe", False))
        self._set_indicator(self.env_torch, "Torch", self.env_status.get("torch", False))
        self._set_indicator(self.env_cam, "Camera", self.env_status.get("camera", False))
        self._set_indicator(self.env_static, "StaticModel", self.env_status.get("static_model_loaded", False))
        self._set_indicator(self.env_dynamic, "DynamicModel", self.env_status.get("dynamic_model_loaded", False))

        if hasattr(self, "btn_start_camera"):
            self.btn_start_camera.setEnabled(True)
            self.btn_start_camera.setText("Start Live Practice")
            if not self.env_status.get("mediapipe", False):
                self._set_practice_feedback("MediaPipe unavailable. Practice may fail until dependencies are fixed.", "#ef4444")
                self._set_feedback_status("MediaPipe warning", "#ef4444")
            elif not self.env_status.get("camera", False):
                self._set_practice_feedback("Camera check warning. Start can still be attempted.", "#facc15")
                self._set_feedback_status("Camera warning", "#facc15")
            else:
                self._set_feedback_status("Ready", "#22c55e")

    @staticmethod
    def _set_indicator(label: QLabel, name: str, ok: bool):
        dot = "●"
        color = "#34d399" if ok else "#f87171"
        label.setText(f"{dot} {name}")
        label.setStyleSheet(f"color:{color}; font-weight:600;")

    def refresh_environment_status(self):
        self.env_status = check_environment(self.config)
        self._apply_environment_status()

    def handle_login(self):
        email = self.login_email.text().strip().lower()
        password = self.login_password.text()
        row = self.db.get_user_by_email(email)
        if not row or not verify_password(password, row["password_hash"]):
            QMessageBox.warning(self, "Login Failed", "Invalid credentials")
            return
        self.session.user_id = row["user_id"]
        self.session.email = row["email"]
        self.session.full_name = row["full_name"]
        self.session.role = row["role"]
        self.session.token = create_access_token(row["user_id"])

        self.sidebar.setVisible(True)
        self._nav_index_map[self.IDX_ADMIN].setVisible(self.session.role == "admin")
        self.navigate_to(self.IDX_DASH)
        self.welcome.setText(f"Welcome, {self.session.full_name}")
        self.feedback_user.setText(f"User: {self.session.full_name}")
        self.feedback_mode.setText("Mode: Dashboard")
        self.feedback_target.setText("Target: -")
        self._set_feedback_status("Ready", "#22c55e")
        self.refresh_after_login()

    def refresh_after_login(self):
        gestures = [dict(g) for g in self.db.get_gestures()]
        self.gesture_rows = gestures
        self.lesson_summary.setText(f"{len(LEVEL_INFO)} guided levels | {len(gestures)} local gestures ready")
        self.feedback_ref_source.setText("Source: isl_videos (local)")
        level_map = {g.display_name: g.level for g in all_gestures()}
        self.level_groups = {lvl: [] for lvl in LEVEL_INFO}
        self.gesture_level_by_id = {}
        for gesture in gestures:
            level = level_map.get(gesture["display_name"], 1)
            self.level_groups.setdefault(level, []).append(gesture)
            self.gesture_level_by_id[str(gesture["gesture_id"])] = level

        self.metric_levels.setText(str(len(LEVEL_INFO)))
        self.metric_gestures.setText(str(len(gestures)))
        self.metric_focus.setText("Level 1")
        self.metric_mode.setText("Live + Local")
        level_one_count = len(self.level_groups.get(1, []))
        self.dashboard_focus_title.setText("Start with Level 1")
        self.dashboard_focus_copy.setText("Letters come first so the hand shapes are stable before you move into words and longer signs.")
        self.dashboard_focus_meta.setText(f"{level_one_count} gestures are ready in Level 1. Study one, then move straight into live practice.")

        for lvl, button in self.dashboard_level_buttons.items():
            info = LEVEL_INFO[lvl]
            count = len(self.level_groups.get(lvl, []))
            label = "gesture" if count == 1 else "gestures"
            button.setText(
                f"{info['emoji']}  Level {lvl} · {info['title']}\n"
                f"{count} {label}\n"
                f"{info['desc']}"
            )

        self._select_study_level(min(LEVEL_INFO) if LEVEL_INFO else 1)
        self._select_practice_level(min(LEVEL_INFO) if LEVEL_INFO else 1)

        self.load_analytics()
        self.load_model_versions()

    def _set_level_button_copy(self, button: QPushButton, level: int, count: int) -> None:
        info = LEVEL_INFO[level]
        label = "gesture" if count == 1 else "gestures"
        button.setText(
            f"{info['emoji']}  Level {level} · {info['title']}\n"
            f"{count} {label}\n"
            f"{info['desc']}"
        )

    def _set_level_button_style(self, button: QPushButton, level: int, active: bool) -> None:
        info = LEVEL_INFO[level]
        background = info["bg"] if active else "#101726"
        border = info["accent"] if active else "#2a3446"
        text_color = "#f8fafc" if active else "#dbeafe"
        button.setStyleSheet(
            f"QPushButton#LevelSelector {{ background: qlineargradient("
            f"x1:0, y1:0, x2:1, y2:1, stop:0 {background}, stop:1 #101726); "
            f"border: 1px solid {border}; color: {text_color}; }}"
            f"QPushButton#LevelSelector:hover {{ border: 1px solid {info['accent']}; }}"
        )

    def _update_level_button_group(self, buttons: Dict[int, QPushButton], current_level: int) -> None:
        for level, button in buttons.items():
            self._set_level_button_copy(button, level, len(self.level_groups.get(level, [])))
            self._set_level_button_style(button, level, level == current_level)

    def _open_study_level(self, level: int) -> None:
        self.navigate_to(self.IDX_STUDY)
        self._select_study_level(level)

    def _reset_study_detail(self) -> None:
        self.selected_gesture = None
        self._study_gesture_id = None
        self._study_started_at = None
        if self.study_ref_thread:
            self.study_ref_thread.set_media(None)
        self.study_ref_label.clear()
        self.study_ref_label.setText("Choose a gesture from the level list to load its reference.")
        self.study_ref_label.setAlignment(Qt.AlignCenter)
        self.study_ref_status.setText("Reference not loaded")
        self.study_name.setText("Select a gesture")
        self.study_type.setText("Type: -")
        self.study_diff.setText("Difficulty: -")
        self.study_desc.setText("Choose a gesture from the level list to load its explanation, tips, and reference media.")

    def _select_study_level(self, level: int, target_id: Optional[str] = None) -> None:
        if level not in LEVEL_INFO:
            return
        self._current_study_level = level
        self._update_level_button_group(self.study_level_buttons, level)
        info = LEVEL_INFO[level]
        rows = [dict(row) for row in self.level_groups.get(level, [])]
        self._current_study_rows = rows
        self.study_level_title.setText(f"{info['emoji']}  Level {level} · {info['title']}")
        self.study_level_caption.setText(info["desc"])
        self.study_level_count.setText(f"{len(rows)} gestures ready in this level")
        self.study_header_meta.setText(f"Level {level} focuses on {info['title'].lower()}.")

        self.study_gesture_list.blockSignals(True)
        self.study_gesture_list.clear()
        selected_row = -1
        for idx, gesture in enumerate(rows):
            icon = "★" if gesture["gesture_mode"] == "static" else "↺"
            self.study_gesture_list.addItem(f"{icon}  {gesture['display_name']}  ·  {gesture['gesture_mode'].capitalize()}")
            if target_id and str(gesture["gesture_id"]) == target_id:
                selected_row = idx
        self.study_gesture_list.blockSignals(False)

        if selected_row >= 0:
            self.study_gesture_list.setCurrentRow(selected_row)
        else:
            self.study_gesture_list.clearSelection()
            self._reset_study_detail()

    def _reset_practice_target(self) -> None:
        self.selected_gesture = None
        if self.practice_ref_thread:
            self.practice_ref_thread.set_media(None)
        self.practice_ref_label.clear()
        self.practice_ref_label.setText("Reference not available")
        self.practice_ref_label.setAlignment(Qt.AlignCenter)
        self.practice_ref_status.setText("Reference not available")
        self.practice_target_stat.setText("Target: -")
        self.practice_target_meta.setText("No target selected")
        self.practice_pred_stat.setText("Prediction: -")
        self.practice_conf_stat.setText("Confidence: 0.00")
        self.practice_model_stat.setText("Model: -")
        self.practice_fps_stat.setText("FPS: 0.0")
        self.live_prediction.setText("Model idle. Choose a gesture and start the camera when you're ready.")
        self._set_practice_feedback("Select a target gesture to prepare the session.", "#94a3b8")

    def _select_practice_level(self, level: int, target_id: Optional[str] = None) -> None:
        if level not in LEVEL_INFO:
            return
        self._current_practice_level = level
        rows: List[Dict[str, object]] = []
        for current_level in sorted(LEVEL_INFO):
            level_rows = sorted(
                [dict(row) for row in self.level_groups.get(current_level, [])],
                key=lambda row: str(row["display_name"]),
            )
            rows.extend(level_rows)
        self._current_practice_rows = rows
        self.practice_target_combo.blockSignals(True)
        self.practice_target_combo.clear()
        selected_row = 0 if rows else -1
        for idx, gesture in enumerate(rows):
            row_level = self.gesture_level_by_id.get(str(gesture["gesture_id"]), 1)
            self.practice_target_combo.addItem(
                f"{gesture['display_name']}  ·  Level {row_level}  ·  {gesture['gesture_mode'].capitalize()}",
                idx,
            )
            if target_id and str(gesture["gesture_id"]) == target_id:
                selected_row = idx
        self.practice_target_combo.blockSignals(False)

        if selected_row >= 0:
            self.practice_target_combo.setCurrentIndex(selected_row)
        else:
            self._reset_practice_target()

    # ------------------------------------------------------------------ sidebar
    def _toggle_sidebar(self):
        if self._sidebar_expanded:
            self.sidebar.setFixedWidth(self._sidebar_collapsed_width)
            self.logo_label.setVisible(False)
            self.logo_sub.setVisible(False)
            collapsed_nav = (
                "QPushButton#NavButton { text-align: center; padding: 10px 0px; }"
                "QPushButton#NavButton:checked { text-align: center; padding: 10px 0px; }"
            )
            collapsed_logout = (
                "QPushButton#LogoutButton { text-align: center; padding: 10px 0px; }"
            )
            for btn in self._sidebar_nav_buttons_list:
                btn.setText("")
                btn.setStyleSheet(collapsed_nav)
            self.btn_logout.setText("")
            self.btn_logout.setStyleSheet(collapsed_logout)
            self.sidebar_toggle.setText("\u276F")
        else:
            self.sidebar.setFixedWidth(self._sidebar_full_width)
            self.logo_label.setVisible(True)
            self.logo_sub.setVisible(True)
            for i, (key, btn) in enumerate(self.nav_buttons.items()):
                btn.setText(key)
                btn.setStyleSheet("")  # revert to global stylesheet
            self.btn_logout.setText("Logout")
            self.btn_logout.setStyleSheet("")
            self.sidebar_toggle.setText("\u276E")
        self._sidebar_expanded = not self._sidebar_expanded

    def logout(self):
        self.stop_camera()
        self._flush_study_timer()
        self.session = SessionState()
        self.feedback_user.setText("User: -")
        self.feedback_mode.setText("Mode: Login")
        self.feedback_target.setText("Target: -")
        self.feedback_prediction.setText("Prediction: -")
        self.feedback_conf.setText("Confidence: 0.00")
        self.feedback_model.setText("Model: -")
        self.feedback_fps.setText("FPS: 0.0")
        self._set_feedback_status("Logged out", "#facc15")
        self.sidebar.setVisible(False)
        self._nav_index_map[self.IDX_ADMIN].setVisible(False)
        self.gesture_rows = []
        self.level_groups = {lvl: [] for lvl in LEVEL_INFO}
        self.gesture_level_by_id = {}
        self._current_study_rows = []
        self._current_practice_rows = []
        self.study_gesture_list.clear()
        self.practice_target_combo.clear()
        self._reset_study_detail()
        self._reset_practice_target()
        self.navigate_to(self.IDX_LOGIN)

    def _difficulty_for(self, gesture: Dict[str, object]) -> str:
        mode = str(gesture.get("gesture_mode", "static"))
        category = str(gesture.get("category", ""))
        if mode == "dynamic":
            return "Intermediate"
        if category in {"emergency", "conversation"}:
            return "Intermediate"
        return "Beginner"

    def _show_text_reference(self, label: QLabel, gesture_name: str, ref_info: Dict[str, str]) -> None:
        """Show text guidance when no reference video is available."""
        desc = ref_info.get("description", "")
        tips = ref_info.get("tips", "")
        difficulty = ref_info.get("difficulty", "")
        hands = ref_info.get("hands", ref_info.get("hand", ""))

        parts = [f"<b style='color:#22c55e; font-size:18px;'>{gesture_name}</b>"]
        if difficulty:
            parts.append(f"<span style='color:#94a3b8; font-size:12px;'>Difficulty: {difficulty.capitalize()}</span>")
        if hands:
            parts.append(f"<span style='color:#94a3b8; font-size:12px;'>Hand(s): {hands}</span>")
        parts.append("")
        if desc:
            parts.append(f"<p style='color:#e2e8f0; font-size:14px; line-height:1.6;'>{desc}</p>")
        if tips:
            parts.append(f"<p style='color:#b45309; font-size:13px;'>💡 {tips}</p>")
        if not desc and not tips:
            parts.append("<p style='color:#94a3b8;'>No reference available for this gesture.</p>")

        label.setText("<br>".join(parts))
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setWordWrap(True)
        label.setStyleSheet(
            "background:#05070a; border:1px solid #334155; border-radius:14px; "
            "padding:18px; font-family:sans-serif;"
        )

    def on_select_study_gesture(self):
        idx = self.study_gesture_list.currentRow()
        if idx < 0 or idx >= len(self._current_study_rows):
            return

        self._flush_study_timer()
        self.selected_gesture = dict(self._current_study_rows[idx])
        g = self.selected_gesture
        ref_info = get_gesture_reference(str(g["display_name"]))
        hand_label = "Two hands" if g.get("requires_two_hands") else "One hand"
        self.study_name.setText(str(g["display_name"]))
        self.study_type.setText(f"Type: {g['gesture_mode'].capitalize()} sign · {hand_label}")
        diff = ref_info.get("difficulty") or self._difficulty_for(g)
        self.study_diff.setText(f"Difficulty: {diff.capitalize()} · Level {self._current_study_level}")
        desc = ref_info.get("description") or str(g.get("description") or "No description available.")
        tips = ref_info.get("tips", "")
        full_desc = desc
        if tips:
            full_desc += f"\n\n💡 Tips: {tips}"
        self.study_desc.setText(full_desc)
        self.feedback_target.setText(f"Target: {g['display_name']} [{g['gesture_mode']}]")

        gesture_key = str(g.get("gesture_code") or g["display_name"])
        self.study_ref_thread.set_media(None)
        media_path = get_media_path(gesture_key)
        if media_path:
            self.study_ref_thread.set_media(media_path)
            self._sync_reference_status("\u25b6 Playing reference video", "study")
        else:
            self._show_text_reference(self.study_ref_label, str(g["display_name"]), ref_info)
            self._sync_reference_status("Reference video not available", "study")
        self._study_gesture_id = str(g["gesture_id"])
        self._study_started_at = time.time()

    def start_practice_from_study(self):
        if not self.selected_gesture:
            QMessageBox.information(self, "Study", "Select a gesture first")
            return
        target_id = str(self.selected_gesture["gesture_id"])
        level = self.gesture_level_by_id.get(target_id, self._current_study_level)
        self._select_practice_level(level, target_id=target_id)
        self.navigate_to(self.IDX_PRACTICE)

    def _flush_study_timer(self):
        if not self.session.is_authenticated() or not self._study_gesture_id or self._study_started_at is None:
            self._study_started_at = None
            return
        elapsed = int(max(time.time() - self._study_started_at, 0))
        if elapsed > 0:
            self.db.record_study_session(self.session.user_id, self._study_gesture_id, elapsed)
        self._study_started_at = None

    def _load_gesture_reference(self, label: QLabel, ref_thread: ReferenceVideoThread, gesture_row: dict, source_tag: str):
        """Unified reference loader: uses only ``isl_videos`` mp4 references."""
        gesture_key = str(gesture_row.get("gesture_code") or gesture_row["display_name"])
        ref_thread.set_media(None)
        media_path = get_media_path(gesture_key)
        if media_path:
            ref_thread.set_media(media_path)
            self._sync_reference_status("\u25b6 Playing reference video", source_tag)
        else:
            ref_info = get_gesture_reference(str(gesture_row["display_name"]))
            self._show_text_reference(label, str(gesture_row["display_name"]), ref_info)
            self._sync_reference_status("Reference video not available", source_tag)

    def on_select_practice_gesture(self, index: Optional[int] = None):
        idx = self.practice_target_combo.currentIndex() if index is None else index
        if idx < 0 or idx >= len(self._current_practice_rows):
            return
        row = dict(self._current_practice_rows[idx])
        self.selected_gesture = row
        self._load_gesture_reference(self.practice_ref_label, self.practice_ref_thread, row, "practice")
        row_level = self.gesture_level_by_id.get(str(row["gesture_id"]), 1)
        level_info = LEVEL_INFO.get(row_level, LEVEL_INFO[min(LEVEL_INFO)])

        self.practice_target_stat.setText(f"Target: {row['display_name']}")
        self.practice_target_meta.setText(
            f"Level {row_level} · {level_info['title']} · {row['gesture_mode'].capitalize()} sign"
        )
        self.feedback_target.setText(f"Target: {row['display_name']} [{row['gesture_mode']}]")
        self.live_prediction.setText(
            f"Ready for {row['display_name']} · {row['gesture_mode'].capitalize()} sign · "
            f"Level {row_level}"
        )
        self._set_practice_feedback(
            f"Target selected: {row['display_name']}. Press Start Live Practice when you are ready.",
            "#22c55e",
        )
        if self.inference_thread:
            self.inference_thread.set_target(str(row["display_name"]))
            self.inference_thread.set_target_mode(str(row["gesture_mode"]))

    def start_camera(self):
        if self.inference_thread and self.inference_thread.isRunning():
            return
        if not self.selected_gesture:
            self._set_practice_feedback("Select a gesture first to start the camera.", "#facc15")
            return

        try:
            self.inference_thread = InferenceThread(self.predictor, self.env_status)
            self.inference_thread.set_target(str(self.selected_gesture["display_name"]))
            self.inference_thread.set_target_mode(str(self.selected_gesture["gesture_mode"]))
            self.inference_thread.frame_signal.connect(self.update_camera_view)
            self.inference_thread.result_signal.connect(self.update_result)
            self.inference_thread.start()
            self._set_feedback_status("Camera started", "#22c55e")
            self.live_prediction.setText(
                f"Camera active for {self.selected_gesture['display_name']} · hold the sign steadily in frame."
            )
            self._set_practice_feedback("Camera active. Show your sign!", "#22c55e")
        except Exception as e:
            self._set_feedback_status("Camera error", "#ef4444")
            self._set_practice_feedback(f"Camera failed: {e}", "#ef4444")

    def stop_camera(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None
            self._set_feedback_status("Camera stopped", "#facc15")
            self.live_prediction.setText("Camera stopped. You can change targets or restart practice.")
            self._set_practice_feedback("Camera stopped.", "#facc15")

    def update_camera_view(self, qimg: QImage):
        self._last_qimage = qimg
        pix = QPixmap.fromImage(qimg)
        # Update practice camera view
        scaled = pix.scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_view.setPixmap(scaled)
        # Also update quiz camera view if it exists
        if hasattr(self, 'quiz_camera_view') and self.quiz_camera_view.isVisible():
            scaled_q = pix.scaled(self.quiz_camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.quiz_camera_view.setPixmap(scaled_q)

    def _update_study_ref_frame(self, qimg: QImage):
        self._last_study_ref = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.study_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.study_ref_label.setPixmap(pix)

    def _update_practice_ref_frame(self, qimg: QImage):
        self._last_practice_ref = qimg
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.practice_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.practice_ref_label.setPixmap(scaled)
        # Always update quiz ref label too (even when hidden, so it's ready on reveal)
        if hasattr(self, 'quiz_ref_label'):
            scaled_q = pix.scaled(self.quiz_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.quiz_ref_label.setPixmap(scaled_q)

    def update_result(self, result: dict):
        self.current_result = result
        status = result.get("status", "-")
        label = result.get("label", "-")
        model_used = result.get("model_used", "-")
        conf = float(result.get("confidence", 0.0))
        fps = float(result.get("fps", 0.0))
        stable = bool(result.get("stable", False))
        warn = result.get("perf_warning", "")

        self.live_prediction.setText(
            f"Target: {self.selected_gesture['display_name'] if self.selected_gesture else '-'} | "
            f"Model: {model_used} | Confidence: {conf:.2f} | FPS: {fps:.1f}"
        )
        self.practice_pred_stat.setText(f"Prediction: {label}")
        self.practice_conf_stat.setText(f"Confidence: {conf:.2f}")
        self.practice_model_stat.setText(f"Model: {model_used}")
        self.practice_fps_stat.setText(f"FPS: {fps:.1f}")
        self.feedback_prediction.setText(f"Prediction: {label}")
        self.feedback_conf.setText(f"Confidence: {conf:.2f}")
        self.feedback_model.setText(f"Model: {model_used}")
        self.feedback_fps.setText(f"FPS: {fps:.1f}")

        # Update quiz prediction label if visible
        if hasattr(self, 'quiz_pred_label') and self.quiz_pred_label.isVisible():
            self.quiz_pred_label.setText(f"Prediction: {label} | Confidence: {conf:.2f}")

        if warn:
            self._set_practice_feedback(warn, "#facc15")
            self._set_feedback_status("Low FPS", "#facc15")
            return

        if self.selected_gesture and status in {"ok", "uncertain"}:
            target = self.selected_gesture["display_name"]
            # Check if this gesture is in the trained model's label map
            predictor = getattr(self, 'predictor', None)
            target_in_model = True
            if predictor and hasattr(predictor, 'label_map'):
                target_in_model = target in predictor.label_map

            if not target_in_model:
                self._set_practice_feedback(
                    f"'{target}' not yet trained. Keep practising with the reference video!",
                    "#60a5fa")
                self._set_feedback_status("Untrained", "#60a5fa")
            elif stable and label == target:
                self._set_practice_feedback("Correct and stable! Well done!", "#22c55e")
                self._set_feedback_status("Correct", "#22c55e")
            elif label == target and not stable:
                self._set_practice_feedback("Detected! Hold steady to confirm...", "#a3e635")
                self._set_feedback_status("Hold steady", "#a3e635")
            elif status == "uncertain":
                self._set_practice_feedback("Analysing your gesture — hold steady...", "#facc15")
                self._set_feedback_status("Uncertain", "#facc15")
            else:
                self._set_practice_feedback(f"Keep trying — showing: {label}. Target: {target}", "#fb923c")
                self._set_feedback_status("Try again", "#fb923c")
        elif status in {"mediapipe_unavailable", "dynamic_model_unavailable"}:
            env_text = (
                f"Environment: MediaPipe: {'❌' if not self.env_status.get('mediapipe') else '✅'} | "
                f"Torch: {'✅' if self.env_status.get('torch') else '❌'} | "
                f"Camera: {'✅' if self.env_status.get('camera') else '❌'}"
            )
            self._set_practice_feedback(env_text, "#ef4444")
            self._set_feedback_status("Environment error", "#ef4444")
        elif status == "no_hand":
            self._set_practice_feedback("No hand detected. Place hand in frame.", "#facc15")
            self._set_feedback_status("No hand", "#facc15")
        elif status == "camera_error":
            self._set_practice_feedback("Camera error. Restart camera.", "#ef4444")
            self._set_feedback_status("Camera error", "#ef4444")

    def record_current_attempt(self):
        if not self.session.is_authenticated() or not self.selected_gesture or not self.current_result:
            return
        predicted = str(self.current_result.get("label", "UNKNOWN"))
        conf = float(self.current_result.get("confidence", 0.0))
        latency = int(self.current_result.get("latency_ms", 0))
        fps = float(self.current_result.get("fps", 0.0))
        stable = bool(self.current_result.get("stable", False))
        is_correct = stable and predicted == self.selected_gesture["display_name"]

        self.db.record_attempt(
            user_id=self.session.user_id,
            gesture_id=self.selected_gesture["gesture_id"],
            target_gesture_id=self.selected_gesture["gesture_id"],
            predicted_label=predicted,
            confidence=conf,
            is_correct=is_correct,
            latency_ms=latency,
            fps=fps,
            attempt_mode="practice",
        )
        self._set_practice_feedback(f"Attempt recorded | correct={is_correct}", "#22c55e" if is_correct else "#ef4444")
        self.load_analytics()

    def start_quiz(self):
        all_gestures = [dict(r) for r in self.db.get_random_gestures(limit=50)]
        # Filter to only gestures the model has been trained on
        if hasattr(self, 'predictor') and hasattr(self.predictor, 'label_map'):
            trained = set(self.predictor.label_map.keys())
            all_gestures = [g for g in all_gestures if g.get("display_name") in trained]
        import random
        random.shuffle(all_gestures)
        self.quiz_queue = all_gestures[:10]
        self.quiz_index = 0
        self.quiz_score = 0
        if not self.quiz_queue:
            self.quiz_target.setText("No gestures available for quiz")
            return
        self._set_quiz_target()
        # Auto-start camera for the quiz
        self.selected_gesture = self.quiz_queue[0]
        if not self.inference_thread or not self.inference_thread.isRunning():
            self.start_camera()
        else:
            self.inference_thread.set_target(str(self.selected_gesture["display_name"]))
            self.inference_thread.set_target_mode(str(self.selected_gesture["gesture_mode"]))

    def _set_quiz_target(self):
        if self.quiz_index >= len(self.quiz_queue):
            self.quiz_target.setText("\u2705 Quiz complete!")
            self.quiz_state.setText(f"Final score: {self.quiz_score}/{len(self.quiz_queue)}")
            self.quiz_progress.setText(f"Done! {len(self.quiz_queue)}/{len(self.quiz_queue)}")
            self.quiz_feedback.setText(f"You scored {self.quiz_score}/{len(self.quiz_queue)}")
            self.quiz_feedback.setStyleSheet(
                "background-color:#0b0f18; border:1px solid #22c55e; border-radius:12px; "
                "padding:6px 10px; color:#22c55e; font-weight:700;"
            )
            return
        target = self.quiz_queue[self.quiz_index]
        self.selected_gesture = target
        self.quiz_target.setText(target['display_name'])
        self.quiz_state.setText(f"Score: {self.quiz_score}/{self.quiz_index}")
        self.quiz_progress.setText(f"Question: {self.quiz_index + 1}/{len(self.quiz_queue)}")
        self.quiz_feedback.setText("Make the sign and press Submit")
        self.quiz_feedback.setStyleSheet(
            "background-color:#0b0f18; border:1px solid #334155; border-radius:12px; "
            "padding:6px 10px; color:#94a3b8; font-weight:700;"
        )
        # Hide reference for new question, pre-load it in background
        self.quiz_ref_card.setVisible(False)
        self.btn_reveal_answer.setText("\U0001F441  Reveal Answer")
        if hasattr(self, 'quiz_ref_label') and hasattr(self, 'practice_ref_thread'):
            self._load_gesture_reference(self.quiz_ref_label, self.practice_ref_thread, target, "quiz")
            self.quiz_ref_status.setText(f"Reference for: {target['display_name']}")
        # Update inference thread target
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.set_target(str(target["display_name"]))
            self.inference_thread.set_target_mode(str(target["gesture_mode"]))

    def submit_quiz_answer(self):
        if self.quiz_index >= len(self.quiz_queue) or not self.current_result:
            if not self.current_result:
                self.quiz_feedback.setText("Camera not active. Start the quiz first!")
                self.quiz_feedback.setStyleSheet(
                    "background-color:#0b0f18; border:1px solid #ef4444; border-radius:12px; "
                    "padding:6px 10px; color:#ef4444; font-weight:700;"
                )
            return
        target = self.quiz_queue[self.quiz_index]
        pred = str(self.current_result.get("label", "UNKNOWN"))
        conf = float(self.current_result.get("confidence", 0.0))
        stable = bool(self.current_result.get("stable", False))
        is_correct = stable and pred == target["display_name"] and conf >= 0.30
        self.quiz_score += int(is_correct)

        if is_correct:
            self.quiz_feedback.setText(f"\u2705 Correct! ({pred}, {conf:.0%})")
            self.quiz_feedback.setStyleSheet(
                "background-color:#0b0f18; border:1px solid #22c55e; border-radius:12px; "
                "padding:6px 10px; color:#22c55e; font-weight:700;"
            )
        else:
            self.quiz_feedback.setText(f"\u274c Incorrect \u2014 predicted {pred} ({conf:.0%})")
            self.quiz_feedback.setStyleSheet(
                "background-color:#0b0f18; border:1px solid #ef4444; border-radius:12px; "
                "padding:6px 10px; color:#ef4444; font-weight:700;"
            )

        self.db.record_attempt(
            user_id=self.session.user_id,
            gesture_id=target["gesture_id"],
            target_gesture_id=target["gesture_id"],
            predicted_label=pred,
            confidence=conf,
            is_correct=is_correct,
            latency_ms=int(self.current_result.get("latency_ms", 0)),
            fps=float(self.current_result.get("fps", 0.0)),
            attempt_mode="quiz",
        )

        self.quiz_index += 1
        # Delay showing next question briefly so user sees feedback
        QTimer.singleShot(1200, lambda: (self._set_quiz_target(), self.load_analytics()))

    def load_analytics(self):
        if not self.session.is_authenticated():
            return
        summary = self.db.get_analytics_summary(self.session.user_id)
        total = int(summary['total_attempts'])
        acc = summary['accuracy']
        avg_conf = summary['avg_confidence']
        avg_lat = summary['avg_latency_ms']

        # Update stat cards
        if hasattr(self, 'stat_total'):
            self.stat_total.setText(str(total))
            self.stat_accuracy.setText(f"{acc:.2f}")
            self.stat_avg_conf.setText(f"{avg_conf:.2f}")
            self.stat_avg_latency.setText(f"{avg_lat:.1f}ms")

        self.analytics_summary.setText(
            f"Attempts: {total} | Accuracy: {acc:.2f} | "
            f"Avg Conf: {avg_conf:.2f} | Avg Latency: {avg_lat:.1f}ms"
        )

        rows = self.db.get_user_attempts(self.session.user_id, limit=120)
        self.analytics_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r["created_at"][:19],
                r["target_name"],
                r["predicted_label"],
                f"{r['confidence']:.2f}",
                "Yes" if r["is_correct"] else "No",
                r["attempt_mode"],
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                # Color-code correct/incorrect
                if c == 4:
                    item.setForeground(QColor("#22c55e") if v == "Yes" else QColor("#ef4444"))
                self.analytics_table.setItem(i, c, item)

        progress_rows = self.db.get_user_progress(self.session.user_id)
        if progress_rows:
            txt = " | ".join([f"{r['title']}: {r['accuracy']:.2f} ({r['attempts_count']} attempts)" for r in progress_rows])
            self.progress_summary.setText(f"Progress: {txt}")
            if hasattr(self, 'progress_detail'):
                parts = [f"<span style='color:#22c55e;font-weight:600;'>{r['title']}</span>: "
                         f"accuracy {r['accuracy']:.0%}, {r['attempts_count']} attempts"
                         for r in progress_rows]
                self.progress_detail.setText("<br>".join(parts))

    def load_confusion_matrix_view(self):
        if not self.session.is_authenticated():
            return
        attempts = self.db.get_user_attempts(self.session.user_id, limit=5000)
        if not attempts:
            self.confusion_note.setText("No attempts available to build confusion matrix.")
            self.confusion_table.setRowCount(0)
            self.confusion_table.setColumnCount(0)
            return

        labels = sorted({str(a["target_name"]) for a in attempts} | {str(a["predicted_label"]) for a in attempts})
        max_labels = 25
        if len(labels) > max_labels:
            freq = {}
            for a in attempts:
                freq[a["target_name"]] = freq.get(a["target_name"], 0) + 1
            labels = [x for x, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:max_labels]]
            self.confusion_note.setText("Showing top 25 target classes by frequency.")
        else:
            self.confusion_note.setText(f"Showing {len(labels)} classes.")

        idx = {name: i for i, name in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int32)
        for a in attempts:
            t = str(a["target_name"])
            p = str(a["predicted_label"])
            if t in idx and p in idx:
                cm[idx[t], idx[p]] += 1

        self.confusion_table.setRowCount(len(labels))
        self.confusion_table.setColumnCount(len(labels))
        self.confusion_table.setVerticalHeaderLabels(labels)
        self.confusion_table.setHorizontalHeaderLabels(labels)
        row_totals = cm.sum(axis=1, keepdims=True).astype(np.float32)
        row_totals[row_totals == 0] = 1.0
        cm_norm = cm / row_totals

        for r in range(len(labels)):
            for c in range(len(labels)):
                val = int(cm[r, c])
                item = QTableWidgetItem(str(val))
                v = float(cm_norm[r, c])
                green = int(60 + 150 * v)
                red = int(30 + 120 * (1.0 - v))
                item.setBackground(QColor(red, green, 70))
                self.confusion_table.setItem(r, c, item)

    def reseed(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        self.db.seed_core_data()
        self.refresh_after_login()
        self.admin_label.setText("Database reseeded successfully")

    def load_model_versions(self):
        rows = self.db.list_model_versions()
        self.model_table.setRowCount(len(rows))
        self._model_rows = [dict(r) for r in rows]
        for i, row in enumerate(rows):
            vals = [
                row["model_name"],
                row["version_tag"],
                row["framework"],
                row["artifact_path"],
                "Yes" if row["is_active"] else "No",
                row["trained_at"][:19],
            ]
            for c, v in enumerate(vals):
                self.model_table.setItem(i, c, QTableWidgetItem(str(v)))

    def activate_selected_model(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        row = self.model_table.currentRow()
        if row < 0 or row >= len(getattr(self, "_model_rows", [])):
            QMessageBox.information(self, "Select Model", "Choose a model row first.")
            return
        model_row = self._model_rows[row]
        ok = self.db.activate_model_version(model_row["model_version_id"])
        if not ok:
            QMessageBox.warning(self, "Activation Failed", "Could not activate model version.")
            return
        self.load_model_versions()

    def reload_predictor_from_registry(self):
        active = self.db.get_active_model_paths()
        model_cfg = self.config.setdefault("model", {})
        changed = []
        for key in ("static_model_path", "dynamic_model_path", "label_map_path", "norm_stats_path"):
            if key in active and model_cfg.get(key) != active[key]:
                model_cfg[key] = active[key]
                changed.append(key)
        if changed:
            save_config(self.config, "config/app.yaml")

        was_running = self.inference_thread is not None and self.inference_thread.isRunning()
        self.stop_camera()
        self.predictor = GesturePredictor(self.config)
        self.refresh_environment_status()
        if was_running:
            self.start_camera()
        self.admin_label.setText("Predictor reloaded" + (f" | updated: {', '.join(changed)}" if changed else ""))

    def _pick_file_into(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            line_edit.setText(path)

    def prefill_model_paths(self):
        active = self.db.get_active_model_paths()
        model_name = self.reg_model_name.currentText()
        if model_name == "static_mlp":
            self.reg_artifact_path.setText(active.get("static_model_path", self.reg_artifact_path.text()))
            self.reg_norm_stats_path.setText(active.get("norm_stats_path", self.reg_norm_stats_path.text()))
            self.reg_label_map_path.setText(active.get("label_map_path", self.reg_label_map_path.text()))
        elif model_name == "dynamic_bigru":
            self.reg_artifact_path.setText(active.get("dynamic_model_path", self.reg_artifact_path.text()))
            self.reg_norm_stats_path.setText("models/registry/dynamic_norm_stats.json")
            self.reg_label_map_path.setText(active.get("label_map_path", self.reg_label_map_path.text()))

    def register_model_version_from_ui(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        model_name = self.reg_model_name.currentText().strip()
        framework = self.reg_framework.text().strip() or "pytorch"
        version_tag = self.reg_version_tag.text().strip() or None
        artifact_path = self.reg_artifact_path.text().strip()
        label_map_path = self.reg_label_map_path.text().strip()
        norm_stats_path = self.reg_norm_stats_path.text().strip()

        if not artifact_path:
            QMessageBox.warning(self, "Invalid Input", "Artifact path is required")
            return
        try:
            metrics = json.loads(self.reg_metrics.toPlainText().strip() or "{}")
        except Exception:
            QMessageBox.warning(self, "Invalid Input", "Metrics JSON is invalid")
            return

        self.db.register_model_version(
            model_name=model_name,
            framework=framework,
            artifact_path=artifact_path,
            label_map_path=label_map_path or "models/registry/label_map.json",
            norm_stats_path=norm_stats_path or "models/registry/norm_stats.json",
            metrics=metrics if isinstance(metrics, dict) else {},
            activate=bool(self.reg_activate.isChecked()),
            version_tag=version_tag,
        )
        self.load_model_versions()
        if self.reg_activate.isChecked():
            self.reload_predictor_from_registry()

    def rollback_model_family_from_ui(self):
        if self.session.role != "admin":
            QMessageBox.warning(self, "Forbidden", "Admin role required")
            return
        model_name = self.reg_model_name.currentText().strip()
        ok = self.db.rollback_model_family(model_name)
        if not ok:
            QMessageBox.information(self, "Rollback", "Rollback not possible (need at least two versions).")
            return
        self.load_model_versions()
        self.reload_predictor_from_registry()

    def closeEvent(self, event):
        self.stop_camera()
        self._flush_study_timer()
        if self.study_ref_thread:
            self.study_ref_thread.stop()
        if self.practice_ref_thread:
            self.practice_ref_thread.stop()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_qimage is not None:
            self.update_camera_view(self._last_qimage)
        if self._last_study_ref is not None:
            self._update_study_ref_frame(self._last_study_ref)
        if self._last_practice_ref is not None:
            self._update_practice_ref_frame(self._last_practice_ref)
            # Also refresh quiz ref if visible
            if hasattr(self, 'quiz_ref_label') and self.quiz_ref_label.isVisible():
                pix = QPixmap.fromImage(self._last_practice_ref).scaled(
                    self.quiz_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.quiz_ref_label.setPixmap(pix)
