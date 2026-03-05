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

            self.predictor.tracker.draw(frame, self.last_result.get("extraction", {}))
            draw_overlay(frame, display_result, packet.fps, target=self.target_name)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        self.selected_gesture: Optional[Dict[str, object]] = None
        self.current_result: Optional[Dict[str, object]] = None
        self.quiz_queue: List[Dict[str, object]] = []
        self.quiz_index = 0
        self.quiz_score = 0

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
            #StatusPill {
                background-color: #0b0f18;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 6px 10px;
                font-weight: 700;
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
        self.sidebar.setFixedWidth(220)
        sbl = QVBoxLayout(self.sidebar)
        sbl.setContentsMargins(12, 14, 12, 14)
        sbl.setSpacing(8)

        logo = QLabel("MUDRA")
        logo.setObjectName("BrandTitle")
        logo.setFont(QFont(self.sys_font, 28, QFont.Bold))
        logo_sub = QLabel("Interactive ISL Learning")
        logo_sub.setObjectName("BrandSub")
        sbl.addWidget(logo)
        sbl.addWidget(logo_sub)
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

        content_wrap = QFrame()
        content_wrap.setObjectName("CenterWrap")
        content = QVBoxLayout(content_wrap)
        content.setContentsMargins(12, 12, 12, 12)
        content.setSpacing(10)
        self.status_bar_widget = self._build_env_header()
        content.addWidget(self.status_bar_widget)

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
        root.addWidget(self.right_panel)
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
        l = QVBoxLayout(page)
        l.addStretch()
        form = QFrame()
        form.setObjectName("InfoCard")
        form.setMaximumWidth(520)
        fl = QVBoxLayout(form)
        fl.setContentsMargins(24, 22, 24, 22)
        fl.setSpacing(10)
        title = QLabel("Login to MUDRA")
        title.setFont(QFont(self.sys_font, 28, QFont.Bold))
        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Email")
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        btn_login = QPushButton("Login")
        btn_demo = QPushButton("Use Demo Account")
        btn_login.clicked.connect(self.handle_login)
        btn_demo.clicked.connect(lambda: self.login_email.setText("demo@mudra.local") or self.login_password.setText("demo123"))
        fl.addWidget(title)
        fl.addWidget(self.login_email)
        fl.addWidget(self.login_password)
        fl.addWidget(btn_login)
        fl.addWidget(btn_demo)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(form)
        row.addStretch()
        l.addLayout(row)
        l.addStretch()
        return page

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        l = QVBoxLayout(page)
        l.setContentsMargins(8, 8, 8, 8)
        l.setSpacing(14)

        # Welcome section
        welcome_card = QFrame()
        welcome_card.setObjectName("InfoCard")
        wc = QVBoxLayout(welcome_card)
        wc.setContentsMargins(18, 14, 18, 14)
        wc.setSpacing(6)
        self.welcome = QLabel("Welcome")
        self.welcome.setFont(QFont(self.sys_font, 24, QFont.Bold))
        self.welcome.setStyleSheet("color:#f8fafc;")
        self.lesson_summary = QLabel("Lessons: -")
        self.lesson_summary.setStyleSheet("color:#94a3b8; font-size:14px;")
        self.progress_summary = QLabel("Progress: -")
        self.progress_summary.setStyleSheet("color:#94a3b8; font-size:13px;")
        wc.addWidget(self.welcome)
        wc.addWidget(self.lesson_summary)
        wc.addWidget(self.progress_summary)
        l.addWidget(welcome_card)

        # Feature cards
        grid = QGridLayout()
        grid.setSpacing(12)
        cards = [
            ("26 Alphabets", "Finger spelling lessons", "#22c55e", "#14532d", self.IDX_STUDY),
            ("50-100 Word Signs", "Common ISL words", "#1cb0f6", "#0c4a6e", self.IDX_STUDY),
            ("Study + Practice", "Reference first, then live camera", "#facc15", "#713f12", self.IDX_PRACTICE),
            ("Analytics", "Accuracy, confidence, confusion matrix", "#a78bfa", "#3b0764", self.IDX_ANALYTICS),
        ]
        for i, (h, s, accent, bg, nav_idx) in enumerate(cards):
            c = QPushButton()
            c.setCursor(Qt.PointingHandCursor)
            c.setStyleSheet(
                f"QPushButton {{ background: {bg}; border-radius: 16px; "
                f"border: 1px solid {accent}40; padding: 20px 18px; text-align: left; }} "
                f"QPushButton:hover {{ background: {accent}30; border: 1px solid {accent}; }}"
            )
            c.clicked.connect(lambda _, idx=nav_idx: self.navigate_to(idx))
            btn_layout = QVBoxLayout(c)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(6)
            hl = QLabel(h)
            hl.setFont(QFont(self.sys_font, 15, QFont.Bold))
            hl.setStyleSheet(f"color:{accent}; background:transparent;")
            hl.setAttribute(Qt.WA_TransparentForMouseEvents)
            sl = QLabel(s)
            sl.setStyleSheet("color:#cbd5e1; font-size:13px; background:transparent;")
            sl.setAttribute(Qt.WA_TransparentForMouseEvents)
            btn_layout.addWidget(hl)
            btn_layout.addWidget(sl)
            grid.addWidget(c, i // 2, i % 2)
        l.addLayout(grid)
        l.addStretch()
        return page

    def _build_study_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        list_card = QFrame()
        list_card.setObjectName("InfoCard")
        ll = QVBoxLayout(list_card)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(8)
        ll.addWidget(QLabel("ISL Gesture Library"))
        self.study_gesture_list = QListWidget()
        self.study_gesture_list.setMinimumWidth(240)
        self.study_gesture_list.itemSelectionChanged.connect(self.on_select_study_gesture)
        ll.addWidget(self.study_gesture_list)

        center_card = QFrame()
        center_card.setObjectName("InfoCard")
        cl = QVBoxLayout(center_card)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(8)
        cl.addWidget(QLabel("Reference Animation"))
        self.study_ref_label = QLabel("Select a gesture to see its reference")
        self.study_ref_label.setAlignment(Qt.AlignCenter)
        self.study_ref_label.setMinimumSize(460, 380)
        self.study_ref_label.setWordWrap(True)
        self.study_ref_label.setStyleSheet("background:#05070a; border:1px solid #334155; border-radius:14px; padding:18px;")
        self.study_ref_status = QLabel("Reference not available")
        self.study_ref_status.setObjectName("SectionMeta")
        cl.addWidget(self.study_ref_label, 1)
        cl.addWidget(self.study_ref_status)

        info = QFrame()
        info.setObjectName("InfoCard")
        il = QVBoxLayout(info)
        il.setContentsMargins(10, 10, 10, 10)
        il.setSpacing(8)
        il.addWidget(QLabel("Gesture Details"))
        self.study_name = QLabel("Gesture: -")
        self.study_type = QLabel("Type: -")
        self.study_diff = QLabel("Difficulty: -")
        self.study_desc = QLabel("Description: Select a gesture to study.")
        self.study_desc.setWordWrap(True)
        self.study_desc.setStyleSheet("color:#cbd5e1; font-size:13px; line-height:1.5;")
        il.addWidget(self.study_name)
        il.addWidget(self.study_type)
        il.addWidget(self.study_diff)
        il.addWidget(self.study_desc)
        self.btn_start_practice_from_study = QPushButton("Start Practice")
        self.btn_start_practice_from_study.setObjectName("AccentButton")
        self.btn_start_practice_from_study.clicked.connect(self.start_practice_from_study)
        il.addWidget(self.btn_start_practice_from_study)
        il.addStretch()

        layout.addWidget(list_card, 2)
        layout.addWidget(center_card, 4)
        layout.addWidget(info, 3)
        return page

    def _build_practice_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        selector_card = QFrame()
        selector_card.setObjectName("InfoCard")
        sl = QVBoxLayout(selector_card)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(8)
        sl.addWidget(QLabel("Select Target Gesture"))
        self.practice_target_list = QListWidget()
        self.practice_target_list.setMaximumHeight(140)
        self.practice_target_list.itemSelectionChanged.connect(self.on_select_practice_gesture)
        sl.addWidget(self.practice_target_list)
        layout.addWidget(selector_card)

        split = QHBoxLayout()
        split.setSpacing(10)

        # ---- Left panel: Reference Gesture ----
        left_card = QFrame()
        left_card.setObjectName("InfoCard")
        left = QVBoxLayout(left_card)
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(8)
        ref_header = QLabel("Reference Gesture")
        ref_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        ref_header.setStyleSheet("color:#10b981;")
        ref_header.setAlignment(Qt.AlignCenter)
        left.addWidget(ref_header)
        self.practice_ref_label = QLabel("Reference not available")
        self.practice_ref_label.setAlignment(Qt.AlignCenter)
        self.practice_ref_label.setMinimumSize(420, 360)
        self.practice_ref_label.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.practice_ref_status = QLabel("Reference not available")
        self.practice_ref_status.setAlignment(Qt.AlignCenter)
        self.practice_ref_status.setObjectName("SectionMeta")
        left.addWidget(self.practice_ref_label)
        left.addWidget(self.practice_ref_status)

        # ---- Right panel: Live Camera Recognition ----
        right_card = QFrame()
        right_card.setObjectName("InfoCard")
        right = QVBoxLayout(right_card)
        right.setContentsMargins(10, 10, 10, 10)
        right.setSpacing(8)
        cam_header = QLabel("Live Camera Recognition")
        cam_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        cam_header.setStyleSheet("color:#1cb0f6;")
        cam_header.setAlignment(Qt.AlignCenter)
        right.addWidget(cam_header)
        self.camera_view = QLabel("Camera preview")
        self.camera_view.setMinimumSize(420, 360)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:14px;")
        self.practice_target_stat = QLabel("Target: -")
        self.practice_pred_stat = QLabel("Prediction: -")
        self.practice_conf_stat = QLabel("Confidence: 0.00")
        self.practice_model_stat = QLabel("Model: -")
        self.practice_fps_stat = QLabel("FPS: 0.0")
        self.live_prediction = QLabel("Prediction: -")
        self.practice_feedback = QLabel("Select target and press Start Live Practice")
        self.practice_feedback.setObjectName("StatusPill")
        self.btn_start_camera = QPushButton("Start Live Practice")
        self.btn_start_camera.setObjectName("AccentButton")
        self.btn_stop_camera = QPushButton("Stop Camera")
        self.btn_stop_camera.setObjectName("GhostButton")
        self.btn_mark_attempt = QPushButton("Record Current Attempt")
        self.btn_mark_attempt.setObjectName("GhostButton")

        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_mark_attempt.clicked.connect(self.record_current_attempt)

        right.addWidget(self.camera_view)
        right.addWidget(self.practice_target_stat)
        right.addWidget(self.practice_pred_stat)
        right.addWidget(self.practice_conf_stat)
        right.addWidget(self.practice_model_stat)
        right.addWidget(self.practice_fps_stat)
        right.addWidget(self.live_prediction)
        right.addWidget(self.practice_feedback)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_start_camera)
        btn_row.addWidget(self.btn_stop_camera)
        right.addLayout(btn_row)
        right.addWidget(self.btn_mark_attempt)

        split.addWidget(left_card, 1)
        split.addWidget(right_card, 1)
        layout.addLayout(split)
        return page

    def _build_quiz_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(10)

        # ---- Top bar: target + score + controls ----
        top_card = QFrame()
        top_card.setObjectName("InfoCard")
        top_l = QHBoxLayout(top_card)
        top_l.setContentsMargins(14, 10, 14, 10)
        top_l.setSpacing(14)
        self.quiz_target = QLabel("Quiz target: -")
        self.quiz_target.setFont(QFont(self.sys_font, 18, QFont.Bold))
        self.quiz_target.setStyleSheet("color:#f8fafc;")
        self.quiz_state = QLabel("Score: 0/0")
        self.quiz_state.setFont(QFont(self.sys_font, 14, QFont.Bold))
        self.quiz_state.setStyleSheet("color:#facc15;")
        self.quiz_progress = QLabel("Question: 0/10")
        self.quiz_progress.setStyleSheet("color:#94a3b8; font-size:13px;")
        b1 = QPushButton("Start Quiz (10 Questions)")
        b1.setObjectName("AccentButton")
        b2 = QPushButton("Submit Answer")
        b2.setObjectName("GhostButton")
        b1.clicked.connect(self.start_quiz)
        b2.clicked.connect(self.submit_quiz_answer)
        top_l.addWidget(self.quiz_target, 1)
        top_l.addWidget(self.quiz_progress)
        top_l.addWidget(self.quiz_state)
        top_l.addWidget(b1)
        top_l.addWidget(b2)
        outer.addWidget(top_card)

        # ---- Main area: reference (left) + camera (right) ----
        split = QHBoxLayout()
        split.setSpacing(10)

        left_card = QFrame()
        left_card.setObjectName("InfoCard")
        left = QVBoxLayout(left_card)
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(8)
        qref_header = QLabel("Reference Gesture")
        qref_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        qref_header.setStyleSheet("color:#10b981;")
        qref_header.setAlignment(Qt.AlignCenter)
        left.addWidget(qref_header)
        self.quiz_ref_label = QLabel("Start a quiz to see the reference")
        self.quiz_ref_label.setAlignment(Qt.AlignCenter)
        self.quiz_ref_label.setMinimumSize(380, 320)
        self.quiz_ref_label.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.quiz_ref_status = QLabel("Waiting for quiz start")
        self.quiz_ref_status.setAlignment(Qt.AlignCenter)
        self.quiz_ref_status.setObjectName("SectionMeta")
        left.addWidget(self.quiz_ref_label)
        left.addWidget(self.quiz_ref_status)

        right_card = QFrame()
        right_card.setObjectName("InfoCard")
        right = QVBoxLayout(right_card)
        right.setContentsMargins(10, 10, 10, 10)
        right.setSpacing(8)
        qcam_header = QLabel("Your Camera")
        qcam_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        qcam_header.setStyleSheet("color:#1cb0f6;")
        qcam_header.setAlignment(Qt.AlignCenter)
        right.addWidget(qcam_header)
        self.quiz_camera_view = QLabel("Camera will start with the quiz")
        self.quiz_camera_view.setAlignment(Qt.AlignCenter)
        self.quiz_camera_view.setMinimumSize(380, 320)
        self.quiz_camera_view.setStyleSheet("background:#05070a; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.quiz_pred_label = QLabel("Prediction: -")
        self.quiz_pred_label.setAlignment(Qt.AlignCenter)
        self.quiz_pred_label.setStyleSheet("color:#94a3b8; font-size:13px;")
        self.quiz_feedback = QLabel("Press 'Start Quiz' to begin")
        self.quiz_feedback.setObjectName("StatusPill")
        right.addWidget(self.quiz_camera_view)
        right.addWidget(self.quiz_pred_label)
        right.addWidget(self.quiz_feedback)

        split.addWidget(left_card, 1)
        split.addWidget(right_card, 1)
        outer.addLayout(split, 1)
        return page

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
        if not hasattr(self, "right_panel") or not hasattr(self, "status_bar_widget"):
            return
        if idx == self.IDX_LOGIN:
            self.status_bar_widget.setVisible(False)
            self.right_panel.setVisible(False)
            for button in self._nav_index_map.values():
                button.setChecked(False)
            return
        self.status_bar_widget.setVisible(True)
        self.right_panel.setVisible(True)
        self.feedback_mode.setText(f"Mode: {self._mode_name(idx)}")
        self._set_active_nav(idx)
        # Auto-refresh analytics when switching to that page
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
        self.right_panel.setVisible(True)
        self.status_bar_widget.setVisible(True)
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
        self.lesson_summary.setText(f"Lessons ready: 3 | Gestures loaded: {len(gestures)}")
        self.feedback_ref_source.setText("Source: isl_videos (local)")

        self.study_gesture_list.clear()
        self.practice_target_list.clear()
        for g in gestures:
            text = f"{g['display_name']}  [{g['gesture_mode']}]"
            self.study_gesture_list.addItem(text)
            self.practice_target_list.addItem(text)

        self.load_analytics()
        self.load_model_versions()

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
        self.right_panel.setVisible(False)
        self.status_bar_widget.setVisible(False)
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
        if idx < 0 or idx >= len(self.gesture_rows):
            return
        self._flush_study_timer()
        self.selected_gesture = dict(self.gesture_rows[idx])
        g = self.selected_gesture
        ref_info = get_gesture_reference(str(g["display_name"]))
        self.study_name.setText(f"Gesture: {g['display_name']}")
        self.study_type.setText(f"Type: {g['gesture_mode']}")
        diff = ref_info.get("difficulty") or self._difficulty_for(g)
        self.study_diff.setText(f"Difficulty: {diff.capitalize()}")
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
        target_id = self.selected_gesture["gesture_id"]
        for i, row in enumerate(self.gesture_rows):
            if row["gesture_id"] == target_id:
                self.practice_target_list.setCurrentRow(i)
                break
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

    def on_select_practice_gesture(self):
        idx = self.practice_target_list.currentRow()
        if idx < 0 or idx >= len(self.gesture_rows):
            return
        row = dict(self.gesture_rows[idx])
        self.selected_gesture = row
        self._load_gesture_reference(self.practice_ref_label, self.practice_ref_thread, row, "practice")

        self.practice_target_stat.setText(f"Target: {row['display_name']}")
        self.feedback_target.setText(f"Target: {row['display_name']} [{row['gesture_mode']}]")
        self._set_practice_feedback(f"Target selected: {row['display_name']} [{row['gesture_mode']}]", "#22c55e")
        if self.inference_thread:
            self.inference_thread.set_target(str(row["display_name"]))
            self.inference_thread.set_target_mode(str(row["gesture_mode"]))
        else:
            self.start_camera()

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
            self._set_practice_feedback("Camera active. Show your sign!", "#22c55e")
        except Exception as e:
            self._set_feedback_status("Camera error", "#ef4444")
            self._set_practice_feedback(f"Camera failed: {e}", "#ef4444")

    def stop_camera(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None
            self._set_feedback_status("Camera stopped", "#facc15")
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
        # Also update quiz ref label if visible (shared ref thread)
        if hasattr(self, 'quiz_ref_label') and self.quiz_ref_label.isVisible():
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
        self.quiz_target.setText(f"Show the sign for: {target['display_name']}")
        self.quiz_state.setText(f"Score: {self.quiz_score}/{self.quiz_index}")
        self.quiz_progress.setText(f"Question: {self.quiz_index + 1}/{len(self.quiz_queue)}")
        self.quiz_feedback.setText("Make the sign and press Submit")
        self.quiz_feedback.setStyleSheet(
            "background-color:#0b0f18; border:1px solid #334155; border-radius:12px; "
            "padding:6px 10px; color:#94a3b8; font-weight:700;"
        )
        # Show reference for quiz target
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
