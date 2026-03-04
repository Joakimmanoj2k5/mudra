from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
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
from utils.gesture_media_mapper import get_media_path, get_gesture_reference, get_gesture_description, get_reference_image_path
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
                    self.status_signal.emit("Text reference shown (no video)")
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
                self.frame_signal.emit(qimg)
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
        if not self.camera.open():
            self.result_signal.emit({"status": "camera_error", "label": "CAMERA_ERROR", "confidence": 0.0, "latency_ms": 0})
            return

        while self._run_flag:
            packet = self.camera.read()
            if packet is None:
                fail_count += 1
                if fail_count > 20:
                    self.result_signal.emit({"status": "camera_error", "label": "CAMERA_DISCONNECTED", "confidence": 0.0, "latency_ms": 0})
                    break
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
            self.frame_signal.emit(qimg)

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
            QMainWindow { background-color: #0f172a; }
            QLabel { color: #f8fafc; font-size: 13px; }
            QPushButton {
                border-radius: 10px;
                padding: 10px;
                font-weight: 600;
                color: #e2e8f0;
                background-color: #1e293b;
                border: 1px solid #334155;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #334155; }
            QLineEdit {
                border-radius: 8px;
                border: 1px solid #334155;
                padding: 8px;
                background-color: #111827;
                color: #f8fafc;
            }
            QListWidget, QTableWidget {
                background-color: #111827;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 10px;
                font-size: 12px;
            }
        """

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(220)
        sbl = QVBoxLayout(self.sidebar)
        logo = QLabel("MUDRA")
        logo.setFont(QFont(self.sys_font, 28, QFont.Bold))
        logo.setStyleSheet("color:#10b981;")
        sbl.addWidget(logo)

        nav = [
            ("Dashboard", self.IDX_DASH),
            ("Study", self.IDX_STUDY),
            ("Practice", self.IDX_PRACTICE),
            ("Quiz", self.IDX_QUIZ),
            ("Analytics", self.IDX_ANALYTICS),
            ("Admin", self.IDX_ADMIN),
        ]
        self.nav_buttons = {}
        for title, idx in nav:
            b = QPushButton(title)
            b.clicked.connect(lambda _=False, i=idx: self.navigate_to(i))
            sbl.addWidget(b)
            self.nav_buttons[title] = b

        self.btn_logout = QPushButton("Logout")
        self.btn_logout.clicked.connect(self.logout)
        sbl.addStretch()
        sbl.addWidget(self.btn_logout)

        content = QVBoxLayout()
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

        root.addWidget(self.sidebar)
        content_wrap = QWidget()
        content_wrap.setLayout(content)
        root.addWidget(content_wrap)
        self.sidebar.setVisible(False)

    def _build_env_header(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet("background:#111827; border:1px solid #334155; border-radius:10px;")
        l = QHBoxLayout(bar)
        self.env_title = QLabel("Environment Health")
        self.env_mp = QLabel("MediaPipe")
        self.env_torch = QLabel("Torch")
        self.env_cam = QLabel("Camera")
        self.env_static = QLabel("StaticModel")
        self.env_dynamic = QLabel("DynamicModel")
        btn_refresh_env = QPushButton("Refresh")
        btn_refresh_env.setFixedWidth(100)
        btn_refresh_env.clicked.connect(self.refresh_environment_status)

        l.addWidget(self.env_title)
        l.addSpacing(12)
        for x in [self.env_mp, self.env_torch, self.env_cam, self.env_static, self.env_dynamic]:
            l.addWidget(x)
        l.addStretch()
        l.addWidget(btn_refresh_env)
        return bar

    def _build_login_page(self) -> QWidget:
        page = QWidget()
        l = QVBoxLayout(page)
        l.addStretch()
        form = QFrame()
        form.setMaximumWidth(520)
        form.setStyleSheet("background:#111827; border-radius:14px; border:1px solid #334155;")
        fl = QVBoxLayout(form)
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
        self.welcome = QLabel("Welcome")
        self.welcome.setFont(QFont(self.sys_font, 22, QFont.Bold))
        self.lesson_summary = QLabel("Lessons: -")
        self.progress_summary = QLabel("Progress: -")

        grid = QGridLayout()
        cards = [
            ("26 Alphabets", "Finger spelling lessons"),
            ("50-100 Word Signs", "Common ISL words"),
            ("Study + Practice", "Reference first, then live camera"),
            ("Analytics", "Accuracy, confidence, confusion matrix"),
        ]
        for i, (h, s) in enumerate(cards):
            c = QFrame()
            c.setStyleSheet("background:#1e293b; border-radius:14px; border:1px solid #334155;")
            cl = QVBoxLayout(c)
            hl = QLabel(h)
            hl.setFont(QFont(self.sys_font, 14, QFont.Bold))
            cl.addWidget(hl)
            cl.addWidget(QLabel(s))
            grid.addWidget(c, i // 2, i % 2)
        l.addWidget(self.welcome)
        l.addWidget(self.lesson_summary)
        l.addWidget(self.progress_summary)
        l.addLayout(grid)
        l.addStretch()
        return page

    def _build_study_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        top = QHBoxLayout()
        self.study_gesture_list = QListWidget()
        self.study_gesture_list.setMaximumHeight(140)
        self.study_gesture_list.itemSelectionChanged.connect(self.on_select_study_gesture)
        top.addWidget(self.study_gesture_list, 2)

        info = QFrame()
        info.setStyleSheet("background:#111827; border:1px solid #334155; border-radius:12px;")
        il = QVBoxLayout(info)
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
        top.addWidget(info, 3)

        layout.addLayout(top)

        panel = QHBoxLayout()
        self.study_ref_label = QLabel("Select a gesture to see its reference")
        self.study_ref_label.setAlignment(Qt.AlignCenter)
        self.study_ref_label.setMinimumHeight(360)
        self.study_ref_label.setWordWrap(True)
        self.study_ref_label.setStyleSheet("background:#000; border:1px solid #334155; border-radius:14px; padding:18px;")
        self.study_ref_status = QLabel("Reference not available")

        right = QVBoxLayout()
        right.addWidget(self.study_ref_label)
        right.addWidget(self.study_ref_status)
        self.btn_start_practice_from_study = QPushButton("Start Practice")
        self.btn_start_practice_from_study.clicked.connect(self.start_practice_from_study)
        right.addWidget(self.btn_start_practice_from_study)
        panel.addLayout(right)

        layout.addLayout(panel)
        return page

    def _build_practice_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.practice_target_list = QListWidget()
        self.practice_target_list.setMaximumHeight(120)
        self.practice_target_list.itemSelectionChanged.connect(self.on_select_practice_gesture)
        layout.addWidget(self.practice_target_list)

        split = QHBoxLayout()

        # ---- Left panel: Reference Gesture ----
        left = QVBoxLayout()
        ref_header = QLabel("Reference Gesture")
        ref_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        ref_header.setStyleSheet("color:#10b981;")
        ref_header.setAlignment(Qt.AlignCenter)
        left.addWidget(ref_header)
        self.practice_ref_label = QLabel("Reference not available")
        self.practice_ref_label.setAlignment(Qt.AlignCenter)
        self.practice_ref_label.setMinimumSize(420, 320)
        self.practice_ref_label.setStyleSheet("background:#000; border-radius:14px; border:1px solid #334155; padding:12px;")
        self.practice_ref_status = QLabel("Reference not available")
        self.practice_ref_status.setAlignment(Qt.AlignCenter)
        left.addWidget(self.practice_ref_label)
        left.addWidget(self.practice_ref_status)

        # ---- Right panel: Live Camera Recognition ----
        right = QVBoxLayout()
        cam_header = QLabel("Live Camera Recognition")
        cam_header.setFont(QFont(self.sys_font, 14, QFont.Bold))
        cam_header.setStyleSheet("color:#38bdf8;")
        cam_header.setAlignment(Qt.AlignCenter)
        right.addWidget(cam_header)
        self.camera_view = QLabel("Camera preview")
        self.camera_view.setMinimumSize(420, 320)
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet("background:#000; border-radius:14px; border:1px solid #334155; padding:14px;")
        self.live_prediction = QLabel("Prediction: -")
        self.practice_feedback = QLabel("Select target and press Start Live Practice")
        self.btn_start_camera = QPushButton("Start Live Practice")
        self.btn_stop_camera = QPushButton("Stop Camera")
        self.btn_mark_attempt = QPushButton("Record Current Attempt")

        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_mark_attempt.clicked.connect(self.record_current_attempt)

        right.addWidget(self.camera_view)
        right.addWidget(self.live_prediction)
        right.addWidget(self.practice_feedback)
        right.addWidget(self.btn_start_camera)
        right.addWidget(self.btn_stop_camera)
        right.addWidget(self.btn_mark_attempt)

        split.addLayout(left, 1)
        split.addLayout(right, 1)
        layout.addLayout(split)
        return page

    def _build_quiz_page(self) -> QWidget:
        page = QWidget()
        l = QVBoxLayout(page)
        self.quiz_target = QLabel("Quiz target: -")
        self.quiz_target.setFont(QFont(self.sys_font, 20, QFont.Bold))
        self.quiz_state = QLabel("Score: 0/0")
        b1 = QPushButton("Start Quiz (10 Questions)")
        b2 = QPushButton("Submit Current Prediction")
        b1.clicked.connect(self.start_quiz)
        b2.clicked.connect(self.submit_quiz_answer)
        l.addWidget(self.quiz_target)
        l.addWidget(self.quiz_state)
        l.addWidget(b1)
        l.addWidget(b2)
        l.addStretch()
        return page

    def _build_analytics_page(self) -> QWidget:
        page = QWidget()
        l = QVBoxLayout(page)
        self.analytics_summary = QLabel("No attempts yet")
        self.analytics_table = QTableWidget(0, 6)
        self.analytics_table.setHorizontalHeaderLabels(["Time", "Target", "Predicted", "Conf", "Correct", "Mode"])
        self.analytics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        btn_refresh = QPushButton("Refresh Analytics")
        btn_conf = QPushButton("Load Confusion Matrix View")
        btn_refresh.clicked.connect(self.load_analytics)
        btn_conf.clicked.connect(self.load_confusion_matrix_view)

        self.confusion_note = QLabel("Confusion matrix appears after attempts are recorded.")
        self.confusion_table = QTableWidget(0, 0)
        self.confusion_table.setMaximumHeight(360)
        self.confusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.confusion_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        l.addWidget(self.analytics_summary)
        l.addWidget(self.analytics_table)
        l.addWidget(btn_refresh)
        l.addWidget(btn_conf)
        l.addWidget(self.confusion_note)
        l.addWidget(self.confusion_table)
        return page

    def _build_admin_page(self) -> QWidget:
        page = QWidget()
        l = QVBoxLayout(page)
        self.admin_label = QLabel("Admin controls")
        btn_seed = QPushButton("Reseed Core Data")
        btn_models = QPushButton("Refresh Model Registry")
        btn_activate = QPushButton("Activate Selected Model")
        btn_reload = QPushButton("Reload Predictor From Active Models")
        btn_seed.clicked.connect(self.reseed)
        btn_models.clicked.connect(self.load_model_versions)
        btn_activate.clicked.connect(self.activate_selected_model)
        btn_reload.clicked.connect(self.reload_predictor_from_registry)

        reg_box = QFrame()
        reg_box.setStyleSheet("background:#111827; border:1px solid #334155; border-radius:10px;")
        reg_layout = QVBoxLayout(reg_box)
        reg_layout.addWidget(QLabel("Register New Model Version"))

        self.reg_model_name = QComboBox()
        self.reg_model_name.addItems(["static_mlp", "dynamic_bigru"])
        self.reg_framework = QLineEdit("pytorch")
        self.reg_version_tag = QLineEdit("")
        self.reg_artifact_path = QLineEdit()
        self.reg_label_map_path = QLineEdit("models/registry/label_map.json")
        self.reg_norm_stats_path = QLineEdit("models/registry/norm_stats.json")
        self.reg_activate = QCheckBox("Activate immediately")
        self.reg_activate.setChecked(True)
        self.reg_metrics = QTextEdit('{"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0}')
        self.reg_metrics.setFixedHeight(80)

        for widget, label in [
            (self.reg_model_name, "Model Name"),
            (self.reg_framework, "Framework"),
            (self.reg_version_tag, "Version Tag (optional)"),
            (self.reg_artifact_path, "Artifact Path"),
            (self.reg_label_map_path, "Label Map Path"),
            (self.reg_norm_stats_path, "Norm Stats Path"),
        ]:
            reg_layout.addWidget(QLabel(label))
            reg_layout.addWidget(widget)

        b_browse_art = QPushButton("Browse Artifact")
        b_browse_map = QPushButton("Browse Label Map")
        b_browse_norm = QPushButton("Browse Norm Stats")
        b_browse_art.clicked.connect(lambda: self._pick_file_into(self.reg_artifact_path))
        b_browse_map.clicked.connect(lambda: self._pick_file_into(self.reg_label_map_path))
        b_browse_norm.clicked.connect(lambda: self._pick_file_into(self.reg_norm_stats_path))
        reg_layout.addWidget(b_browse_art)
        reg_layout.addWidget(b_browse_map)
        reg_layout.addWidget(b_browse_norm)

        reg_layout.addWidget(QLabel("Metrics JSON"))
        reg_layout.addWidget(self.reg_metrics)
        reg_layout.addWidget(self.reg_activate)
        b_prefill = QPushButton("Prefill From Active Paths")
        b_register = QPushButton("Register Model Version")
        b_rollback = QPushButton("Rollback Active Version (Selected Family)")
        b_prefill.clicked.connect(self.prefill_model_paths)
        b_register.clicked.connect(self.register_model_version_from_ui)
        b_rollback.clicked.connect(self.rollback_model_family_from_ui)
        reg_layout.addWidget(b_prefill)
        reg_layout.addWidget(b_register)
        reg_layout.addWidget(b_rollback)

        self.model_table = QTableWidget(0, 6)
        self.model_table.setHorizontalHeaderLabels(["Model", "Version", "Framework", "Artifact", "Active", "Trained At"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        l.addWidget(self.admin_label)
        l.addWidget(btn_seed)
        l.addWidget(btn_models)
        l.addWidget(btn_activate)
        l.addWidget(btn_reload)
        l.addWidget(reg_box)
        l.addWidget(self.model_table)
        l.addStretch()
        return page

    def _start_reference_threads(self):
        self.study_ref_thread = ReferenceVideoThread()
        self.study_ref_thread.frame_signal.connect(self._update_study_ref_frame)
        self.study_ref_thread.status_signal.connect(self.study_ref_status.setText)
        self.study_ref_thread.start()

        self.practice_ref_thread = ReferenceVideoThread()
        self.practice_ref_thread.frame_signal.connect(self._update_practice_ref_frame)
        self.practice_ref_thread.status_signal.connect(self.practice_ref_status.setText)
        self.practice_ref_thread.start()

    def _on_stack_changed(self, idx: int):
        if idx != self.IDX_STUDY:
            self._flush_study_timer()

    def navigate_to(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def _apply_environment_status(self):
        self._set_indicator(self.env_mp, "MediaPipe", self.env_status.get("mediapipe", False))
        self._set_indicator(self.env_torch, "Torch", self.env_status.get("torch", False))
        self._set_indicator(self.env_cam, "Camera", self.env_status.get("camera", False))
        self._set_indicator(self.env_static, "StaticModel", self.env_status.get("static_model_loaded", False))
        self._set_indicator(self.env_dynamic, "DynamicModel", self.env_status.get("dynamic_model_loaded", False))

        can_practice = self.env_status.get("mediapipe", False) and self.env_status.get("camera", False)
        if hasattr(self, "btn_start_camera"):
            self.btn_start_camera.setEnabled(can_practice)
            if not can_practice:
                self.btn_start_camera.setText("Start Live Practice (Unavailable)")
                self.practice_feedback.setText("Practice disabled: MediaPipe and camera are required.")
            else:
                self.btn_start_camera.setText("Start Live Practice")

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
        self.navigate_to(self.IDX_DASH)
        self.welcome.setText(f"Welcome, {self.session.full_name}")
        self.refresh_after_login()

    def refresh_after_login(self):
        gestures = [dict(g) for g in self.db.get_gestures()]
        self.gesture_rows = gestures
        self.lesson_summary.setText(f"Lessons ready: 3 | Gestures loaded: {len(gestures)}")

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
        self.sidebar.setVisible(False)
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
        """Show a reference image + text description when no video is available."""
        # Try to show a reference image
        img_path = get_reference_image_path(gesture_name)
        if img_path:
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(
                    "background:#111827; border:1px solid #334155; border-radius:14px; padding:8px;"
                )
                return

        # Fallback: styled HTML text
        desc = ref_info.get("description", "")
        tips = ref_info.get("tips", "")
        difficulty = ref_info.get("difficulty", "")
        hands = ref_info.get("hands", ref_info.get("hand", ""))

        parts = [f"<b style='color:#10b981; font-size:18px;'>{gesture_name}</b>"]
        if difficulty:
            parts.append(f"<span style='color:#94a3b8; font-size:12px;'>Difficulty: {difficulty.capitalize()}</span>")
        if hands:
            parts.append(f"<span style='color:#94a3b8; font-size:12px;'>Hand(s): {hands}</span>")
        parts.append("")
        if desc:
            parts.append(f"<p style='color:#e2e8f0; font-size:14px; line-height:1.6;'>{desc}</p>")
        if tips:
            parts.append(f"<p style='color:#fbbf24; font-size:13px;'>💡 {tips}</p>")
        if not desc and not tips:
            parts.append("<p style='color:#94a3b8;'>No reference available for this gesture.</p>")

        label.setText("<br>".join(parts))
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        label.setWordWrap(True)
        label.setStyleSheet(
            "background:#111827; border:1px solid #334155; border-radius:14px; "
            "padding:18px; font-family:sans-serif;"
        )

    def _load_static_media(self, label: QLabel, media_path: str) -> None:
        """Display a static image (png/jpg) on a QLabel."""
        pixmap = QPixmap(media_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "background:#111827; border:1px solid #334155; border-radius:14px; padding:8px;"
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

        media_path = get_media_path(str(g["display_name"]))
        if media_path and media_path.lower().endswith((".mp4", ".gif")):
            # Video / animated media — hand off to the looping reference thread
            self.study_ref_thread.set_media(media_path)
            self.study_ref_status.setText("▶ Loading reference animation…")
        elif media_path and media_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            # Static image — display directly, stop video thread
            self.study_ref_thread.set_media(None)
            self._load_static_media(self.study_ref_label, media_path)
            self.study_ref_status.setText("Showing reference image")
        else:
            # No media at all — show text fallback
            self.study_ref_thread.set_media(None)
            self._show_text_reference(self.study_ref_label, str(g["display_name"]), ref_info)
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

    def on_select_practice_gesture(self):
        idx = self.practice_target_list.currentRow()
        if idx < 0 or idx >= len(self.gesture_rows):
            return
        row = dict(self.gesture_rows[idx])
        self.selected_gesture = row

        media_path = get_media_path(str(row["display_name"]))
        if media_path and media_path.lower().endswith((".mp4", ".gif")):
            self.practice_ref_thread.set_media(media_path)
            self.practice_ref_status.setText("▶ Loading reference animation…")
        elif media_path and media_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            self.practice_ref_thread.set_media(None)
            self._load_static_media(self.practice_ref_label, media_path)
            self.practice_ref_status.setText("Showing reference image")
        else:
            self.practice_ref_thread.set_media(None)
            ref_info = get_gesture_reference(str(row["display_name"]))
            self._show_text_reference(self.practice_ref_label, str(row["display_name"]), ref_info)

        self.practice_feedback.setText(f"Target selected: {row['display_name']} [{row['gesture_mode']}]")
        if self.inference_thread:
            self.inference_thread.set_target(str(row["display_name"]))
            self.inference_thread.set_target_mode(str(row["gesture_mode"]))

    def start_camera(self):
        if not (self.env_status.get("mediapipe") and self.env_status.get("camera")):
            QMessageBox.warning(self, "Unavailable", "Cannot start practice. MediaPipe or camera is unavailable.")
            return
        if self.inference_thread and self.inference_thread.isRunning():
            return
        if not self.selected_gesture:
            QMessageBox.information(self, "Select Target", "Choose a gesture first.")
            return

        self.inference_thread = InferenceThread(self.predictor, self.env_status)
        self.inference_thread.set_target(str(self.selected_gesture["display_name"]))
        self.inference_thread.set_target_mode(str(self.selected_gesture["gesture_mode"]))
        self.inference_thread.frame_signal.connect(self.update_camera_view)
        self.inference_thread.result_signal.connect(self.update_result)
        self.inference_thread.start()

    def stop_camera(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread = None

    def update_camera_view(self, qimg: QImage):
        self._last_qimage = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_view.setPixmap(pix)

    def _update_study_ref_frame(self, qimg: QImage):
        self._last_study_ref = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.study_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.study_ref_label.setPixmap(pix)

    def _update_practice_ref_frame(self, qimg: QImage):
        self._last_practice_ref = qimg
        pix = QPixmap.fromImage(qimg).scaled(self.practice_ref_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.practice_ref_label.setPixmap(pix)

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

        if warn:
            self.practice_feedback.setText(warn)
            self.practice_feedback.setStyleSheet("color:#fbbf24; font-weight:600;")
            return

        if self.selected_gesture and status in {"ok", "uncertain"}:
            target = self.selected_gesture["display_name"]
            if stable and label == target:
                self.practice_feedback.setText("Correct and stable")
                self.practice_feedback.setStyleSheet("color:#34d399; font-weight:700;")
            elif status == "uncertain":
                self.practice_feedback.setText("Hold steady. Confidence not stable yet.")
                self.practice_feedback.setStyleSheet("color:#fbbf24; font-weight:600;")
            else:
                self.practice_feedback.setText(f"Incorrect. Predicted {label}, target {target}")
                self.practice_feedback.setStyleSheet("color:#f87171; font-weight:700;")
        elif status in {"mediapipe_unavailable", "dynamic_model_unavailable"}:
            env_text = (
                f"Environment: MediaPipe: {'❌' if not self.env_status.get('mediapipe') else '✅'} | "
                f"Torch: {'✅' if self.env_status.get('torch') else '❌'} | "
                f"Camera: {'✅' if self.env_status.get('camera') else '❌'}"
            )
            self.practice_feedback.setText(env_text)
            self.practice_feedback.setStyleSheet("color:#f87171; font-weight:700;")
        elif status == "no_hand":
            self.practice_feedback.setText("No hand detected. Place hand in frame.")
            self.practice_feedback.setStyleSheet("color:#fbbf24; font-weight:600;")
        elif status == "camera_error":
            self.practice_feedback.setText("Camera error. Restart camera.")
            self.practice_feedback.setStyleSheet("color:#f87171; font-weight:700;")

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
        self.practice_feedback.setText(f"Attempt recorded | correct={is_correct}")
        self.load_analytics()

    def start_quiz(self):
        self.quiz_queue = [dict(r) for r in self.db.get_random_gestures(limit=10)]
        self.quiz_index = 0
        self.quiz_score = 0
        self._set_quiz_target()

    def _set_quiz_target(self):
        if self.quiz_index >= len(self.quiz_queue):
            self.quiz_target.setText("Quiz complete")
            self.quiz_state.setText(f"Final score: {self.quiz_score}/{len(self.quiz_queue)}")
            return
        target = self.quiz_queue[self.quiz_index]
        self.quiz_target.setText(f"Quiz target: {target['display_name']}")
        self.quiz_state.setText(f"Score: {self.quiz_score}/{self.quiz_index}")

    def submit_quiz_answer(self):
        if self.quiz_index >= len(self.quiz_queue) or not self.current_result:
            return
        target = self.quiz_queue[self.quiz_index]
        pred = str(self.current_result.get("label", "UNKNOWN"))
        conf = float(self.current_result.get("confidence", 0.0))
        stable = bool(self.current_result.get("stable", False))
        is_correct = stable and pred == target["display_name"] and conf >= 0.65
        self.quiz_score += int(is_correct)

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
        self._set_quiz_target()
        self.load_analytics()

    def load_analytics(self):
        if not self.session.is_authenticated():
            return
        summary = self.db.get_analytics_summary(self.session.user_id)
        self.analytics_summary.setText(
            f"Attempts: {int(summary['total_attempts'])} | Accuracy: {summary['accuracy']:.2f} | "
            f"Avg Conf: {summary['avg_confidence']:.2f} | Avg Latency: {summary['avg_latency_ms']:.1f}ms"
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
                self.analytics_table.setItem(i, c, QTableWidgetItem(str(v)))

        progress_rows = self.db.get_user_progress(self.session.user_id)
        if progress_rows:
            txt = " | ".join([f"{r['title']}: {r['accuracy']:.2f} ({r['attempts_count']} attempts)" for r in progress_rows])
            self.progress_summary.setText(f"Progress: {txt}")

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
