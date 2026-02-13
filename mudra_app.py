import sys
import cv2
import json
import base64
import requests
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QStackedWidget, QFrame, QScrollArea, QGridLayout,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPalette

# --- CONFIGURATION ---
API_KEY = "AIzaSyCRMPnJYi14wpFCOFHn1QxmaX5EpcxgrP4"  # Your Gemini API Key
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    raw_frame_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2=0
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.raw_frame_signal.emit(cv_img)
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = qt_img.scaled(1280, 720, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MudraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mudra - ISL Pro Tutor")
        self.resize(1200, 850)
        
        # UI Setup for macOS/Windows
        self.sys_font = ".AppleSystemUIFont" if sys.platform == "darwin" else "Segoe UI"
        self.setStyleSheet(self.get_dark_theme())
        
        self.current_frame = None
        self.is_analyzing = False
        self.target_sign = "" 
        
        # Library of signs (Mock data for descriptions)
        self.sign_guide = {
            "A": "Make a fist with the thumb resting against the side of the index finger.",
            "B": "All fingers straight up and touching, thumb tucked across the palm.",
            "C": "Curve your fingers and thumb to form the letter 'C' shape.",
            "Namaste": "Press your palms together in front of your chest and bow slightly.",
            "Thank You": "Touch your fingertips to your chin, then move your hand forward.",
            "I Love You": "Extend index, pinky, and thumb. Keep middle and ring fingers down.",
            "Sorry": "Make a fist and rub it in a circular motion over your heart."
        }
        
        self.init_ui()

    def get_dark_theme(self):
        """A modern 'Pro' Dark Theme using Slate and Emerald palettes."""
        return f"""
            QMainWindow {{ background-color: #0f172a; }}
            #Sidebar {{ background-color: #1e293b; border-right: 1px solid #334155; }}
            #Logo {{ color: #10b981; margin: 20px; font-weight: bold; font-family: {self.sys_font}; }}
            
            QLabel {{ color: #f8fafc; font-family: {self.sys_font}; }}
            
            QPushButton {{ 
                border-radius: 12px; 
                padding: 12px; 
                font-weight: bold; 
                font-family: {self.sys_font};
                font-size: 14px;
                text-align: left;
                padding-left: 20px;
                color: #94a3b8;
                border: none;
                background-color: transparent;
            }}
            QPushButton:hover {{ background-color: #334155; color: #f8fafc; }}
            
            #ActiveNavBtn {{ 
                background-color: #10b981; 
                color: white; 
                border-radius: 12px; 
            }}
            
            #Card {{ 
                background-color: #1e293b; 
                border: 1px solid #334155; 
                border-radius: 20px;
                color: #f8fafc;
            }}
            
            #AlphabetBtn {{
                background-color: #1e293b;
                border: 2px solid #334155;
                text-align: center;
                padding-left: 0;
                font-size: 18px;
                min-width: 65px;
                min-height: 65px;
                color: #f8fafc;
            }}
            #AlphabetBtn:hover {{ border-color: #10b981; color: #10b981; background-color: #0f172a; }}
            
            #ReferenceBox {{
                background-color: #1e293b;
                border-radius: 20px;
                border: 1px solid #334155;
            }}
            
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{ border: none; background: #0f172a; width: 10px; }}
            QScrollBar::handle:vertical {{ background: #334155; border-radius: 5px; }}
        """

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 30, 15, 30)

        logo = QLabel("🟩 MUDRA")
        logo.setObjectName("Logo")
        logo.setFont(QFont(self.sys_font, 26, QFont.Bold))
        sidebar_layout.addWidget(logo)
        sidebar_layout.addSpacing(20)

        self.nav_buttons = []
        self.btn_learn = self.create_nav_btn("  Learn", 0, active=True)
        self.btn_rankings = self.create_nav_btn("  Rankings", 1)
        self.btn_profile = self.create_nav_btn("  Profile", 2)

        sidebar_layout.addWidget(self.btn_learn)
        sidebar_layout.addWidget(self.btn_rankings)
        sidebar_layout.addWidget(self.btn_profile)
        sidebar_layout.addStretch()

        # Streak indicator
        streak_box = QFrame()
        streak_box.setStyleSheet("background-color: #064e3b; border-radius: 15px; padding: 15px;")
        streak_lyt = QVBoxLayout(streak_box)
        st_title = QLabel("🔥 5 Day Streak!")
        st_title.setStyleSheet("color: #34d399; font-weight: bold;")
        st_sub = QLabel("Mastering ISL daily")
        st_sub.setStyleSheet("color: #10b981; font-size: 11px;")
        streak_lyt.addWidget(st_title)
        streak_lyt.addWidget(st_sub)
        sidebar_layout.addWidget(streak_box)

        # --- MAIN CONTENT ---
        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_curriculum_view())
        self.stack.addWidget(self.create_placeholder("Rankings"))
        self.stack.addWidget(self.create_placeholder("Profile"))

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack)

    def create_nav_btn(self, text, index, active=False):
        btn = QPushButton(text)
        btn.setObjectName("ActiveNavBtn" if active else "NavBtn")
        btn.clicked.connect(lambda: self.switch_tab(index))
        self.nav_buttons.append(btn)
        return btn

    def switch_tab(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setObjectName("ActiveNavBtn" if i == index else "NavBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def create_curriculum_view(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 40)

        title = QLabel("Study Room")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # Alphabet Grid
        alpha_title = QLabel("Letters (Finger Spelling)")
        alpha_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 20px 0; color: #94a3b8;")
        content_layout.addWidget(alpha_title)
        
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            btn = QPushButton(char)
            btn.setObjectName("AlphabetBtn")
            btn.clicked.connect(lambda checked, c=char: self.open_camera_lesson(c))
            grid.addWidget(btn, i // 6, i % 6)
        content_layout.addLayout(grid)

        # Expressions
        words_title = QLabel("Common Phrases")
        words_title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 30px 0; color: #94a3b8;")
        content_layout.addWidget(words_title)
        
        for word in ["Namaste", "Thank You", "I Love You", "Sorry"]:
            w_btn = QPushButton(f"  Practice '{word}'")
            w_btn.setObjectName("Card")
            w_btn.setMinimumHeight(75)
            w_btn.clicked.connect(lambda checked, w=word: self.open_camera_lesson(w))
            content_layout.addWidget(w_btn)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        return page

    def create_placeholder(self, text):
        w = QWidget()
        l = QVBoxLayout(w)
        lbl = QLabel(f"{text} View Loaded")
        lbl.setAlignment(Qt.AlignCenter)
        l.addWidget(lbl)
        return w

    def open_camera_lesson(self, sign_name):
        """Opens a Practice Session with a side-by-side Camera and Reference."""
        self.target_sign = sign_name
        self.lesson_window = QMainWindow(self)
        self.lesson_window.setWindowTitle(f"Practicing: {sign_name}")
        self.lesson_window.resize(1100, 800)
        self.lesson_window.setStyleSheet(self.get_dark_theme())

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QHBoxLayout()
        back = QPushButton("← Back to Study Room")
        back.setFixedWidth(200)
        back.clicked.connect(self.close_lesson)
        header.addWidget(back)
        header.addStretch()
        title_lbl = QLabel(f"SIGN TARGET: {sign_name.upper()}")
        title_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #10b981;")
        header.addWidget(title_lbl)
        header.addStretch()
        layout.addLayout(header)

        # Workspace (Reference + Camera)
        workspace = QHBoxLayout()
        workspace.setSpacing(25)

        # LEFT: Reference Panel (The "Preview")
        ref_panel = QFrame()
        ref_panel.setObjectName("ReferenceBox")
        ref_panel.setFixedWidth(320)
        ref_lyt = QVBoxLayout(ref_panel)
        ref_lyt.setContentsMargins(20, 20, 20, 20)

        ref_header = QLabel("How to do it:")
        ref_header.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 14px;")
        ref_lyt.addWidget(ref_header)

        # Placeholder for Image (Using Emoji/Icon style)
        self.ref_visual = QLabel(self.get_emoji_for_sign(sign_name))
        self.ref_visual.setAlignment(Qt.AlignCenter)
        self.ref_visual.setStyleSheet("font-size: 100px; background-color: #0f172a; border-radius: 15px; margin: 10px 0;")
        self.ref_visual.setMinimumHeight(200)
        ref_lyt.addWidget(self.ref_visual)

        self.ref_text = QLabel(self.sign_guide.get(sign_name, "Keep your hand clear and visible to the camera."))
        self.ref_text.setWordWrap(True)
        self.ref_text.setStyleSheet("color: #cbd5e1; font-size: 15px; line-height: 1.5;")
        ref_lyt.addWidget(self.ref_text)
        ref_lyt.addStretch()
        
        workspace.addWidget(ref_panel)

        # RIGHT: Camera
        cam_container = QVBoxLayout()
        self.camera_label = QLabel("Waiting for Camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000000; border-radius: 20px; border: 2px solid #334155;")
        self.camera_label.setMinimumHeight(450)
        cam_container.addWidget(self.camera_label)

        # Feedback
        self.feedback_label = QLabel("Position your hand in the center of the frame")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("font-size: 18px; color: #94a3b8; margin-top: 15px;")
        cam_container.addWidget(self.feedback_label)
        
        workspace.addLayout(cam_container, 2)
        layout.addLayout(workspace)

        # Bottom Check Button
        self.check_btn = QPushButton("ANALYZE MY SIGN")
        self.check_btn.setFixedSize(300, 65)
        self.check_btn.setStyleSheet("background-color: #10b981; color: white; font-size: 18px; text-align: center; padding: 0; margin-top: 10px;")
        self.check_btn.clicked.connect(self.analyze_current_frame)
        
        btn_center = QHBoxLayout()
        btn_center.addStretch()
        btn_center.addWidget(self.check_btn)
        btn_center.addStretch()
        layout.addLayout(btn_center)

        self.lesson_window.setCentralWidget(container)
        
        # Logic
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.raw_frame_signal.connect(self.store_frame)
        self.thread.start()
        self.lesson_window.show()

    def get_emoji_for_sign(self, name):
        """Returns a placeholder emoji for visual reference."""
        emojis = {"Namaste": "🙏", "I Love You": "🤟", "Thank You": "🤚", "Sorry": "✊", "Water": "💧"}
        return emojis.get(name, "🖐️")

    def close_lesson(self):
        if hasattr(self, 'thread'):
            self.thread.stop()
        self.lesson_window.close()

    def store_frame(self, frame):
        self.current_frame = frame

    def update_image(self, qt_img):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

    def analyze_current_frame(self):
        if self.current_frame is None or self.is_analyzing: return
        if not API_KEY:
            self.feedback_label.setText("⚠️ Recognition Unavailable (Missing API Key)")
            return

        self.is_analyzing = True
        self.feedback_label.setText("Checking against ISL standards...")
        self.check_btn.setEnabled(False)
        threading.Thread(target=self.call_gemini_api).start()

    def call_gemini_api(self):
        try:
            _, buffer = cv2.imencode('.jpg', self.current_frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
            prompt = f"Target ISL Sign: '{self.target_sign}'. Check the user's hand position. Return JSON: {{\"isCorrect\": bool, \"feedback\": \"str\"}}"
            payload = {"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}], "generationConfig": {"responseMimeType": "application/json"}}
            resp = requests.post(url, json=payload, timeout=10)
            result = json.loads(resp.json()['candidates'][0]['content']['parts'][0]['text'])
            QTimer.singleShot(0, lambda: self.show_ai_feedback(result))
        except Exception:
            QTimer.singleShot(0, lambda: self.show_ai_feedback({"isCorrect": False, "feedback": "Connection lag. Try again."}))

    def show_ai_feedback(self, result):
        self.is_analyzing = False
        self.check_btn.setEnabled(True)
        if result['isCorrect']:
            self.feedback_label.setText(f"✅ PERFECT: {result['feedback']}")
            self.feedback_label.setStyleSheet("color: #34d399; font-weight: bold; font-size: 18px;")
        else:
            self.feedback_label.setText(f"❌ TRY AGAIN: {result['feedback']}")
            self.feedback_label.setStyleSheet("color: #f87171; font-weight: bold; font-size: 18px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MudraApp()
    window.show()
    sys.exit(app.exec_())