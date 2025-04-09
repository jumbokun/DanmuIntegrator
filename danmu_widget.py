import sys
import os

# *** Remove pure QSS imports, revert to WebEngine imports ***
# from PyQt6.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea, QSizePolicy,
#     QFrame
# )
# from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QRect, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QPoint
# from PyQt6.QtGui import QPalette, QColor, QFont, QFontDatabase

# Re-add WebEngine imports
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtCore import QUrl, Qt, pyqtSlot, QTimer, QRect
import random

# --- Remove DanmuMessageWidget class --- 
# class DanmuMessageWidget(QWidget): ...

# --- Revert DanmuWidget to use QWebEngineView --- 

class DanmuWebPage(QWebEnginePage):
    """自定义 WebPage 以处理 console.log 等"""
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        # Filter out the specific GPU/raster errors if they appear here
        ignore_errors = [
            "shared_image_factory.cc",
            "native_skia_output_device.cpp",
            "raster_decoder.cc"
        ]
        if not any(err in message for err in ignore_errors):
            print(f"JS Console ({sourceID}:{lineNumber}): {message}")

class DanmuWidget(QWidget):
    def __init__(self, initial_rect=None):
        super().__init__()

        self.web_view = QWebEngineView(self)
        self.web_page = DanmuWebPage(self.web_view)
        self.web_view.setPage(self.web_page)

        # --- Window Setup (Keep transparency) ---
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                          Qt.WindowType.WindowStaysOnTopHint |
                          Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.web_view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.web_page.setBackgroundColor(Qt.GlobalColor.transparent)

        # --- Layout (Manual positioning for WebEngineView) ---
        # No QVBoxLayout needed for this version
        # Set initial geometry for the WebView manually
        initial_width = 400
        initial_height = 600
        if initial_rect and isinstance(initial_rect, QRect):
             self.setGeometry(initial_rect)
             initial_width = initial_rect.width()
             initial_height = initial_rect.height()
        else:
            self.resize(initial_width, initial_height)
        self.web_view.setGeometry(0, 0, initial_width, initial_height)


        # --- Load HTML --- 
        # Decide which HTML to load (original or alternating)
        html_file = "danmu_alternating.html" # Or "danmu.html"
        html_path = os.path.abspath(html_file)
        if not os.path.exists(html_path):
             print(f"错误: 找不到 {html_file} 文件于 {html_path}")
             # Create placeholder
             try:
                 with open(html_file, "w", encoding="utf-8") as f:
                     f.write(f"<!DOCTYPE html><html><head><title>Error</title></head><body style='background-color: rgba(50,0,0,0.8); color: white; padding: 10px;'><h1>Error</h1><p>Could not find {html_file}. Please create it.</p></body></html>")
                 print(f"已创建占位符 {html_file}")
             except Exception as e:
                 print(f"创建占位符 {html_file} 失败: {e}")
                 return

        print(f"Loading HTML from: {QUrl.fromLocalFile(html_path).toString()}")
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))

        # --- Mock Data Timer ---
        self.mock_timer = QTimer(self)
        self.mock_timer.timeout.connect(self.add_mock_danmu)
        # self.mock_timer.start(1500)

    # Keep apply_styles (does nothing in this version, but harmless)
    def apply_styles(self):
        pass

    @pyqtSlot(str, str, str, int)
    def add_danmu_message(self, username, message, user_type='normal', guard_level=0):
        """将弹幕消息发送到 QWebEngineView"""
        # Escape special characters for JS string insertion
        js_username = username.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        js_message = message.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        js_user_type = user_type.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

        js_code = f"addDanmu('{js_username}', '{js_message}', '{js_user_type}', {guard_level});"
        # Use page() to run JavaScript
        if self.web_page:
            self.web_page.runJavaScript(js_code)
        else:
            print("Error: web_page not initialized.")

    def add_mock_danmu(self):
        """添加模拟弹幕数据"""
        users = ["用户A", "张三", "Commenter", "路人甲", "测试员"]
        messages = ["哈哈哈", "666", "主播好棒", "这条消息\n有换行", "Test message!"]
        user_types = ['normal', 'twitter', 'weibo', 'streamer', 'system']
        guard_levels = [0, 1, 2, 3]

        username = random.choice(users)
        message = random.choice(messages)
        user_type = random.choice(user_types)
        guard_level = random.choice(guard_levels) if user_type == 'normal' else 0

        self.add_danmu_message(username, message, user_type, guard_level)

    # --- Allow dragging the frameless window ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'drag_position') and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    # --- Handle Resize --- 
    def resizeEvent(self, event):
        """Resize the web view when the main widget is resized."""
        super().resizeEvent(event)
        # Resize the web_view to fill the widget
        self.web_view.setGeometry(0, 0, self.width(), self.height())


# --- Test Section --- 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = DanmuWidget()
    widget.show()
    widget.add_danmu_message("System", "弹幕窗口已启动 (WebEngine)", "system")
    widget.mock_timer.start(1500)
    sys.exit(app.exec())
