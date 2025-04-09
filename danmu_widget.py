import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtCore import QUrl, Qt, pyqtSlot, QTimer, QRect # Import QRect
import random

class DanmuWebPage(QWebEnginePage):
    """自定义 WebPage 以处理 console.log 等"""
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS Console ({sourceID}:{lineNumber}): {message}")

class DanmuWidget(QWidget):
    def __init__(self, initial_rect=None): # 添加 initial_rect 参数
        super().__init__()
        self.web_view = QWebEngineView(self)
        self.web_page = DanmuWebPage(self.web_view)
        self.web_view.setPage(self.web_page)

        # --- Window Setup ---
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |      # 无边框
                          Qt.WindowType.WindowStaysOnTopHint |     # 保持置顶
                          Qt.WindowType.Tool)                      # 不在任务栏显示 (Tool 类型可能自带置顶)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) # 设置背景透明
        self.web_view.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.web_page.setBackgroundColor(Qt.GlobalColor.transparent) # Web 背景透明

        # --- Layout ---
        # QWidget 没有 layout，需要手动设置 web_view 大小
        self.web_view.setGeometry(0, 0, 400, 600) # 初始大小
        self.resize(400, 600)                     # 设置窗口大小

        # 设置初始位置和大小 (如果提供了)
        if initial_rect and isinstance(initial_rect, QRect):
            self.setGeometry(initial_rect)
            self.web_view.setGeometry(0, 0, initial_rect.width(), initial_rect.height())


        # --- Load HTML ---
        html_path = os.path.abspath("danmu.html")
        if not os.path.exists(html_path):
             print(f"错误: 找不到 danmu.html 文件于 {html_path}")
             # 在当前目录下创建一个简单的占位符 HTML
             try:
                 with open("danmu.html", "w", encoding="utf-8") as f:
                     f.write("<!DOCTYPE html><html><head><title>Error</title></head><body style='background-color: rgba(50,0,0,0.8); color: white; padding: 10px;'><h1>Error</h1><p>Could not find danmu.html. Please create it.</p></body></html>")
                 print("已创建占位符 danmu.html")
             except Exception as e:
                 print(f"创建占位符 danmu.html 失败: {e}")
                 return # Exit if we can't even create a placeholder


        print(f"Loading HTML from: {QUrl.fromLocalFile(html_path).toString()}")
        self.web_view.setUrl(QUrl.fromLocalFile(html_path))

        # --- Enable DevTools (optional) ---
        # os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = "9223" # Choose a port
        # settings = self.web_page.settings()
        # settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        # settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        # settings.setAttribute(QWebEngineSettings.WebAttribute.ErrorPageEnabled, True)
        # settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True) # If needed


        # --- Mock Data Timer ---
        self.mock_timer = QTimer(self)
        self.mock_timer.timeout.connect(self.add_mock_danmu)
        # self.mock_timer.start(2000) # 每 2 秒添加一条

    @pyqtSlot(str, str, str, int)
    def add_danmu_message(self, username, message, user_type='normal', guard_level=0):
        """将弹幕消息发送到 QWebEngineView"""
        # 转义特殊字符以安全地插入到 JS 字符串中
        js_username = username.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        js_message = message.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        js_user_type = user_type.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

        js_code = f"addDanmu('{js_username}', '{js_message}', '{js_user_type}', {guard_level});"
        self.web_page.runJavaScript(js_code)

    def add_mock_danmu(self):
        """添加模拟弹幕数据"""
        users = ["用户A", "张三", "Commenter", "路人甲", "测试员"]
        messages = ["哈哈哈", "666", "主播好棒", "这条消息\n有换行", "Test message!"]
        user_types = ['normal', 'twitter', 'weibo', 'streamer']
        guard_levels = [0, 1, 2, 3]

        username = random.choice(users)
        message = random.choice(messages)
        user_type = random.choice(user_types)
        # 只有普通用户才有 guard level
        guard_level = random.choice(guard_levels) if user_type == 'normal' else 0

        self.add_danmu_message(username, message, user_type, guard_level)

    # --- Allow dragging the frameless window ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
         # Check if drag_position exists and the left button is pressed
        if hasattr(self, 'drag_position') and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def resizeEvent(self, event):
        """窗口大小改变时，调整 WebView 大小"""
        super().resizeEvent(event)
        self.web_view.setGeometry(0, 0, self.width(), self.height())


# --- Test Section ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 可以传入一个 QRect 来设置初始位置和大小
    # initial_geometry = QRect(100, 100, 350, 500)
    # widget = DanmuWidget(initial_geometry)
    widget = DanmuWidget() # 使用默认大小和位置
    widget.show()

    # 添加一些初始测试弹幕
    widget.add_danmu_message("System", "弹幕窗口已启动", "system")
    widget.add_danmu_message("我是谁", "第一条消息", "normal", 3)

    # 启动模拟弹幕定时器
    widget.mock_timer.start(1500)


    sys.exit(app.exec())
