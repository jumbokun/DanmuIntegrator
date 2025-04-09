import sys
import os

# *** Force software rendering BEFORE importing PyQt ***
# Try ANGLE's D3D11 backend.
# os.environ["QT_OPENGL"] = "software"
# print("尝试设置 QT_OPENGL=software 以使用软件渲染")
# os.environ["QT_ANGLE_PLATFORM"] = "warp" # <--- Comment this out
# print("尝试设置 QT_ANGLE_PLATFORM=warp 以使用 ANGLE WARP 渲染")
# os.environ["QT_ANGLE_PLATFORM"] = "d3d9" # <--- Uncomment and enable this
print("尝试设置 QT_ANGLE_PLATFORM=d3d11 以使用 ANGLE D3D11 渲染")
# os.environ["QT_ANGLE_PLATFORM"] = "d3d9"

import time
import threading
import mss
import pytesseract
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QRect, pyqtSlot # Import QRect and pyqtSlot
from danmu_widget import DanmuWidget
from region_selector import RegionSelector
import random # For mock data
import json

# --- Configuration ---
CONFIG_FILE = "config.json"
TESSERACT_CMD = r'D:\OCR\tesseract.exe' # <--- 修改这里
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    print(f"Tesseract 命令路径已设置为: {TESSERACT_CMD}") # 添加打印确认
except Exception as e:
    print(f"无法设置 Tesseract 路径: {e}. 请确保 Tesseract 已安装并在系统 PATH 中，或在代码中指定 TESSERACT_CMD。")


class OcrWorker(QThread):
    """在单独的线程中执行截图和 OCR"""
    new_comment_signal = pyqtSignal(str, str, str, int) # username, message, source ('weibo'/'twitter'), guard_level (0 for now)
    error_signal = pyqtSignal(str)

    def __init__(self, weibo_rect=None, twitter_rect=None, interval=2.0):
        super().__init__()
        self.weibo_rect = self._qrect_to_dict(weibo_rect)
        self.twitter_rect = self._qrect_to_dict(twitter_rect)
        self.interval = interval
        self._running = False
        self.sct = None
        self.last_texts = {"weibo": "", "twitter": ""} # 存储上次识别的完整文本，用于简单差分

    def _qrect_to_dict(self, qrect):
        """将 QRect 转换为 mss 兼容的字典"""
        if qrect and isinstance(qrect, QRect):
            return {"top": qrect.y(), "left": qrect.x(), "width": qrect.width(), "height": qrect.height()}
        return None

    def set_regions(self, weibo_rect=None, twitter_rect=None):
         self.weibo_rect = self._qrect_to_dict(weibo_rect)
         self.twitter_rect = self._qrect_to_dict(twitter_rect)
         print(f"OCR Worker regions updated: Weibo={self.weibo_rect}, Twitter={self.twitter_rect}")

    def run(self):
        self._running = True
        self.sct = mss.mss()
        print("OCR Worker started.")

        while self._running:
            start_time = time.time()

            if self.weibo_rect:
                self.process_region("weibo", self.weibo_rect)

            if self.twitter_rect:
                self.process_region("twitter", self.twitter_rect)

            # --- MOCK DATA Section (REMOVE or COMMENT OUT for real OCR) ---
            # self.generate_mock_comment()
            # --- End of MOCK DATA Section ---


            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time) # Use time.sleep for thread blocking

        self.sct = None # Release mss context
        print("OCR Worker stopped.")


    def process_region(self, source, region_dict):
         """处理单个区域的截图和 OCR"""
         if not region_dict:
             return
         try:
            # 截图
             sct_img = self.sct.grab(region_dict)
             img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX") # Convert to PIL Image

            # OCR
             # 使用中文简体 + 英文语言包
             # '--psm 6' 假设是一个统一的文本块
             # 配置可以根据实际情况调整
             text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6').strip()
             # print(f"--- OCR Result ({source}) --- \n{text}\n------------------------")


             if text and text != self.last_texts[source]:
                # 非常基础的差分：只发送整个识别到的新文本块
                # TODO: 实现更智能的解析和差分逻辑来提取单条新评论
                diff_text = self.find_diff(self.last_texts[source], text)

                if diff_text:
                    print(f"New text detected ({source}):\n{diff_text}")
                    # 目前将整个差异块作为一条消息发送
                    # 需要改进解析逻辑以分离用户名和消息
                    username = f"{source.capitalize()} Update" # Placeholder username
                    message = diff_text # Send the entire diff
                    self.new_comment_signal.emit(username, message, source, 0)

                self.last_texts[source] = text # 更新最后识别的文本


         except ImportError:
             self.error_signal.emit("错误：找不到 pytesseract 库。请确保已安装。")
             self.stop()
         except pytesseract.TesseractNotFoundError:
             self.error_signal.emit("错误：找不到 Tesseract OCR 引擎。\n请确保已安装 Tesseract 并将其添加到系统 PATH，\n或者在 danmu_integrator.py 中设置 TESSERACT_CMD 变量。")
             self.stop()
         except mss.ScreenShotError as e:
             print(f"截图错误 ({source}): {e}. 区域可能无效: {region_dict}")
             # 可以选择停止或忽略此错误继续尝试
             # self.stop()
         except Exception as e:
             print(f"处理区域时发生未知错误 ({source}): {e}")
             import traceback
             traceback.print_exc()
             # self.stop() # Consider stopping on unknown errors

    def find_diff(self, old_text, new_text):
        """简单的查找新文本的方法 (从末尾比较)"""
        if not old_text:
            return new_text
        if new_text.startswith(old_text):
             # 如果新文本以旧文本开头，返回新增部分
             return new_text[len(old_text):].strip()
        # 更复杂的场景：文本可能滚动了，或者中间有变化
        # 这是一个非常基础的实现，仅当新评论完美附加在末尾时有效
        # 更好的方法可能涉及比较行列表或使用 difflib
        # 暂时返回完整的新文本，如果它不是以旧文本开头
        return new_text # Fallback: return the whole new text if not simply appended

    def generate_mock_comment(self):
         """生成模拟评论信号 (用于测试，替代真实 OCR)"""
         time.sleep(random.uniform(0.5, 3.0)) # 模拟处理时间
         sources = ['weibo', 'twitter']
         users = ["微博用户", "TwitterUser", "测试评论员"]
         messages = ["这是来自模拟器的消息", "Mock comment!", "另一条测试"]
         source = random.choice(sources)
         username = random.choice(users)
         message = random.choice(messages)
         print(f"Mock emitting: {username}, {message}, {source}")
         self.new_comment_signal.emit(username, message, source, 0)


    def stop(self):
        self._running = False
        print("Stopping OCR Worker...")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("弹幕集成器控制台")
        self.setGeometry(100, 100, 500, 400) # 主控制窗口大小

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Danmu Widget ---
        self.danmu_widget = None # Initially None
        self.danmu_window_rect = None # Store position/size

        # --- Region Selection ---
        self.region_selector = None
        self.weibo_rect = None
        self.twitter_rect = None
        self.load_config() # Load saved regions

        # --- Controls ---
        self.controls_layout = QHBoxLayout()
        self.select_region_button = QPushButton("选择/重选区域")
        self.start_button = QPushButton("启动 OCR")
        self.stop_button = QPushButton("停止 OCR")
        self.show_danmu_button = QPushButton("显示/隐藏弹幕窗口")
        self.controls_layout.addWidget(self.select_region_button)
        self.controls_layout.addWidget(self.start_button)
        self.controls_layout.addWidget(self.stop_button)
        self.controls_layout.addWidget(self.show_danmu_button)
        self.stop_button.setEnabled(False)

        # --- Status/Log ---
        self.status_label = QLabel("状态: 未启动")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # --- Layout Setup ---
        self.layout.addLayout(self.controls_layout)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(QLabel("日志:"))
        self.layout.addWidget(self.log_output)

        # --- OCR Worker ---
        self.ocr_worker = OcrWorker()
        self.ocr_worker.new_comment_signal.connect(self.add_comment_to_widget)
        self.ocr_worker.error_signal.connect(self.log_message) # Connect error signal

        # --- Connect Signals ---
        self.select_region_button.clicked.connect(self.open_region_selector)
        self.start_button.clicked.connect(self.start_ocr)
        self.stop_button.clicked.connect(self.stop_ocr)
        self.show_danmu_button.clicked.connect(self.toggle_danmu_widget)


        # --- Initial Danmu Widget State ---
        # Create but don't show initially unless requested or loaded from config?
        self.ensure_danmu_widget_exists()
        # self.danmu_widget.show() # Optional: show on startup

    def ensure_danmu_widget_exists(self):
        """创建弹幕窗口实例 (如果不存在)"""
        if self.danmu_widget is None:
            self.log_message("创建弹幕窗口...")
             # Pass saved geometry if available
            self.danmu_widget = DanmuWidget(self.danmu_window_rect)
             # Connect signal to save position when moved/resized (optional but good)
            self.danmu_widget.moveEvent = lambda event: self.save_danmu_window_rect()
            self.danmu_widget.resizeEvent = lambda event: self.save_danmu_window_rect_and_resize(event)

    def save_danmu_window_rect_and_resize(self, event):
         """处理调整大小事件并保存窗口几何信息"""
         # 调用原始的 resizeEvent 处理
         # QWidget.resizeEvent(self.danmu_widget, event) # This might cause recursion if not careful
         self.danmu_widget.web_view.setGeometry(0, 0, event.size().width(), event.size().height())
         self.save_danmu_window_rect()

    def save_danmu_window_rect(self):
         """保存弹幕窗口的当前位置和大小"""
         if self.danmu_widget and self.danmu_widget.isVisible():
             self.danmu_window_rect = self.danmu_widget.geometry()
             self.save_config() # Save position along with regions


    def toggle_danmu_widget(self):
        """切换弹幕窗口的可见性"""
        self.ensure_danmu_widget_exists()
        if self.danmu_widget.isVisible():
            self.save_danmu_window_rect() # Save position before hiding
            self.danmu_widget.hide()
            self.log_message("弹幕窗口已隐藏")
        else:
            # Restore position if available
            if self.danmu_window_rect:
                 self.danmu_widget.setGeometry(self.danmu_window_rect)
            self.danmu_widget.show()
            self.log_message("弹幕窗口已显示")


    def open_region_selector(self):
        """打开区域选择器"""
        if self.region_selector is None or not self.region_selector.isVisible():
            self.log_message("打开区域选择器...")
            self.region_selector = RegionSelector()
            self.region_selector.weibo_region.connect(self.set_weibo_rect)
            self.region_selector.twitter_region.connect(self.set_twitter_rect)
            self.region_selector.selection_done.connect(self.regions_selected)
            self.region_selector.show()
        else:
            self.region_selector.activateWindow()

    def set_weibo_rect(self, rect):
        self.weibo_rect = rect
        self.log_message(f"微博区域更新: {rect.x()},{rect.y()},{rect.width()},{rect.height()}")

    def set_twitter_rect(self, rect):
        self.twitter_rect = rect
        self.log_message(f"推特区域更新: {rect.x()},{rect.y()},{rect.width()},{rect.height()}")

    def regions_selected(self):
        """区域选择完成后的回调"""
        self.log_message("区域选择完成.")
        self.save_config() # 保存选择的区域
        # 更新 OCR worker 的区域 (如果 worker 正在运行，可能需要先停止再更新)
        was_running = self.ocr_worker.isRunning()
        if was_running:
            self.stop_ocr()

        self.ocr_worker.set_regions(self.weibo_rect, self.twitter_rect)

        if was_running:
             # Optionally restart if it was running before
             # self.start_ocr()
             self.log_message("OCR 线程区域已更新。请手动重启 OCR。")
        else:
            self.status_label.setText("状态: 区域已设置，可以启动 OCR")


    def start_ocr(self):
        """启动 OCR 线程"""
        if not self.weibo_rect and not self.twitter_rect:
            self.log_message("错误: 请先选择至少一个监控区域!")
            return

        if not self.ocr_worker.isRunning():
            self.log_message("启动 OCR 线程...")
             # Ensure regions are set before starting
            self.ocr_worker.set_regions(self.weibo_rect, self.twitter_rect)
            self.ocr_worker.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("状态: OCR 运行中...")
        else:
            self.log_message("OCR 线程已在运行中。")

    def stop_ocr(self):
        """停止 OCR 线程"""
        if self.ocr_worker.isRunning():
            self.log_message("正在停止 OCR 线程...")
            self.ocr_worker.stop()
            self.ocr_worker.wait() # 等待线程完全结束
            self.log_message("OCR 线程已停止。")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("状态: OCR 已停止")
        else:
            self.log_message("OCR 线程未运行。")

    @pyqtSlot(str, str, str, int)
    def add_comment_to_widget(self, username, message, source, guard_level):
        """将评论添加到弹幕窗口"""
        if self.danmu_widget and self.danmu_widget.isVisible():
            self.danmu_widget.add_danmu_message(username, message, source, guard_level)
        # 同时也记录到日志
        self.log_message(f"[{source.upper()}] {username}: {message[:50]}{'...' if len(message)>50 else ''}")


    @pyqtSlot(str)
    def log_message(self, message):
        """在日志窗口显示消息"""
        print(message) # Also print to console
        self.log_output.append(message)

    def load_config(self):
        """从 JSON 文件加载区域和窗口位置"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    wb_coords = config.get('weibo_rect')
                    tw_coords = config.get('twitter_rect')
                    win_coords = config.get('danmu_window_rect')

                    if wb_coords:
                        self.weibo_rect = QRect(wb_coords['x'], wb_coords['y'], wb_coords['width'], wb_coords['height'])
                        self.log_message(f"已加载微博区域: {self.weibo_rect}")
                    if tw_coords:
                        self.twitter_rect = QRect(tw_coords['x'], tw_coords['y'], tw_coords['width'], tw_coords['height'])
                        self.log_message(f"已加载推特区域: {self.twitter_rect}")
                    if win_coords:
                         self.danmu_window_rect = QRect(win_coords['x'], win_coords['y'], win_coords['width'], win_coords['height'])
                         self.log_message(f"已加载弹幕窗口位置: {self.danmu_window_rect}")

        except Exception as e:
            self.log_message(f"加载配置文件 {CONFIG_FILE} 失败: {e}")

    def save_config(self):
        """将当前区域和窗口位置保存到 JSON 文件"""
        config = {}
        if self.weibo_rect:
            config['weibo_rect'] = {'x': self.weibo_rect.x(), 'y': self.weibo_rect.y(), 'width': self.weibo_rect.width(), 'height': self.weibo_rect.height()}
        if self.twitter_rect:
            config['twitter_rect'] = {'x': self.twitter_rect.x(), 'y': self.twitter_rect.y(), 'width': self.twitter_rect.width(), 'height': self.twitter_rect.height()}
        if self.danmu_window_rect:
             config['danmu_window_rect'] = {'x': self.danmu_window_rect.x(), 'y': self.danmu_window_rect.y(), 'width': self.danmu_window_rect.width(), 'height': self.danmu_window_rect.height()}

        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.log_message(f"配置已保存到 {CONFIG_FILE}")
        except Exception as e:
            self.log_message(f"保存配置文件 {CONFIG_FILE} 失败: {e}")

    def closeEvent(self, event):
        """关闭应用程序前停止 OCR 线程并保存配置"""
        self.log_message("正在关闭应用程序...")
        self.stop_ocr()
        if self.danmu_widget:
            self.save_danmu_window_rect() #确保最后的位置被保存
            self.danmu_widget.close()
        self.save_config()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
