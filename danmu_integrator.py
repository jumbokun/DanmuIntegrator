import sys
import os
import time
import threading
import mss
from PIL import Image
import numpy as np  # 添加 numpy 导入
import google.generativeai as genai
# Keep necessary PyQt imports for the GUI control panel
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel, QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QSpinBox, QMessageBox, QLineEdit
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QRect, pyqtSlot, QObject # Keep QObject for signals
# Remove DanmuWidget import
# from danmu_widget import DanmuWidget
from region_selector import RegionSelector
import random
import json
# Added imports for web server and websockets
import asyncio
import websockets
import http.server
import socketserver
import json # Ensure json is imported
import io
import jieba

# --- Configuration ---
CONFIG_FILE = "config.json"
WEBSOCKET_PORT = 8765
HTTP_PORT = 8080 # Port for serving the HTML file
HTML_FILE = "danmu_alternating_transparent.html" # 改为透明版本的 HTML 文件
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 从环境变量获取 API 密钥

# 可用的模型列表
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-lite",
    "gemini-pro"  # 保留一个基础模型作为备选
]

# --- WebSocket Server Logic ---
connected_clients = set()

async def register(websocket):
    """Registers a new client connection."""
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}. Total clients: {len(connected_clients)}")
    try:
        await websocket.wait_closed() # Keep connection open until client disconnects
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected: {websocket.remote_address}. Total clients: {len(connected_clients)}")

async def broadcast_message(message_json):
    """Broadcasts a JSON message to all connected clients."""
    if connected_clients:
        # Use asyncio.gather for concurrent sending
        await asyncio.gather(*[client.send(message_json) for client in connected_clients])

# Global variable to hold the running websocket server instance
websocket_server_instance = None
# Global asyncio event loop reference
asyncio_loop = None

# --- Signal Bridge (to communicate from Qt thread to asyncio thread) ---
class SignalBridge(QObject):
    # Define a signal that carries the dict payload
    broadcast_signal = pyqtSignal(dict)

# Global instance of the bridge
signal_bridge = SignalBridge()

# --- Sensitive Filter ---
class SensitiveFilter:
    def __init__(self):
        # 初始化 DFA 过滤器
        self.dfa_filter = DFAFilter()
        
        # 加载敏感词文件夹中的所有敏感词
        self.load_sensitive_words_from_directory('sw')
        
        # 将所有词添加到结巴分词词典
        for word in self.get_all_words():
            jieba.add_word(word)

    def load_sensitive_words_from_directory(self, directory):
        """从指定目录加载所有敏感词文件"""
        if not os.path.exists(directory):
            print(f"警告: 敏感词目录 {directory} 不存在")
            return

        print(f"\n=== 开始加载敏感词文件 ===")
        loaded_count = 0
        
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                try:
                    # 尝试以不同编码读取文件
                    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
                    content = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        print(f"无法读取文件 {filename}，跳过")
                        continue
                    
                    # 按行分割并去除空白字符
                    words = [line.strip() for line in content.splitlines() if line.strip()]
                    
                    # 添加到过滤器
                    for word in words:
                        self.dfa_filter.add_word(word)
                        loaded_count += 1
                    
                    print(f"已加载 {filename}: {len(words)} 个词")
                    
                except Exception as e:
                    print(f"加载文件 {filename} 时出错: {e}")
                    continue

        print(f"=== 敏感词加载完成，共加载 {loaded_count} 个词 ===\n")

    def get_all_words(self):
        """获取所有已加载的敏感词"""
        return self.dfa_filter.get_all_words()

    def filter_text(self, text):
        """过滤文本中的敏感词"""
        if not text:
            return text
            
        # 使用 DFA 过滤器过滤文本
        filtered_text = self.dfa_filter.filter(text, "*")
        
        # 如果文本被修改，打印日志
        if filtered_text != text:
            print(f"原文: {text}")
            print(f"过滤后: {filtered_text}")
            # 找出所有被过滤的词
            original_words = jieba.lcut(text)
            filtered_words = jieba.lcut(filtered_text)
            sensitive_words = set()
            
            for word in original_words:
                if '*' * len(word) in filtered_words:
                    sensitive_words.add(word)
            
            if sensitive_words:
                print(f"发现敏感词: {list(sensitive_words)}")
            
        return filtered_text

class DFAFilter:
    def __init__(self):
        self.keyword_chains = {}  # 关键词链
        self.delimit = '\x00'     # 结束符
        self._keywords = set()    # 保存所有敏感词

    def add_word(self, word):
        """添加敏感词"""
        if not word:
            return
        
        # 将词语添加到集合
        self._keywords.add(word)
        
        # 构建 DFA 树
        level = self.keyword_chains
        for char in word:
            if char not in level:
                level[char] = {}
            level = level[char]
        level[self.delimit] = 0

    def filter(self, message, repl="*"):
        """过滤文本，使用最长匹配和完整遍历"""
        if not message:
            return message
            
        result = list(message)
        i = 0
        while i < len(message):
            level = self.keyword_chains
            temp_i = i
            longest_match = -1  # 记录最长匹配的位置
            
            # 尝试从当前位置开始匹配最长的敏感词
            while temp_i < len(message) and message[temp_i] in level:
                level = level[message[temp_i]]
                if self.delimit in level:  # 找到一个匹配
                    longest_match = temp_i
                temp_i += 1
            
            # 如果找到匹配，替换最长的匹配结果
            if longest_match != -1:
                for j in range(i, longest_match + 1):
                    result[j] = repl
                i = longest_match + 1
            else:
                i += 1
        
        return ''.join(result)

    def get_all_words(self):
        """返回所有已添加的敏感词"""
        return list(self._keywords)

# --- OcrWorker remains mostly the same, but uses Gemini ---
class OcrWorker(QThread):
    # Modify signal to emit a dict
    new_comment_signal = pyqtSignal(dict) # Changed from (str, str, str, int)
    error_signal = pyqtSignal(str)

    def __init__(self, weibo_interval=15.0, twitter_interval=30.0):
        super().__init__()
        self.running = True
        self.weibo_interval = weibo_interval
        self.twitter_interval = twitter_interval
        self.last_weibo_time = 0
        self.last_twitter_time = 0
        self.weibo_region = None
        self.twitter_region = None
        self.model = None
        self.selected_model = AVAILABLE_MODELS[0]
        self.last_messages = {"weibo": [], "twitter": []}  # 初始化last_messages字典
        
        # 初始化敏感词过滤器
        self.sensitive_filter = SensitiveFilter()
        
        # 初始化Gemini模型
        self.initialize_model()

    def _qrect_to_dict(self, qrect):
        """将QRect转换为截图所需的字典格式"""
        if not qrect:
            return None
            
        # 获取所有显示器信息
        with mss.mss() as sct:
            monitors = sct.monitors[1:]  # 跳过第一个monitors[0]，因为它是全部显示器的组合
            print(f"\n显示器信息:")
            for i, m in enumerate(monitors, 1):
                print(f"显示器 {i}: {m}")

        # 将QRect坐标转换为截图坐标
        x, y = qrect.x(), qrect.y()
        
        # 调试信息
        print(f"\n区域信息:")
        print(f"原始坐标: x={x}, y={y}, width={qrect.width()}, height={qrect.height()}")
        
        # 创建截图区域字典
        region = {
            "top": y,
            "left": x,
            "width": qrect.width(),
            "height": qrect.height()
        }
        
        print(f"转换后的区域: {region}")
        return region

    def initialize_model(self):
        """初始化 Gemini 模型"""
        try:
            if not GEMINI_API_KEY:
                raise ValueError("未设置 GEMINI_API_KEY 环境变量")
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.selected_model)
            print("Gemini 模型初始化成功")
            return True
        except Exception as e:
            print(f"Gemini 模型初始化失败: {e}")
            self.error_signal.emit(f"错误：Gemini 模型初始化失败: {e}")
            return False

    def set_regions(self, weibo_rect=None, twitter_rect=None):
        """设置要监控的区域"""
        if weibo_rect:
            self.weibo_region = self._qrect_to_dict(weibo_rect)
        if twitter_rect:
            self.twitter_region = self._qrect_to_dict(twitter_rect)
        print(f"OCR Worker regions updated: Weibo={self.weibo_region}, Twitter={self.twitter_region}")

    def normalize_json_response(self, text):
        """统一处理不同格式的 JSON 响应"""
        try:
            # 清理 Gemini 响应中的 markdown 格式
            if isinstance(text, str):
                # 移除 ```json 和 ``` 标记
                text = text.replace("```json", "").replace("```", "").strip()
            
            # 解析 JSON
            if isinstance(text, str):
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}")
                    print(f"尝试解析的文本: {text}")
                    return []
            else:
                data = text

            normalized_messages = []
            
            # 如果数据是列表
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # 处理格式 1: {"1": "comment"} 或 {"username": "comment"}
                        if len(item) == 1:
                            username = list(item.keys())[0]
                            comment = item[username]
                            # 跳过数字用户名
                            if not username.isdigit():
                                normalized_messages.append({
                                    "username": username,
                                    "comment": comment
                                })
                        # 处理格式 2: {"username": "user", "comment": "text"}
                        elif "username" in item and "comment" in item:
                            normalized_messages.append(item)
            # 如果数据是字典
            elif isinstance(data, dict):
                for username, comment in data.items():
                    # 跳过数字用户名
                    if not username.isdigit():
                        normalized_messages.append({
                            "username": username,
                            "comment": comment
                        })

            return normalized_messages
        except Exception as e:
            print(f"JSON 格式化错误: {e}")
            print(f"错误的输入文本: {text}")
            return []

    def find_new_messages(self, current_messages, last_messages):
        """找出新消息，从上一次的最后一条开始"""
        if not last_messages:
            return current_messages

        # 将消息转换为元组以便比较
        current_tuples = [(msg["username"], msg["comment"]) for msg in current_messages]
        last_tuples = [(msg["username"], msg["comment"]) for msg in last_messages]

        # 如果找到最后一条消息的位置
        if last_tuples and last_tuples[-1] in current_tuples:
            last_index = current_tuples.index(last_tuples[-1])
            # 返回最后一条之后的所有新消息
            return current_messages[last_index + 1:]
        
        # 如果没找到最后一条消息，检查是否有任何重叠
        for last_msg_tuple in reversed(last_tuples):
            if last_msg_tuple in current_tuples:
                last_index = current_tuples.index(last_msg_tuple)
                return current_messages[last_index + 1:]

        # 如果没有找到任何重叠，认为全部都是新消息
        return current_messages

    def filter_sensitive_content(self, text):
        """使用敏感词过滤器过滤内容"""
        return self.sensitive_filter.filter_text(text)

    def run(self):
        while self.running:
            current_time = time.time()
            
            # 处理微博区域
            if self.weibo_region:
                if current_time - self.last_weibo_time >= self.weibo_interval:
                    self.process_region(self.weibo_region, "weibo")
                    self.last_weibo_time = current_time
            
            # 处理推特区域
            if self.twitter_region:
                if current_time - self.last_twitter_time >= self.twitter_interval:
                    self.process_region(self.twitter_region, "twitter")
                    self.last_twitter_time = current_time
            
            # 短暂休眠以减少CPU使用
            time.sleep(0.1)

    def process_region(self, region_dict, source):
         if not region_dict or not self.model:  # 确保模型已初始化
             return
         try:
             # 每次截图时创建新的 mss 实例
             with mss.mss() as sct:
                 # 截图前打印区域信息
                 print(f"\n=== 开始处理区域 ({source}) ===")
                 print(f"截图区域: {region_dict}")
                 
                 # 截图
                 sct_img = sct.grab(region_dict)
                 print(f"截图尺寸: {sct_img.size}")
                 
                 # 将截图转换为 PNG 格式
                 img_pil = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                 
                 # 保存调试图片，只保留最新的一张
                 debug_dir = "debug_images"
                 if not os.path.exists(debug_dir):
                     os.makedirs(debug_dir)
                 
                 # 删除旧的调试图片
                 for old_file in os.listdir(debug_dir):
                     if old_file.startswith(f"{source}_"):
                         os.remove(os.path.join(debug_dir, old_file))
                 
                 # 保存新的调试图片
                 debug_image_path = os.path.join(debug_dir, f"{source}_latest.png")
                 img_pil.save(debug_image_path)
                 print(f"调试图片已保存: {debug_image_path}")
                 
                 img_byte_arr = io.BytesIO()
                 img_pil.save(img_byte_arr, format='PNG')
                 img_byte_arr = img_byte_arr.getvalue()
             
             # 使用 Gemini 模型识别文本
             instruction = self.system_instruction if hasattr(self, 'system_instruction') and self.system_instruction else "请识别这张图片中的所有文本内容，只返回识别到的文本，不要添加任何其他描述。"
             
             print("\n=== 发送到 Gemini 的请求 ===")
             print(f"使用的模型: {self.selected_model}")
             print(f"系统指令: {instruction}")
             print(f"图片大小: {len(img_byte_arr)} bytes")
             
             response = self.model.generate_content([
                 instruction,
                 {"mime_type": "image/png", "data": img_byte_arr}
             ])
             
             print("\n=== Gemini 的响应 ===")
             print(f"原始响应:\n{response.text}")
             
             if response and response.text:
                 text = response.text.strip()
                 
                 if text:
                     # 统一化 JSON 格式
                     current_messages = self.normalize_json_response(text)
                     
                     # 找出新消息
                     new_messages = self.find_new_messages(current_messages, self.last_messages[source])
                     
                     if new_messages:
                         print(f"\n=== 检测到新消息 ({source}) ===")
                         print(f"当前消息列表: {json.dumps(current_messages, ensure_ascii=False, indent=2)}")
                         print(f"新消息: {json.dumps(new_messages, ensure_ascii=False, indent=2)}")
                         
                         # 逐条发送新消息，添加内容过滤
                         for message in new_messages:
                             # 过滤用户名和评论内容
                             filtered_username = self.filter_sensitive_content(message["username"])
                             filtered_comment = self.filter_sensitive_content(message["comment"])
                             
                             payload = {
                                 "username": filtered_username,
                                 "message": filtered_comment,
                                 "source": source,
                                 "guard_level": 0
                             }
                             print(f"\n=== 发送到前端的数据 ===")
                             print(json.dumps(payload, ensure_ascii=False, indent=2))
                             self.new_comment_signal.emit(payload)
                     
                     # 更新上一次的消息列表（使用原始消息，而不是过滤后的）
                     self.last_messages[source] = current_messages
             print("\n=== 处理完成 ===\n")
             
         except Exception as e:
             print(f"\n=== 处理错误 ({source}) ===")
             print(f"错误类型: {type(e).__name__}")
             print(f"错误信息: {str(e)}")
             import traceback
             print("详细错误信息:")
             traceback.print_exc()
             print("=== 错误处理完成 ===\n")

    def stop(self):
        self.running = False
        print("Stopping OCR Worker...")

    def update_model(self, model_name):
        """更新模型名称并重新初始化模型"""
        self.selected_model = model_name
        if self.running:
            return self.initialize_model()
        return True

    def set_weibo_interval(self, interval):
        """设置微博截图间隔"""
        self.weibo_interval = max(0.1, float(interval))

    def set_twitter_interval(self, interval):
        """设置推特截图间隔"""
        self.twitter_interval = max(0.1, float(interval))

# --- HTTP Server Thread --- 
class HttpServerThread(QThread):
    def run(self):
        Handler = http.server.SimpleHTTPRequestHandler
        # Ensure the server serves from the correct directory where HTML_FILE resides
        # Change directory if HTML file is not in the script's directory
        # os.chdir("path/to/html/directory")
        try:
            with socketserver.TCPServer(("", HTTP_PORT), Handler) as httpd:
                print(f"HTTP server started on port {HTTP_PORT}. Serving {HTML_FILE}")
                print(f"Please add a Browser Source in OBS with URL: http://localhost:{HTTP_PORT}/{HTML_FILE}")
                httpd.serve_forever()
        except OSError as e:
             print(f"Error starting HTTP server on port {HTTP_PORT}: {e}")
             print("Port might be in use, or permissions are denied.")
        except Exception as e:
            print(f"HTTP server failed: {e}")

# --- WebSocket Server Thread --- 
class WebSocketServerThread(QThread):
    def run(self):
        global asyncio_loop, websocket_server_instance
        try:
            # Create and set a new event loop for this thread
            asyncio_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(asyncio_loop)

            # Define the WebSocket server startup coroutine
            async def start_server():
                global websocket_server_instance
                async with websockets.serve(register, "localhost", WEBSOCKET_PORT) as server:
                    websocket_server_instance = server # Store the instance
                    print(f"WebSocket server started on port {WEBSOCKET_PORT}")
                    await asyncio.Future() # Run forever

            # Run the server within the event loop
            asyncio_loop.run_until_complete(start_server())

        except OSError as e:
            print(f"Error starting WebSocket server on port {WEBSOCKET_PORT}: {e}")
            print("Port might be in use.")
            # Signal main thread or handle error appropriately
        except Exception as e:
            print(f"WebSocket server failed: {e}")
        finally:
            if asyncio_loop:
                asyncio_loop.close()
            print("WebSocket server loop stopped.")

    def stop_server(self):
        global asyncio_loop, websocket_server_instance
        if asyncio_loop and websocket_server_instance:
            print("Stopping WebSocket server...")
            # Stop the server from the loop it's running in
            asyncio_loop.call_soon_threadsafe(websocket_server_instance.close)
            # Optionally wait for it to close fully
            # asyncio_loop.run_until_complete(websocket_server_instance.wait_closed())
            # Stop the loop itself
            if asyncio_loop.is_running():
                 asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)

# --- Main GUI Window --- 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("弹幕集成器控制台")
        self.setGeometry(100, 100, 500, 400)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 初始化配置字典
        self.config = {}

        # --- API Key 设置 ---
        self.api_key_layout = QHBoxLayout()
        self.api_key_input = QTextEdit()
        self.api_key_input.setFixedHeight(30)  # 限制高度
        self.api_key_input.setPlaceholderText("请输入 Gemini API Key")
        if GEMINI_API_KEY:
            self.api_key_input.setText(GEMINI_API_KEY)
        self.api_key_button = QPushButton("保存 API Key")
        self.api_key_layout.addWidget(QLabel("API Key:"))
        self.api_key_layout.addWidget(self.api_key_input)
        self.api_key_layout.addWidget(self.api_key_button)
        self.layout.addLayout(self.api_key_layout)

        # --- 模型选择 ---
        self.model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(AVAILABLE_MODELS)
        self.model_layout.addWidget(QLabel("选择模型:"))
        self.model_layout.addWidget(self.model_selector)
        self.layout.addLayout(self.model_layout)

        # 创建微博截图间隔设置布局
        weibo_interval_layout = QHBoxLayout()
        weibo_interval_label = QLabel("微博截图间隔(秒):")
        self.weibo_interval_input = QLineEdit()
        self.weibo_interval_input.setText("15.0")
        self.weibo_interval_input.setFixedWidth(60)
        self.save_weibo_interval_button = QPushButton("保存间隔")
        self.save_weibo_interval_button.clicked.connect(self.save_weibo_interval)
        weibo_interval_layout.addWidget(weibo_interval_label)
        weibo_interval_layout.addWidget(self.weibo_interval_input)
        weibo_interval_layout.addWidget(self.save_weibo_interval_button)
        weibo_interval_layout.addStretch()

        # 创建推特截图间隔设置布局
        twitter_interval_layout = QHBoxLayout()
        twitter_interval_label = QLabel("推特截图间隔(秒):")
        self.twitter_interval_input = QLineEdit()
        self.twitter_interval_input.setText("30.0")
        self.twitter_interval_input.setFixedWidth(60)
        self.save_twitter_interval_button = QPushButton("保存间隔")
        self.save_twitter_interval_button.clicked.connect(self.save_twitter_interval)
        twitter_interval_layout.addWidget(twitter_interval_label)
        twitter_interval_layout.addWidget(self.twitter_interval_input)
        twitter_interval_layout.addWidget(self.save_twitter_interval_button)
        twitter_interval_layout.addStretch()

        # 将间隔设置布局添加到主布局
        self.layout.addLayout(weibo_interval_layout)
        self.layout.addLayout(twitter_interval_layout)

        # --- 系统指令设置 ---
        self.system_instruction_layout = QVBoxLayout()
        self.system_instruction_label = QLabel("系统指令:")
        self.system_instruction_input = QTextEdit()
        self.system_instruction_input.setFixedHeight(80)  # 给更多空间显示指令
        default_instruction = """发给你一张图片，其中是用户在直播中的评论。请逐条分析是否是合法（不含不宜展示的内容）并且完整的评论内容。请仅提取合法的内容，将粗体显示的用户名（形如jumbokun@Jumbokun2024请仅提取jumbokun）以及评论输出为一个json[{"username": "jumbokun", "comment": "评论"}]的格式返回"""
        self.system_instruction_input.setText(default_instruction)
        self.system_instruction_button = QPushButton("保存指令")
        self.system_instruction_layout.addWidget(self.system_instruction_label)
        self.system_instruction_layout.addWidget(self.system_instruction_input)
        self.system_instruction_layout.addWidget(self.system_instruction_button)
        self.layout.addLayout(self.system_instruction_layout)

        # --- 字体设置 ---
        self.font_settings_layout = QHBoxLayout()
        
        # 字体大小设置
        self.font_size_layout = QHBoxLayout()
        self.font_size_input = QTextEdit()
        self.font_size_input.setFixedHeight(30)
        self.font_size_input.setFixedWidth(50)
        self.font_size_input.setPlaceholderText("16")
        self.font_size_button = QPushButton("设置字号")
        self.font_size_layout.addWidget(QLabel("字体大小:"))
        self.font_size_layout.addWidget(self.font_size_input)
        self.font_size_layout.addWidget(self.font_size_button)
        
        # 字体颜色设置
        self.font_color_layout = QHBoxLayout()
        self.font_color_input = QTextEdit()
        self.font_color_input.setFixedHeight(30)
        self.font_color_input.setFixedWidth(80)
        self.font_color_input.setPlaceholderText("#FFFFFF")
        self.font_color_button = QPushButton("设置颜色")
        self.font_color_layout.addWidget(QLabel("字体颜色:"))
        self.font_color_layout.addWidget(self.font_color_input)
        self.font_color_layout.addWidget(self.font_color_button)
        
        # 将字体设置添加到布局
        self.font_settings_layout.addLayout(self.font_size_layout)
        self.font_settings_layout.addLayout(self.font_color_layout)
        self.layout.addLayout(self.font_settings_layout)

        # 在布局中添加弹幕显示设置部分
        danmu_display_group = QGroupBox("弹幕显示设置")
        danmu_display_layout = QGridLayout()

        # 宽度设置
        self.width_label = QLabel("宽度:")
        self.width_input = QSpinBox()
        self.width_input.setRange(200, 800)  # 设置宽度范围
        self.width_input.setValue(400)  # 默认宽度
        self.width_input.setSuffix("px")  # 添加单位后缀

        # 行数设置
        self.lines_label = QLabel("显示行数:")
        self.lines_input = QSpinBox()
        self.lines_input.setRange(1, 10)  # 设置行数范围
        self.lines_input.setValue(3)  # 默认行数

        # 确认按钮
        self.display_confirm_btn = QPushButton("确认设置")
        self.display_confirm_btn.clicked.connect(self.update_display_settings)

        # 添加到布局
        danmu_display_layout.addWidget(self.width_label, 0, 0)
        danmu_display_layout.addWidget(self.width_input, 0, 1)
        danmu_display_layout.addWidget(self.lines_label, 1, 0)
        danmu_display_layout.addWidget(self.lines_input, 1, 1)
        danmu_display_layout.addWidget(self.display_confirm_btn, 2, 0, 1, 2)

        danmu_display_group.setLayout(danmu_display_layout)
        self.layout.addWidget(danmu_display_group)

        # --- Remove Danmu Widget management --- 
        # self.danmu_widget = None
        # self.danmu_window_rect = None

        self.region_selector = None
        self.weibo_region = None
        self.twitter_region = None

        # --- Controls --- 
        self.controls_layout = QHBoxLayout()
        self.select_region_button = QPushButton("选择/重选区域")
        self.start_weibo_button = QPushButton("启动微博 OCR") # 新的微博启动按钮
        self.start_twitter_button = QPushButton("启动推特 OCR") # 新的推特启动按钮
        self.stop_button = QPushButton("停止所有服务")
        self.controls_layout.addWidget(self.select_region_button)
        self.controls_layout.addWidget(self.start_weibo_button)
        self.controls_layout.addWidget(self.start_twitter_button)
        self.controls_layout.addWidget(self.stop_button)
        self.stop_button.setEnabled(False)

        # --- Status/Log --- 
        self.status_label = QLabel("状态: 未启动")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # --- Layout Setup --- 
        self.layout.addLayout(self.controls_layout)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(QLabel("日志/说明:"))
        self.layout.addWidget(self.log_output)

        # --- OCR Worker --- 
        self.ocr_worker = OcrWorker()
        # Connect OCR signal to the signal bridge
        self.ocr_worker.new_comment_signal.connect(signal_bridge.broadcast_signal)
        self.ocr_worker.error_signal.connect(self.log_message)

        # --- Servers --- 
        self.http_server_thread = None
        self.websocket_server_thread = None

        # --- Connect GUI Signals --- 
        self.api_key_button.clicked.connect(self.save_api_key)
        self.select_region_button.clicked.connect(self.open_region_selector)
        self.start_weibo_button.clicked.connect(lambda: self.start_service("weibo"))
        self.start_twitter_button.clicked.connect(lambda: self.start_service("twitter"))
        self.stop_button.clicked.connect(self.stop_services)
        self.system_instruction_button.clicked.connect(self.save_instruction)  # 连接新的指令保存按钮
        self.font_size_button.clicked.connect(self.update_font_size)
        self.font_color_button.clicked.connect(self.update_font_color)
        self.display_confirm_btn.clicked.connect(self.update_display_settings)

        # --- Connect Bridge Signal --- 
        signal_bridge.broadcast_signal.connect(self.handle_broadcast)

        # --- Load config --- 
        self.load_config()
        # Log initial instructions
        self.log_message("欢迎使用弹幕集成器!")
        self.log_message(f"请先点击 [选择/重选区域] 按钮框选评论区。")
        self.log_message(f"然后在 OBS 中添加浏览器来源，URL 设置为: http://localhost:{HTTP_PORT}/{HTML_FILE}")
        self.log_message(f"完成后点击 [启动 OCR & 服务器] 按钮。")

        # --- 初始化变量 ---
        self.selected_model = self.model_selector.currentText()

        # --- 连接信号 ---
        self.model_selector.currentTextChanged.connect(self.on_model_changed)

    # --- Remove methods related to managing DanmuWidget --- 
    # def ensure_danmu_widget_exists(self):
    # def on_danmu_widget_geometry_changed(self, event):
    # def save_danmu_window_rect(self):
    # def toggle_danmu_widget(self):

    # --- Region Selection Methods (remain the same) --- 
    def open_region_selector(self):
        if self.region_selector is None or not self.region_selector.isVisible():
            self.log_message("打开区域选择器...")
            self.region_selector = RegionSelector()
            self.region_selector.weibo_region.connect(self.set_weibo_region)
            self.region_selector.twitter_region.connect(self.set_twitter_region)
            self.region_selector.selection_done.connect(self.regions_selected)
            self.region_selector.show()
        else:
            self.region_selector.activateWindow()

    def set_weibo_region(self, rect):
        self.weibo_region = rect
        self.log_message(f"微博区域更新: {rect.x()},{rect.y()},{rect.width()},{rect.height()}")

    def set_twitter_region(self, rect):
        self.twitter_region = rect
        self.log_message(f"推特区域更新: {rect.x()},{rect.y()},{rect.width()},{rect.height()}")

    def regions_selected(self):
        self.log_message("区域选择完成.")
        self.save_config() # Only saves region rects now
        was_running = self.ocr_worker.isRunning()
        if was_running:
            self.stop_ocr() # Only stop OCR worker
        self.ocr_worker.set_regions(self.weibo_region, self.twitter_region)
        if was_running:
             self.log_message("OCR 线程区域已更新。请手动重启 OCR & 服务器。")
        else:
            self.status_label.setText("状态: 区域已设置，可以启动")

    # --- Start/Stop OCR and Servers --- 
    def start_service(self, service_type):
        """启动指定的 OCR 服务"""
        print(f"\n=== 开始启动{service_type}服务 ===")
        
        # 检查API Key
        if not self.api_key_input.toPlainText().strip():
            self.log_message("错误: 请先设置 Gemini API Key!")
            return
            
        # 检查区域设置
        if service_type == "weibo" and not self.weibo_region:
            self.log_message("错误: 请先选择微博监控区域!")
            return
        elif service_type == "twitter" and not self.twitter_region:
            self.log_message("错误: 请先选择推特监控区域!")
            return

        # 设置系统指令
        instruction = self.system_instruction_input.toPlainText().strip()
        if instruction:
            self.ocr_worker.system_instruction = instruction
            self.log_message("已更新系统指令")
        else:
            self.log_message("警告: 使用默认系统指令")

        # 启动 HTTP 和 WebSocket 服务器（如果还没启动）
        if not self.http_server_thread or not self.http_server_thread.isRunning():
            self.log_message("启动 HTTP 服务器线程...")
            self.http_server_thread = HttpServerThread()
            self.http_server_thread.start()

        if not self.websocket_server_thread or not self.websocket_server_thread.isRunning():
            self.log_message("启动 WebSocket 服务器线程...")
            self.websocket_server_thread = WebSocketServerThread()
            self.websocket_server_thread.start()

        # 如果 OCR 线程已在运行，先停止它
        if self.ocr_worker.isRunning():
            self.log_message(f"停止当前 OCR 线程以更新配置...")
            self.ocr_worker.stop()
            self.ocr_worker.wait()

        # 确保 Gemini 模型已初始化
        if not self.ocr_worker.model:
            self.log_message("初始化 Gemini 模型...")
            if not self.ocr_worker.initialize_model():
                self.log_message("错误: Gemini 模型初始化失败!")
                return

        # 根据服务类型设置区域
        if service_type == "weibo":
            self.log_message("启动微博 OCR 线程...")
            self.ocr_worker.set_regions(self.weibo_region, None)
            self.start_weibo_button.setEnabled(False)
            # 设置微博截图间隔
            try:
                interval = float(self.weibo_interval_input.text())
                self.ocr_worker.set_weibo_interval(interval)
                self.log_message(f"微博截图间隔设置为: {interval}秒")
            except ValueError:
                self.log_message("警告: 使用默认微博截图间隔")
        else:
            self.log_message("启动推特 OCR 线程...")
            self.ocr_worker.set_regions(None, self.twitter_region)
            self.start_twitter_button.setEnabled(False)
            # 设置推特截图间隔
            try:
                interval = float(self.twitter_interval_input.text())
                self.ocr_worker.set_twitter_interval(interval)
                self.log_message(f"推特截图间隔设置为: {interval}秒")
            except ValueError:
                self.log_message("警告: 使用默认推特截图间隔")

        # 启动 OCR 线程
        self.ocr_worker.running = True  # 确保运行标志被设置
        self.ocr_worker.start()
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"状态: {service_type.capitalize()} OCR 服务运行中...")
        self.log_message(f"{service_type.capitalize()} OCR 服务已启动")

    def stop_services(self):
        """停止所有服务"""
        stopped_something = False
        if self.ocr_worker.isRunning():
            self.log_message("正在停止 OCR 线程...")
            self.ocr_worker.stop()
            self.ocr_worker.wait()
            self.log_message("OCR 线程已停止。")
            stopped_something = True
            # 重新启用启动按钮
            self.start_weibo_button.setEnabled(True)
            self.start_twitter_button.setEnabled(True)
        else:
            self.log_message("OCR 线程未运行。")

        if self.websocket_server_thread and self.websocket_server_thread.isRunning():
            self.log_message("正在停止 WebSocket 服务器...")
            self.websocket_server_thread.stop_server()
            self.websocket_server_thread.wait()
            self.websocket_server_thread = None
            self.log_message("WebSocket 服务器已停止。")
            stopped_something = True
        else:
            self.log_message("WebSocket 服务器未运行。")

        if self.http_server_thread and self.http_server_thread.isRunning():
            self.log_message("正在停止 HTTP 服务器...")
            print("HTTP 服务器将在程序退出时停止。")
        else:
            self.log_message("HTTP 服务器未运行。")

        if stopped_something:
            self.stop_button.setEnabled(False)
            self.status_label.setText("状态: 服务已停止")

    # --- Slot to handle signal from bridge --- 
    @pyqtSlot(dict)
    def handle_broadcast(self, payload):
        """Receives payload from signal bridge and sends via WebSocket."""
        # Ensure this runs in the asyncio loop's thread
        global asyncio_loop
        if asyncio_loop:
             message_data = {"type": "danmu", "payload": payload}
             message_json = json.dumps(message_data, ensure_ascii=False)
             # Schedule the broadcast coroutine in the asyncio loop
             asyncio.run_coroutine_threadsafe(broadcast_message(message_json), asyncio_loop)
        else:
            print("Error: Asyncio loop not available for broadcasting.")


    @pyqtSlot(str)
    def log_message(self, message):
        print(message)
        # Ensure log_output exists before appending
        if hasattr(self, 'log_output'):
             self.log_output.append(message)
        else:
            print("(Log output widget not ready yet)")

    def load_config(self):
        """从文件加载配置"""
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # 加载API Key
                if 'api_key' in config:
                    self.api_key_input.setText(config['api_key'])
                
                # 加载选中的模型
                if 'selected_model' in config:
                    model_name = config['selected_model']
                    if model_name in AVAILABLE_MODELS:
                        self.model_selector.setCurrentText(model_name)
                        if self.ocr_worker:
                            self.ocr_worker.selected_model = model_name
                
                # 加载截图间隔
                if 'weibo_interval' in config:
                    self.weibo_interval_input.setText(str(config['weibo_interval']))
                if 'twitter_interval' in config:
                    self.twitter_interval_input.setText(str(config['twitter_interval']))
                
                # 加载区域设置
                wb_coords = config.get('weibo_rect')
                tw_coords = config.get('twitter_rect')
                if wb_coords:
                    self.weibo_region = QRect(wb_coords['x'], wb_coords['y'], 
                                            wb_coords['width'], wb_coords['height'])
                    self.log_message(f"已加载微博区域: {self.weibo_region}")
                if tw_coords:
                    self.twitter_region = QRect(tw_coords['x'], tw_coords['y'], 
                                              tw_coords['width'], tw_coords['height'])
                    self.log_message(f"已加载推特区域: {self.twitter_region}")
                
                # 加载系统指令
                if 'system_instruction' in config:
                    self.system_instruction_input.setPlainText(config['system_instruction'])
                
                self.log_message("配置已加载")
        except FileNotFoundError:
            self.log_message("未找到配置文件，将使用默认设置")
        except json.JSONDecodeError:
            self.log_message("配置文件格式错误，将使用默认设置")
        except Exception as e:
            self.log_message(f"加载配置时出错: {str(e)}")

    def save_config(self):
        """保存配置到文件"""
        config = {}
        
        # 保存API Key
        if self.api_key_input.toPlainText():
            config['api_key'] = self.api_key_input.toPlainText()
        
        # 保存选中的模型
        config['selected_model'] = self.ocr_worker.selected_model if self.ocr_worker else AVAILABLE_MODELS[0]
        
        # 保存截图间隔
        config['weibo_interval'] = float(self.weibo_interval_input.text())
        config['twitter_interval'] = float(self.twitter_interval_input.text())
        
        # 保存区域设置
        if self.weibo_region:
            config['weibo_rect'] = {
                'x': self.weibo_region.x(),
                'y': self.weibo_region.y(),
                'width': self.weibo_region.width(),
                'height': self.weibo_region.height()
            }
        if self.twitter_region:
            config['twitter_rect'] = {
                'x': self.twitter_region.x(),
                'y': self.twitter_region.y(),
                'width': self.twitter_region.width(),
                'height': self.twitter_region.height()
            }
        
        # 保存系统指令
        if self.system_instruction_input.toPlainText():
            config['system_instruction'] = self.system_instruction_input.toPlainText()

        # 保存显示设置
        config['display'] = {
            'width': self.width_input.value(),
            'lines': self.lines_input.value()
        }
        
        # 写入配置文件
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        self.log_message("配置已保存")

    def save_api_key(self):
        """保存 API Key 并重新初始化 Gemini 模型"""
        api_key = self.api_key_input.toPlainText().strip()
        if not api_key:
            self.log_message("错误：API Key 不能为空")
            return

        # 更新配置文件中的 API Key
        config = {}
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
        except Exception:
            pass

        config['api_key'] = api_key
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.log_message("API Key 已保存到配置文件")
        except Exception as e:
            self.log_message(f"保存 API Key 失败: {e}")
            return

        # 重新初始化 Gemini 模型
        if hasattr(self.ocr_worker, 'model'):
            try:
                genai.configure(api_key=api_key)
                self.ocr_worker.model = genai.GenerativeModel(self.ocr_worker.selected_model)
                self.log_message("Gemini 模型已使用新的 API Key 重新初始化")
            except Exception as e:
                self.log_message(f"重新初始化 Gemini 模型失败: {e}")
                return

        global GEMINI_API_KEY
        GEMINI_API_KEY = api_key

    def on_model_changed(self, model_name):
        """当用户选择新的模型时调用"""
        self.ocr_worker.selected_model = model_name
        self.log_message(f"已选择模型: {model_name}")
        
        # 更新 OCR Worker 的模型
        if hasattr(self, 'ocr_worker') and self.ocr_worker:
            try:
                if self.ocr_worker.update_model(model_name):
                    self.log_message("已更新运行中的模型")
                else:
                    self.log_message("模型更新失败")
            except Exception as e:
                self.log_message(f"更新模型失败: {e}")

        # 保存配置
        self.save_config()

    def save_instruction(self):
        """保存系统指令"""
        instruction = self.system_instruction_input.toPlainText().strip()
        if not instruction:
            self.log_message("错误：系统指令不能为空")
            return
        
        # 保存到配置文件
        config = {}
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
        except Exception:
            pass

        config['system_instruction'] = instruction
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            self.log_message("系统指令已保存到配置文件")
        except Exception as e:
            self.log_message(f"保存系统指令失败: {e}")
            return

    def update_font_size(self):
        """更新字体大小"""
        try:
            size = int(self.font_size_input.toPlainText().strip())
            if size < 8:  # 最小字号
                size = 8
                self.font_size_input.setText("8")
            elif size > 72:  # 最大字号
                size = 72
                self.font_size_input.setText("72")
            
            # 发送样式更新到前端
            self.broadcast_style_update()
            self.log_message(f"字体大小已更新为: {size}px")
            
            # 保存到配置
            self.save_config()
        except ValueError:
            self.log_message("错误: 字体大小必须是有效的数字!")

    def update_font_color(self):
        """更新字体颜色"""
        color = self.font_color_input.toPlainText().strip()
        if not color.startswith('#'):
            color = '#' + color
        
        # 简单的颜色格式验证
        if len(color) == 7 and all(c in '0123456789ABCDEFabcdef#' for c in color):
            # 发送样式更新到前端
            self.broadcast_style_update()
            self.log_message(f"字体颜色已更新为: {color}")
            
            # 保存到配置
            self.save_config()
        else:
            self.log_message("错误: 请输入有效的颜色代码 (例如: #FFFFFF)")

    def broadcast_style_update(self):
        """发送样式更新到前端"""
        try:
            size = int(self.font_size_input.toPlainText().strip())
            color = self.font_color_input.toPlainText().strip()
            if not color.startswith('#'):
                color = '#' + color

            style_data = {
                "type": "style",
                "payload": {
                    "fontSize": size,
                    "textColor": color
                }
            }
            
            # 使用 WebSocket 发送样式更新
            global asyncio_loop
            if asyncio_loop:
                message_json = json.dumps(style_data, ensure_ascii=False)
                asyncio.run_coroutine_threadsafe(broadcast_message(message_json), asyncio_loop)
        except Exception as e:
            self.log_message(f"更新样式失败: {e}")

    def update_display_settings(self):
        """更新弹幕显示设置"""
        width = self.width_input.value()
        lines = self.lines_input.value()
        
        # 更新样式
        style_data = {
            'type': 'style',
            'payload': {
                'containerWidth': width,
                'lines': lines
            }
        }
        
        # 使用 WebSocket 发送样式更新
        global asyncio_loop
        if asyncio_loop:
            message_json = json.dumps(style_data, ensure_ascii=False)
            asyncio.run_coroutine_threadsafe(broadcast_message(message_json), asyncio_loop)
        
        # 确保 config 字典存在
        if not hasattr(self, 'config'):
            self.config = {}
        
        # 保存设置到配置文件
        self.config['display'] = {
            'width': width,
            'lines': lines
        }
        self.save_config()
        
        QMessageBox.information(self, "设置更新", "显示设置已更新")

    def save_weibo_interval(self):
        """保存微博截图间隔"""
        try:
            interval = float(self.weibo_interval_input.text())
            if interval < 0.1:
                self.log_message("错误: 间隔不能小于0.1秒")
                self.weibo_interval_input.setText("0.1")
                interval = 0.1
            if self.ocr_worker:
                self.ocr_worker.set_weibo_interval(interval)
                self.log_message(f"已更新微博截图间隔为: {interval}秒")
            self.save_config()
        except ValueError:
            self.log_message("错误: 请输入有效的数字")
            self.weibo_interval_input.setText("15.0")

    def save_twitter_interval(self):
        """保存推特截图间隔"""
        try:
            interval = float(self.twitter_interval_input.text())
            if interval < 0.1:
                self.log_message("错误: 间隔不能小于0.1秒")
                self.twitter_interval_input.setText("0.1")
                interval = 0.1
            if self.ocr_worker:
                self.ocr_worker.set_twitter_interval(interval)
                self.log_message(f"已更新推特截图间隔为: {interval}秒")
            self.save_config()
        except ValueError:
            self.log_message("错误: 请输入有效的数字")
            self.twitter_interval_input.setText("30.0")

    def closeEvent(self, event):
        self.log_message("正在关闭应用程序...")
        # Stop services first
        self.stop_services()
        # Save final config
        self.save_config()
        event.accept()


if __name__ == '__main__':
    # Note: QApplication is created implicitly by importing QApplication above
    # Ensure Qt event loop integration with asyncio if needed (more advanced)
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
