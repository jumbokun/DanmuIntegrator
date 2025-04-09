import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtGui import QScreen, QPixmap, QPainter, QPen, QBrush, QColor, QGuiApplication
from PyQt6.QtCore import Qt, QRect, pyqtSignal

class OverlayWidget(QWidget):
    """一个半透明的覆盖窗口，用于提示用户选择区域"""
    region_selected = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # 获取所有屏幕的组合几何信息
        geometry = QRect()
        for screen in QGuiApplication.screens():
            geometry = geometry.united(screen.geometry())
        self.setGeometry(geometry)

        self.start_point = None
        self.end_point = None
        self.selecting = False
        # Define overlay and selection colors/pen
        self.overlay_color = QColor(0, 0, 0, 100) # Darker semi-transparent overlay (0-255 alpha)
        self.selection_pen = QPen(QColor(255, 0, 0, 200), 3, Qt.PenStyle.SolidLine) # Thicker, slightly transparent red
        self.selection_brush = QBrush(QColor(255, 0, 0, 30)) # Very light red fill inside selection

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the semi-transparent overlay over the entire screen
        painter.fillRect(self.rect(), self.overlay_color)

        if self.selecting and self.start_point and self.end_point:
            rect = QRect(self.start_point, self.end_point).normalized()
            # Clear the overlay within the selected rectangle to show original screen content
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Draw the selection rectangle border and fill
            painter.setPen(self.selection_pen)
            painter.setBrush(self.selection_brush)
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.selecting = True
            self.update() # 触发重绘

    def mouseMoveEvent(self, event):
        if self.selecting:
            self.end_point = event.position().toPoint()
            self.update() # 触发重绘

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False
            if self.start_point and self.end_point and self.start_point != self.end_point:
                selected_rect = QRect(self.start_point, self.end_point).normalized()
                # print(f"Region selected: {selected_rect}")
                self.region_selected.emit(selected_rect)
            self.close()

    def keyPressEvent(self, event):
        # 按 Esc 取消选择
        if event.key() == Qt.Key.Key_Escape:
            self.close()

class RegionSelector(QWidget):
    """用于启动区域选择过程的辅助窗口"""
    weibo_region = pyqtSignal(QRect)
    twitter_region = pyqtSignal(QRect)
    selection_done = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.initUI()
        self.start = None
        self.end = None
        self.is_selecting = False
        self.current_mode = "weibo"  # 或 "twitter"
        self.weibo_rect = None  # 保存微博区域
        self.twitter_rect = None  # 保存推特区域
        self.show_selection = True  # 控制是否显示选择框
        self.selection_completed = False  # 添加标志来追踪是否完成选择
        
        # 获取所有屏幕的组合几何信息
        self.total_geometry = QRect()
        for screen in QGuiApplication.screens():
            self.total_geometry = self.total_geometry.united(screen.geometry())
        print(f"总显示区域: {self.total_geometry}")

    def initUI(self):
        # 设置窗口标志
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        # 设置窗口透明
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # 设置窗口大小为所有屏幕的组合区域
        self.total_geometry = QRect()
        for screen in QGuiApplication.screens():
            self.total_geometry = self.total_geometry.united(screen.geometry())
        self.setGeometry(self.total_geometry)

        # 创建说明标签
        self.label = QLabel("请框选微博评论区域\n按Enter/Space确认，按ESC取消", self)
        self.label.setStyleSheet("color: white; font-size: 16px; background-color: rgba(0, 0, 0, 150); padding: 10px;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 创建垂直布局并添加标签
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 设置半透明背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        
        # 如果正在选择或已经选择完成且需要显示选择框
        if self.start and self.end and self.show_selection:
            # 绘制当前选择区域
            current_rect = self.getCurrentRect()
            if current_rect:
                # 清除选择区域的半透明背景
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                painter.fillRect(current_rect, Qt.GlobalColor.transparent)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                
                # 根据当前模式选择颜色
                color = QColor(255, 0, 0) if self.current_mode == "weibo" else QColor(0, 255, 0)
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(current_rect)
        
        # 如果已保存的区域存在且需要显示，也绘制它们
        if self.weibo_rect and self.show_selection:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self.weibo_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            pen = QPen(QColor(255, 0, 0))  # 微博区域用红色
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.weibo_rect)
            
        if self.twitter_rect and self.show_selection:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self.twitter_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            pen = QPen(QColor(0, 255, 0))  # 推特区域用绿色
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(self.twitter_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.pos()
            self.end = self.start
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.end = event.pos()
            current_rect = self.getCurrentRect()
            
            if current_rect and current_rect.width() > 10 and current_rect.height() > 10:
                if self.current_mode == "weibo":
                    self.weibo_rect = current_rect
                    self.weibo_region.emit(current_rect)
                    self.label.setText("请框选推特评论区域\n按Enter/Space确认，按ESC取消\n按ESC两次可退出选择")
                    self.current_mode = "twitter"
                else:
                    self.twitter_rect = current_rect
                    self.twitter_region.emit(current_rect)
                    self.selection_done.emit()
                    self.label.setText("选择完成！\n按ESC退出")
                    self.selection_completed = True  # 标记选择已完成
            
            self.start = None
            self.end = None
            self.update()

    def getCurrentRect(self):
        if self.start and self.end:
            # 获取相对于整个虚拟屏幕的坐标
            return QRect(min(self.start.x(), self.end.x()),
                        min(self.start.y(), self.end.y()),
                        abs(self.start.x() - self.end.x()),
                        abs(self.start.y() - self.end.y()))
        return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            # 如果选择已完成，直接关闭窗口
            if self.selection_completed:
                self.close()
            # 如果是第一次按ESC且已经选择了微博区域，但还未完成整个选择过程
            elif self.current_mode == "twitter" and self.weibo_rect and not self.selection_completed:
                self.current_mode = "weibo"
                self.label.setText("请框选微博评论区域\n按Enter/Space确认，按ESC取消")
                self.weibo_rect = None
                self.start = None
                self.end = None
                self.update()
            else:
                self.close()
        elif event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Space]:
            if self.start and self.end:
                current_rect = self.getCurrentRect()
                if current_rect and current_rect.width() > 10 and current_rect.height() > 10:
                    if self.current_mode == "weibo":
                        self.weibo_rect = current_rect
                        self.weibo_region.emit(current_rect)
                        self.label.setText("请框选推特评论区域\n按Enter/Space确认，按ESC取消")
                        self.current_mode = "twitter"
                    else:
                        self.twitter_rect = current_rect
                        self.twitter_region.emit(current_rect)
                        self.selection_done.emit()
                        self.label.setText("选择完成！\n按ESC退出")
                        self.selection_completed = True  # 标记选择已完成
                    
                    self.start = None
                    self.end = None
                    self.update()

    def focusOutEvent(self, event):
        # 确保窗口始终保持焦点
        self.activateWindow()
        super().focusOutEvent(event)

# --- Test Section ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    selector_widget = RegionSelector()

    def print_regions(wb_rect, tw_rect):
        print("选择完成:")
        if wb_rect:
            print(f"  微博: {wb_rect}")
        if tw_rect:
            print(f"  推特: {tw_rect}")

    def on_done():
        print("选择流程结束")
        # app.quit() # Example usage might keep app running

    selector_widget.weibo_region.connect(lambda r: print(f"主程序收到微博区域: {r}"))
    selector_widget.twitter_region.connect(lambda r: print(f"主程序收到推特区域: {r}"))
    selector_widget.selection_done.connect(on_done)

    selector_widget.show()
    sys.exit(app.exec())