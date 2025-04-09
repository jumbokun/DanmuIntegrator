import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtGui import QScreen, QPixmap, QPainter, QPen, QGuiApplication
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

    def paintEvent(self, event):
        if self.selecting and self.start_point and self.end_point:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            # 半透明覆盖层
            painter.fillRect(self.rect(), Qt.GlobalColor.transparent) # Qt.GlobalColor.black with alpha? No, let mouse pass through
            # 绘制选框
            rect = QRect(self.start_point, self.end_point).normalized()
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
        self.setWindowTitle("区域选择")
        layout = QVBoxLayout()
        self.info_label = QLabel("请点击按钮并框选区域")
        self.select_weibo_button = QPushButton("选择微博评论区")
        self.select_twitter_button = QPushButton("选择推特评论区")
        self.done_button = QPushButton("完成选择")

        layout.addWidget(self.info_label)
        layout.addWidget(self.select_weibo_button)
        layout.addWidget(self.select_twitter_button)
        layout.addWidget(self.done_button)
        self.setLayout(layout)

        self.overlay = None
        self._weibo_rect = None
        self._twitter_rect = None

        self.select_weibo_button.clicked.connect(lambda: self.start_selection("weibo"))
        self.select_twitter_button.clicked.connect(lambda: self.start_selection("twitter"))
        self.done_button.clicked.connect(self.finish_selection)

    def start_selection(self, target):
        if self.overlay is None or not self.overlay.isVisible():
             # 稍微隐藏主窗口，避免干扰选择
            self.hide()
            self.overlay = OverlayWidget()
            # 将信号连接到不同的槽
            if target == "weibo":
                self.overlay.region_selected.connect(self.on_weibo_region_selected)
            elif target == "twitter":
                 self.overlay.region_selected.connect(self.on_twitter_region_selected)
            self.overlay.show()
            # 当覆盖窗口关闭时，重新显示主窗口
            self.overlay.destroyed.connect(self.show)


    def on_weibo_region_selected(self, rect):
        self._weibo_rect = rect
        print(f"微博区域选定: {rect.x()}, {rect.y()}, {rect.width()}, {rect.height()}")
        self.info_label.setText(f"微博区域已选: ({rect.x()}, {rect.y()}, {rect.width()}, {rect.height()})\n请选择推特区域或点击完成。")
        if self.overlay:
            self.overlay.region_selected.disconnect(self.on_weibo_region_selected) # 断开连接避免重复触发

    def on_twitter_region_selected(self, rect):
        self._twitter_rect = rect
        print(f"推特区域选定: {rect.x()}, {rect.y()}, {rect.width()}, {rect.height()}")
        self.info_label.setText(f"推特区域已选: ({rect.x()}, {rect.y()}, {rect.width()}, {rect.height()})\n请选择微博区域或点击完成。")
        if self.overlay:
            self.overlay.region_selected.disconnect(self.on_twitter_region_selected) # 断开连接

    def finish_selection(self):
        if self._weibo_rect:
            self.weibo_region.emit(self._weibo_rect)
        if self._twitter_rect:
            self.twitter_region.emit(self._twitter_rect)
        self.selection_done.emit()
        self.close()


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