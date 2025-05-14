from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QValueAxis,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import ConfigManager


class GUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.config = ConfigManager.get()

        self.setWindowTitle(self.config["app"]["title"])
        self.setGeometry(100, 100, 1200, 800)

        self.classes_list = self.config["classes_list"]

        self.img_path_list: List[Path] = []
        self.model_path_list: List[Path] = []

        self.img_idx: int = -1
        self.ort_session: Optional[ort.InferenceSession] = None
        self.probs: Optional[np.ndarray] = None

        self.init_main_ui()

        self.connet_signals()

        self.update_buttons()

    def init_main_ui(self) -> None:
        # --- 主容器 ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        self.init_view_panels(main_layout)
        self.init_control_panels(main_layout)
        self.init_status_bar()

    def init_view_panels(self, parent_layout: QVBoxLayout) -> None:
        ## --- 视图面板 ---
        panels_widget = QGroupBox()
        panels_layout = QHBoxLayout(panels_widget)

        self.init_left_view_panel(panels_layout)
        self.init_right_view_panel(panels_layout)

        parent_layout.addWidget(panels_widget, 1)

    def init_left_view_panel(self, parent_layout: QHBoxLayout) -> None:
        ### --- 左侧面板 ---
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)

        #### --- 左侧面板标题 ---
        panel_label = QLabel(self.config["img_panel"]["label"])

        #### --- 左侧面板内容 ---
        self.img_panel = QLabel("")
        self.img_panel.setMinimumSize(
            self.config["panel"]["minw"], self.config["panel"]["minh"]
        )
        self.img_panel.setFrameShape(QFrame.Shape.StyledPanel)
        self.img_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        #### --- 左侧面板信息 ---
        info_widget = QGroupBox()
        info_layout = QHBoxLayout(info_widget)

        self.img_name_label = QLabel(self.config["img_panel"]["name_null"])
        self.img_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.img_idx_label = QLabel(self.config["img_panel"]["idx_null"])
        self.img_idx_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        info_layout.addWidget(self.img_name_label)
        info_layout.addWidget(self.img_idx_label)

        panel_layout.addWidget(panel_label)
        panel_layout.addWidget(self.img_panel, 1)
        panel_layout.addWidget(info_widget)

        parent_layout.addWidget(panel_widget, 1)

    def init_right_view_panel(self, parent_layout: QHBoxLayout) -> None:
        ### --- 右侧面板 ---
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)

        #### --- 右侧面板标题 ---
        panel_label = QLabel(self.config["pred_panel"]["label"])

        #### --- 右侧面板内容 ---
        self.pred_chart = QChart()
        self.pred_chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

        chart_view = QChartView(self.pred_chart)
        chart_view.setMinimumSize(
            self.config["panel"]["minw"],
            self.config["panel"]["minh"],
        )

        #### --- 右侧面板信息 ---
        info_widget = QGroupBox()
        info_layout = QHBoxLayout(info_widget)

        self.pred_name_label = QLabel(self.config["pred_panel"]["name_null"])
        self.pred_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.pred_conf_label = QLabel(self.config["pred_panel"]["conf_null"])
        self.pred_conf_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        info_layout.addWidget(self.pred_name_label)
        info_layout.addWidget(self.pred_conf_label)

        panel_layout.addWidget(panel_label)
        panel_layout.addWidget(chart_view, 1)
        panel_layout.addWidget(info_widget)

        parent_layout.addWidget(panel_widget, 1)

    def init_control_panels(self, parent_layout: QVBoxLayout) -> None:
        ## --- 控制面板 ---
        panels_widget = QGroupBox()
        panels_layout = QHBoxLayout(panels_widget)

        self.init_left_control_panel(panels_layout)
        self.init_center_control_panel(panels_layout)
        self.init_right_control_panel(panels_layout)

        parent_layout.addWidget(panels_widget)

    def init_left_control_panel(self, parent_layout: QHBoxLayout) -> None:
        ### --- 左侧面板 ---
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)

        panel_label = QLabel(self.config["load_panel"]["label"])
        self.load_img_btn = QPushButton(self.config["load_panel"]["img_btn"])
        self.load_img_btn.setIcon(QIcon.fromTheme("folder"))
        self.load_model_btn = QPushButton(self.config["load_panel"]["model_btn"])
        self.load_model_btn.setIcon(QIcon.fromTheme("folder"))

        panel_layout.addWidget(panel_label)
        panel_layout.addWidget(self.load_img_btn)
        panel_layout.addWidget(self.load_model_btn)

        parent_layout.addWidget(panel_widget)

    def init_center_control_panel(self, parent_layout: QHBoxLayout) -> None:
        ### --- 中间面板 ---
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)

        panel_label = QLabel(self.config["nav_panel"]["label"])
        self.prev_btn = QPushButton(self.config["nav_panel"]["prev_btn"])
        self.prev_btn.setIcon(QIcon.fromTheme("go-previous"))
        self.next_btn = QPushButton(self.config["nav_panel"]["next_btn"])
        self.next_btn.setIcon(QIcon.fromTheme("go-next"))

        panel_layout.addWidget(panel_label)
        panel_layout.addWidget(self.prev_btn)
        panel_layout.addWidget(self.next_btn)

        parent_layout.addWidget(panel_widget)

    def init_right_control_panel(self, parent_layout: QHBoxLayout) -> None:
        ### --- 右侧面板 ---
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)

        panel_label = QLabel(self.config["model_panel"]["label"])
        self.model_combobox = QComboBox()
        self.model_combobox.addItem(self.config["model_panel"]["box_null"])
        self.pred_btn = QPushButton(self.config["model_panel"]["pred_btn"])
        self.pred_btn.setIcon(QIcon.fromTheme("media-playback-start"))

        panel_layout.addWidget(panel_label)
        panel_layout.addWidget(self.model_combobox)
        panel_layout.addWidget(self.pred_btn)

        parent_layout.addWidget(panel_widget)

    def init_status_bar(self) -> None:
        ## --- 状态栏 ---
        status_bar = self.statusBar()

        self.status_bar_label = QLabel(self.config["status_bar"]["null"])
        self.status_bar_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.version_label = QLabel(self.config["status_bar"]["version"])
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        status_bar.addPermanentWidget(self.status_bar_label, 1)
        status_bar.addPermanentWidget(self.version_label)

        status_bar.setStyleSheet("QStatusBar { border: 1px solid #c0c0c0; }")

    def connet_signals(self) -> None:
        self.load_img_btn.clicked.connect(self.load_images)
        self.load_model_btn.clicked.connect(self.load_models)
        self.prev_btn.clicked.connect(self.show_prev_img)
        self.next_btn.clicked.connect(self.show_next_img)
        self.model_combobox.currentIndexChanged.connect(self.select_model)
        self.pred_btn.clicked.connect(self.predict_image)

    def load_images(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self)
        if dir_path:
            extensions = self.config["load"]["img_extensions"]
            self.img_path_list = [
                img_path for ext in extensions for img_path in Path(dir_path).glob(ext)
            ]

            self.img_idx = 0 if self.img_path_list else -1

            self.update_image_panel()
            self.update_image_info()
            self.update_buttons()
            self.update_status_bar()

    def load_models(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self)
        if dir_path:
            extensions = self.config["load"]["model_extensions"]
            self.model_path_list = [
                model_path
                for ext in extensions
                for model_path in Path(dir_path).glob(ext)
            ]

            if self.model_path_list:
                self.model_combobox.clear()
                self.model_combobox.addItem(self.config["model_panel"]["box_item"])
                self.model_combobox.addItems(
                    [path.name for path in self.model_path_list]
                )

            self.update_buttons()
            self.update_status_bar()

    def select_model(self, index: int) -> None:
        if index > 0:
            self.ort_session = ort.InferenceSession(self.model_path_list[index - 1])

        self.update_buttons()

    def update_image_panel(self) -> None:
        img_path = self.img_path_list[self.img_idx]
        pixmap = QPixmap(img_path)
        self.img_panel.setPixmap(
            pixmap.scaled(
                self.img_panel.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def update_image_info(self) -> None:
        name = self.img_path_list[self.img_idx].name
        idx = self.img_idx + 1
        total = len(self.img_path_list)

        self.img_name_label.setText(f"{self.config['img_panel']['name_label']} {name}")
        self.img_idx_label.setText(
            f"{self.config['img_panel']['idx_label']} {idx}/{total}"
        )

    def update_buttons(self) -> None:
        total = len(self.img_path_list)

        if total <= 1:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(self.img_idx > 0)
            self.next_btn.setEnabled(self.img_idx < total - 1)

        self.model_combobox.setEnabled(bool(self.model_path_list))
        self.pred_btn.setEnabled(bool(self.ort_session and self.img_path_list))

    def update_status_bar(self) -> None:
        img_count = len(self.img_path_list)
        model_count = len(self.model_path_list)

        self.status_bar_label.setText(
            f"{self.config['status_bar']['label']} {img_count} {self.config['status_bar']['img']} | {model_count} {self.config['status_bar']['model']}"
        )

    def show_prev_img(self) -> None:
        if self.img_idx > 0:
            self.img_idx -= 1

            self.update_image_panel()
            self.update_image_info()

            self.update_buttons()

    def show_next_img(self) -> None:
        if self.img_idx < len(self.img_path_list) - 1:
            self.img_idx += 1

            self.update_image_panel()
            self.update_image_info()
            self.update_buttons()

    def preprocess_image(self, img_path: Path) -> np.ndarray:
        with Image.open(img_path) as img:
            # 1. ToImage
            img = img.convert("RGB")

            # 2. Resize
            img = img.resize((256, 256), Image.Resampling.BILINEAR)

            # 3. CenterCrop
            img = img.crop((16, 16, 240, 240))

            # 4. ToTensor
            img_np = np.array(img) / 255.0
            img_np = img_np.transpose((2, 0, 1))

            # 5. Normalize
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_np = (img_np - mean) / std

            # 6. ToBatch
            img_np = np.expand_dims(img_np, axis=0).astype(np.float32)

            return img_np

    def predict_image(self) -> None:
        img_path = self.img_path_list[self.img_idx]

        input_value = self.preprocess_image(img_path)
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name

        output = self.ort_session.run([output_name], {input_name: input_value})[0][0]

        # Softmax
        e_x = np.exp(output - np.max(output))
        self.probs = e_x * 100 / np.sum(e_x)

        idx = np.argmax(output)
        self.pred_name_label.setText(
            f"{self.config['pred_panel']['name_label']} {self.config['classes_list'][idx]}"
        )
        self.pred_conf_label.setText(
            f"{self.config['pred_panel']['conf_label']} {self.probs[idx]:.2f}%"
        )

        self.plot_barchart()

    def plot_barchart(self) -> None:
        # 0. 清除旧的 Series 和 Axes
        self.pred_chart.removeAllSeries()
        for axis in self.pred_chart.axes():
            self.pred_chart.removeAxis(axis)

        # 1. 创建 Set
        bar_set = QBarSet("")
        for val in self.probs:
            bar_set.append(val)

        # 2. 创建 Series 并添加到图表
        series = QBarSeries()
        series.append(bar_set)
        self.pred_chart.addSeries(series)

        # 3. X 轴
        axis_x = QBarCategoryAxis()
        axis_x.append(self.config["classes_list"])
        axis_x.setTitleText(self.config["plot"]["xlabel"])
        self.pred_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        # 4. Y 轴
        axis_y = QValueAxis()
        axis_y.setTitleText(self.config["plot"]["ylabel"])
        axis_y.setRange(0, 100)
        self.pred_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

        # 5. 图例
        self.pred_chart.legend().setVisible(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        if self.img_idx != -1:
            self.update_image_panel()

    def closeEvent(self, event: Any) -> None:
        reply = QMessageBox.question(
            self,
            self.config["question"]["title"],
            self.config["question"]["text"],
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()
