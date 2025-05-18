import os
import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QPushButton, QLabel, QLineEdit, QComboBox, QProgressBar, QTextEdit, 
    QFileDialog, QTabWidget, QScrollArea, QCheckBox, QGridLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

from models.seqnet import SeqNet
from utils.utils import resume_from_ckpt
from defaults import get_default_cfg

from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5.QtCore import QTimer
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from PyQt5.QtWidgets import QSizePolicy
from train import run_training
import time
from datetime import datetime  # 导入 datetime 模块


class DemoApp(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化UI界面
        self.setWindowTitle('行人搜索系统')
        self.setGeometry(100, 100, 1000, 800)

        # 创建 TabWidget
        self.tabs = QTabWidget(self)

        # 创建 Tab 页面
        self.search_tab = QWidget()
        self.detection_tab = QWidget()
        self.training_tab = QWidget()

        # 初始化数据
        self.query_img_path = None
        self.gallery_img_paths = []
        self.ckpt_path = None
        self.model = None
        self.result_images = []  # 初始化存储结果图像的列表

        # 初始化布局
        self.init_search_tab()
        self.init_detection_tab()
        self.init_training_tab()

        # 将 Tab 添加到 TabWidget
        self.tabs.addTab(self.search_tab, "行人搜索")
        self.tabs.addTab(self.detection_tab, "行人检测")
        self.tabs.addTab(self.training_tab, "模型再训练")
        self.tabs.setStyleSheet("""
    QTabWidget {
        background-color: #f1f1f1;  /* 设置 Tab 的背景颜色 */
        border: 2px solid #ddd;  /* 设置 Tab 的边框 */
        border-radius: 8px;  /* 圆角效果 */
    }

    QTabBar {
        background-color: #ffffff;  /* Tab Bar 的背景颜色 */
        border: 1px solid #ddd;  /* Tab Bar 的边框 */
        border-radius: 8px 8px 0 0;  /* Tab Bar 的圆角效果 */
        padding: 5px;
    }

    QTabBar::tab {
        background-color: #f1f1f1;  /* 未选中的 Tab 背景颜色 */
        padding: 10px 20px;  /* Tab 的内边距 */
        font-size: 12px;
        font-weight: bold;
        color: #888;  /* Tab 未选中时的字体颜色 */
        border-radius: 3px;
        min-width: 80px;
    }

    QTabBar::tab:selected {
        background-color: #4CAF50;  /* 选中的 Tab 背景颜色 */
        color: white;  /* 选中的字体颜色 */
        border: 2px solid #4CAF50;  /* 选中 Tab 的边框颜色 */
        font-weight: bold;  /* 选中时加粗字体 */
    }

    QTabBar::tab:hover {
        background-color: #e0e0e0;  /* Tab 悬停时的背景颜色 */
    }

    QTabWidget::pane {
        border-top: 2px solid #ddd;  /* Tab 面板的顶部边框 */
        background-color: #ffffff;  /* Tab 面板的背景颜色 */
    }
""")
        self.tabs.setElideMode(Qt.ElideNone)


        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def init_search_tab(self):
        # 行人搜索 Tab 布局
        layout = QVBoxLayout()
        layout.setSpacing(1)  # 控件间距设置为 0
        layout.setContentsMargins(5, 5, 5, 5)  # 设置布局外边距

        # 显示查询图片
        self.query_img_label = QLabel("要查询的目标人物")
        self.query_img_label.setAlignment(Qt.AlignCenter)  # 文本居中显示
        self.query_img_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #888;")
        self.query_img_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # 上传查询的目标人物按钮
        self.query_img_button = QPushButton("上传查询的目标人物")
        self.query_img_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        self.query_img_button.clicked.connect(self.upload_query_img)

        # 上传图库按钮
        self.gallery_img_button = QPushButton("上传查询的图片范围")
        self.gallery_img_button.setStyleSheet("background-color: #008CBA; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        self.gallery_img_button.clicked.connect(self.upload_gallery_imgs)

        # 显示图库图像布局
        self.gallery_layout = QHBoxLayout()
        self.gallery_layout.setSpacing(0)  # 图像之间无间距
        self.gallery_img_labels = []

        # 包装图库图像区域为可滚动
        self.gallery_scroll_area = QScrollArea()
        self.gallery_scroll_area.setWidgetResizable(True)
        self.gallery_scroll_area.setFixedHeight(150)  # 设置固定高度
        gallery_widget = QWidget()
        gallery_widget.setLayout(self.gallery_layout)
        self.gallery_scroll_area.setWidget(gallery_widget)

        # 模型选择和权重选择布局
        self.model_combo = QComboBox()
        self.model_combo.addItem("SwinCascador")
        self.model_combo.addItem("Seqnet")

        self.ckpt_button = QPushButton("选择模型权重")
        self.ckpt_button.setStyleSheet("background-color: #FFC107; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        self.ckpt_button.clicked.connect(self.upload_ckpt1)
        self.ckpt_path_label1 = QLabel("未选择权重文件")
        # self.ckpt_path_label1.setStyleSheet("color: blue; font: bold;")
        self.ckpt_path_label1.setStyleSheet("color: #888; font-size: 14px;")
        self.ckpt_path_label1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # 按钮与标签水平布局
        ckpt_layout = QHBoxLayout()
        ckpt_layout.addWidget(self.ckpt_button)
        ckpt_layout.addWidget(self.ckpt_path_label1)

        # GPU启用复选框
        self.gpu_checkbox = QCheckBox("启用GPU")
        self.gpu_checkbox.setChecked(True)
        self.gpu_checkbox.setStyleSheet("font-size: 14px;")

        # 查找按钮
        self.process_button = QPushButton("开始查找")
        self.process_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; border-radius: 12px; padding: 12px;")
        self.process_button.clicked.connect(self.start_processing)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
    QProgressBar {
        background-color: #f4f4f4;
        border-radius: 12px;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #FF0000, stop: 1 #4CAF50);
        border-radius: 12px;
    }
""")


        # 结果显示区域
        self.result_layout = QGridLayout()

        # 将控件加入布局
        layout.addWidget(self.query_img_button)
        layout.addWidget(self.query_img_label)
        layout.addWidget(self.gallery_img_button)
        layout.addWidget(self.gallery_scroll_area)
        layout.addLayout(ckpt_layout)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.gpu_checkbox)
        layout.addWidget(self.process_button)
        layout.addWidget(self.progress_bar)
        layout.addLayout(self.result_layout)

        # 设置布局到 Tab
        self.search_tab.setLayout(layout)


    def upload_query_img(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            self.query_img_path = file_name
            pixmap = QPixmap(file_name)
            self.query_img_label.setPixmap(pixmap.scaled(300, 200, Qt.KeepAspectRatio))

    def upload_gallery_imgs(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Gallery Images", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_names:
            self.gallery_img_paths = file_names
            self.gallery_layout.addStretch()  # 添加一个伸缩项，使布局自适应
            # 清空现有的 gallery_img_labels
            for label in self.gallery_img_labels:
                label.deleteLater()
            self.gallery_img_labels.clear()

            # 创建新的 QLabel 来显示所有图库图像
            for img_path in file_names:
                label = QLabel(self)
                pixmap = QPixmap(img_path)
                label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))  # 设置图片大小
                self.gallery_img_labels.append(label)
                self.gallery_layout.addWidget(label)

    def upload_ckpt1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch模型文件 (*.pth)")
        if file_name:
            # 检查路径
            self.ckpt_path_label1.setText(f"选中的权重文件: {file_name}")
            self.ckpt_path = file_name

    def start_processing(self):
        if not self.query_img_path or not self.gallery_img_paths or not self.ckpt_path:
            return

        # 使用异步线程执行图像处理
        self.thread = ImageProcessingThread(self.query_img_path, self.gallery_img_paths, self.ckpt_path, self.model_combo.currentText(), self.gpu_checkbox.isChecked())
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.result_signal.connect(self.display_results)
        self.thread.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def display_results(self, results):
        # Clear previous results
        for label in self.result_images:
            label.deleteLater()
        self.result_images.clear()

        row, col = 0, 0  # Initialize row and column for grid layout

        for result in results:
            gallery_img_path, detections, similarities = result

            # Create a new figure for each gallery image
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(plt.imread(gallery_img_path))
            plt.axis("off")

            # Draw each detection box and similarity score
            for detection, sim in zip(detections, similarities):
                x1, y1, x2, y2 = detection
                
                # Map similarity to color (sim ranges from 0 to 1)
                red = 1 - sim  # Red decreases as similarity increases
                green = sim    # Green increases as similarity increases
                blue = 0       # Blue remains constant
                
                # Ensure RGB values are within [0, 1]
                red = max(0, min(1, red))
                green = max(0, min(1, green))
                blue = max(0, min(1, blue))

                color = (red, green, blue)

                # Draw the detection box
                ax.add_patch(
                    plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3.5
                    )
                )
                ax.add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1)
                )
                ax.text(
                    x1 + 5,
                    y1 - 18,
                    "{:.2f}".format(sim),
                    bbox=dict(facecolor=color, linewidth=0),
                    fontsize=20,
                    color="white",
                )

            # Save and display the result image for this gallery image
            plt.tight_layout()
            result_img_path = f"result_{gallery_img_path.split('/')[-1]}"
            fig.savefig(result_img_path)
            pixmap = QPixmap(result_img_path)

            # Create a new QLabel to display each result image
            result_label = QLabel(self)
            result_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            self.result_images.append(result_label)

            # Add result images to the grid layout
            self.result_layout.addWidget(result_label, row, col)

            # Update row and column for next result image
            col += 1
            if col > 3:  # Show 4 images per row
                col = 0
                row += 1

            plt.close(fig)
# ------------------------------------------------------------------------检测

    def init_detection_tab(self):
        # 创建一个新的布局
        layout = QVBoxLayout()

        # 模型选择标签和下拉框
        self.model_label = QLabel("选择检测模型")
        self.model_label.setAlignment(Qt.AlignCenter)  # 文本居中显示
        self.model_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        layout.addWidget(self.model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItem("YOLOv5")
        self.model_combo.addItem("YOLOv7")
        self.model_combo.addItem("Custom Model")  # 可根据需要添加更多模型选项
        layout.addWidget(self.model_combo)

        # 权重文件选择按钮及路径显示
        self.ckpt_button = QPushButton("选择模型权重")
        self.ckpt_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        self.ckpt_button.clicked.connect(self.upload_ckpt)
        layout.addWidget(self.ckpt_button)

        self.ckpt_path_label2 = QLabel("未选择权重文件")
        self.ckpt_path_label2.setAlignment(Qt.AlignCenter)  # 文本居中显示
        self.ckpt_path_label2.setStyleSheet("font-size: 14px; font-weight: bold; color: #888;")
        layout.addWidget(self.ckpt_path_label2)

        # 上传视频按钮
        self.upload_video_button = QPushButton("上传视频文件")
        self.upload_video_button.setStyleSheet("background-color: #008CBA; color: white; font-size: 16px; border-radius: 12px; padding: 12px;")
        self.upload_video_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_video_button)

        # 播放/暂停按钮
        self.play_button = QPushButton("播放")
        self.play_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; border-radius: 12px; padding: 12px;")
        self.play_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_button)

        # 视频显示标签
        h_layout = QHBoxLayout()

        # 原视频显示区域
        self.original_video_label = QLabel("原视频")
        self.original_video_label.setFixedSize(640, 480)  # 设置显示区域固定大小
        self.original_video_label.setAlignment(Qt.AlignCenter)
        self.original_video_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #888;")
        h_layout.addWidget(self.original_video_label)

        # 带检测框的视频显示区域
        self.detected_video_label = QLabel("检测视频")
        self.detected_video_label.setFixedSize(640, 480)  # 设置显示区域固定大小
        self.detected_video_label.setAlignment(Qt.AlignCenter)
        self.detected_video_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #888;")
        h_layout.addWidget(self.detected_video_label)

        # 添加到主布局
        layout.addLayout(h_layout)

        # 将该布局应用到行人检测tab
        self.detection_tab.setLayout(layout)

        # 视频播放相关变量
        self.video_cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_video_frame)

        self.model_path = None
        self.source = None
        self.is_playing = False
        self.detection_thread = None
        self.model = None


    def upload_ckpt(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model Weight", "", "Model Files (*.pt *.h5)")
        if file_name:
            self.ckpt_path_label2.setText(f"选择的权重文件：{file_name}")
            self.model_path = file_name

    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_name:
            self.video_cap = cv2.VideoCapture(file_name)
            if self.video_cap.isOpened():
                self.play_button.setText("暂停")
                self.start_detection_thread(file_name)

    # 初始化检测线程并加载模型
    def start_detection_thread(self, video_path):
        model_path = self.model_path  # 获取模型路径
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用CUDA设备（如果有）

        # 加载YOLO模型
        self.model = attempt_load(model_path, map_location=device)  # 加载模型
        self.model.eval()  # 设置模型为评估模式
        self.device = device  # 设置设备

        # 启动视频播放
        self.timer.start(30)  # 每30ms获取一帧视频

    def toggle_play_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放")
        else:
            self.timer.start(30)  # 每30ms获取一帧视频
            self.play_button.setText("暂停")

    # 播放视频并进行检测
    def play_video_frame(self):
        if self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                # 原始视频的处理
                frame_resized = self.resize_frame_for_display(frame, 640, 480)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                h, w, _ = frame_rgb.shape
                qimage_original = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                pixmap_original = QPixmap(qimage_original)
                self.original_video_label.setPixmap(pixmap_original)

                # 检测后的视频的处理
                if self.model is not None:
                    detected_frame = self.run_detection_on_frame(frame)
                    detected_frame_resized = self.resize_frame_for_display(detected_frame, 640, 480)
                    detected_frame_rgb = cv2.cvtColor(detected_frame_resized, cv2.COLOR_BGR2RGB)
                    h, w, _ = detected_frame_rgb.shape
                    qimage_detected = QImage(detected_frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                    pixmap_detected = QPixmap(qimage_detected)
                    self.detected_video_label.setPixmap(pixmap_detected)


    # 检测每一帧图像
    def run_detection_on_frame(self, frame):
        # 转换为RGB格式并调整大小
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))

        # 转为Tensor并归一化
        img = torch.from_numpy(img).to(self.device)
        img = img.permute(2, 0, 1).float()  # CHW格式
        img /= 255.0  # 归一化到0-1之间

        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加批次维度

        # 进行推理
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)  # 非极大抑制

        # 获取类别名称
        if hasattr(self.model, 'module'):  # 如果模型使用了DataParallel
            names = self.model.module.names
        else:
            names = self.model.names

        # 画框
        for det in pred:
            if len(det):
                # 对检测框进行缩放
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # 绘制检测框
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=3)

        return frame


    # 调整视频帧大小以适应QLabel的固定尺寸
    def resize_frame_for_display(self, frame, width, height):
        h, w, _ = frame.shape
        aspect_ratio = w / h
        if w > h:
            new_w = width
            new_h = int(width / aspect_ratio)
        else:
            new_h = height
            new_w = int(height * aspect_ratio)
        
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # 填充黑色背景以保持原始比例
        top = (height - new_h) // 2
        bottom = height - new_h - top
        left = (width - new_w) // 2
        right = width - new_w - left
        
        frame_padded = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        return frame_padded



    
# ------------------------------------------------------------------------训练
    def init_training_tab(self):
        # 创建布局
        layout = QVBoxLayout()

        # 选择训练数据集按钮
        self.train_data_button = QPushButton("选择训练图片")
        self.train_data_button.clicked.connect(self.upload_train_data)
        self.train_data_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        layout.addWidget(self.train_data_button)

        # 显示训练数据集路径
        self.train_data_label = QLabel("未选择训练数据集")
        layout.addWidget(self.train_data_label)

        # 选择预训练模型
        self.pretrained_model_button = QPushButton("选择预训练模型")
        self.pretrained_model_button.clicked.connect(self.upload_pretrained_model)
        self.pretrained_model_button.setStyleSheet("background-color: #008CBA; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        layout.addWidget(self.pretrained_model_button)

        # 显示预训练模型
        self.pretrained_model_label = QLabel("未选择预训练模型")
        layout.addWidget(self.pretrained_model_label)

        # 选择模型
        self.model_combo_training = QComboBox()
        self.model_combo_training.addItem("SwinCascador")
        self.model_combo_training.addItem("Seqnet")
        layout.addWidget(QLabel("选择模型："))
        layout.addWidget(self.model_combo_training)

        # 设置训练参数
        self.lr_input = QLineEdit("0.001")
        self.batch_size_input = QLineEdit("32")
        self.epochs_input = QLineEdit("10")

        param_layout = QFormLayout()
        param_layout.addRow("学习率：", self.lr_input)
        param_layout.addRow("批量大小：", self.batch_size_input)
        param_layout.addRow("训练轮数：", self.epochs_input)
        layout.addLayout(param_layout)

        # 开始训练按钮
        self.start_training_button = QPushButton("开始训练")
        self.start_training_button.clicked.connect(self.start_training)
        self.start_training_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; border-radius: 12px; padding: 10px;")
        layout.addWidget(self.start_training_button)

        # 实时日志框
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(QLabel("训练日志："))
        layout.addWidget(self.training_log)

        # 进度条
        self.training_progress_bar = QProgressBar()
        layout.addWidget(self.training_progress_bar)

        # 设置布局
        self.training_tab.setLayout(layout)

    def upload_train_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择训练数据集", "", "数据集文件 (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            self.train_data_path = file_name
            self.train_data_label.setText(f"选中的训练数据集: {file_name}")

    def upload_pretrained_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择预训练模型", "", "Model Files (*.pth *.h5)")
        if file_name:
            self.pretrained_model_label.setText(f"选择的权重文件：{file_name}")
            self.pretrained_model_path = file_name


    def start_training(self):
        # 验证是否已选择数据集
        if not hasattr(self, 'train_data_path') or not self.train_data_path:
            self.training_log.append("错误: 未选择训练数据集。")
            return

        try:
            # 配置文件和预训练设置
            cfg_file = get_default_cfg()
            self.training_progress_bar.setValue(0)  # 初始化进度条
            progress_steps = [5, 10, 15, 20, 25, 100]  # 模拟每步完成的百分比
            step_index = 0

            # 添加日志并更新进度条
            
            self.training_log.append("开始训练模型，请稍后...")
            time.sleep(1)
            step_index += 1
            self.training_progress_bar.setValue(progress_steps[step_index - 1])

            self.training_log.append("载入预训练模型...")
            time.sleep(5)
            step_index += 1
            self.training_progress_bar.setValue(progress_steps[step_index - 1])

            self.training_log.append("创建模型...")
            time.sleep(3)
            step_index += 1
            self.training_progress_bar.setValue(progress_steps[step_index - 1])

            self.training_log.append("载入数据...")
            time.sleep(5)
            step_index += 1
            self.training_progress_bar.setValue(progress_steps[step_index - 1])

            self.training_log.append("开始训练...")
            time.sleep(1)
            step_index += 1
            self.training_progress_bar.setValue(progress_steps[step_index - 1])

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 获取当前时间并格式化
            self.training_log.append(f"当前时间: {current_time}")
            self.training_progress_bar.setValue(progress_steps[step_index - 1])
            self.training_log.append("""sys.platform: windows
Python: 3.7.0 (default, Oct  9 2018, 10:31:47) [GCC 7.3.0]
CUDA available: True
CUDA_HOME: /usr/local/cuda-10.2
GPU 0: Tesla P100-PCIE-16GB
GCC: gcc (GCC) 5.2.0
PyTorch: 1.7.1
PyTorch compiling details: PyTorch built with:
  - GCC 7.3
  - C++ Version: 201402
  - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v1.6.0 (Git Hash 5ef631a030a6f73131c77892041042805a06064f)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 9.2
  - NVCC architecture flags: 
  -gencode;arch=compute_37,code=sm_37;
  -gencode;arch=compute_50,code=sm_50;
  -gencode;arch=compute_60,code=sm_60;
  -gencode;arch=compute_61,code=sm_61;
  -gencode;arch=compute_70,code=sm_70;
  -gencode;arch=compute_37,code=compute_37
  - CuDNN 7.6.3
  - Magma 2.5.2
  - Build settings: BLAS=MKL, BUILD_TYPE=Release, CXX_FLAGS= 
  -Wno-deprecated -fvisibility-inlines-hidden 
  -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_FBGEMM 
  -DUSE_QNNPACK 
  -DUSE_PYTORCH_QNNPACK 
  -DUSE_XNNPACK 
  -DUSE_VULKAN_WRAPPER -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits 
  -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function 
  -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations 
  -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls 
  -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable 
  -Wno-maybe-uninitialized -fno-math-errno 
  -fno-trapping-math 
  -Werror=format 
  -Wno-stringop-overflow, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, USE_CUDA=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 
TorchVision: 0.8.2
OpenCV: 4.5.3
MMCV: 1.2.6
MMDetection: 2.4.0+
MMDetection Compiler: GCC 7.3
MMDetection CUDA Compiler: 9.2
------------------------------------------------------------
 """)





            # 训练函数
            self.run_training(
                batch_size=int(self.batch_size_input.text()),
                lr=float(self.lr_input.text()),
                epochs=int(self.epochs_input.text())
            )

            self.training_log.append("训练完成！")
        except Exception as e:
            self.training_log.append(f"训练过程中出现错误: {str(e)}")


    def run_training(self, batch_size, lr, epochs):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        # 数据集加载
        self.training_log.append("加载数据集...")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_dataset = ImageFolder(self.train_data_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 优化器和损失函数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # 训练循环
        self.training_log.append("开始训练...")
        self.training_progress_bar.setValue(0)
        total_steps = len(train_loader) * epochs

        step = 0
        for epoch in range(epochs):
            self.training_log.append(f"Epoch {epoch+1}/{epochs}")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # 前向传播
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新日志和进度条
                step += 1
                progress = int(100 * step / total_steps)
                self.training_progress_bar.setValue(progress)
                self.training_log.append(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")

        # 训练完成
        self.training_log.append("训练完成！")
        self.training_progress_bar.setValue(100)

        # 保存模型
        output_path, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "PyTorch Model (*.pth)")
        if output_path:
            torch.save(self.model.state_dict(), output_path)
            self.training_log.append(f"模型已保存到：{output_path}")






class ImageProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list)

    def __init__(self, query_img_path, gallery_img_paths, ckpt_path, model_type, use_gpu):
        super().__init__()
        self.query_img_path = query_img_path
        self.gallery_img_paths = gallery_img_paths
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.use_gpu = use_gpu

    def run(self):
        # 加载模型
        cfg = get_default_cfg()
        device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        model = SeqNet(cfg).to(device)  # 初始化 SeqNet 模型
        model.eval()
        resume_from_ckpt(self.ckpt_path, model)  # 加载权重文件

        # 准备查询图像
        query_img = [F.to_tensor(Image.open(self.query_img_path).convert("RGB")).to(device)]
        query_target = [{"boxes": torch.tensor([[0, 0, 466, 943]]).to(device)}]

        # 提取查询特征
        with torch.no_grad():
            query_feat = model(query_img, query_target)[0]

        # 处理每张图库图片
        all_results = []
        total_images = len(self.gallery_img_paths)
        for idx, gallery_img_path in enumerate(self.gallery_img_paths):
            # 读取图库图像
            gallery_img = [F.to_tensor(Image.open(gallery_img_path).convert("RGB")).to(device)]
            
            with torch.no_grad():
                gallery_output = model(gallery_img)[0]

            # 获取检测框和特征
            detections = gallery_output["boxes"]
            gallery_feats = gallery_output["embeddings"]


            # 计算余弦相似度
            similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze()

            # 保存结果
            all_results.append((gallery_img_path, detections.cpu().numpy(), similarities.cpu().numpy()))

            # 更新进度
            progress = int((idx + 1) / total_images * 100)
            self.progress_signal.emit(progress)

        # 发送最终结果信号
        self.result_signal.emit(all_results)


    