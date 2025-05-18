import os
import cv2
import torch
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QScrollArea, 
    QGridLayout, QProgressBar, QComboBox, QCheckBox, QTabWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

from models.seqnet import SeqNet
from utils.utils import resume_from_ckpt
from defaults import get_default_cfg

from pedestrian_detection import DetectionThread 
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from PyQt5.QtCore import QTimer
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box

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

        # 初始化布局
        self.init_search_tab()
        self.init_detection_tab()
        self.init_training_tab()

        # 将 Tab 添加到 TabWidget
        self.tabs.addTab(self.search_tab, "行人搜索")
        self.tabs.addTab(self.detection_tab, "行人检测")
        self.tabs.addTab(self.training_tab, "模型训练")

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def init_search_tab(self):
        # 行人搜索 Tab 布局
        layout = QVBoxLayout()

        # 显示查询图片
        self.query_img_label = QLabel("Query Image")

        # 上传查询的目标人物按钮
        self.query_img_button = QPushButton("上传查询的目标人物")
        self.query_img_button.clicked.connect(self.upload_query_img)

        # 显示图库图像
        self.gallery_layout = QHBoxLayout()  # 使用 QHBoxLayout 来横向显示图库图像
        self.gallery_img_labels = []  # 用来存储所有gallery图像的QLabel

        # 结果显示标签
        self.result_layout = QGridLayout()  # 使用QGridLayout来显示多个结果图像
        self.result_images = []  # 用来存储所有结果图像的QLabel

        # 上传图库按钮
        self.gallery_img_button = QPushButton("上传查询的图片范围")
        self.gallery_img_button.clicked.connect(self.upload_gallery_imgs)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # 模型选择
        self.model_combo = QComboBox()
        self.model_combo.addItem("SeqNet")
        self.model_combo.addItem("ResNet")

        # 是否启用GPU
        self.gpu_checkbox = QCheckBox("启用GPU")
        self.gpu_checkbox.setChecked(True)

        # 按钮
        self.ckpt_button = QPushButton("选择模型权重")
        self.ckpt_button.clicked.connect(self.upload_ckpt1)

        # 显示权重路径的标签
        self.ckpt_path_label = QLabel("未选择权重文件")
        self.ckpt_path_label.setStyleSheet("color: blue; font: bold;")  # 设置样式使其更显眼

        self.process_button = QPushButton("开始查找")
        self.process_button.clicked.connect(self.start_processing)

        # 将按钮和标签放入 QHBoxLayout 水平布局
        ckpt_layout = QHBoxLayout()
        ckpt_layout.addWidget(self.ckpt_button)
        ckpt_layout.addWidget(self.ckpt_path_label)

        # 将其他控件加入布局
        layout.addWidget(self.query_img_button)
        layout.addWidget(self.query_img_label)
        layout.addWidget(self.gallery_img_button)

        # 将图库图片区域包装进QScrollArea以便支持滚动
        self.gallery_scroll_area = QScrollArea()
        self.gallery_scroll_area.setWidgetResizable(True)  # 设置为可调节大小
        gallery_widget = QWidget()
        gallery_widget.setLayout(self.gallery_layout)
        self.gallery_scroll_area.setWidget(gallery_widget)

        # 布局设置
        layout.addWidget(self.query_img_button)
        layout.addWidget(self.query_img_label)
        layout.addWidget(self.gallery_img_button)
        layout.addWidget(self.gallery_scroll_area)

        layout.addLayout(ckpt_layout)

        layout.addWidget(self.process_button)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.gpu_checkbox)
        layout.addWidget(self.progress_bar)
        layout.addLayout(self.result_layout)

        self.search_tab.setLayout(layout)

    def upload_query_img(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            self.query_img_path = file_name
            pixmap = QPixmap(file_name)
            self.query_img_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))

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
                label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))  # 设置图片大小
                self.gallery_img_labels.append(label)
                self.gallery_layout.addWidget(label)

    def upload_ckpt1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch模型文件 (*.pth)")
        if file_name:
            # 检查路径
            file_name = file_name.replace("\\", "/")  # 替换反斜杠为正斜杠，以避免路径格式问题
            self.ckpt_path_label.setText(file_name)

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
            result_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
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
        layout.addWidget(self.model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItem("YOLOv5")
        self.model_combo.addItem("YOLOv7")
        self.model_combo.addItem("Custom Model")  # 可根据需要添加更多模型选项
        layout.addWidget(self.model_combo)

        # 权重文件选择按钮及路径显示
        self.ckpt_button = QPushButton("选择模型权重")
        self.ckpt_button.clicked.connect(self.upload_ckpt)
        layout.addWidget(self.ckpt_button)

        self.ckpt_path_label = QLabel("未选择权重文件")
        layout.addWidget(self.ckpt_path_label)

        # 上传视频按钮
        self.upload_video_button = QPushButton("上传视频文件")
        self.upload_video_button.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_video_button)

        # 播放/暂停按钮
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_button)

        # 视频显示标签
        h_layout = QHBoxLayout()

        # 原视频显示区域
        self.original_video_label = QLabel("原视频")
        self.original_video_label.setFixedSize(640, 480)  # 设置显示区域固定大小
        self.original_video_label.setAlignment(Qt.AlignCenter)
        h_layout.addWidget(self.original_video_label)

        # 带检测框的视频显示区域
        self.detected_video_label = QLabel("检测视频")
        self.detected_video_label.setFixedSize(640, 480)  # 设置显示区域固定大小
        self.detected_video_label.setAlignment(Qt.AlignCenter)
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
            self.ckpt_path_label.setText(f"选择的权重文件：{file_name}")
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
        layout = QVBoxLayout()
        label = QLabel("模型训练功能尚未实现")
        layout.addWidget(label)

        # 选择训练数据集按钮
        self.train_data_button = QPushButton("选择训练数据集")
        self.train_data_button.clicked.connect(self.upload_train_data)
        layout.addWidget(self.train_data_button)

        # 显示训练数据集路径的标签
        self.train_data_label = QLabel("未选择训练数据集")
        layout.addWidget(self.train_data_label)

        self.training_tab.setLayout(layout)
    
    def upload_train_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Training Dataset", "", "Dataset Files (*.zip *.tar.gz)")
        if file_name:
            self.train_data_path = file_name
            self.train_data_label.setText(f"选中的训练数据集: {file_name}")


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
        # Load the model
        cfg = get_default_cfg()
        device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        if self.model_type == 'SeqNet':
            model = SeqNet(cfg).to(device)
        else:
            # Placeholder for other models like ResNet, can be added here
            pass
        model.eval()
        resume_from_ckpt(self.ckpt_path, model)

        # Prepare query image
        query_img = [F.to_tensor(Image.open(self.query_img_path).convert("RGB")).to(device)]
        query_target = [{"boxes": torch.tensor([[0, 0, 466, 943]]).to(device)}]
        
        with torch.no_grad():  # Disable gradient computation for inference
            query_feat = model(query_img, query_target)[0]

        # Process each gallery image
        all_results = []
        total_images = len(self.gallery_img_paths)
        for idx, gallery_img_path in enumerate(self.gallery_img_paths):
            gallery_img = [F.to_tensor(Image.open(gallery_img_path).convert("RGB")).to(device)]
            
            with torch.no_grad():  # Disable gradient computation for inference
                gallery_output = model(gallery_img)[0]
            
            detections = gallery_output["boxes"]
            gallery_feats = gallery_output["embeddings"]

            # Compute cosine similarities
            similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze()
            
            all_results.append((gallery_img_path, detections.cpu().numpy(), similarities.cpu().numpy()))

            # Update progress
            progress = int((idx + 1) / total_images * 100)
            self.progress_signal.emit(progress)

        self.result_signal.emit(all_results)

    