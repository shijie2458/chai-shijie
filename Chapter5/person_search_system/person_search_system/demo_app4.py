import os
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
        self.ckpt_button.clicked.connect(self.upload_ckpt)

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

    def upload_ckpt(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "Checkpoint Files (*.pth)")
        if file_name:
            self.ckpt_path_label.setText(f"选中的权重文件: {file_name}")
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
        layout = QVBoxLayout()
        label = QLabel("行人检测功能尚未实现")
        layout.addWidget(label)

        # 选择图像按钮
        self.detect_img_button = QPushButton("上传检测图像")
        self.detect_img_button.clicked.connect(self.upload_detection_img)
        layout.addWidget(self.detect_img_button)

        # 显示检测图像的标签
        self.detect_img_label = QLabel("Detection Image")
        layout.addWidget(self.detect_img_label)

        self.detection_tab.setLayout(layout)

    def upload_detection_img(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Detection Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            self.detect_img_path = file_name
            pixmap = QPixmap(file_name)
            self.detect_img_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))


    
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

    