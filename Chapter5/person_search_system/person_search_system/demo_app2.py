import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F
from models.seqnet import SeqNet
from utils.utils import resume_from_ckpt
from defaults import get_default_cfg

class DemoApp(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化UI界面
        self.setWindowTitle('Image Search Demo')
        self.setGeometry(100, 100, 1000, 800)

        # 布局
        self.layout = QVBoxLayout()

        # 显示查询图片
        self.query_img_label = ImageLabel(self)
        self.query_img_label.setAlignment(Qt.AlignCenter)
        # self.query_img_label = QLabel("Query Image")
        
        # 显示图库图片
        self.gallery_layout = QHBoxLayout()  # 使用 QHBoxLayout 来横向显示图库图像
        self.gallery_img_labels = []  # 用来存储所有gallery图像的QLabel
        
        # 结果显示标签
        self.result_layout = QGridLayout()  # 使用QGridLayout来显示多个结果图像
        self.result_images = []  # 用来存储所有结果图像的QLabel
        
        # 按钮
        self.query_img_button = QPushButton("Upload Query Image")
        self.gallery_img_button = QPushButton("Upload Gallery Images")
        self.ckpt_button = QPushButton("Upload Model Checkpoint")
        self.process_button = QPushButton("Process Images")
        
        # 连接信号与槽函数
        self.query_img_button.clicked.connect(self.upload_query_img)
        self.gallery_img_button.clicked.connect(self.upload_gallery_imgs)
        self.ckpt_button.clicked.connect(self.upload_ckpt)
        self.process_button.clicked.connect(self.process_images)

        # 布局设置
        self.layout.addWidget(self.query_img_button)
        self.layout.addWidget(self.query_img_label)
        self.layout.addWidget(self.gallery_img_button)
        
        # 将图库图片区域包装进QScrollArea以便支持滚动
        self.gallery_scroll_area = QScrollArea()
        self.gallery_scroll_area.setWidgetResizable(True)  # 设置为可调节大小
        gallery_widget = QWidget()
        gallery_widget.setLayout(self.gallery_layout)
        self.gallery_scroll_area.setWidget(gallery_widget)
        self.layout.addWidget(self.gallery_scroll_area)

        self.layout.addWidget(self.ckpt_button)
        self.layout.addWidget(self.process_button)

        # 结果显示区域
        self.layout.addLayout(self.result_layout)
        self.setLayout(self.layout)

        # 初始化数据
        self.query_img_path = None
        self.gallery_img_paths = []
        self.ckpt_path = None
        self.model = None

        # 存储框选区域的坐标
        self.selection_rect = None

    def upload_query_img(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Query Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_name:
            self.query_img_path = file_name
            pixmap = QPixmap(file_name)
            self.query_img_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))

            # 清除框选区域
            self.selection_rect = None
            self.query_img_label.set_selection_rect(self.selection_rect)  # 更新矩形框
            self.update()

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
            self.ckpt_path = file_name

    def mousePressEvent(self, event):
        """捕获鼠标按下事件，用于记录鼠标起始位置"""
        if self.query_img_path and event.button() == Qt.LeftButton:
            self.selection_rect = QRect(event.pos(), event.pos())
            self.query_img_label.set_selection_rect(self.selection_rect)  # 更新矩形框
            self.update()

    def mouseMoveEvent(self, event):
        """捕获鼠标移动事件，用于动态绘制矩形框"""
        if self.selection_rect and event.buttons() == Qt.LeftButton:
            self.selection_rect.setBottomRight(event.pos())
            self.query_img_label.set_selection_rect(self.selection_rect)  # 更新矩形框
            self.update()

    def mouseReleaseEvent(self, event):
        """捕获鼠标释放事件，用于确定最终的矩形框"""
        if self.selection_rect:
            self.selection_rect = QRect(self.selection_rect.topLeft(), event.pos())
            self.query_img_label.set_selection_rect(self.selection_rect)  # 更新矩形框
            self.update()
   

    def process_images(self):
        if not self.query_img_path or not self.gallery_img_paths or not self.ckpt_path:
            self.result_label.setText("Please upload all necessary files.")
            return

        if self.selection_rect is None:
            self.result_label.setText("Please select a target in the query image.")
            return

        # Load model
        cfg = get_default_cfg()
        device = torch.device(cfg.DEVICE)
        self.model = SeqNet(cfg).to(device)
        self.model.eval()
        resume_from_ckpt(self.ckpt_path, self.model)

        # Prepare query image and selected region
        query_img = Image.open(self.query_img_path).convert("RGB")
        query_img_cropped = query_img.crop((self.selection_rect.left(), self.selection_rect.top(), self.selection_rect.right(), self.selection_rect.bottom()))
        query_img_tensor = F.to_tensor(query_img_cropped).unsqueeze(0).to(device)
        query_target = [{"boxes": torch.tensor([[0, 0, 466, 943]]).to(device)}]  # This box is ignored, we rely on selection_rect
        
        with torch.no_grad():  # Disable gradient computation for inference
            query_feat = self.model(query_img_tensor, query_target)[0]

        # Process each gallery image
        all_results = []
        for gallery_img_path in self.gallery_img_paths:
            gallery_img = [F.to_tensor(Image.open(gallery_img_path).convert("RGB")).to(device)]
            
            with torch.no_grad():  # Disable gradient computation for inference
                gallery_output = self.model(gallery_img)[0]
            
            detections = gallery_output["boxes"]
            gallery_feats = gallery_output["embeddings"]

            # Compute cosine similarities
            similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze()
            
            all_results.append((gallery_img_path, detections.cpu().numpy(), similarities.cpu().numpy()))

        self.display_results(all_results)

    def display_results(self, all_results):
        row, col = 0, 0
        self.result_images = []  # 存储所有的结果图像标签

        # 遍历每个图库图像的匹配结果
        for result in all_results:
            gallery_img_path, detections, similarities = result
            
            # 创建 matplotlib 图像
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.imshow(plt.imread(gallery_img_path))
            plt.axis("off")

            for detection, sim in zip(detections, similarities):
                x1, y1, x2, y2 = detection
                
                # 将相似度映射到 RGB 颜色空间：相似度越高，颜色越绿，越低则越红
                red = 1 - sim  # Red decreases as similarity increases
                green = sim    # Green increases as similarity increases
                blue = 0       # Blue is always 0 for this red-green gradient

                # 确保 RGB 值在 [0, 1] 范围内
                red = max(0, min(1, red))
                green = max(0, min(1, green))
                blue = max(0, min(1, blue))

                color = (red, green, blue)  # RGB 颜色元组

                # 绘制框和相似度
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

            # 保存并显示结果图像
            plt.tight_layout()
            result_img_path = f"result_{gallery_img_path.split('/')[-1]}"
            fig.savefig(result_img_path)
            
            # 将结果图像加载到 QPixmap 中
            pixmap = QPixmap(result_img_path)

            # 创建新的 QLabel 来显示每个结果图像
            result_label = QLabel(self)
            result_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.result_images.append(result_label)

            # 使用 QGridLayout 显示结果图像
            self.result_layout.addWidget(result_label, row, col)

            # 更新行列，准备放置下一个结果图像
            col += 1
            if col > 3:  # 每行显示4个图像
                col = 0
                row += 1

            plt.close(fig)
            
class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_rect = None

    def set_selection_rect(self, rect):
        self.selection_rect = rect
        self.update()  # 更新界面，触发重新绘制

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.selection_rect:
            pixmap = self.pixmap()
            if pixmap:
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                label_rect = self.rect()

                # Compute the scale factor based on the QLabel and pixmap sizes
                scale_x = label_rect.width() / pixmap_width
                scale_y = label_rect.height() / pixmap_height

                # Find the correct scaling to preserve the aspect ratio
                scale = min(scale_x, scale_y)  # Keep the aspect ratio intact

                # Find the top-left corner of the pixmap within the QLabel
                offset_x = (label_rect.width() - pixmap_width * scale) / 2
                offset_y = (label_rect.height() - pixmap_height * scale) / 2

                # Scale the selection rectangle according to the QLabel size
                scaled_rect = QRect(
                    (self.selection_rect.left() - offset_x) / scale,
                    (self.selection_rect.top() - offset_y) / scale,
                    self.selection_rect.width() / scale,
                    self.selection_rect.height() / scale
                )

                # Draw the scaled rectangle
                painter = QPainter(self)
                painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red rectangle
                painter.drawRect(scaled_rect)
                painter.end()


