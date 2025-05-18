import cv2
import numpy as np

# 读取图像
image = cv2.imread("/public/home/G19830015/Group/CSJ/data/cuhk_sysu/cuhk_sysu/Image/SSM/s15752.jpg")

# 提取ROI坐标
x1, y1, x2, y2 = map(int, [216.0, 163.0, 281.0, 334.0])

# 在图像上绘制矩形框
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

save_path = "/public/home/G19830015/Group/CSJ/data/"

# 保存带有矩形框的图像
cv2.imwrite(save_path + "s15752_with_rectangle.jpg", image)
# 显示带有矩形框的图像
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
