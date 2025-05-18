import cv2
import numpy as np

# 读取图像

image = cv2.imread("E:/person_search/CGPS-main/CGPS-main/show/bigdata/c1s4_069011.jpg")

# 提取ROI坐标
x1, y1, x2, y2 = map(int, [1754.0, 409.0, 1813.0, 559.0])

# 在图像上绘制矩形框
# cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  #绿
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  #红
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2) # 黄

save_path = "E:/person_search/CGPS-main/CGPS-main/show/bigdata/"

# 保存带有矩形框的图像
cv2.imwrite(save_path + "c1s4_069011_with_query.jpg", image)
# 显示带有矩形框的图像
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()