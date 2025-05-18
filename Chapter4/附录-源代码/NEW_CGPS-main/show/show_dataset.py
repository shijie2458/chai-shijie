import cv2

# 读取图像
image = cv2.imread("E:/person_search/CGPS-main/CGPS-main/show/bigdata/c1s3_009921.jpg")

# 提取ROI坐标 (左上角 x1, y1 和宽高)
x1, y1, width, height = 244, 364, 159, 334
# 计算右下角坐标
x2 = x1 + width
y2 = y1 + height

# 在图像上绘制矩形框
# cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)



# 保存带有矩形框的图像
save_path = "E:/person_search/CGPS-main/CGPS-main/show/bigdata/"
cv2.imwrite(save_path + "c1s3_009921_with_query.jpg", image)

# 显示带有矩形框的图像
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()





