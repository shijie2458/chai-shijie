from PIL import Image, ImageDraw

# 打开图片
image_path = "E:\\person_search\\CGPS-main\\CGPS-main\\show\\show_solider\\s14262.jpg"  # 替换为你的图片路径
output_path = "E:\\person_search\\CGPS-main\\CGPS-main\\show\\show_solider\\s14262_bw.jpg"  # 输出图片路径
img = Image.open(image_path)
img = img.convert("RGB")  # 确保图片为RGB模式

# 创建绘图对象
draw = ImageDraw.Draw(img)

# 输入框的坐标
# 示例：左上角(x1, y1)，右下角(x2, y2)
x1, y1, x2, y2 = 415, 204, 476, 369  # 替换为你的坐标

# 绘制矩形框
draw.rectangle([x1, y1, x2, y2], outline="black", width=3)  # outline设置框的颜色，width设置线条粗细

# 转换为黑白图像
img_bw = img.convert("L")  # 转换为灰度模式（黑白）

# 保存图片
img_bw.save(output_path)
print(f"图片保存成功：{output_path}")
