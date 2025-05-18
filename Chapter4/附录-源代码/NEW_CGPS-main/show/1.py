import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 替换为你的 .pkl 文件路径
file_path = "E:/person_search/CGPS-main/CGPS-main/show/probe_features.pkl"

# 打开并读取 .pkl 文件
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print("文件内容：", data)  # 输出内容

        # 如果数据是 NumPy 数组或可以转换为 NumPy 数组
        if isinstance(data, np.ndarray):
            features = data
        else:
            # 如果数据是其他格式，比如列表或字典，可以先转换成 NumPy 数组
            features = np.array(data)
        
        # 绘制热图
        plt.figure(figsize=(12, 8))  # 设置图像大小
        sns.heatmap(features, cmap='viridis', annot=False)  # annot=True 会显示每个单元格的数值
        plt.title('特征的热图可视化')
        plt.xlabel('特征维度')
        plt.ylabel('样本')
        plt.show()

except FileNotFoundError:
    print(f"文件未找到：{file_path}")
except pickle.UnpicklingError:
    print("文件格式可能有问题，无法解包。")
except Exception as e:
    print(f"发生错误：{e}")
