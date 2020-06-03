import numpy as np

# 只包含一个数字的张量叫做标量(scalar), 也称零维张量、标量张量
from keras.datasets import mnist
import matplotlib.pyplot as plt

x = np.array(12)
print(f"是否为标量: {x.ndim == 0}")

# 数字组成的数组叫做向量(vector), 也称一维张量
x = np.array([12, 3, 6, 14, 7])
print(f"是否为向量: {x.ndim == 1}")

# 向量组成的数组叫做矩阵(matrix), 也称二维张量
# 第一个轴上的元素叫做行(row), 第二个轴上的元素叫做列(column)
x = np.array([[7, 78, 2, 34, 0], [8, 79, 3, 35, 1], [9, 80, 4, 36, 2]])
print(f"是否为矩阵: {x.ndim == 2}")

# 张量的形状(shape)是一个整数元祖, 表示沿每个轴的维度大小
print(f"矩阵的形状为: {x.shape}")
row, column = x.shape
print(f"即矩阵有 {row} 行 {column} 列")

# 数据类型(dtype), 是张量中包含的数据类型, 例如float32、uint8, float64
print(f"此张量的数据类型为: {x.dtype}")

# 使用plot预览数据集
(train_images, train_lables), (test_images, test_lables) = mnist.load_data()
plt.imshow(train_images[4])
plt.show()
