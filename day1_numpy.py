import numpy as np

# 1. 创建数组
a = np.array([1, 2, 3, 4, 5])
print("a =", a)
print("a 的形状:", a.shape)
print("a 的类型:", a.dtype)

# 2. 矩阵运算
m = np.array([[1, 2], [3, 4]])
print("\n矩阵 m:")
print(m)
print("m 的转置:")
print(m.T)
print("m 的行列式:", np.linalg.det(m))

# 3. 广播
b = np.array([10, 20])
print("\n广播: m + b =")
print(m + b)

# 4. 统计
print("\n统计:")
print("平均值:", a.mean())
print("标准差:", a.std())
print("最大值:", a.max())

# 5. 生成序列
t = np.linspace(0, 10, 100)   # 0到10之间100个点
print("\nlinspace 前5个:", t[:5])
print("linspace 后5个:", t[-5:])