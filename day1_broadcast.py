import numpy as np

# 1. 标量广播
a = np.array([1, 2, 3, 4, 5])
print("a + 100 =", a + 100)

# 2. 一维广播到二维
m = np.array([[1, 2, 3], [4, 5, 6]])    # (2, 3)
v = np.array([10, 20, 30])              # (3,)
print("\nm + v:")
print(m + v)      # v 广播到每一行

# 3. 列广播（列向量 + 行向量）
col = np.array([[1], [2], [3]])         # (3, 1)
row = np.array([[10, 20, 30, 40]])      # (1, 4)
print("\ncol 形状:", col.shape)
print("row 形状:", row.shape)
print("col + row 形状:", (col + row).shape)
print(col + row)

# 4. 标准化实例
np.random.seed(42)
X = np.random.randn(5, 3)              # 5个样本，3个特征
mean = X.mean(axis=0)                  # (3,)
std = X.std(axis=0)                    # (3,)
X_norm = (X - mean) / std              # 广播生效
print("\n原始 X:")
print(X)
print("\n标准化后（均值应接近0，标准差应接近1）:")
print(X_norm)
print("每列均值:", X_norm.mean(axis=0))
print("每列标准差:", X_norm.std(axis=0))