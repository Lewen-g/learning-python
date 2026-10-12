# -*- coding: utf-8 -*-
"""
day5 最小版：手写两层神经网络。

目标：**这 30 行要能默写。**

砍掉了梯度检验、逻辑回归对照组、画图、防溢出细节——那些都是脚手架，
不是核心。真正的东西只有：前向 → 损失 → 反向 → 更新。
"""
import numpy as np

# ---------- 1. 造数据：两个交错的新月（一条直线分不开）----------
np.random.seed(0)
t = np.linspace(0, np.pi, 200)
X = np.vstack([np.c_[np.cos(t), np.sin(t)],
               np.c_[1 - np.cos(t), .5 - np.sin(t)]])
X = X + np.random.normal(0, .12, X.shape)
y = np.r_[np.zeros(200), np.ones(200)].reshape(-1, 1)

# ---------- 2. 初始化参数：2 → 16 → 1 ----------
W1, b1 = np.random.randn(2, 16) * .5, np.zeros(16)
W2, b2 = np.random.randn(16, 1) * .3, np.zeros(1)
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# ---------- 3. 训练 ----------
for epoch in range(3000):
    a1 = np.tanh(X @ W1 + b1)                # 前向：隐藏层
    a2 = sig(a1 @ W2 + b2)                   # 前向：输出概率
    loss = -np.mean(y * np.log(a2 + 1e-12) + (1 - y) * np.log(1 - a2 + 1e-12))

    dz2 = a2 - y                             # 反向：输出层误差 = 预测 - 真实
    dW2, db2 = a1.T @ dz2 / 400, dz2.mean(0)
    dz1 = (dz2 @ W2.T) * (1 - a1 ** 2)       # 反向：传回隐藏层，乘 tanh 导数
    dW1, db1 = X.T @ dz1 / 400, dz1.mean(0)

    W1, b1 = W1 - .5 * dW1, b1 - .5 * db1    # 梯度下降
    W2, b2 = W2 - .5 * dW2, b2 - .5 * db2

    if epoch % 1000 == 0:
        print(f"epoch {epoch:4d}   loss {loss:.4f}   acc {((a2 > .5) == y).mean():.3f}")

print(f"最终       loss {loss:.4f}   acc {((a2 > .5) == y).mean():.3f}")
