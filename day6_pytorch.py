# -*- coding: utf-8 -*-
"""
day6：用 PyTorch 写同一个网络，和 day5_minimal.py 逐行对照。

重点看这三行替掉了 day5 里一大段手写推导：

    loss.backward()          ← 反向传播（day5 手写的 6 行链式法则）
    optimizer.step()         ← 参数更新（day5 手写的 4 行 W -= lr*dW）
    optimizer.zero_grad()    ← 清空上一轮梯度

数据和初始权重都和 day5_minimal.py 完全一致，所以两边的损失曲线应该重合。
"""
import numpy as np
import torch
import torch.nn as nn

# ---------- 1. 数据和初始权重：与 day5_minimal.py 逐行相同 ----------
np.random.seed(0)
t = np.linspace(0, np.pi, 200)
X = np.vstack([np.c_[np.cos(t), np.sin(t)],
               np.c_[1 - np.cos(t), .5 - np.sin(t)]])
X = X + np.random.normal(0, .12, X.shape)
y = np.r_[np.zeros(200), np.ones(200)].reshape(-1, 1)

W1, b1 = np.random.randn(2, 16) * .5, np.zeros(16)
W2, b2 = np.random.randn(16, 1) * .3, np.zeros(1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# ---------- 2. 定义网络：2 → 16 → 1 ----------
# 这两行就替代了 day5 里的 forward() 和 init_params()
model = nn.Sequential(
    nn.Linear(2, 16), nn.Tanh(),
    nn.Linear(16, 1), nn.Sigmoid(),
)

# ★ 正常写代码时不需要下面这一段。这里只是把 numpy 算好的初始权重
#   搬进 PyTorch，好让两边的损失曲线能严格对比。
#   注意 nn.Linear 的 weight 形状是 (输出, 输入)，所以要转置。
with torch.no_grad():
    model[0].weight.copy_(torch.tensor(W1.T, dtype=torch.float32))
    model[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
    model[2].weight.copy_(torch.tensor(W2.T, dtype=torch.float32))
    model[2].bias.copy_(torch.tensor(b2, dtype=torch.float32))

loss_fn = nn.BCELoss()                                        # = day5 的 bce_loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)       # = day5 的 W -= lr*dW

# ---------- 3. 训练 ----------
for epoch in range(3000):
    pred = model(X)               # 前向传播
    loss = loss_fn(pred, y)       # 算损失

    optimizer.zero_grad()         # 清空上一轮梯度（不清会累加起来）
    loss.backward()               # ★ 反向传播：day5 手写的那一段
    optimizer.step()              # ★ 更新参数：day5 手写的那一段

    if epoch % 1000 == 0:
        acc = ((pred > .5).float() == y).float().mean().item()
        print(f"epoch {epoch:4d}   loss {loss.item():.4f}   acc {acc:.3f}")

acc = ((pred > .5).float() == y).float().mean().item()
print(f"最终       loss {loss.item():.4f}   acc {acc:.3f}")
