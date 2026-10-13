# -*- coding: utf-8 -*-
"""
day10：一个完整的 CNN

结构（经典 LeNet 风格）：

    输入 1x24x24
      ↓ Conv(1→8, 3x3) + ReLU      卷积：提取局部特征
      ↓ MaxPool(2)                 池化：降采样，24→12
      ↓ Conv(8→16, 3x3) + ReLU     再提取更高级的特征
      ↓ MaxPool(2)                 再降采样，12→6
      ↓ Flatten                    拉平成一维
      ↓ Linear(576→1) + Sigmoid    全连接：用特征做判断
    输出 概率

数据是自己生成的：**随机位置、随机大小的圆 vs 方**。
不用下载任何数据集，而且"位置随机"能顺便验证 CNN 的平移不变性。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

SIZE = 24          # 图片边长
N_TOTAL = 1200

# ============================================================
# 1. 生成数据：圆 vs 方，位置和大小都随机
# ============================================================
print("=" * 62)
print("生成数据")
print("=" * 62)


def make_shapes(n, size=SIZE, noise=0.18, seed=1):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]

    for i in range(n):
        cy, cx = rng.randint(6, size - 6, 2)      # ★ 位置随机 → 考验平移不变性
        r = rng.randint(4, 7)                     # ★ 大小也随机
        if i % 2 == 0:                            # 圆
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        else:                                     # 方
            mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
        X[i, 0][mask] = 1.0
        y[i, 0] = i % 2

    X += rng.normal(0, noise, X.shape).astype(np.float32)   # 加噪声
    idx = rng.permutation(n)                                # 打乱（不然前半圆后半方）
    return X[idx], y[idx]


X, y = make_shapes(N_TOTAL)


def split(X, y, n_tr, n_va):
    a, b = n_tr, n_tr + n_va
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


(Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, 600, 300)
print(f"  训练 {Xtr.shape}   验证 {Xva.shape}   测试 {Xte.shape}")
print(f"  每张图是 1 通道 {SIZE}x{SIZE}（圆和方的位置、大小都随机）")


class ShapeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================================================
# 2. 完整 CNN
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)    # 24x24 -> 24x24
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)   # 12x12 -> 12x12
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 6 * 6, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))    # 24 -> 12
        x = self.pool(torch.relu(self.conv2(x)))    # 12 -> 6
        x = x.flatten(1)                            # (N,16,6,6) -> (N,576)
        return torch.sigmoid(self.fc(x))


model = SmallCNN()

print()
print("=" * 62)
print("每层有多少参数（= 训练时要优化的数字）")
print("=" * 62)
total = 0
for name, p in model.named_parameters():
    n = p.numel()
    total += n
    shape = tuple(p.shape)
    print(f"  {name:12s} {str(shape):22s} {n:6d} 个")
print(f"  {'合计':12s} {'':22s} {total:6d} 个")
print("""
  注意：conv1 只有 8*(1*3*3)=72 个权重，却要扫描整张图 ——
  这就是【参数共享】。换成同样感受野的全连接层，参数量会大几个数量级。
""")

# ============================================================
# 3. 训练
# ============================================================
EPOCHS = 30
BATCH = 32
LR = 1e-2


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32))
        t = torch.tensor(y, dtype=torch.float32)
        loss = nn.functional.binary_cross_entropy(p, t).item()
        acc = ((p > 0.5).float() == t).float().mean().item()
    return loss, acc


loader = DataLoader(ShapeDataset(Xtr, ytr), batch_size=BATCH, shuffle=True)
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)

hist = {"train": [], "val": [], "val_acc": []}
best_val, best_epoch, best_state = float("inf"), 0, None

print("=" * 62)
print(f"训练（Adam, lr={LR}, batch={BATCH}, {EPOCHS} epoch）")
print("=" * 62)

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(Xtr)

    val_loss, val_acc = evaluate(model, Xva, yva)
    hist["train"].append(ep_loss)
    hist["val"].append(val_loss)
    hist["val_acc"].append(val_acc)

    if val_loss < best_val:
        import copy
        best_val, best_epoch = val_loss, epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 5 == 0 or epoch == 1:
        tr_acc = evaluate(model, Xtr, ytr)[1]
        print(f"  epoch {epoch:3d}   训练损失 {ep_loss:.4f}   验证损失 {val_loss:.4f}"
              f"   训练准确率 {tr_acc:.3f}   验证准确率 {val_acc:.3f}")

model.load_state_dict(best_state)

print()
print("=" * 62)
print("最终成绩")
print("=" * 62)
print(f"  验证损失最低点：第 {best_epoch} 轮")
for tag, (Xs, ys) in [("训练集", (Xtr, ytr)), ("验证集", (Xva, yva)), ("测试集", (Xte, yte))]:
    _, a = evaluate(model, Xs, ys)
    print(f"  {tag}准确率：{a:.3f}")
print("""
  ★ 训练集和测试集准确率接近 → 没有明显过拟合
    而且数据是随机位置的，能分对说明 CNN 学到了
    『形状』而不是『位置』—— 这就是平移不变性。
""")

# ============================================================
# 4. 看看它到底学到了什么
# ============================================================
print("=" * 62)
print("第一层学到的 8 个卷积核")
print("=" * 62)
kernels = model.conv1.weight.data.numpy()      # (8, 1, 3, 3)
for i in range(8):
    k = kernels[i, 0]
    print(f"  核 {i}: " + "  ".join(f"{v:+.2f}" for v in k.ravel()))

print("""
  ★ 这些数字就是 day9 里你手写的那种卷积核 ——
    只不过 Sobel 是【人设计】的，这 8 个是【梯度下降学出来】的。
    它们检测什么，人看不懂，但网络自己知道有用。
""")

# ============================================================
# 5. 画图
# ============================================================
plt.rcParams.update({
    "font.size": 9,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 8, height_ratios=[1.1, 1, 1], hspace=0.45, wspace=0.25)

# --- 第一行左：样本图 ---
ax = fig.add_subplot(gs[0, 0:2])
# 数据已经打乱过，所以按标签挑，保证左边两张是圆、右边两张是方
idx_circle = np.where(ytr.ravel() == 0)[0][:2]
idx_square = np.where(ytr.ravel() == 1)[0][:2]
strip = np.concatenate([Xtr[i, 0] for i in list(idx_circle) + list(idx_square)], axis=1)
ax.imshow(strip, cmap="gray")
ax.set_title("训练样本：左两张是圆，右两张是方\n位置、大小都随机，还加了噪声", fontsize=9)
ax.axis("off")

# --- 第一行右：损失曲线 ---
ax = fig.add_subplot(gs[0, 2:8])
ax.plot(hist["train"], label="训练损失", color="tab:red")
ax.plot(hist["val"], label="验证损失", color="tab:blue")
ax.axvline(best_epoch - 1, color="green", linestyle="--", linewidth=1.2,
           label=f"验证最低点 (第 {best_epoch} 轮)")
ax.set_xlabel("epoch")
ax.set_ylabel("损失")
ax.set_title("训练曲线", fontsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# --- 第二行：8 个卷积核 ---
for i in range(8):
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(kernels[i, 0], cmap="RdBu_r")
    ax.set_title(f"核{i}", fontsize=8)
    ax.axis("off")
fig.text(0.02, 0.60, "第一层\n卷积核\n(3x3)", fontsize=10, va="center")

# --- 第三行：一张图经过第一层后的 8 个特征图 ---
model.eval()
with torch.no_grad():
    fmap = torch.relu(model.conv1(torch.tensor(Xtr[1:2], dtype=torch.float32)))[0].numpy()
for i in range(8):
    ax = fig.add_subplot(gs[2, i])
    ax.imshow(fmap[i], cmap="viridis")
    ax.set_title(f"特征图{i}", fontsize=8)
    ax.axis("off")
fig.text(0.02, 0.22, "同一张图\n的 8 个\n特征图", fontsize=10, va="center")

plt.savefig("day10_cnn.png", dpi=120, bbox_inches="tight")
print("图片已保存：day10_cnn.png")

print("""
============================================================
一张图看懂完整 CNN 的数据流
------------------------------------------------------------
  原图 (1x24x24)
    ↓ 8 个 3x3 卷积核           ← 参数 72 个（学出来的）
  8 个特征图 (8x24x24)          ← 每个核"看到"一种局部模式
    ↓ ReLU + MaxPool(2)
  8x12x12
    ↓ 16 个 3x3x8 卷积核        ← 参数 1152 个
  16x12x12
    ↓ ReLU + MaxPool(2)
  16x6x6
    ↓ Flatten
  576 维向量
    ↓ Linear(576→1)             ← 参数 577 个
  1 个概率

  【卷积层】负责提取特征，【全连接层】负责用特征判断。
  所有数字都是训练出来的 —— 包括那 8 个"看不懂"的卷积核。
============================================================
""")
