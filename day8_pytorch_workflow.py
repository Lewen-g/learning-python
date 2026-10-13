# -*- coding: utf-8 -*-
"""
day8：用 PyTorch 重做 day7

day7 是一个脚本从头跑到尾。真实项目不是这样 —— 它是分层结构：

    Dataset        管数据：怎么取第 i 个样本、一共多少个
    DataLoader     管批：怎么打包成小批、要不要打乱
    nn.Module      管模型：参数放哪、前向怎么写
    optimizer      管更新：用哪个优化器、学习率多少、要不要正则化
    训练循环        管流程：训练 → 验证 → 早停

**3DGS、SLAM、视觉的官方实现，全是这套骨架。**

数据和划分与 day7 完全一致，方便对照。
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 造数据（和 day7 逐字相同，保证可比）
# ============================================================
np.random.seed(0)


def make_moons(n, noise=0.15, seed=1):
    rng = np.random.RandomState(seed)
    n_up = n // 2
    t1 = rng.uniform(0, np.pi, n_up)
    t2 = rng.uniform(0, np.pi, n - n_up)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)],
                   np.c_[1 - np.cos(t2), .5 - np.sin(t2)]])
    X = X + rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n_up), np.ones(n - n_up)].reshape(-1, 1)
    return X, y


def split_dataset(X, y, n_train, n_val, seed=2):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    a, b = n_train, n_train + n_val
    return ((X[idx[:a]], y[idx[:a]]),
            (X[idx[a:b]], y[idx[a:b]]),
            (X[idx[b:]], y[idx[b:]]))


# ★ 测试集放大到 1000 个：150 个样本时，1 个样本 = 0.67%，
#   模型间的真实差异会被这种"离散噪声"完全淹没。评估指标越细，
#   过拟合造成的损失才越看得出来。
X_all, y_all = make_moons(1500, noise=0.15)
(Xtr, ytr), (Xva, yva), (Xte, yte) = split_dataset(X_all, y_all, 100, 400)

print("=" * 64)
print("数据划分")
print("=" * 64)
print(f"  训练集  {Xtr.shape[0]:4d} 个样本")
print(f"  验证集  {Xva.shape[0]:4d} 个样本")
print(f"  测试集  {Xte.shape[0]:4d} 个样本")


# ============================================================
# 2. ★ Dataset：告诉 PyTorch「数据长什么样」
# ============================================================
# 只要实现两个方法，任何数据都能喂给 PyTorch：
#     __len__      一共多少个样本
#     __getitem__  第 i 个样本是什么（返回 (特征, 标签)）
#
# 真实项目里这一步通常干的是：读图片、做数据增强、转成 tensor。
# 这里数据已经在内存里，所以只是包一层。


class MoonsDataset(Dataset):
    def __init__(self, X, y):
        # torch 只认 float32（numpy 默认是 float64，不转会报错）
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_ds = MoonsDataset(Xtr, ytr)
val_ds = MoonsDataset(Xva, yva)
test_ds = MoonsDataset(Xte, yte)

# ============================================================
# 3. ★ DataLoader：告诉 PyTorch「怎么取数据」
# ============================================================
# 它负责两件事：
#   1. 打包成小批（mini-batch）—— 每步只用一个批次的梯度，比全批量更快更稳
#   2. 每轮打乱顺序 —— 否则模型会学到"样本顺序"这种无关信息
#
# ★ 注意：验证集和测试集【不要打乱】，而且通常用更大的批。
BATCH = 32

print()
print("=" * 64)
print("DataLoader")
print("=" * 64)
_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
print(f"  批大小 {BATCH}  →  每个 epoch 有 {len(_loader)} 个批次")
print(f"  （day7 是全批量训练：1 个 epoch = 1 次梯度更新）")


# ============================================================
# 4. ★ nn.Module：告诉 PyTorch「模型长什么样」
# ============================================================
# 对比 day7：那边参数存在一个 dict 里，前向是个普通函数。
# 这里参数是【模块的属性】，PyTorch 会自动追踪它们，好交给优化器。
#
# ★ 千万不要自己调用 forward()，要写 model(x) ——
#   因为 __call__ 里还有 hooks、train/eval 模式等一堆机制。


class Net(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()                    # 必须先调父类构造
        self.fc1 = nn.Linear(2, hidden)       # = day7 的 W1, b1
        self.fc2 = nn.Linear(hidden, 1)       # = day7 的 W2, b2

    def forward(self, x):
        x = torch.tanh(self.fc1(x))           # 隐藏层
        return torch.sigmoid(self.fc2(x))     # 输出概率


def build_model(state, hidden=128):
    """根据一份 state_dict 造出一个模型。"""
    m = Net(hidden)
    m.load_state_dict(state)
    return m


# ============================================================
# 5. 评估函数
# ============================================================
# ★ @torch.no_grad() 有两个作用：
#     1. 不记录计算图 —— 省内存、更快
#     2. 防止误把验证集也算进梯度（写论文时的低级错误）
# ★ model.eval() 切换成"评估模式"（影响 Dropout / BatchNorm 等层的行为）


@torch.no_grad()
def evaluate(model, ds):
    model.eval()
    pred = model(ds.X)                        # 数据小，一次性算完，不用分批
    loss = nn.functional.binary_cross_entropy(pred, ds.y).item()
    acc = ((pred > 0.5).float() == ds.y).float().mean().item()
    return loss, acc


# ============================================================
# 6. 训练循环
# ============================================================
def train_model(train_ds, val_ds, hidden=128, weight_decay=0.0,
                lr=0.1, batch=32, epochs=1500, seed=0):
    # ★ 注意 lr 从 day7 的 0.5 降到了 0.1。
    #   原因：小批量（32 个样本）的梯度噪声比全批量（100 个样本）大得多，
    #   还用 0.5 的步长会不断"跨过头"，损失曲线剧烈震荡、模型来回乱跳。
    #   规律：批越小 → 学习率要越小。真实项目里 batch 和学习率要一起调。
    torch.manual_seed(seed)
    model = Net(hidden)

    loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    loss_fn = nn.BCELoss()

    # ★ L2 正则化在这里！day7 手写的 lam 惩罚项，PyTorch 只要一个参数
    #   注意：weight_decay 会作用到优化器收到【所有】参数上，包括偏置。
    #   day7 我们只惩罚了权重。想完全一致的话，要分组传参（下面注释那一版）。
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 严格版（只惩罚权重，不惩罚偏置）：
    # optimizer = torch.optim.SGD([
    #     {"params": [model.fc1.weight, model.fc2.weight], "weight_decay": weight_decay},
    #     {"params": [model.fc1.bias,   model.fc2.bias],   "weight_decay": 0.0},
    # ], lr=lr)

    hist = {"train": [], "val": []}
    best_val, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        # ---------- 训练：一批一批地走 ----------
        model.train()                          # ← 切到训练模式
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)                   # 前向（自动调用 forward）
            loss = loss_fn(pred, yb)           # 损失
            optimizer.zero_grad()              # 清空上一轮梯度（不清会累加）
            loss.backward()                    # ★ 反向传播
            optimizer.step()                   # ★ 更新参数
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)

        # ---------- 验证（只看，绝不参与梯度）----------
        val_loss, _ = evaluate(model, val_ds)
        hist["train"].append(epoch_loss)
        hist["val"].append(val_loss)

        # ---------- 早停：保存验证最好的那个模型 ----------
        # ★ 必须 deepcopy state_dict，不能直接存 model 引用
        #   （引用会跟着训练一直变，最后存的还是"最后一个模型"）
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

    # ★ 关键顺序：先把"训练到最后"的权重存下来，再恢复最佳模型。
    #   顺序反了就会丢掉最后那个模型（我第一版就写错在这里）。
    last_state = copy.deepcopy(model.state_dict())
    return last_state, best_state, hist, best_epoch, best_val


def accuracy_of(model, ds):
    return evaluate(model, ds)[1]


# ============================================================
# 7. 跑两个实验（和 day7 对照）
# ============================================================
HIDDEN = 128
EPOCHS = 12000

experiments = [("无正则化", 0.0), ("L2 正则化", 1e-3)]
results = {}

for label, wd in experiments:
    print()
    print("=" * 64)
    print(f"实验：{label}   (隐藏层 {HIDDEN}, {EPOCHS} epoch, weight_decay={wd})")
    print("=" * 64)
    last_state, best_state, hist, best_ep, best_val = train_model(
        train_ds, val_ds, hidden=HIDDEN, weight_decay=wd, epochs=EPOCHS)

    results[label] = dict(model_best=build_model(best_state, HIDDEN),
                          model_last=build_model(last_state, HIDDEN),
                          hist=hist, best_epoch=best_ep, best_val=best_val)

    print(f"  验证损失最低点：第 {best_ep} 轮（val loss = {best_val:.4f}）")
    print(f"  训练损失：最低点 {hist['train'][best_ep]:.4f}"
          f"  →  最后一轮 {hist['train'][-1]:.4f}")
    print(f"  验证损失：最低点 {hist['val'][best_ep]:.4f}"
          f"  →  最后一轮 {hist['val'][-1]:.4f}"
          f"   （{hist['val'][-1] - hist['val'][best_ep]:+.4f}）")

# ============================================================
# 8. 报告
# ============================================================
print()
print("=" * 64)
print("损失（灵敏的指标）")
print("=" * 64)
print(f"  {'':22s}{'训练损失':>10s}{'验证损失':>10s}{'测试损失':>10s}")
print("  " + "-" * 52)
for label, r in results.items():
    for tag, m in [("训练到最后", r["model_last"]), ("早停（最佳点）", r["model_best"])]:
        print(f"  {label + ' ' + tag:22s}"
              f"{evaluate(m, train_ds)[0]:10.4f} "
              f"{evaluate(m, val_ds)[0]:10.4f} "
              f"{evaluate(m, test_ds)[0]:10.4f}")
    print("  " + "-" * 52)

print()
print("=" * 64)
print("准确率（迟钝的指标）")
print("=" * 64)
print(f"  {'':22s}{'训练集':>10s}{'验证集':>10s}{'测试集':>10s}")
print("  " + "-" * 52)
for label, r in results.items():
    for tag, m in [("训练到最后", r["model_last"]), ("早停（最佳点）", r["model_best"])]:
        print(f"  {label + ' ' + tag:22s}"
              f"{accuracy_of(m, train_ds):9.3f} "
              f"{accuracy_of(m, val_ds):9.3f} "
              f"{accuracy_of(m, test_ds):9.3f}")
    print("  " + "-" * 52)

print("""
  两张表放一起看，能发现一件很重要的事：

  『损失』说：无正则化的模型过拟合了，验证损失从 0.0295 涨到 0.0460（+56%）
  『准确率』说：没啥区别，0.987 vs 0.989（1000 个测试样本里差 2 个）

  两个都对 —— 它们量的是不同的东西：
    损失  连续、灵敏，反映"模型有多自信/多准"
    准确率 离散、迟钝，只看"分类对不对"

  在这个二维问题上，决策边界即使过拟合也基本还是对的，所以准确率不怎么掉。
  但换成图像、三维重建那种高维问题，过拟合会把结果彻底搞坏。

  ★ 结论：评估模型不能只看准确率。损失（或 AUC、F1 之类）往往先暴露出问题。
""")

# ============================================================
# 9. 画图（和 day7 同样的布局，方便对比）
# ============================================================
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (13, 9),
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(2, 2)

pad = 0.4
gx, gy = np.meshgrid(
    np.linspace(X_all[:, 0].min() - pad, X_all[:, 0].max() + pad, 250),
    np.linspace(X_all[:, 1].min() - pad, X_all[:, 1].max() + pad, 250),
)
grid_t = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)


@torch.no_grad()
def draw_boundary(ax, model, title):
    model.eval()
    probs = model(grid_t).numpy().reshape(gx.shape)
    ax.contourf(gx, gy, probs, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.contour(gx, gy, probs, levels=[0.5], colors="black", linewidths=1.8)
    ax.scatter(Xtr[ytr.ravel() == 0, 0], Xtr[ytr.ravel() == 0, 1],
               s=10, c="tab:blue", edgecolors="k", linewidths=0.2)
    ax.scatter(Xtr[ytr.ravel() == 1, 0], Xtr[ytr.ravel() == 1, 1],
               s=10, c="tab:red", edgecolors="k", linewidths=0.2)
    ax.scatter(Xva[yva.ravel() == 0, 0], Xva[yva.ravel() == 0, 1],
               s=26, facecolors="none", edgecolors="blue", linewidths=0.9)
    ax.scatter(Xva[yva.ravel() == 1, 0], Xva[yva.ravel() == 1, 1],
               s=26, facecolors="none", edgecolors="red", linewidths=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")


for row, (label, r) in enumerate(results.items()):
    hist = r["hist"]

    ax = axes[row, 0]
    ax.plot(hist["train"], label="训练损失", color="tab:red", linewidth=1.6)
    ax.plot(hist["val"], label="验证损失", color="tab:blue", linewidth=1.6)
    ep = r["best_epoch"]
    ax.axvline(ep, color="green", linestyle="--", linewidth=1.2,
               label=f"验证最低点 (第 {ep} 轮)")
    ax.scatter([ep], [hist["val"][ep]], color="green", s=45, zorder=5)
    ax.set_title(f"{label}：训练 vs 验证损失（PyTorch）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("损失")
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=8)

    draw_boundary(axes[row, 1], r["model_last"],
                  f"{label}：训练到最后的决策边界\n"
                  f"测试集准确率 {accuracy_of(r['model_last'], test_ds):.1%}"
                  f"（早停为 {accuracy_of(r['model_best'], test_ds):.1%}）")

plt.tight_layout()
plt.savefig("day8_pytorch.png", dpi=120)
print("\n图片已保存：day8_pytorch.png")

# ============================================================
# 10. day7 手写版 ↔ day8 PyTorch 版
# ============================================================
print("""
============================================================
day7 手写                       day8 PyTorch
------------------------------------------------------------
params = {"W1": ..., ...}       class Net(nn.Module)
forward(X, p)                   model(x)  ← 自动调 forward
dx = a2 - y  （手推链式法则）    loss.backward()
p[k] -= lr * grads[k]           optimizer.step()
+ lam * W   （手写 L2）          weight_decay=lam  ← 一个参数
全批量：一次用 100 个样本         DataLoader(batch_size=32)  ← 小批量
验证：手动算                      @torch.no_grad() + model.eval()
保存最佳模型：复制 dict           copy.deepcopy(model.state_dict())
------------------------------------------------------------
左边 200 行 → 右边 80 行。
但你【知道】那 80 行背后在干什么 —— 这才是区别。
============================================================
""")
