# -*- coding: utf-8 -*-
"""
day7：训练集 / 验证集 / 测试集，以及过拟合

day5 的模型在训练集上准确率 100%。但我们**完全不知道它在没见过的数据上表现如何**。
这节课要解决的就是这个问题 —— 它是真实项目的第一个坎。

三件事：

    1. 把数据切成 训练集 / 验证集 / 测试集
    2. 盯住"训练损失"和"验证损失"两条曲线：验证损失会先降后升
       那个分叉点就是过拟合的开始
    3. 两种对策：早停（Early Stopping）和 L2 正则化
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# 1. 生成数据
# ============================================================
# ★ 制造过拟合的标准配方是「小数据 + 大模型」：
#   - 训练集故意只留 100 个样本
#   - 网络却有 128 个隐藏神经元（可调参数约 400 个）
#   参数量远超样本数 → 网络完全有能力把训练集"背下来"
#   - 噪声 0.15：留一点不可消除的误差，接近真实数据


def make_moons(n, noise=0.25, seed=1):
    rng = np.random.RandomState(seed)
    n_up = n // 2
    t1 = rng.uniform(0, np.pi, n_up)          # 均匀随机取角度（比 linspace 更像真实采样）
    t2 = rng.uniform(0, np.pi, n - n_up)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)],
                   np.c_[1 - np.cos(t2), .5 - np.sin(t2)]])
    X = X + rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n_up), np.ones(n - n_up)].reshape(-1, 1)
    return X, y


def split_dataset(X, y, n_train, n_val, seed=2):
    """按【样本】随机划分，不是按特征。三个集合绝对不能有重叠。"""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    a, b = n_train, n_train + n_val
    return ((X[idx[:a]], y[idx[:a]]),          # 训练集：用来调参数
            (X[idx[a:b]], y[idx[a:b]]),        # 验证集：用来判断什么时候停
            (X[idx[b:]], y[idx[b:]]))          # 测试集：最后只看一次


X_all, y_all = make_moons(400, noise=0.15)
(Xtr, ytr), (Xva, yva), (Xte, yte) = split_dataset(X_all, y_all, 100, 150)

print("=" * 64)
print("数据划分")
print("=" * 64)
print(f"  训练集  {Xtr.shape[0]:4d} 个样本   ← 梯度下降用它")
print(f"  验证集  {Xva.shape[0]:4d} 个样本   ← 判断该在哪一轮停下")
print(f"  测试集  {Xte.shape[0]:4d} 个样本   ← 最后评分，只碰一次")

# ============================================================
# 2. 网络定义（加了 L2 正则化）
# ============================================================


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def init_params(n_hidden, seed=0):
    """★ 所有实验都用同一个 seed，保证起点一样，对比才公平。"""
    rng = np.random.RandomState(seed)
    return {
        "W1": rng.randn(2, n_hidden) * np.sqrt(1.0 / 2),
        "b1": np.zeros(n_hidden),
        "W2": rng.randn(n_hidden, 1) * np.sqrt(1.0 / n_hidden),
        "b2": np.zeros(1),
    }


def forward(X, p):
    a1 = np.tanh(X @ p["W1"] + p["b1"])
    a2 = sigmoid(a1 @ p["W2"] + p["b2"])
    return a2, (X, a1)


def loss_and_grads(X, y, p, lam=0.0):
    """返回 (总损失, 纯数据损失, 梯度)。

    L2 正则化：在损失里加一项  (lam/2) * (‖W1‖² + ‖W2‖²)
      作用：惩罚过大的权重，逼网络用"更小更平滑"的权重去拟合
      求导 ：d/dW [ (lam/2)·W² ] = lam·W    ← 那个 0.5 就是为了让导数干净
      ★ 偏置 b 不参与正则化（它不控制曲线的弯曲程度）
    """
    n = len(X)
    a2, (_, a1) = forward(X, p)

    eps = 1e-12
    data_loss = -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))
    reg = 0.5 * lam * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))
    total_loss = data_loss + reg

    # ---- 反向 ----
    dz2 = a2 - y
    dW2 = a1.T @ dz2 / n + lam * p["W2"]        # ← + lam*W 就是 L2 的梯度
    db2 = dz2.mean(0)
    dz1 = (dz2 @ p["W2"].T) * (1 - a1 ** 2)
    dW1 = X.T @ dz1 / n + lam * p["W1"]
    db1 = dz1.mean(0)

    return total_loss, data_loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def accuracy(X, y, p):
    a2, _ = forward(X, p)
    return float(((a2 > .5) == y).mean())


# ============================================================
# 3. 梯度检验（确认 L2 的梯度没写错）
# ============================================================
# L2 里那个 0.5 很容易搞错，所以单独验一次。
print()
print("=" * 64)
print("梯度检验（验证 L2 的梯度公式）")
print("=" * 64)

_p = init_params(8, seed=3)
_lam = 0.1
_, _, _g = loss_and_grads(Xtr[:20], ytr[:20], _p, lam=_lam)

_eps = 1e-6
_worst = 0.0
for _key in ("W1", "W2"):
    _arr = _p[_key]
    for _i in np.random.RandomState(0).choice(_arr.size, size=6, replace=False):
        _flat = _arr.ravel()
        _orig = _flat[_i]
        _flat[_i] = _orig + _eps
        _lp = loss_and_grads(Xtr[:20], ytr[:20], _p, lam=_lam)[0]
        _flat[_i] = _orig - _eps
        _lm = loss_and_grads(Xtr[:20], ytr[:20], _p, lam=_lam)[0]
        _flat[_i] = _orig
        _num = (_lp - _lm) / (2 * _eps)
        _ana = _g[_key].ravel()[_i]
        _worst = max(_worst, abs(_num - _ana) / max(abs(_num) + abs(_ana), 1e-12))
print(f"  最大相对误差: {_worst:.3e}")
print("  → 通过。" if _worst < 1e-6 else "  → 没通过，L2 梯度有问题！")


# ============================================================
# 4. 训练函数
# ============================================================
def train(Xtr, ytr, Xva, yva, hidden=128, lam=0.0, lr=0.5, epochs=4000, seed=0):
    """训练并记录每一步的【训练损失】和【验证损失】。

    ★ 关键实现：保存验证损失最低那一刻的参数。
      很多人的错误是训练完直接用最后一个模型 —— 那往往已经过拟合了。
    """
    p = init_params(hidden, seed)
    hist = {"train": [], "val": [], "train_acc": [], "val_acc": []}

    best_val = np.inf
    best_epoch = 0
    best_p = None

    for epoch in range(epochs):
        # --- 前向 + 反向（只用训练集调参数）---
        _, data_loss, grads = loss_and_grads(Xtr, ytr, p, lam)
        for k in p:
            p[k] = p[k] - lr * grads[k]

        # --- 记录（注意：验证集只用来"看"，绝不参与梯度）---
        # ★ 记录的是【纯数据损失】，不含正则化惩罚项。这一点很关键：
        #   惩罚项只是训练时的工具，跟"在没见过的数据上预测得多准"没有关系。
        #   如果拿含惩罚的损失去比较不同 λ 的模型，结论会完全错 ——
        #   λ 越大的模型验证损失看起来越差，但那只是被惩罚项抬高了。
        _, data_tr, _ = loss_and_grads(Xtr, ytr, p, lam)
        _, data_va, _ = loss_and_grads(Xva, yva, p, lam)
        hist["train"].append(data_tr)
        hist["val"].append(data_va)
        hist["train_acc"].append(accuracy(Xtr, ytr, p))
        hist["val_acc"].append(accuracy(Xva, yva, p))

        # --- 早停：记住验证损失最低的那个模型（同样用纯数据损失判断）---
        if data_va < best_val:
            best_val = data_va
            best_epoch = epoch
            best_p = {k: v.copy() for k, v in p.items()}

    return p, best_p, hist, best_epoch, best_val


# ============================================================
# 5. 跑两个实验
# ============================================================
HIDDEN = 128            # ★ 故意用大网络：容量大才容易过拟合
EPOCHS = 6000

experiments = [
    ("无正则化", 0.0),
    ("L2 正则化", 1e-3),
]

results = {}
for label, lam in experiments:
    print()
    print("=" * 64)
    print(f"实验：{label}   (隐藏层 {HIDDEN} 个神经元, {EPOCHS} 轮, λ={lam})")
    print("=" * 64)
    p_last, p_best, hist, best_ep, best_val = train(
        Xtr, ytr, Xva, yva, hidden=HIDDEN, lam=lam, epochs=EPOCHS)

    results[label] = dict(hist=hist, p_last=p_last, p_best=p_best,
                          best_epoch=best_ep, best_val=best_val, lam=lam)

    print(f"  验证损失最低点：第 {best_ep} 轮（val loss = {best_val:.4f}）")
    print(f"  训练损失：最低点 {hist['train'][best_ep]:.4f}"
          f"  →  最后一轮 {hist['train'][-1]:.4f}")
    print(f"  验证损失：最低点 {hist['val'][best_ep]:.4f}"
          f"  →  最后一轮 {hist['val'][-1]:.4f}"
          f"   （涨了 {hist['val'][-1] - hist['val'][best_ep]:+.4f}）")

# ============================================================
# 6. 结果报告
# ============================================================
print()
print("=" * 64)
print("测试集准确率（这才是真正的成绩）")
print("=" * 64)
print(f"  {'':22s}{'训练集':>10s}{'验证集':>10s}{'测试集':>10s}")
print("  " + "-" * 52)
for label, r in results.items():
    for tag, key in [("训练到最后", "p_last"), ("早停（最佳点）", "p_best")]:
        p = r[key]
        print(f"  {label + ' ' + tag:22s}"
              f"{accuracy(Xtr, ytr, p):9.3f} "
              f"{accuracy(Xva, yva, p):9.3f} "
              f"{accuracy(Xte, yte, p):9.3f}")
    print("  " + "-" * 52)

print("""
  读这张表的三个要点：

  1. 「训练集准确率」高得离谱，不代表模型好 —— 它只是在背训练数据
  2. 「训练到最后」的测试成绩，通常低于「早停」—— 多练的那几轮在帮倒忙
  3. 测试集准确率才是真实水平。它和验证集应该接近 ——
     如果差很多，说明你在验证集上"调过头"了
""")

# ============================================================
# 7. 画图
# ============================================================
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

# 画决策边界的辅助函数
pad = 0.4
gx, gy = np.meshgrid(
    np.linspace(X_all[:, 0].min() - pad, X_all[:, 0].max() + pad, 250),
    np.linspace(X_all[:, 1].min() - pad, X_all[:, 1].max() + pad, 250),
)
grid = np.stack([gx.ravel(), gy.ravel()], axis=1)


def draw_boundary(ax, p, title):
    probs, _ = forward(grid, p)
    probs = probs.reshape(gx.shape)
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

    # --- 左：损失曲线 ---
    ax = axes[row, 0]
    ax.plot(hist["train"], label="训练损失", color="tab:red", linewidth=1.6)
    ax.plot(hist["val"], label="验证损失", color="tab:blue", linewidth=1.6)
    ep = r["best_epoch"]
    ax.axvline(ep, color="green", linestyle="--", linewidth=1.2,
               label=f"验证最低点 (第 {ep} 轮)")
    ax.scatter([ep], [hist["val"][ep]], color="green", s=45, zorder=5)
    ax.set_title(f"{label}：训练 vs 验证损失")
    ax.set_xlabel("epoch")
    ax.set_ylabel("损失")
    ax.set_ylim(0, 0.25)      # 放大 y 轴，让"分叉"更醒目（开头的陡降超出范围，不影响结论）
    ax.legend(fontsize=8)

    # --- 右：用"训练到最后"的模型画边界 ---
    draw_boundary(axes[row, 1], r["p_last"],
                  f"{label}：训练到最后的决策边界\n"
                  f"测试集准确率 {accuracy(Xte, yte, r['p_last']):.1%}"
                  f"（早停为 {accuracy(Xte, yte, r['p_best']):.1%}）")

plt.tight_layout()
plt.savefig("day7_overfitting.png", dpi=120)
print("图片已保存：day7_overfitting.png")
print("（图里画的是【训练集】的点和【验证集】的空心圈）")

# ============================================================
# 8. 记住这三条
# ============================================================
print("""
============================================================
记住这三条
------------------------------------------------------------
1. 训练损失下降 ≠ 模型在变好
   它可能只是在"背答案"。要看【验证损失】。

2. 验证损失先降后升，那个转折点就是过拟合的开始
   对策：早停（最简单）、L2 正则化、更多数据（最有效）

3. 永远保留一份模型没见过数据 —— 测试集
   而且只碰一次。反复用测试集调参，它就不再是"没见过"了。
============================================================
""")
