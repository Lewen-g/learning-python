# -*- coding: utf-8 -*-
"""
day5：用 NumPy 从零手写一个两层神经网络

目的不是"会用神经网络"，而是把这条链路彻底打通：

    前向传播 → 算损失 → 反向传播（链式法则）→ 梯度下降 → 更新参数

只用 numpy，不调任何深度学习框架。跑通之后再去看 PyTorch，你会发现
框架替你做的，恰恰就是这里手写的这几步。

网络结构（二分类）：

    输入 x (N, 2)
      │
      ├─ z1 = x @ W1 + b1        (N, H)
      ├─ a1 = tanh(z1)           ← 隐藏层：非线性在这里被引入
      ├─ z2 = a1 @ W2 + b2       (N, 1)
      └─ a2 = sigmoid(z2)        ← 输出"属于类别 1 的概率"

    损失 L = 二元交叉熵 BCE(a2, y)
"""
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 生成数据：两个交错的新月
# ============================================================
# 为什么不用线性可分的数据？因为一条直线就能分开的数据，
# 根本不需要隐藏层 —— 逻辑回归就够了。
# 必须用"一条直线分不开"的数据，才看得出隐藏层的价值。
np.random.seed(42)

N_PER_CLASS = 200


def make_two_moons(n, noise=0.12):
    """生成两个交错的新月，形状有点像太极图。"""
    t = np.linspace(0, np.pi, n)
    # 上半月：单位圆的上半部分
    x1 = np.stack([np.cos(t), np.sin(t)], axis=1)
    # 下半月：水平翻转 + 右移 + 下移，让它和上半月咬合
    x2 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.vstack([x1, x2])
    X = X + np.random.normal(0, noise, X.shape)   # 加点噪声，更像真实数据
    y = np.concatenate([np.zeros(n), np.ones(n)])
    return X, y


X, y = make_two_moons(N_PER_CLASS)
y = y.reshape(-1, 1)                     # (N, 1)

# 打乱顺序。不打乱的话前一半全是类别 0，训练会很不稳定。
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

print("=" * 60)
print(f"数据  X {X.shape}   y {y.shape}")
print(f"类别 0 有 {(y == 0).sum()} 个，类别 1 有 {(y == 1).sum()} 个")
print("=" * 60)

# ============================================================
# 2. 初始化参数
# ============================================================
H = 16          # 隐藏层神经元个数


def init_params(hidden=H, seed=0):
    rng = np.random.RandomState(seed)

    # ★ 绝对不能用全 0 初始化！
    #   如果所有隐藏神经元的初始权重相同，它们算出的梯度也完全相同，
    #   更新后依然相同 —— "16 个神经元"会退化成一个。
    #   随机初始化的作用就是打破这种对称性。
    W1 = rng.randn(2, hidden) * np.sqrt(1.0 / 2)
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden, 1) * np.sqrt(1.0 / hidden)
    b2 = np.zeros(1)
    return W1, b1, W2, b2


# ============================================================
# 3. 前向传播
# ============================================================
def sigmoid(z):
    z = np.clip(z, -500, 500)        # 防止 exp 溢出
    return 1.0 / (1.0 + np.exp(-z))


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1                 # (N, H)  线性变换
    a1 = np.tanh(z1)                 # (N, H)  ★非线性激活
    z2 = a1 @ W2 + b2                # (N, 1)
    a2 = sigmoid(z2)                 # (N, 1)  概率
    # 把中间结果缓存下来，反向传播要用
    cache = (X, z1, a1, z2, a2)
    return a2, cache


# ============================================================
# 4. 损失函数：二元交叉熵
# ============================================================
def bce_loss(a2, y):
    # 交叉熵衡量"预测分布和真实分布的差距"，比均方误差更适合分类
    eps = 1e-12                      # 防止 log(0)
    return -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))


# ============================================================
# 5. 反向传播 ★ 整个脚本的核心 ★
# ============================================================
def backward(cache, y, W2):
    """从损失出发，用链式法则一层层往回求梯度。"""
    X, z1, a1, z2, a2 = cache
    N = X.shape[0]

    # ---------- 输出层 ----------
    # ★ 这里有个优美的化简：
    #   dL/dz2 = dL/da2 · da2/dz2
    #          = [ -(y/a2) + (1-y)/(1-a2) ] · a2(1-a2)
    #          = a2 - y
    #   交叉熵配 sigmoid，导数正好是"预测值减真实值"。
    #   这不是巧合，是这两个函数被设计成一对的结果。
    dz2 = a2 - y                     # (N, 1)

    dW2 = a1.T @ dz2 / N             # (H, 1)  = 隐藏层输出^T · 输出层误差
    db2 = dz2.mean(axis=0)           # (1,)

    # ---------- 隐藏层：误差往回传 ----------
    da1 = dz2 @ W2.T                 # (N, H)  输出层的误差按权重比例"分发"回来
    dz1 = da1 * (1 - a1 ** 2)        # (N, H)  ★乘以 tanh 的导数 1 - tanh²

    dW1 = X.T @ dz1 / N              # (2, H)
    db1 = dz1.mean(axis=0)           # (H,)

    return dW1, db1, dW2, db2


# ============================================================
# 6. 梯度检验：证明上面那些公式确实是对的
# ============================================================
# 导数的定义就是  (f(w+ε) - f(w-ε)) / (2ε)。
# 用这个"数值梯度"去对比反向传播算出的"解析梯度"，
# 两者接近就说明公式没写错。这是真实工程里调试自定义层的标准手段。
def pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def unpack(theta, hidden):
    i = 0
    W1 = theta[i:i + 2 * hidden].reshape(2, hidden);  i += 2 * hidden
    b1 = theta[i:i + hidden];                         i += hidden
    W2 = theta[i:i + hidden].reshape(hidden, 1);      i += hidden
    b2 = theta[i:i + 1]
    return W1, b1, W2, b2


def loss_of_theta(theta, X, y, hidden):
    W1, b1, W2, b2 = unpack(theta, hidden)
    a2, _ = forward(X, W1, b1, W2, b2)
    return bce_loss(a2, y)


def gradient_check(X, y, W1, b1, W2, b2, n_samples=30, eps=1e-5):
    hidden = W1.shape[1]
    theta = pack(W1, b1, W2, b2)

    a2, cache = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(cache, y, W2)
    analytic = pack(dW1, db1, dW2, db2)

    rng = np.random.RandomState(7)
    idxs = rng.choice(len(theta), size=min(n_samples, len(theta)), replace=False)

    rel_errors = []
    for i in idxs:
        orig = theta[i]
        theta[i] = orig + eps
        lp = loss_of_theta(theta, X, y, hidden)
        theta[i] = orig - eps
        lm = loss_of_theta(theta, X, y, hidden)
        theta[i] = orig                        # 还原！
        num = (lp - lm) / (2 * eps)
        ana = analytic[i]
        rel_errors.append(abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
    return np.array(rel_errors)


W1, b1, W2, b2 = init_params()
errs = gradient_check(X, y, W1, b1, W2, b2)

print("\n=== 梯度检验（数值梯度 vs 反向传播）===")
print(f"抽样 {len(errs)} 个参数")
print(f"最大相对误差: {errs.max():.3e}")
print(f"平均相对误差: {errs.mean():.3e}")
if errs.max() < 1e-6:
    print("→ 通过。反向传播的公式是正确的。")
else:
    print("→ 没通过！反向传播某处写错了。")

# ============================================================
# 7. 训练：梯度下降
# ============================================================
LR = 0.5          # 学习率
EPOCHS = 3000

W1, b1, W2, b2 = init_params()          # 重新初始化
loss_history = []

print("\n=== 训练 ===")
for epoch in range(EPOCHS + 1):
    # --- 前向 ---
    a2, cache = forward(X, W1, b1, W2, b2)
    loss = bce_loss(a2, y)
    loss_history.append(loss)

    # --- 反向 ---
    dW1, db1, dW2, db2 = backward(cache, y, W2)

    # --- 沿梯度的反方向走一小步 ---
    W1 -= LR * dW1
    b1 -= LR * db1
    W2 -= LR * dW2
    b2 -= LR * db2

    if epoch % 500 == 0:
        acc = ((a2 > 0.5) == y).mean()
        print(f"epoch {epoch:5d}    loss {loss:.4f}    acc {acc:.3f}")

# --- 最终结果 ---
a2, _ = forward(X, W1, b1, W2, b2)
final_acc = ((a2 > 0.5) == y).mean()
print(f"\n最终 loss = {loss_history[-1]:.4f}   准确率 = {final_acc:.3f}")

# 对照：只用逻辑回归（没有隐藏层）能做到什么水平
def logistic_regression_baseline(X, y, lr=0.5, epochs=3000):
    w = np.zeros((2, 1)); b = np.zeros(1)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        d = p - y
        w -= lr * (X.T @ d / len(X))
        b -= lr * d.mean(axis=0)
    p = sigmoid(X @ w + b)
    return ((p > 0.5) == y).mean()


baseline_acc = logistic_regression_baseline(X, y)
print(f"对照：纯逻辑回归（无隐藏层）准确率 = {baseline_acc:.3f}")
print("→ 差距就是隐藏层带来的非线性能力。")

# ============================================================
# 8. 画图
# ============================================================
plt.rcParams.update({
    "figure.figsize": (15, 4.5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(1, 3)

# (1) 原始数据
ax = axes[0]
ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1],
           s=14, c="tab:blue", edgecolors="k", linewidths=0.3, label="类别 0")
ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1],
           s=14, c="tab:red", edgecolors="k", linewidths=0.3, label="类别 1")
ax.set_title("原始数据：两个交错的新月\n（一条直线分不开）")
ax.set_xlabel("特征 1")
ax.set_ylabel("特征 2")
ax.legend(fontsize=9)

# (2) 损失曲线
ax = axes[1]
ax.plot(loss_history, color="tab:red", linewidth=2)
ax.set_yscale("log")
ax.set_title("训练损失（对数坐标）")
ax.set_xlabel("epoch")
ax.set_ylabel("BCE loss")

# (3) 决策边界
ax = axes[2]
pad = 0.3
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
    np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
)
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
probs, _ = forward(grid, W1, b1, W2, b2)
probs = probs.reshape(xx.shape)

ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.75)
ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2)
ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1],
           s=10, c="tab:blue", edgecolors="k", linewidths=0.2)
ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1],
           s=10, c="tab:red", edgecolors="k", linewidths=0.2)
ax.set_title(f"学习到的决策边界（准确率 {final_acc:.1%}）\n黑线 = 决策面 P=0.5")
ax.set_xlabel("特征 1")
ax.set_ylabel("特征 2")

plt.tight_layout()
plt.savefig("day5_network.png", dpi=120)
print("\n图片已保存：day5_network.png")

# ============================================================
# 9. 手写版 ↔ PyTorch 对照
# ============================================================
print("""
============================================================
手写版                      PyTorch 里对应什么
------------------------------------------------------------
forward()                   y_pred = model(x)
bce_loss()                  nn.BCELoss() / nn.BCEWithLogitsLoss()
backward()                  loss.backward()          ← 自动微分
W1 -= LR * dW1              optimizer.step()         ← 自动更新
init_params()               各层默认的随机初始化
------------------------------------------------------------
你手写的这 5 件事，框架全都替你做了。
但只有自己写过一遍，看到 loss.backward() 时才知道它背后在算什么。
============================================================
""")
