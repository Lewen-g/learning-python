# -*- coding: utf-8 -*-
"""
day4：模拟运动轨迹 + 手写一维卡尔曼滤波

一次交付三件事：
  - 第2周输出：用 Python 模拟一条运动轨迹并画出来
  - 第5周实践：一维卡尔曼滤波（全部手写，不调滤波库）
  - 第6周素材：预测 / 更新 / 噪声 / 协方差 / 卡尔曼增益 的直观理解

数学模型（匀速 CV 模型）：
  状态  x = [位置 p, 速度 v]^T
  观测  z = p + 噪声          （传感器只能测位置，测不到速度）
  转移  p' = p + v*dt,  v' = v
"""
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 配置与随机种子
# ============================================================
np.random.seed(42)

DT = 0.1        # 采样间隔（秒）
N = 200         # 采样点数，总时长 N*DT = 20 秒
P0 = 0.0        # 初始位置（米）
V0 = 2.0        # 真实速度（米/秒）

SIGMA_A = 0.3   # 加速度扰动标准差（过程噪声）——让"真值"不是完美匀速
SIGMA_Z = 1.5   # 观测噪声标准差（传感器精度，米）

t = np.arange(N) * DT

# ============================================================
# 2. 生成"真值"轨迹
# ============================================================
# 真实世界不是完美匀速：每一步都有一点随机加速度
a_true = np.random.normal(0.0, SIGMA_A, size=N)

p_true = np.zeros(N)
v_true = np.zeros(N)
p_true[0] = P0
v_true[0] = V0
for k in range(1, N):
    v_true[k] = v_true[k - 1] + a_true[k] * DT
    p_true[k] = p_true[k - 1] + v_true[k - 1] * DT + 0.5 * a_true[k] * DT ** 2

# 如果天真地相信"完美匀速"，轨迹会是这样：
p_ideal = P0 + V0 * t

# ============================================================
# 3. 生成观测（带噪声的位置测量）
# ============================================================
z = p_true + np.random.normal(0.0, SIGMA_Z, size=N)

# ============================================================
# 4. 手写一维卡尔曼滤波
# ============================================================
F = np.array([[1.0, DT],
              [0.0, 1.0]])          # 状态转移矩阵
H = np.array([[1.0, 0.0]])          # 观测矩阵：只测位置

q = SIGMA_A ** 2
Q = q * np.array([[DT ** 4 / 4.0, DT ** 3 / 2.0],
                  [DT ** 3 / 2.0, DT ** 2]])   # 过程噪声协方差（由加速度驱动）
R = np.array([[SIGMA_Z ** 2]])                 # 观测噪声协方差

I = np.eye(2)

x_est = np.zeros((N, 2))        # 每步的状态估计
P_est = np.zeros((N, 2, 2))     # 每步的协方差
K_hist = np.zeros((N, 2))       # 每步的卡尔曼增益

x = np.array([0.0, 0.0])        # 初始猜测：位置 0、速度 0（故意猜错，看它能不能收敛）
P = np.diag([10.0, 10.0])       # 初始不确定度：给得很大

for k in range(N):
    # ---------- 预测（Predict）----------
    x = F @ x
    P = F @ P @ F.T + Q

    # ---------- 更新（Update）----------
    y = z[k] - (H @ x)[0]              # 新息 = 观测 - 预测
    S = H @ P @ H.T + R                # 新息协方差
    K = P @ H.T @ np.linalg.inv(S)     # 卡尔曼增益
    x = x + (K * y).ravel()            # 用新息修正状态
    P = (I - K @ H) @ P                # 修正不确定度

    x_est[k] = x
    P_est[k] = P
    K_hist[k] = K.ravel()

# ============================================================
# 5. 评估：滤波到底有没有用
# ============================================================
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

rmse_ideal = rmse(p_ideal, p_true)
rmse_z = rmse(z, p_true)
rmse_kf = rmse(x_est[:, 0], p_true)

print("=== 位置误差（RMSE，单位：米）===")
print(f"理想匀速模型（不滤波）: {rmse_ideal:.4f}")
print(f"原始观测（不滤波）    : {rmse_z:.4f}")
print(f"卡尔曼滤波            : {rmse_kf:.4f}")
print(f"相对原始观测改善      : {(1 - rmse_kf / rmse_z) * 100:.1f}%")

print("\n=== 速度：滤波器能否估出看不见的量 ===")
print(f"真实末速度  : {v_true[-1]:.4f} m/s")
print(f"KF 估计末速度: {x_est[-1, 1]:.4f} m/s")
print(f"速度 RMSE   : {rmse(x_est[:, 1], v_true):.4f} m/s")

print("\n=== 卡尔曼增益收敛情况 ===")
print(f"K 第一步   : {K_hist[0]}")
print(f"K 最后一步 : {K_hist[-1]}")
print("（K 收敛到稳态，说明滤波器进入稳定工作状态）")

# ============================================================
# 6. 画图
# ============================================================
plt.rcParams.update({
    "figure.figsize": (11, 9),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(3, 1)

# (1) 位置对比
ax = axes[0]
ax.plot(t, p_true, color="black", linewidth=2, label="真值")
ax.plot(t, z, ".", color="tab:blue", markersize=4, alpha=0.45, label="观测（带噪声）")
ax.plot(t, x_est[:, 0], color="tab:red", linewidth=2, label="卡尔曼滤波估计")
ax.plot(t, p_ideal, "--", color="gray", linewidth=1, label="理想匀速模型")
ax.set_ylabel("位置 (m)")
ax.set_title("一维卡尔曼滤波：位置估计")
ax.legend(loc="upper left", fontsize=9, ncol=2)

# (2) 误差对比
ax = axes[1]
ax.plot(t, z - p_true, ".", color="tab:blue", markersize=4, alpha=0.45, label="观测误差")
ax.plot(t, x_est[:, 0] - p_true, color="tab:red", linewidth=1.5, label="滤波后误差")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("误差 (m)")
ax.set_title(f"误差对比：RMSE 从 {rmse_z:.2f} m 降到 {rmse_kf:.2f} m")
ax.legend(loc="upper left", fontsize=9)

# (3) 卡尔曼增益收敛
ax = axes[2]
ax.plot(t, K_hist[:, 0], label="$K_1$（位置修正权重）")
ax.plot(t, K_hist[:, 1], label="$K_2$（速度修正权重）")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("卡尔曼增益")
ax.set_title("卡尔曼增益随时间收敛到稳态")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("day4_kalman.png", dpi=120)
print("\n图片已保存：day4_kalman.png")
plt.show()
