import numpy as np
import matplotlib.pyplot as plt

# 1. 数据
t = np.linspace(0, 2 * np.pi, 200)
y_sin = np.sin(t)
y_cos = np.cos(t)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# 2. 画图
plt.figure()
plt.plot(t, y_sin, label="sin(t)", color="blue", linewidth=2)
plt.plot(t, y_cos, label="cos(t)", color="orange", linestyle="--", linewidth=2)

plt.title("Sine and Cosine", fontsize=14)
plt.xlabel("t (radians)", fontsize=12)
plt.ylabel("value", fontsize=12)

# ★ 关键：X 轴用 π 做刻度
plt.xticks(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    ["0", "π/2", "π", "3π/2", "2π"],
)

# ★ 关键：Y 轴留出上下空白
plt.ylim(-1.5, 1.5)

# ★ 关键：过原点画两条参考线
plt.axhline(0, color="gray", linewidth=0.8)      # 水平零线
plt.axvline(0, color="gray", linewidth=0.8)      # 垂直零线

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()                               # 自动排版，避免标签被截断
plt.savefig("day2_sine.png", dpi=120)
plt.show()

print("图片已保存：day2_sine.png")