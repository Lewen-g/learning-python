# -*- coding: utf-8 -*-
"""
day9：卷积神经网络（CNN）—— 先手写一遍卷积

CNN 听起来复杂，但【核心只有 6 行】：

    for i in range(H - kh + 1):
        for j in range(W - kw + 1):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)

一个小窗口在图上滑来滑去，每滑到一个位置就做一次「逐元素相乘再求和」。
**卷积的全部内容就是这句话。** 剩下的（多通道、步长、填充、池化、反向传播）
全是这句话的工程包装。

本脚本只做一件事：把这句话看清楚。
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 1. 核心：手写二维卷积
# ============================================================
# img    输入图，形状 (H, W)
# kernel 卷积核，形状 (kh, kw)
# 返回    (H-kh+1, W-kw+1) 的特征图
#
# ★ 注意输出尺寸为什么变小了：
#   窗口必须完整落在图内，所以可滑动的起点只有 H-kh+1 个。


def conv2d(img, kernel):
    kh, kw = kernel.shape
    H, W = img.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(H - kh + 1):
        for j in range(W - kw + 1):
            # 取出一块和卷积核同样大的区域，逐元素相乘，全部加起来
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out


# ============================================================
# 2. 用一个小例子手算看清楚
# ============================================================
print("=" * 60)
print("手工演示：3x3 的图，2x2 的核")
print("=" * 60)

tiny = np.array([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])
tiny_k = np.array([[1., 1.],
                   [1., 1.]])

print("输入图：")
print(tiny)
print("\n卷积核（全 1，等于求 2x2 邻域的和）：")
print(tiny_k)

print("\n窗口滑动过程（一共 (3-2+1)^2 = 4 个位置）：")
for i in range(2):
    for j in range(2):
        patch = tiny[i:i + 2, j:j + 2]
        print(f"  位置({i},{j})  取到 {patch.ravel()}  "
              f"相乘后求和 = {np.sum(patch * tiny_k):.0f}")

print("\n输出特征图：")
print(conv2d(tiny, tiny_k))

# ============================================================
# 3. 造一张图（不用下载任何数据）
# ============================================================
H = W = 64
img = np.zeros((H, W))
img[16:48, 16:40] = 1.0                       # 一个方块

yy, xx = np.mgrid[0:H, 0:W]
mask = (yy - 32) ** 2 + (xx - 50) ** 2 < 90   # 一个圆
img[mask] = 0.65

print()
print("=" * 60)
print("在 64x64 的合成图上试几种卷积核")
print("=" * 60)

kernels = {
    "竖直边缘\n(Sobel)": np.array([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]]),
    "水平边缘\n(Sobel)": np.array([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]]),
    "模糊\n(3x3 均值)": np.ones((3, 3)) / 9.0,
    "锐化":            np.array([[0., -1., 0.],
                                 [-1., 5., -1.],
                                 [0., -1., 0.]]),
}

results = {}
for name, k in kernels.items():
    results[name] = conv2d(img, k)
    print(f"  {name.replace(chr(10), ' '):16s} 卷积核形状 {k.shape}  "
          f"→ 输出形状 {results[name].shape}")

# 边缘强度 = 竖直边缘和水平边缘平方和开根号
edge_v = results["竖直边缘\n(Sobel)"]
edge_h = results["水平边缘\n(Sobel)"]
edge_mag = np.sqrt(edge_v ** 2 + edge_h ** 2)
print(f"\n  合并两者得到「边缘强度」 → 形状 {edge_mag.shape}")
print(f"  数值范围 {edge_mag.min():.2f} ~ {edge_mag.max():.2f}")

# ============================================================
# 4. 和 PyTorch 对照：同一件事
# ============================================================
print()
print("=" * 60)
print("PyTorch 的 nn.Conv2d 干的是同一件事")
print("=" * 60)

k = kernels["竖直边缘\n(Sobel)"]
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                 padding=0, bias=False)
conv.weight.data = torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)

img_t = torch.tensor(img, dtype=torch.float32).view(1, 1, H, W)
with torch.no_grad():
    out_t = conv(img_t).numpy()[0, 0]

print(f"  手写 conv2d 输出形状 : {conv2d(img, k).shape}")
print(f"  nn.Conv2d  输出形状 : {out_t.shape}")
print(f"  两者是否一致         : {np.allclose(conv2d(img, k), out_t)}")
print("""
  nn.Conv2d 做的事 = 刚才那 6 行 + 三件工程包装：
     in_channels / out_channels  多通道（每个输出通道有一组核）
     kernel_size / stride / padding  窗口大小、步长、边缘补零
     bias                        每个输出通道加一个偏置
  ★ 反向传播也是自动的 —— 卷积核的梯度由 autograd 算，
    公式和全连接层一样都是「输入.T @ 误差」，只是卷积有参数共享。
""")

# ============================================================
# 5. 画图
# ============================================================
plt.rcParams.update({
    "figure.figsize": (13, 7),
    "font.size": 10,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(2, 3)
panels = [
    (img,        "原图（灰度高 = 亮）",                    "gray"),
    (results["竖直边缘\n(Sobel)"], "竖直边缘：亮的地方是竖直边界", "RdBu_r"),
    (results["水平边缘\n(Sobel)"], "水平边缘：亮的地方是水平边界", "RdBu_r"),
    (results["模糊\n(3x3 均值)"],  "模糊：邻域求平均",              "gray"),
    (results["锐化"],             "锐化：强化自身、削弱邻域",       "gray"),
    (edge_mag,   "边缘强度 = √(竖直² + 水平²)",           "hot"),
]
for ax, (data, title, cmap) in zip(axes.ravel(), panels):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("day9_convolution.png", dpi=120)
print("图片已保存：day9_convolution.png")

print("""
============================================================
记住这一条
------------------------------------------------------------
**卷积核就是一组权重。** 它的数值决定了它检测什么：

    全 1 的核         → 求邻域均值（模糊）
    [-1,0,1] 的核     → 检测竖直边缘（左右差异）
    中心正、四周负     → 锐化（强化自身）

在 CNN 里，这些数值【不是人设计的，是训练出来的】。
你只需要给出「用几个核、多大」，网络自己会学出它该检测什么。

这就是 CNN 相对全连接层的两个关键优势：
    1. 参数共享  —— 同一个核扫全图，参数少得多
    2. 平移不变  —— 物体挪个位置，照样检测得到
============================================================
""")
