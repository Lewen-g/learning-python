# learning-python

研一阶段编程基础重建的学习仓库。记录从 Python 基础到机器人 / 自动驾驶方向的学习轨迹。

## 环境

| | |
|---|---|
| Python | 3.13.1 |
| 虚拟环境 | `.venv/`（已在 `.gitignore` 中忽略，不提交） |
| 依赖 | 见 `requirements.txt` |

## 文件说明

| 文件 | 内容 | 对应计划 |
|---|---|---|
| `day1_numpy.py` | NumPy 基础：数组创建、矩阵运算、转置、行列式、广播、统计、linspace | 第 1 周 |
| `day1_broadcast.py` | 广播专题：标量广播、一维到二维、行列广播、特征标准化实例 | 第 1 周 |
| `day2_matplotlib.py` | Matplotlib 绘图：sin/cos、π 刻度、rcParams 全局样式、参考线 | 第 2 周 |
| `day2_sine.png` | 上一脚本的输出图片 | 第 2 周 |
| `day3_matrix.ipynb` | 矩阵运算：乘法（手写三重循环验证 `@`）、非交换性、特征值与特征向量、线性方程组求解 | 第 3 周 |
| `day4_trajectory_kalman.py` | 运动轨迹模拟 + 手写一维卡尔曼滤波（预测 / 更新 / 协方差 / 增益） | 第 2、5、6 周 |
| `day4_kalman.png` | 上一脚本的输出：位置估计、误差对比、卡尔曼增益收敛 | 第 2、5、6 周 |

## 快速开始

```bash
git clone https://github.com/Lewen-g/learning-python.git
cd learning-python

python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

运行任意脚本：

```bash
python day4_trajectory_kalman.py
```

## 学习进度

- [x] NumPy 基础与广播
- [x] Matplotlib 绘图
- [x] 矩阵运算与线性代数
- [x] 卡尔曼滤波（一维，手写实现）
- [x] Git 基本工作流
- [ ] C++ 基础
- [ ] Linux 基础
- [ ] ROS2 入门
- [ ] 深度学习 / PyTorch 基础

## 说明

本仓库只放学习代码。个人规划、成长档案等文档不在本仓库（见 `.gitignore`）。
