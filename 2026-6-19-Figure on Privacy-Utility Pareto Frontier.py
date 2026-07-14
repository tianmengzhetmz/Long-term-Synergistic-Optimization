import matplotlib.pyplot as plt
import numpy as np

# 数据点（与 TikZ 坐标完全一致）
data = {
    "Greedy": {
        "x": [30, 50, 80, 100],
        "y": [0.45, 0.52, 0.60, 0.62],
        "color": "red",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2,
    },
    "PPO": {
        "x": [25, 45, 70, 90],
        "y": [0.50, 0.62, 0.71, 0.73],
        "color": "blue",
        "marker": "s",
        "linestyle": "-",
        "linewidth": 2,
    },
    "Opt-Once": {
        "x": [28, 48, 75, 95],
        "y": [0.48, 0.58, 0.65, 0.67],
        "color": "green",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2,
    },
    "MAMRL-PDQ (Ours)": {
        "x": [20, 35, 55, 80],
        "y": [0.60, 0.75, 0.84, 0.86],
        "color": "black",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 2,
    },
}

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(6, 4.8))  # 宽高比与 TikZ 中 width=0.48\textwidth, height=6cm 相近

# 绘制每条曲线
for label, params in data.items():
    ax.plot(
        params["x"],
        params["y"],
        color=params["color"],
        marker=params["marker"],
        linestyle=params["linestyle"],
        linewidth=params["linewidth"],
        markersize=8,
        label=label,
    )

# 设置坐标轴范围和刻度
ax.set_xlim(0, 120)
ax.set_ylim(0.3, 0.9)
ax.set_xticks(np.arange(0, 121, 20))
ax.set_yticks(np.arange(0.3, 0.91, 0.1))

# 标题和轴标签（与 TikZ 完全一致）
ax.set_title("Privacy-Utility Pareto Frontier", fontsize=12)
ax.set_xlabel("Total Privacy Budget Consumed $\\sum \\epsilon_t$", fontsize=11)
ax.set_ylabel("Average Data Quality", fontsize=11)

# 网格（与 TikZ 的 grid=both 一致）
ax.grid(True, linestyle="--", alpha=0.7)

# 图例（位置 south east）
ax.legend(loc="lower right", fontsize=10, frameon=True, edgecolor="black")

# 紧凑布局
plt.tight_layout()

# 保存为 PNG（也可改为 PDF）
plt.savefig("pareto.png", dpi=300, bbox_inches="tight")
plt.show()