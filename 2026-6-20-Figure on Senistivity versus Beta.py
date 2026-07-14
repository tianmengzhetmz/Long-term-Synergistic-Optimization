import matplotlib.pyplot as plt
import numpy as np

# ========== 硬编码数据（来自论文 Figure 8） ==========
# 每种算法在四个β值下的累计利润（GeoLife, T-Drive, YJMoB100K）
data = {
    'GeoLife': {
        'beta': [0.5, 1.0, 2.0, 3.0, 4.0],
        'Greedy': [92, 96, 100, 88, 78],
        'PPO (No Meta)': [128, 136, 142, 134, 126],
        'Opt-Once': [108, 114, 118, 110, 102],
        'MAMRL-PDQ (Ours)': [180, 188, 191, 185, 175]
    },
    'T-Drive': {
        'beta': [0.5, 1.0, 2.0, 3.0, 4.0],
        'Greedy': [89, 93, 100, 85, 76],
        'PPO (No Meta)': [125, 132, 138, 130, 122],
        'Opt-Once': [105, 110, 115, 107, 99],
        'MAMRL-PDQ (Ours)': [175, 182, 186, 180, 170]
    },
    'YJMoB100K': {
        'beta': [0.5, 1.0, 2.0, 3.0, 4.0],
        'Greedy': [87, 91, 100, 83, 74],
        'PPO (No Meta)': [122, 129, 135, 127, 119],
        'Opt-Once': [103, 108, 112, 105, 97],
        'MAMRL-PDQ (Ours)': [172, 179, 183, 177, 167]
    }
}

# 颜色与标记
styles = {
    'Greedy': {'color': 'red', 'marker': '*', 'linestyle': '-'},
    'PPO (No Meta)': {'color': 'blue', 'marker': 's', 'linestyle': '-'},
    'Opt-Once': {'color': 'green', 'marker': '^', 'linestyle': '-'},
    'MAMRL-PDQ (Ours)': {'color': 'black', 'marker': 'D', 'linestyle': '-'}
}

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
fig.subplots_adjust(wspace=0.3)

for ax, (dataset_name, vals) in zip(axes, data.items()):
    beta = vals['beta']
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Cumulative Profit')
    ax.set_xlim(0.4, 4.1)
    ax.set_ylim(70, 210)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(dataset_name)

    # 画四条曲线
    for algo in ['Greedy', 'PPO (No Meta)', 'Opt-Once', 'MAMRL-PDQ (Ours)']:
        y = vals[algo]
        style = styles[algo]
        ax.plot(beta, y,
                label=algo,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=2,
                markersize=8)
    # 图例只放在第一张子图
    if dataset_name == 'GeoLife':
        ax.legend(loc='upper left', fontsize=9)

# 保存图片
plt.tight_layout()
plt.savefig('sensitivity_beta_reproduced.png', dpi=300, bbox_inches='tight')
plt.show()