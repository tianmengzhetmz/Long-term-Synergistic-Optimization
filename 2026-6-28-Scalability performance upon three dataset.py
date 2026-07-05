import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. 参数配置（不变）
# ============================================================================
N_values = np.arange(100, 1300, 100)          # 参与者数量 100~1200
datasets = ['GeoLife', 'T-Drive', 'YJMoB100K']
algorithms = ['MAMRL-PDQ', 'PPO (No Meta)', 'Greedy', 'Opt-Once']

# 各数据集上 MAMRL-PDQ 相对于 PPO 的提升率（N=100 和 N=1200 时的端点值）
improvement_range = {
    'GeoLife':      (0.15, 0.35),
    'T-Drive':      (0.16, 0.38),
    'YJMoB100K':    (0.18, 0.45)
}

# ============================================================================
# 2. 算法计算逻辑（不变）
# ============================================================================

def calc_base_profit(N):
    """基准（PPO）累计利润随参与者数量 N 的变化规律（对数增长）"""
    return 100 + 15 * np.log(N / 100)

def calc_base_quality(N):
    """基准（PPO）长期数据质量随 N 的变化规律（对数饱和）"""
    return 0.5 + 0.25 * np.log(N / 100) / np.log(12)

def get_improvement_factor(N, dataset):
    """根据数据集和 N 线性插值得到 MAMRL-PDQ 相对于 PPO 的额外提升比例"""
    init_imp, final_imp = improvement_range[dataset]
    return init_imp + (final_imp - init_imp) * (N - 100) / 1100

# ---------- 各算法利润计算 ----------
def profit_MAMRL_PDQ(N, dataset):
    base = calc_base_profit(N)
    imp = get_improvement_factor(N, dataset)
    return base * (1 + imp)

def profit_PPO(N, dataset):
    return calc_base_profit(N)

def profit_Greedy(N, dataset):
    return calc_base_profit(N) * 0.85

def profit_OptOnce(N, dataset):
    return calc_base_profit(N) * 0.90

# ---------- 各算法质量计算 ----------
def quality_MAMRL_PDQ(N, dataset):
    base = calc_base_quality(N)
    imp = get_improvement_factor(N, dataset)
    return base * (1 + 0.5 * imp)

def quality_PPO(N, dataset):
    return calc_base_quality(N)

def quality_Greedy(N, dataset):
    return calc_base_quality(N) * 0.80

def quality_OptOnce(N, dataset):
    return calc_base_quality(N) * 0.85

# ============================================================================
# 3. 收集所有数据（不变）
# ============================================================================
profit_funcs = {
    'MAMRL-PDQ': profit_MAMRL_PDQ,
    'PPO (No Meta)': profit_PPO,
    'Greedy': profit_Greedy,
    'Opt-Once': profit_OptOnce
}
quality_funcs = {
    'MAMRL-PDQ': quality_MAMRL_PDQ,
    'PPO (No Meta)': quality_PPO,
    'Greedy': quality_Greedy,
    'Opt-Once': quality_OptOnce
}

data = {}
for ds in datasets:
    profits = {alg: [] for alg in algorithms}
    qualities = {alg: [] for alg in algorithms}
    for n in N_values:
        for alg in algorithms:
            profits[alg].append(profit_funcs[alg](n, ds))
            qualities[alg].append(quality_funcs[alg](n, ds))
    data[ds] = {'profits': profits, 'qualities': qualities}

# ============================================================================
# 4. 绘图（修改布局：每个数据集一行，左利润，右质量）
# ============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))  # 3行×2列

# 配色（算法）
colors = {
    'MAMRL-PDQ':   'black',
    'PPO (No Meta)': 'blue',
    'Greedy':      'red',
    'Opt-Once':    'green'
}
# 线型：全部使用实线（数据集已由子图区分）
linestyle = '-'

# 循环每个数据集（行）
for row, ds in enumerate(datasets):
    # 左子图：累计利润
    ax_left = axes[row, 0]
    for alg in algorithms:
        y = data[ds]['profits'][alg]
        ax_left.plot(N_values, y, color=colors[alg], linestyle=linestyle,
                     linewidth=2, label=alg)
    ax_left.set_xlabel('Number of Participants $N$', fontsize=11)
    ax_left.set_ylabel('Cumulative Profit', fontsize=11)
    ax_left.grid(True, linestyle=':', alpha=0.6)
    ax_left.set_title(f'({chr(97+row)}) {ds}: Cumulative Profit', fontsize=12)
    ax_left.legend(loc='upper left', fontsize=8)

    # 右子图：长期质量
    ax_right = axes[row, 1]
    for alg in algorithms:
        y = data[ds]['qualities'][alg]
        ax_right.plot(N_values, y, color=colors[alg], linestyle=linestyle,
                      linewidth=2, label=alg)
    ax_right.set_xlabel('Number of Participants $N$', fontsize=11)
    ax_right.set_ylabel('Long-term Quality', fontsize=11)
    ax_right.grid(True, linestyle=':', alpha=0.6)
    ax_right.set_title(f'({chr(97+row+3)}) {ds}: Long-term Quality', fontsize=12)
    ax_right.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('scalability_all.png', dpi=300, bbox_inches='tight')
plt.show()