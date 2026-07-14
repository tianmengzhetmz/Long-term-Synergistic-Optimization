import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果系统有支持中文的字体，否则可注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义曲线参数：每条曲线的最终值和时间常数（tau）
curves = [
    {'final': 100.0,  'tau': 27.96, 'color': 'red',   'label': 'Greedy'},
    {'final': 142.3,  'tau': 19.89, 'color': 'blue',  'label': 'PPO (No Meta)'},
    {'final': 118.5,  'tau': 11.18, 'color': 'green', 'label': 'Opt-Once'},
    {'final': 191.4,  'tau': 7.46,  'color': 'black', 'label': 'MAMRL-PDQ (Ours)'},
]

# 生成 x 轴数据（0 到 500，共 100 个点）
x = np.linspace(0, 500, 100)

# 创建图形
fig, ax = plt.subplots(figsize=(5.5, 4.0))  # 宽高比与 LaTeX 图片匹配

# 绘制每条曲线
for c in curves:
    y = c['final'] * (1 - np.exp(-x / c['tau']))
    ax.plot(x, y, color=c['color'], linewidth=2.5, label=c['label'])

# 设置坐标轴范围和刻度
ax.set_xlim(0, 500)
ax.set_ylim(0, 220)
ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('Cumulative Profit', fontsize=12)

# 设置标题（可加，但原图没有显式标题，只有图形上方caption）
ax.set_title('Convergence of Cumulative Profit', fontsize=13, pad=10)

# 网格
ax.grid(True, linestyle='--', alpha=0.6)

# 图例（右下角）
ax.legend(loc='lower right', fontsize=10)

# 紧凑布局
plt.tight_layout()

# 保存图片到当前目录
plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
plt.savefig('convergence.pdf', bbox_inches='tight')

# 显示（如果需要）
# plt.show()

print("图片已保存为 convergence.png 和 convergence.pdf")