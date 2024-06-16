import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文字体（例如宋体、黑体等）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 柱子的名称
labels = ['真实值', '套索回归', '决策树', '随机森林', 'k近邻算法']

# 柱子的高度
heights = [132, 138, 109, 125, 133]

# 创建柱状图
pcolors = plt.cm.Blues(np.linspace(0.3, 1, len(labels)))  # 生成渐变色列表，从浅蓝到深蓝

# 创建柱状图
plt.bar(range(len(labels)), heights, width=0.7, color=pcolors )
# 添加标题和标签
plt.title('各模型预测结果比较')
plt.xlabel('模型名')
plt.ylabel('房价单位（万）')
for i in range(len(labels)):
    plt.text(i, heights[i] + 3, str(heights[i]), ha='center', va='bottom')
mid_points = [(i, heights[i]) for i in range(len(labels))]  # 获取每个柱子中间点的坐标
plt.plot(*zip(*mid_points), marker='o', linestyle='-', color='black')  # 连接曲线
# 设置x轴刻度和标签
plt.xticks(range(len(labels)), labels)

# 设置y轴刻度从100开始
plt.ylim(100, max(heights) + 10)

# 显示图形
plt.show()
