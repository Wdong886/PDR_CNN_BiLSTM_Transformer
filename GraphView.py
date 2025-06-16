import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# 1. 读取数据（请根据实际文件名调整路径）
df = pd.read_excel(r'C:\Users\Lenovo\Desktop\1.xlsx')

# 2. 选取前 7 个误差指标列（假设它们是第 2~8 列）
metrics = df.columns[1:8]

# 3. 直方图 + KDE
for metric in metrics:
    data = df[metric].dropna()
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=20, density=True, alpha=0.6, edgecolor='k')
    pd.Series(data).plot.kde()
    plt.title(f'Histogram & KDE of {metric}', fontsize=14)
    plt.xlabel(metric, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.show()

# 4. 箱线图
plt.figure(figsize=(8,5))
df[metrics].boxplot()
plt.title('Boxplot of First 7 Metrics')
plt.ylabel('Error Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 5. 散点矩阵
scatter_matrix(df[metrics], alpha=0.5, diagonal='hist', figsize=(10,10))
plt.suptitle('Scatter Matrix of First 7 Metrics')
plt.tight_layout()
plt.show()

# 6. 相关系数热力图
corr = df[metrics].corr()
plt.figure(figsize=(6,5))
plt.imshow(corr, cmap='viridis', aspect='equal')
plt.colorbar(label='Correlation')
plt.xticks(range(len(metrics)), metrics, rotation=45, ha='right')
plt.yticks(range(len(metrics)), metrics)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# 7. CDF 曲线（以第3和第5个指标为例）
for idx in [1,2, 3,4,5,6]:
    metric = metrics[idx]
    data = np.sort(df[metric].dropna())
    cdf = np.arange(1, len(data)+1) / len(data)
    plt.figure(figsize=(6,4))
    plt.plot(data, cdf, marker='.', linestyle='none')
    plt.title(f'CDF of {metric}')
    plt.xlabel(metric)
    plt.ylabel('CDF')
    plt.tight_layout()
    plt.show()
