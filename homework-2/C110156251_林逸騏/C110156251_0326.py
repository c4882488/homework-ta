# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:44:46 2024

@author: a0998
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
df = pd.read_csv('laptops.csv')

# 选择特征和目标变量
X = df[['Rating', 'Price']]  # 根据你的数据集的实际列名调整
y = (df['Price'] > 50000).astype(int)  # 将价格超过30000的笔记本标记为1，其他标记为0

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化并训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 可视化决策边界
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    # 设置最小和最大值并加上一点额外空间
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # 网格步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测整个网格的值
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.title(title)
    plt.xlabel('Rating (scaled)')
    plt.ylabel('Price (scaled)')

plot_decision_boundary(X_scaled, y, model)

# 输出模型准确度
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))
plt.show()



