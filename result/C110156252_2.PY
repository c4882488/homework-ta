import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 创建示例数据表
data = pd.read_csv('Final_Autism_data_encoded.csv')
data['jundice'] = pd.to_numeric(data['jundice'], errors='coerce')
data['age'] = pd.to_numeric(data['age'], errors='coerce')         
data.fillna(0, inplace=True)
# 定义特征和目标变量
X = data[['jundice', 'age']].values  # 转换为 NumPy 数组
y = data['Class/ASD'].values  # 转换为 NumPy 数组

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练 Logistic Regression 模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 打印模型在测试集上的准确率
print("模型的准确率：", model.score(X_test, y_test))

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
