import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 生成模擬數據集
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.random.choice([0, 1], 100)

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 Logistic Regression 模型
model = LogisticRegression()

# 將模型與訓練數據擬合
model.fit(X_train, y_train)

# 在測試集上做出預測
y_pred = model.predict(X_test)

# 繪製訓練數據
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='viridis', label='Training data')

# 繪製測試數據
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', cmap='viridis', label='Test data')

# 繪製決策邊界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
