import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# 讀取資料
data = pd.read_csv("laptops.csv")

# 選擇特徵
X = data[['Price', 'ram_memory']]
y = data['Rating']

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立並訓練Logistic Regression模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算準確率
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# 繪製決策邊界
plt.figure(figsize=(10, 6))

# 繪製訓練資料點
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Training Data')

# 繪製測試資料點
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=80, edgecolors='k', label='Test Data')

# 繪製決策邊界
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')

plt.xlabel('Price')
plt.ylabel('ram_memory')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.colorbar()
plt.show()
