import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 假設您的資料集是一個CSV檔案，名為'shopping_trends_updated.csv'
# 請將下面的路徑更改為您實際資料集的路徑
file_path = "C:\portfolio\機器學習\乙班\邏輯斯回歸\shopping_trends_updated.csv"

# 讀取資料集到DataFrame中
df = pd.read_csv(file_path)

# 將Discount Applied和Subscription Status的資料型態轉換為1表示Yes、0表示No
df['Discount Applied'] = df['Discount Applied'].map({'Yes': 1, 'No': 0})
df['Subscription Status'] = df['Subscription Status'].map({'Yes': 1, 'No': 0})

# 提取特徵和目標變量
X = df[['Purchase Amount (USD)', 'Discount Applied']]
y = df['Subscription Status']

# 將資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立並訓練邏輯斯回歸模型
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 進行預測
y_pred = model.predict(X_test_scaled)

# 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f'模型準確度：{accuracy}')

# 繪製決策邊界圖
plt.figure(figsize=(10, 6))

# 繪製訓練集資料點
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='viridis', label='Training Data')

# 繪製測試集資料點
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='viridis', marker='x', label='Test Data')

# 繪製決策邊界
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Discount Applied')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.colorbar(label='Subscription Status')
plt.show()
