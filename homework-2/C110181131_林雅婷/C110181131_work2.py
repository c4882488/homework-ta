import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 讀取資料
data = pd.read_csv('AirQualityUCI_new.csv')

# 選擇兩個特徵和目標標籤
features = ['NOx(GT)', 'NO2(GT)']
target = 'CO(GT)_Category'

# 分割訓練集和測試集
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 填補缺失值
imputer = SimpleImputer(strategy='mean')
X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)
X_test_scaled_imputed = imputer.transform(X_test_scaled)

# 建立並訓練 Logistic Regression 模型
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled_imputed, y_train)

# 訓練集準確度
train_accuracy = logistic_model.score(X_train_scaled_imputed, y_train)
print("訓練集準確度:", train_accuracy)

# 測試集準確度
test_accuracy = logistic_model.score(X_test_scaled_imputed, y_test)
print("測試集準確度:", test_accuracy)

# 繪製決策邊界
plt.figure(figsize=(8, 6))

# 繪製訓練集散點圖
plt.scatter(X_train_scaled_imputed[:, 0], X_train_scaled_imputed[:, 1], c=y_train, cmap='viridis', marker='o', edgecolors='k', label='Training Set')

# 繪製測試集散點圖
plt.scatter(X_test_scaled_imputed[:, 0], X_test_scaled_imputed[:, 1], c=y_test, cmap='viridis', marker='^', edgecolors='k', label='Test Set')

# 繪製決策邊界
x_min, x_max = X_train_scaled_imputed[:, 0].min() - 1, X_train_scaled_imputed[:, 0].max() + 1
y_min, y_max = X_train_scaled_imputed[:, 1].min() - 1, X_train_scaled_imputed[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

plt.xlabel('NOx(GT) (Scaled)')
plt.ylabel('NO2(GT) (Scaled)')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.colorbar(label='CO(GT)_Category')
plt.show()
