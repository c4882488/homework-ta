# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:39:12 2024

@author: a3690
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 讀取資料
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# 只保留需要的特徵和目標變量
data = data[['Height', 'Weight', 'Nobeyesdad']]

# 將目標變量轉換為二元變量
data['Nobeyesdad'] = data['Nobeyesdad'].apply(lambda x: 1 if x == 'Obesity_Type_I' else 0)

# 拆分特徵和目標變量
X = data[['Height', 'Weight']]
y = data['Nobeyesdad']

# 將資料集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化Logistic Regression模型
model = LogisticRegression()

# 訓練模型
model.fit(X_train_scaled, y_train)

# 預測測試集
y_pred = model.predict(X_test_scaled)

# 計算準確度
accuracy = accuracy_score(y_test, y_pred)
print("準確度:", accuracy)

# 繪製訓練集
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, marker='o', cmap='bwr', label='Training set')

# 繪製測試集
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, marker='x', cmap='bwr', label='Test set')

# 繪製決策邊界
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

plt.xlabel('Height (standardized)')
plt.ylabel('Weight (standardized)')
plt.title('Logistic Regression Classifier')
plt.legend()
plt.show()

