# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:57:16 2024

@author: Genuine
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#顯示全部
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 讀取 Excel 
data = pd.read_excel('Real estate valuation data set.xlsx')

# 顯示數據
print(data)

#刪除遺失值
data.dropna(inplace=True)
#填補遺失值
data.fillna(data.mean(), inplace=True)
# 檢查遺失值數量
missing_values = data.isnull().sum()
print("遺失值：")
print(missing_values)

#每個特徵與Label的敘述統計 
print(data.describe())

# 選擇特徵
numeric_features  = data[['Y house price of unit area']]

# 標準化特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

data[['Y house price of unit area']] = scaled_features

# 顯示特徵縮放結果
print(data['Y house price of unit area'])
"""
# 讀熱編碼
data_encoded = pd.get_dummies(data, columns=['Y house price of unit area'])

# 讀熱編碼結果
print(data_encoded)
"""
#設標籤
X = data.drop(columns=['Y house price of unit area']) 
y = data[['Y house price of unit area']]  # Lable

# 將數據拆成訓練集、測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 顯示大小
print("訓練集大小:", X_train.shape)
print("測試集大小:", X_test.shape)


# 選擇兩個特徵（X3和X4）
X = data[['X3 distance to the nearest MRT station', 'X4 number of convenience stores']]


# 訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 初始化線性回歸模型
lr = LinearRegression()

# 使用訓練集對模型進行訓練
lr.fit(X_train, y_train)


# 繪製實際值與預測值的散點圖
y_pred = lr.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title('x vs  Y ')
plt.show()