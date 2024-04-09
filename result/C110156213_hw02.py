#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:11:22 2024

@author: huangzhizhen
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# 讀取CSV檔
data = pd.read_csv('laptops.csv')

# 處理空值資料
data.dropna(inplace=True)
data.replace('No information', pd.NA, inplace=True)
data.dropna(inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # 將無窮大值替換為 NaN
data.dropna(inplace=True)  # 刪除含有 NaN 值的列

# 敘述統計
statistics = data.describe()
print("Descriptive Statistics:")
print(statistics)

# 類別資料處理
categorical_columns = ['brand', 'Model', 'processor_brand', 'processor_tier', 'primary_storage_type',
                       'secondary_storage_type', 'gpu_brand', 'gpu_type', 'OS']
data_categorical = pd.get_dummies(data[categorical_columns])
print(data_categorical)
# 特徵縮放 - 資料正規化
numeric_columns = ['Price', 'Rating', 'num_cores', 'num_threads', 'ram_memory',
                   'primary_storage_capacity', 'secondary_storage_capacity', 'is_touch_screen',
                   'display_size', 'resolution_width', 'resolution_height', 'year_of_warranty']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numeric_columns])
data_scaled = pd.DataFrame(data_scaled, columns=numeric_columns)
print(data_scaled)

# 合併類別資料和縮放後的數值資料
processed_data = pd.concat([data_categorical, data_scaled], axis=1)

# 資料分隔
X = processed_data.drop('Price', axis=1) # 特徵
y = processed_data['Price'] # 目標
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data preprocessing completed.")
# 建立 Logistic Regression 模型
model = LogisticRegression()

# 使用兩個特徵進行訓練
# 在這裡，我們假設選擇 'Rating' 和 'ram_memory' 這兩個特徵進行訓練
X_train_two_features = X_train[['Rating', 'ram_memory']]
X_test_two_features = X_test[['Rating', 'ram_memory']]

# 訓練模型
model.fit(X_train_two_features, y_train)

# 顯示模型的係數
print("Model Coefficients:")
print(model.coef_)

# 繪製訓練資料的散點圖
plt.scatter(X_train_two_features['Rating'], X_train_two_features['ram_memory'], c=y_train, cmap='viridis')
plt.xlabel('Rating')
plt.ylabel('RAM Memory')
plt.title('Scatter plot of Training Data')
plt.colorbar(label='Price')
plt.show()

print("Logistic Regression Model training completed.")
