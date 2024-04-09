# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:35:56 2024

@author: USER
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from sklearn.impute import SimpleImputer

# 載入資料
data = pd.read_csv("OnlineNewsPopularity.csv")
data1 = pd.read_csv("mushrooms.csv")

# 1.處理missing data
# 檢查資料中是否有缺失值
missing_values = data.isnull().sum()

# 列出所有欄位的缺失值總數
print("各欄位缺失值總數：")
print(missing_values)

# 總結是否有缺失值
if missing_values.sum() == 0:
    print("資料中沒有缺失值。")
else:
    print("資料中有缺失值。")

# 2.敘述統計
# Label的敘述統計
label_stats = data[' shares'].describe()

# 每個特徵的敘述統計
features_stats = data.describe()

# 3.類別特徵轉換
label_encoders = {}
for column in data1.columns:
    encoder = LabelEncoder()
    data1[column] = encoder.fit_transform(data1[column])
    label_encoders[column] = encoder

# 將特徵與標籤分開
X = data1.drop('class', axis=1)
y = data1['class']

# 4.訓練與測試資料切割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5.特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 顯示訓練與測試資料的大小
print("訓練資料大小：", X_train_scaled.shape, y_train.shape)
print("測試資料大小：", X_test_scaled.shape, y_test.shape)




