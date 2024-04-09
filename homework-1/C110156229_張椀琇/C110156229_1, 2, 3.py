# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:34:13 2024

@author: user
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('laptops.csv',encoding='big5')




#1 空值資料處理
df.isnull().sum()
df.dropna()
print(df)





#2 敘述統計
print(df.describe())





#3 類別資料處理
# 創建LabelEncoder對象
label_encoder = LabelEncoder()

# 對所有類別資料進行編碼
categorical_columns = ['brand', 'Model', 'processor_brand', 'processor_tier', 'primary_storage_type', 'secondary_storage_type', 'gpu_brand', 'gpu_type', 'OS']
for column in categorical_columns:
    df[column + '_LabelEncoded'] = label_encoder.fit_transform(df[column])

# 查看編碼後的結果
print(df.head())

# 使用get_dummies函數進行One-hot編碼
one_hot_encoded = pd.get_dummies(df[categorical_columns], prefix=categorical_columns)

# 將One-hot編碼的結果合併到原始數據中
df = pd.concat([df, one_hot_encoded], axis=1)

# 刪除原始的類別資料欄位
df.drop(columns=categorical_columns, inplace=True)

# 查看編碼後的結果
print(df.head())








