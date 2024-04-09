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



#4 特徵縮放
# 初始化標準化器
scaler = StandardScaler()

# 初始化標籤編碼器
label_encoder = LabelEncoder()

# 將 '處理器品牌' 特徵進行標籤編碼
df['processor_brand_LabelEncoded'] = label_encoder.fit_transform(df['processor_brand'])

# 移除 '處理器品牌' 列
df.drop(columns=['processor_brand'], inplace=True)

# 對所有數值型特徵進行標準化
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 顯示縮放後的數據
print(df.head())






#5 資料分割
# 選擇特徵
features = ['processor_tier', 'num_cores', 'num_threads', 'ram_memory', 'primary_storage_capacity', 'gpu_type', 'display_size', 'resolution_width', 'resolution_height']

# 將特徵和目標值（價格和評分）分開
X = df[features]
y_price = df['Price']
y_rating = df['Rating']

# 分割資料集
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)
X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(X, y_rating, test_size=0.2, random_state=42)

# 顯示分割後的資料集大小
print("價格預測資料集大小：")
print("訓練集大小：", X_train_price.shape[0])
print("測試集大小：", X_test_price.shape[0])

print("\n評分預測資料集大小：")
print("訓練集大小：", X_train_rating.shape[0])
print("測試集大小：", X_test_rating.shape[0])






