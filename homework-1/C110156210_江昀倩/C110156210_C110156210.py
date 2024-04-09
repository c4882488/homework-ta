import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 读取数据
data = pd.read_csv('data.csv')


#1 删除或填补缺失数据
data.dropna(0)
print(data)

#2 每個特徵與Label的敘述統計
description = data.describe(include='all')
print(description)

#3 特徵縮放
features_to_scale = ['Amount']  
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

print(data.head())

#4 類別特徵轉換
data = pd.get_dummies(data, columns=['City', 'Card Type', 'Exp Type', 'Gender'])
print(data.head())

#5 訓練與測試split data
print(data.describe(include='all'))

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print("train_data:", train_data.shape)
print("test_data:", test_data.shape)