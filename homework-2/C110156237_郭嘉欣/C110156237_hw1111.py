# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:25:08 2024

@author: DK
"""
# 預測黃疸是否會影響罹患自閉症機率
import pandas as pd
import numpy as np

df = pd.read_csv('Autism-Child-Data.csv')

# 1.Missing Data (無缺失值)
df.isnull().sum()
# 刪除A1_Score~A10_Score的欄位
df = df.drop(df.columns[0:10], axis=1)
df.values


# 將 '?' 值轉換為 NaN
df['age'] = df['age'].replace('?', np.nan)

# 將 'age' 欄位轉換為數值型態
df['age'] = df['age'].astype(float)

# 計算 'age' 欄位的平均值
mean_age = df['age'].mean()

# 用平均值填補 '?' 值
df['age'] = df['age'].fillna(round(mean_age))


# 2.敘述統計
df.describe()


# 3.類別資料處理
jundice_mapping = {'yes':1,'no':0,'?':2}
df['jundice'] = df['jundice'].map(jundice_mapping)
df

ASD_mapping = {'YES':1, 'NO':0,'?':2}
df['Class/ASD'] = df['Class/ASD'].map(ASD_mapping)


# 4.特徵縮放

from sklearn.preprocessing import StandardScaler

# 創建標準化的實例
scaler = StandardScaler()

# 要縮放的特徵欄位
features_to_scale = ['age', 'jundice', 'Class/ASD']

# 使用標準化器進行標準化
scaled_features = scaler.fit_transform(df[features_to_scale])

# 將縮放後的特徵資料放回原始資料框
df[features_to_scale] = scaled_features


# 5. Split Data
# 將 'age_desc' 和 'relation' 欄位從資料中移除，因為它們不是特徵
features = df.drop(['age_desc', 'relation', 'Class/ASD'], axis=1)

# 將 'Class/ASD' 欄位作為目標
target = df['Class/ASD']

# 顯示分割後的特徵和目標資料
print("Features:")
print(features)
print("\nTarget:")
print(target)

#---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features[['age', 'result']], target, test_size=0.2, random_state=42)

# 創建線性回歸模型
model = LinearRegression()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 繪製散點圖
plt.scatter(X_test['age'], X_test['result'], c=y_test, cmap='bwr', marker='x', label='Actual')
plt.xlabel('Age (scaled)')
plt.ylabel('Jundice (scaled)')
plt.title('Actual values')
plt.legend()
plt.show()

# 繪製預測結果
plt.scatter(X_test['age'], X_test['result'], c=y_pred, cmap='bwr', marker='o', label='Predicted')
plt.xlabel('Age (scaled)')
plt.ylabel('Jundice (scaled)')
plt.title('Predicted values')
plt.legend()
plt.show()


