# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:30:13 2024

@author: a0998
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# 加载数据
df = pd.read_csv('laptops.csv')

df.head()

df_filled = df.fillna(0)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df[numeric_cols].mean())

# 描述性统计，以查看缩放后的数值特征
print("\n描述性统计：")
print(df_filled.describe())

scaler = MinMaxScaler()

df_filled[numeric_cols] = scaler.fit_transform(df_filled[numeric_cols])

# 显示缩放后的数据
print("\n缩放后的数据预览：")
print(df_filled.head())


categorical_cols = df_filled.select_dtypes(include=['object']).columns.tolist()

# 实例化OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')

# 对类别型特征进行独热编码
encoded_features = encoder.fit_transform(df_filled[categorical_cols])

# 将编码后的特征转换为DataFrame，并使用原特征名作为列名的一部分
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

# 重置索引，以便后续能够按行拼接
df_filled.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)

# 按行将编码后的DataFrame拼接回原DataFrame
df_final = pd.concat([df_filled.drop(categorical_cols, axis=1), encoded_df], axis=1)

# 显示处理后的数据预览
print("\n处理后的数据预览：")
print(df_final.head())

X = df_final.drop('Price', axis=1)
y = df_final['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 输出划分结果的维度以确认
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
