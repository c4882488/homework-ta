# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:24:21 2024

@author: shes9
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Credit card transactions - India.csv')

#1-----------------------------------------------------------------------------
#Missing Data
df = df.dropna()

#2-----------------------------------------------------------------------------
#數據-敘述統計
result = df.describe()
print(result)

#3-----------------------------------------------------------------------------
#特徵縮放
scaler = StandardScaler()
df2 = df[['Amount']]
df2_scaled = scaler.fit_transform(df2)
df2_scaled_df = pd.DataFrame(df2_scaled, columns=df2.columns)
print(df2_scaled_df.head())

#4-----------------------------------------------------------------------------
#類別-Code
df4 = df[['City', 'Card Type', 'Exp Type', 'Gender']]
df4 = pd.get_dummies(df4)
print(df4)

#5-----------------------------------------------------------------------------
#Split Data
df5 = pd.concat([df2, df4], axis=1)
x_train,x_test,y_train,y_test = train_test_split(df5,df[['Amount']],test_size=0.2,random_state=0)
print(len(x_train),len(x_test))


