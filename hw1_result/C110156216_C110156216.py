import numpy as np
import pandas as pd
from sklearn import datasets
from io import StringIO


df = pd.read_csv('salary.csv')

#1 missing data
print(df.isnull().sum())
print("---------------------------")

#2 敘述統計
print(df.describe())
print("---------------------------")

#3 類別
salary1 = {
    ' <=50K':0,
    ' >50K':1
}
df["salary"] = df['salary'].map(salary1)
print(df)

workclass = {
    ' Federal-gov':1,
    ' Local-gov':2,
    ' Never-worked':3,
    ' Private':4,
    ' Self-emp-inc':5,
    ' Self-emp-not-inc':6,
    ' State-gov':7,
    ' Without-pay':8,
    ' ?':0
}
df["workclass"] = df['workclass'].map(workclass)
inv_price_mapping = {v: k for k, v in workclass.items()}
print(inv_price_mapping)
print(df)

occupation = {
    ' ?':0,
    ' Adm-clerical':1,
    ' Armed-Forces':2,
    ' Craft-repair':3,
    ' Exec-managerial':4,
    ' Farming-fishing':5,
    ' Handlers-cleaners':6,
    ' Machine-op-inspct':7,
    ' Other-service':8,
    ' Priv-house-serv':9,
    ' Prof-specialty':10,
    ' Protective-serv':11
}
df["occupation"] = df['occupation'].map(occupation)
inv_price_mapping = {v: k for k, v in occupation.items()}
print(inv_price_mapping)
print(df)

maritalstatus = {
    ' Divorced':0,
    ' Married-AF-Spouse':1,
    ' Married-civ-spouse':2,
    ' Married-spouse-absent':3,
    ' Never-married':4,
    ' Separated':5,
    ' Widowed':6,
}
df["maritalstatus"] = df['maritalstatus'].map(maritalstatus)
inv_price_mapping = {v: k for k, v in maritalstatus.items()}
print(inv_price_mapping)
print(df)
print("---------------------------")

#4 Split Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("salary.csv")

X = data.drop(columns=['salary'])  # 特徵變量
y = data['salary']  # 目標變量

numeric_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=60)

print("訓練集樣本數量:", len(X_train))
print("測試集樣本數量:", len(X_test))
print("---------------------------")

#5 特徵縮放
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("salary.csv")

numeric_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

scaler = MinMaxScaler()

data[numeric_features] = scaler.fit_transform(data[numeric_features])

print(data)
