import numpy as np
import pandas as pd
df=pd.read_csv('shopping.csv')
#1
print(df.isnull().sum())
#2
print(df.describe())
#3
conditions = [
    (df['Purchase Amount (USD)'] <= 39.00),
    (df['Purchase Amount (USD)'] > 39.00) & (df['Purchase Amount (USD)'] <= 80.00),
    (df['Purchase Amount (USD)'] > 81.00)
]
choices = ['低消費金額', '中消費金額', '高消費金額']

df['price_type'] = pd.np.select(conditions, choices, default='Unknown')

price = {
    '低消費金額': 1,
    '中消費金額': 2,
    '高消費金額': 3
}

df['price_type'] = df['price_type'].map(price)
inv_price_mapping = {v: k for k, v in price.items()}
print(inv_price_mapping)

print(df)
#4
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age', 'Review Rating', 'Previous Purchases']] = scaler.fit_transform(df[['Age', 'Review Rating', 'Previous Purchases']])
print(df)
#5
from sklearn.model_selection import train_test_split

# 切分特徵和目標變量
X = df.drop(columns=['price_type','Purchase Amount (USD)'])  # 特徵變量
y = df['Purchase Amount (USD)']  # 目標變量

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=80)

print("訓練集樣本數量:", len(X_train))
print("測試集樣本數量:", len(X_test))