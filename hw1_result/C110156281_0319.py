import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. missing data
df = pd.read_excel('1.xlsx')
print(df.isnull().sum())
print("----------------------------------------------")

# 2. 每個特徵、label的描述統計
print(df.describe())
print("----------------------------------------------")

# 3. 根據'house price of unit area'列的值，轉成類別型態，再轉為
conditions = [
    (df['Y house price of unit area'] <= 30.00),
    (df['Y house price of unit area'] > 30.00) & (df['Y house price of unit area'] <= 55.00),
    (df['Y house price of unit area'] > 55.00)
]
choices = ['低面積房價', '中面積房價', '高面積房價']

df['price_type'] = np.select(conditions, choices, default='Unknown')
print(df)

# 打印 price_mapping 的反向映射
price_mapping = {
    '低面積房價': 1,
    '中面積房價': 2,
    '高面積房價': 3
}
df['price_type'] = df['price_type'].map(price_mapping)

inv_price_mapping = {v: k for k, v in price_mapping.items()}
print(inv_price_mapping)
print(df)
print("----------------------------------------------")
# 4. 特徵縮放
scaler = MinMaxScaler()
df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area']] = scaler.fit_transform(df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'Y house price of unit area']])
print(df)

#5.split data
from sklearn.model_selection import train_test_split

# 切分特徵和目標變量
X = df.drop(columns=['price_type','Y house price of unit area'])  # 特徵變量
y = df['Y house price of unit area']  # 目標變量

# 切分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=60)

print("訓練集樣本數量:", len(X_train))
print("測試集樣本數量:", len(X_test))