import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 讀取資料
data = pd.read_csv("AirQualityUCI.csv", delimiter=";")

# 將-200替換成NaN
data.replace(-200, np.nan, inplace=True)

# 替換逗號為點
data.replace(',', '.', regex=True, inplace=True)

# 定義特徵和標籤
features = data.drop(columns=["CO(GT)", "Date", "Time"])  # 移除非數值的欄位
label = data["CO(GT)"]

# 處理missing data
imputer = SimpleImputer(strategy="mean")
imputed_features = imputer.fit_transform(features)

# 將numpy array轉換為DataFrame
imputed_features = pd.DataFrame(imputed_features, columns=features.columns)

# 重新設置索引
data.reset_index(drop=True, inplace=True)

# 將特徵和標籤重新連接
data = pd.concat([data[['Date', 'Time']], imputed_features, label], axis=1)

# 敘述統計
description = imputed_features.describe()

# 特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)

# 訓練與測試split data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, label, test_size=0.2, random_state=42)

# 完整程式碼
print("Imputed Features:")
print(imputed_features.head())
print("\nDescription Statistics:")
print(description)
print("\nScaled Features:")
print(scaled_features[:5])
print("\nTraining and Testing Data Shapes:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
