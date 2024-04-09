import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 讀取資料
data = pd.read_csv('AirQualityUCI.csv', delimiter=';')

# 將逗號替換為點，並將數據類型轉換為浮點數
for column in data.columns[2:]:
    data[column] = data[column].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x).astype(float)

# 處理missing data
data['CO(GT)'] = data['CO(GT)'].mask(data['CO(GT)'] == -200, pd.NA)  # 將-200替換為NaN
data.dropna(subset=['CO(GT)'], inplace=True)  # 刪除含有NaN的列

# 描述統計
description = data.describe()


# 特徵縮放
scaler = StandardScaler()
features = data.drop(columns=['Date', 'Time', 'CO(GT)'])  # 移除日期、時間和標籤
scaled_features = scaler.fit_transform(features)
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)



# 將 'CO(GT)' 分配給 y
y = data['CO(GT)']

# 確保 scaled_data 和 y 的樣本數相同
scaled_data = scaled_data.iloc[:len(y)]

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=42)

# 輸出結果
print("處理過的資料:")
print(data.head())
print("\n描述統計:")
print(description)
print("\n縮放後的特徵:")
print(scaled_data.head())
print("\n訓練集和測試集大小:")
print("訓練集大小:", len(X_train))
print("測試集大小:", len(X_test))
