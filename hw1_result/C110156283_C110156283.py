import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: 載入數據
file_path = 'freedom_index.csv'  # 請根據您的檔案路徑修改此處
data = pd.read_csv(file_path)

# Step 2: 敘述統計分析（這裡數據已經是乾淨的，無遺失值）
print(data.describe())

# Step 3: 特徵縮放 - 使用MinMaxScaler對數值特徵進行縮放
scaler = MinMaxScaler()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Step 4: 類別特徵轉換 - 對'Country'和'Region'進行獨熱編碼
data = pd.get_dummies(data, columns=['Country', 'Region'])

# Step 5: 分割數據 - 設定70%為訓練集，30%為測試集
X = data.drop(['Overall Score'], axis=1)
y = data['Overall Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 顯示訓練集和測試集的形狀
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")