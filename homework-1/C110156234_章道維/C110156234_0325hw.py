import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 讀取資料
data = pd.read_csv("森林大火 Forest Fires.csv")

# 1.1. 檢查缺失值
missing_values = data.isnull().sum()
print("缺失值:")
print(missing_values)

# 1.2. 填補缺失值（數值型特徵）
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
imputer_numeric = SimpleImputer(strategy="mean")
data[numeric_features] = imputer_numeric.fit_transform(data[numeric_features])

# 1.3. 填補缺失值（非數值型特徵）
non_numeric_features = data.select_dtypes(exclude=['float64', 'int64']).columns
imputer_non_numeric = SimpleImputer(strategy="most_frequent")
data[non_numeric_features] = imputer_non_numeric.fit_transform(data[non_numeric_features])

# 2. 特徵與標籤(Label)的描述統計
# 假設 "area" 是標籤(Label)，其他列是特徵
features = data.drop(columns=['area'])  # 排除 'area' 列
label = data['area']

print("\n特徵的描述統計:")
print(features.describe())
print("\nLabel 的描述統計:")
print(label.describe())

# 3. 類別特徵轉換
# 在這個 CSV 中，'month' 和 'day' 是類別特徵，我們將它們轉換成數值形式
encoder = LabelEncoder()
features['month'] = encoder.fit_transform(features['month'])
features['day'] = encoder.fit_transform(features['day'])

# 4. 特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

# 5. 訓練與測試 split data
X_train, X_test, y_train, y_test = train_test_split(scaled_data, label, test_size=0.2, random_state=42)