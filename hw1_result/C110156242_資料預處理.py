import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 讀取資料
data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")  

# 1.1. 檢查缺失值
missing_values = data.isnull().sum()
print("缺失值:")
print(missing_values)

# 1.2. 填補缺失值（數值型特徵）
imputer_numeric = SimpleImputer(strategy="mean")
data_numeric = imputer_numeric.fit_transform(data.select_dtypes(include=['float64', 'int64']))
data[data.select_dtypes(include=['float64', 'int64']).columns] = data_numeric

# 1.3. 填補缺失值（非數值型特徵）
imputer_non_numeric = SimpleImputer(strategy="most_frequent")
data_non_numeric = imputer_non_numeric.fit_transform(data.select_dtypes(exclude=['float64', 'int64']))
data[data.select_dtypes(exclude=['float64', 'int64']).columns] = data_non_numeric

# 2. 特徵與標籤(Label)的敘述統計
# 假設 'Nobeyesdad' 是標籤(Label)，其他列是特徵
features = data.drop(columns=['Nobeyesdad'])  # 排除 'Nobeyesdad' 列
label = data['Nobeyesdad']

print("\n特徵的敘述統計:")
print(features.describe())
print("\nLabel 的敘述統計:")
print(label.describe())

# 3. 類別特徵轉換
# 使用 LabelEncoder 將類別特徵轉換為數值形式
encoder = LabelEncoder()
categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
for feature in categorical_features:
    features[feature] = encoder.fit_transform(features[feature])

# 印出轉換後的特徵
print("轉換後的特徵:")
print(features.head())

# 4. 特徵縮放
# 使用 StandardScaler 對特徵進行縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

# 印出縮放後的特徵
print("\n縮放後的特徵:")
print(scaled_data.head())

# 印出標籤(Label)
print("\n標籤(Label):")
print(label)




# 5. 訓練與測試分割數據
X_train, X_test, y_train, y_test = train_test_split(scaled_data, label, test_size=0.2, random_state=42)










