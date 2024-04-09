import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 載入數據
pokemon_data = pd.read_csv('Pokemon.csv')

# 2. 處理missing data
pokemon_data.dropna(inplace=True)

# 3. 每個特徵與Label的描述性統計
print("描述性統計：\n", pokemon_data.describe())

# 4. 特徵縮放
scaler = StandardScaler()
numeric_features = pokemon_data.select_dtypes(include=['float64', 'int64'])
scaled_features = scaler.fit_transform(numeric_features)

# 5. 類別特徵轉換
label_encoder = LabelEncoder()
pokemon_data['Legendary'] = label_encoder.fit_transform(pokemon_data['Legendary'])

# 分割特徵和標籤
X = scaled_features
y = pokemon_data['Legendary']

# 訓練和測試分割數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印訓練和測試數據的形狀
print("訓練數據特徵形狀:", X_train.shape)
print("測試數據特徵形狀:", X_test.shape)
print("訓練數據標籤形狀:", y_train.shape)
print("測試數據標籤形狀:", y_test.shape)
