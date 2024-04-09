import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. 處理missing data
def handle_missing_data(data):
    # 假設有缺失值的特徵是以NaN表示的
    data.dropna(inplace=True)  # 或者使用其他填補策略
    return data

# 2. 每個特徵與Label的敘述統計
def describe_statistics(data):
    # 描述性統計
    statistics = data.describe()
    return statistics

# 3. 特徵縮放
def feature_scaling(data, numeric_features):
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    return data

# 4. 類別特徵轉換
def transform_categorical_features(data, categorical_features):
    transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )
    data = transformer.fit_transform(data)
    return data

# 5. 訓練與測試split data
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# 假設data包含特徵和標籤
# 假設數據中有缺失值，且所有特徵都需要縮放和轉換
# 要先把data分成特徵和標籤
X = data.drop(columns=['Label'])  # 特徵
y = data['Label']  # 標籤

# 執行每個步驟
X = handle_missing_data(X)
statistics = describe_statistics(X)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X = feature_scaling(X, numeric_features)

categorical_features = X.select_dtypes(include=['object']).columns
X = transform_categorical_features(X, categorical_features)

# 分割數據
X_train, X_test, y_train, y_test = split_data(X, y)

# 現在X_train, X_test, y_train, y_test都是處理過的數據，可以用於機器學習模型的訓練和測試
