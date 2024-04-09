import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('forestfires.csv')

# 檢查缺失值
missing_values = df.isnull().sum()

# 對特徵和標籤 'area' 進行敘述性統計分析
descriptive_stats = df.describe(include='all')

# 定義特徵和目標變量
X = df.drop('area', axis=1)  
y = df['area'] 

# 數值特徵的預處理
numeric_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) 
])

# 類別特徵的預處理
categorical_features = ['month', 'day']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())  
])

# 結合數值和類別特徵的預處理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 對X進行預處理：應用轉換
X_processed = preprocessor.fit_transform(X)

# 將數據集切分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f'訓練集形狀: {X_train.shape}')
print(f'測試集形狀: {X_test.shape}')
