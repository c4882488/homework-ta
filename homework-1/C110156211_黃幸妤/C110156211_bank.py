import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 讀取CSV文件
df = pd.read_csv('bank.csv')

# 處理缺失數據：用中位數填補數值型特徵，用最常見類別填補分類特徵
num_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[num_features] = num_imputer.fit_transform(df[num_features])
df[cat_features] = cat_imputer.fit_transform(df[cat_features])

# 每個特徵與標籤的描述統計
print("\n特徵與標籤的描述統計:")
print(df.describe(include='all'))

# 特徵縮放
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])
print('特徵縮放')
print(df[num_features])

# 類別特徵轉換：使用獨熱編碼
df = pd.get_dummies(df, columns=cat_features, drop_first=True)


# 訓練與測試split data
X = df.drop('balance', axis=1)  # 特徵集（假設 'balance' 是目標特徵）
y = df['balance']  # 目標特徵

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
