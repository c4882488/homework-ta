import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 讀取資料 (這裡假設是CSV檔案)
df = pd.read_csv('car.csv')

# 1. 處理遺失資料
# 使用 SimpleImputer 填補 'Customer Name' 欄位的遺失值
imputer = SimpleImputer(strategy='constant', fill_value='1')
df['Customer Name'] = imputer.fit_transform(df['Customer Name'].values.reshape(-1, 1))[:, 0]

# 計算每個欄位的缺失值數量
missing_values = df.isnull().sum()

# 印出每個欄位的缺失值數量
print(missing_values)



# 2. 每個特徵與Label的敘述統計
# 使用describe()获取所有数值型特征的描述性统计信息
numeric_stats = df.describe()

# 对于类别型数据，我们需要加入参数include=['O']来获取描述性统计信息
categorical_stats = df.describe(include=['O'])
print(df.describe(include='all'))

# 3. 特徵縮放 - 將特徵標準化為平均值為0、變異數為1
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print(df[numerical_features])

# 4. 類別特徵轉換 - 對類別變數進行獨熱編碼
encoder = OneHotEncoder(drop='first')
categorical_features = df.select_dtypes(include=['object']).columns.difference(['Price ($)']) # 假設'Label'是您的標籤欄位名稱
encoded_df = pd.DataFrame(encoder.fit_transform(df[categorical_features]).toarray(), columns=encoder.get_feature_names_out(categorical_features))
df.drop(columns=categorical_features, inplace=True)
df = pd.concat([df, encoded_df], axis=1)
print(df)

# 5. 將資料分割為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Price ($)']), df['Annual Income'], test_size=0.2) # 假設'Label'是您的標籤欄位名稱

