import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 讀取數據集
file = 'Credit card transactions-India.csv'
data = pd.read_csv(file)
data = data.drop('index', axis=1)

#預處理

#檢查缺失值
print(data.isnull().sum())
# 將 'Date' 轉換為 datetime 格式
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].apply(lambda x: x.toordinal())

#敘述統計
print("數據預處理前的敘述統計:")
features = ['City', 'Date', 'Card Type', 'Exp Type', 'Gender']
for feature in features:
    if data[feature].dtype == 'object' or feature == 'City':
        print(f"\n{feature}:")
        print(data[feature].describe())
    if feature == 'Date':
        print(f"\n{feature} range:")
        print(data[feature].min(), "to", data[feature].max())
print(data['Amount'].describe())
#標籤編碼
data = pd.get_dummies(data, columns=features)
print(data)


# 分割數據為訓練集和測試集
X = data.drop(['Amount'], axis=1)
y = data['Amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)