import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 讀取資料
data = pd.read_csv('data.csv')

# 處理missing data (用眾數填補資料)
data.fillna(data.mode().iloc[0], inplace=True)

# 只保留 'month' 和 'day' 兩個特徵
data = data[['area', 'temp', 'Y']]

# 類別特徵轉換
label_encoders = {}

for feature in ['area', 'temp']:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# 特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(['Y'], axis=1))  # Y是標籤，不進行縮放
scaled_data = pd.DataFrame(scaled_features, columns=['area', 'temp'])  # 重新構建DataFrame，僅包含 'month' 和 'day'

# 訓練與測試split data
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['Y'], test_size=0.2, random_state=42)

# 打印部分處理後的資料
print("\n經處理後的資料 (部分示例):")
print(X_train.head())

# 保存處理後的資料
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
