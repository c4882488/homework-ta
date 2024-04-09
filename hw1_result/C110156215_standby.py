import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 讀取資料
data = pd.read_csv('data.csv')

# 處理missing data (用眾數填補資料)
data.fillna(data.mode().iloc[0], inplace=True)

# 特徵與Label的敘述統計
print("描述性統計:")
print(data.describe())

# 類別特徵轉換
label_encoders = {}
categorical_features = ['month', 'day']

# 加入對area欄位的類別特徵轉換
area_labels = ['no_fire', 'small_fire', 'medium_fire', 'large_fire']
data['area_category'] = pd.cut(data['area'], bins=[-1, 0, 5, 25, 1090.84], labels=area_labels, right=False)

le_area = LabelEncoder()
data['area_category'] = le_area.fit_transform(data['area_category'])
label_encoders['area_category'] = le_area

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# 特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(['Y'], axis=1))  # Y是標籤，不進行縮放
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])  # 重新構建DataFrame，不包括Y

# 加入area_category欄位
scaled_data['area_category'] = data['area_category']

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
