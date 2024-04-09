import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

df = pd.read_csv('shopping_trends_updated.csv',index_col=0)
#1.處理missing data
missing_values = df.isnull().sum()
print(missing_values)

# df['購買金額（美元）'] = pd.cut(df['購買金額（美元）'], bins=[float('-inf'), 39, 80, float('inf')], labels=['低價商品', '中價商品', '高價商品'])

#2.敘述統計
description = df.describe()
print(description)
#3.特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['年齡', '先前購買']])
df[['年齡', '先前購買']] = scaled_features
print(df)

#4.類別特徵轉換
target_encoder = TargetEncoder()
encoded_df = target_encoder.fit_transform(df[['購買的商品', '類別', '地點', '顏色', '季節', '運送類型', '付款方式']], df['購買金額（美元）'])
df_encoded = pd.concat([df.drop(columns=['購買的商品', '類別', '地點', '顏色', '季節', '運送類型', '付款方式']), encoded_df], axis=1)

print(df_encoded.head())

# features_onehot = ['購買的商品', '類別', '地點', '顏色', '季節', '運送類型', '付款方式']
# onehot_encoder = OneHotEncoder(sparse=False)
# encoded_features = onehot_encoder.fit_transform(df[features_onehot])
# encoded_columns = onehot_encoder.get_feature_names_out(features_onehot)
# encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)
# df_encoded = pd.concat([df.drop(columns=features_onehot), encoded_df], axis=1)

label_encoder = LabelEncoder()
df_encoded['性別'] = label_encoder.fit_transform(df_encoded['性別'])
df_encoded['尺寸'] = label_encoder.fit_transform(df_encoded['尺寸'])
df_encoded['訂閱狀態'] = label_encoder.fit_transform(df_encoded['訂閱狀態'])
df_encoded['應用折扣'] = label_encoder.fit_transform(df_encoded['應用折扣'])
df_encoded['使用的促銷代碼'] = label_encoder.fit_transform(df_encoded['使用的促銷代碼'])
df_encoded['購買頻率'] = label_encoder.fit_transform(df_encoded['購買頻率'])
print(df_encoded)
#5.訓練資料
df['購買金額（美元）'] = pd.cut(df['購買金額（美元）'], bins=[float('-inf'), 39, 80, float('inf')], labels=['低價商品', '中價商品', '高價商品'])
X = df_encoded.drop(columns=['購買金額（美元）'])
y = df_encoded['購買金額（美元）']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

print(X_train.head())
print(y_train.head())
# X_train.to_csv('X_train.csv')
# y_train.to_csv('y_train.csv', header=['購買金額（美元）'], index_label='客戶ID')
