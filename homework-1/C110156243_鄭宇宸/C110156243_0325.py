import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('car_data.csv') 

missing_values_count = df.isnull().sum()
print(missing_values_count)

df=df.drop(['Car_id','Customer Name','Dealer_Name','Phone'],axis=1)
numeric_stats = df.describe()
categorical_stats = df.describe(include=['O'])

print("\n數值型特徵或label的描述性統計信息:")
print(numeric_stats)
print("\n類別型特徵或label的描述性統計信息:")
print(categorical_stats)

features = df[['Annual Income']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
print("\n特徵縮放")
print(features_scaled_df)

categorical_features = ['Gender', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Dealer_No', 'Dealer_Region']

df_encoded = pd.get_dummies(df, columns=categorical_features)
print("\n類別特徵轉換後")
print(df_encoded)

X = df_encoded.drop(['Price ($)'], axis=1)  
y = df_encoded['Price ($)']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("訓練集特徵形狀:", X_train.shape)
print("測試集特徵形狀:", X_test.shape)
print("訓練集標籤形狀:", y_train.shape)
print("測試集標籤形狀:", y_test.shape)

