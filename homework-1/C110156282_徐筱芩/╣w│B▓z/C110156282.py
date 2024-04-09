import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 載入資料集shopping_trends_updated.csv
df = pd.read_csv("C:\portfolio\機器學習\預處理\shopping_trends_updated.csv")

# 1. 缺失值处理（刪除），列印出結果
df.dropna(inplace=True)
print("缺失值處理後的資料集:\n", df)

# 2. 針對: Age、Purchase Amount (USD)、Review Rating、Previous Purchases 做敘述統計
cols_for_stats = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
stats_df = df[cols_for_stats].describe()
print("\n針對特定欄位的敘述統計:\n", stats_df)

# 3. 針對非數值型欄位做 Label Encoding
non_numeric_cols = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color',
                    'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied',
                    'Promo Code Used', 'Payment Method', 'Frequency of Purchases']
label_enc = LabelEncoder()
encoded_df = df.copy()
for col in non_numeric_cols:
    encoded_df[col] = label_enc.fit_transform(df[col])

print("\nLabel Encoding 結果:\n", encoded_df)

# 4. 特徵縮放:使用 StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(encoded_df.drop(non_numeric_cols, axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=encoded_df.drop(non_numeric_cols, axis=1).columns)
scaled_df[non_numeric_cols] = encoded_df[non_numeric_cols]
print("\n特徵縮放後的資料集:\n", scaled_df)

# 5. 切割成訓練和測試集:Purchase Amount (USD)
X = scaled_df.drop('Purchase Amount (USD)', axis=1)  
y = scaled_df['Purchase Amount (USD)']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n訓練集特徵形狀:", X_train.shape)
print("測試集特徵形狀:", X_test.shape)
print("訓練集目標形狀:", y_train.shape)
print("測試集目標形狀:", y_test.shape)
