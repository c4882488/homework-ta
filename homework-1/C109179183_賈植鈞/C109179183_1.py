import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# 讀取CSV檔案
df = pd.read_csv('carsale_data.csv')

# 顯示前幾行資料
print(df.head())

# 將 'Date' 欄位轉換為日期時間型態
df['Date'] = pd.to_datetime(df['Date'])

# 從 'Date' 欄位中提取年份和月份，並將其作為新的 'Year' 和 'Month' 欄位
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 檢查資料是否有缺失值
missing_values = df.isnull().sum()

# 顯示缺失值統計
print(missing_values)

print("敘述統計")

# 使用describe()函數進行敘述統計
stats = df.describe()

# 顯示敘述統計
print(stats)

import numpy as np

# 將 pandas DataFrame 轉換為 numpy array
car_data_np = df.to_numpy()

# 顯示 numpy array 的形狀以了解其結構
print("numpy array 的形狀:", car_data_np.shape)

# 顯示第一行的示例
print("第一行的示例:", car_data_np[0])

# 顯示個欄位名稱
print(df.columns)

# 顯示 'Gender' 列中每個唯一值的計數
print(df['Gender'].value_counts())

# 顯示 'Dealer_Name' 列中每個唯一值的計數
print(df['Dealer_Name'].value_counts())

# 顯示 'Company' 列中每個唯一值的計數
print(df['Company'].value_counts())

# 顯示 'Model' 列中每個唯一值的計數
print(df['Model'].value_counts())

# 顯示 'Engine' 列中每個唯一值的計數
print(df['Engine'].value_counts())

# 顯示 'Transmission' 列中每個唯一值的計數
print(df['Transmission'].value_counts())

# 顯示 'Color' 列中每個唯一值的計數
print(df['Color'].value_counts())

# 顯示 'Dealer_No ' 列中每個唯一值的計數
print(df['Dealer_No '].value_counts())

# 顯示 'Body Style' 列中每個唯一值的計數
print(df['Body Style'].value_counts())

# 顯示 'Dealer_Region' 列中每個唯一值的計數
print(df['Dealer_Region'].value_counts())

# 對類別型資料進行獨熱編碼
df_encoded = pd.get_dummies(df, columns=['Gender', 'Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Dealer_No ', 'Body Style', 'Dealer_Region'])

# 顯示編碼後的資料框架
print(df_encoded)

encoded_stats = df_encoded.describe()
print("對處理過後的資料進行敘述統計")
print(encoded_stats)


# 進行特徵縮放的數值型特徵
numeric_features = ['Annual Income', 'Price ($)']

# 初始化標準化器
scaler = StandardScaler()

# 對數值型特徵進行標準化
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 選擇使用的特徵
selected_features = ['Annual Income', 'Gender', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Body Style', 'Dealer_Region']

# 將選擇的特徵作為特徵集 X
X = df[selected_features]

# 以 'Price ($)' 欄位作為目標 y
y = df['Price ($)']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train),len(X_test))


#以中位數劃分價格為低價格和高價格為目標特徵
# 計算汽車價格的中位數
price_median = df_encoded['Price ($)'].median()

# 使用中位數劃分價格為低價格和高價格
df_encoded['Price_Category'] = df_encoded['Price ($)'].apply(lambda x: 'Low' if x <= price_median else 'High')

# 將除了價格和價格分類以外的列作為特徵（X）
X = df_encoded.drop(['Price ($)', 'Price_Category'], axis=1)

# 將連續型特徵列名稱保存下來
continuous_features = ['Annual Income']

# 對連續型特徵進行標準化
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# 將目標變量 'Price_Category' 提取為 y
y = df_encoded['Price_Category']

# 將特徵 X 和目標 y 分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(len(X_train),len(X_test))















