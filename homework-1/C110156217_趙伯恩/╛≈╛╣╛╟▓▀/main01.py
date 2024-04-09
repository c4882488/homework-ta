import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("./AirQualityUCI.csv", delimiter=";")

# 資料預處理
# 1.處理缺失值，將-200跟-200,0替換為 nan
data.replace(-200, float("nan"), inplace=True)
data.replace("-200,0", float("nan"), inplace=True) 
# 用中位數填補nan
filled_data = data.fillna(data.median())
# 刪除 'Unnamed: 15' 和 'Unnamed: 16' 這兩欄
filled_data.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)


# 2. 每個特徵與 Label 的敘述統計
description = filled_data.describe()
print(description)


# 3. 特徵縮放
filled_data = filled_data.replace(',', '.', regex=True) # 因為原始資料是用","來當作小數點，我這裡把它改回來
# 轉換特徵欄位為浮點數
for col in filled_data.columns[2:]:
    filled_data[col] = filled_data[col].astype(float)
# 特徵縮放(標準化)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(filled_data.iloc[:, 2:])
# 將縮放後的特徵重新轉換為 DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=filled_data.columns[2:])
scaled_data['Date'] = filled_data['Date']
scaled_data['Time'] = filled_data['Time']
# print(scaled_data.head())


# 4. 數值轉類別的特徵轉換 (CO(GT))
def categorize_CO(value):
    if 0 <= value <= 4:
        return 0  # 低濃度
    elif 5 <= value <= 8:
        return 1  # 中濃度
    elif 9 <= value <= 12:
        return 2  # 高濃度
    else:
        return -1  # 無效值

scaled_data['CO_category'] = filled_data["CO(GT)"].apply(categorize_CO)

print(scaled_data.head())


# 5. 訓練與測試 split data (70% train，30% test)
X = scaled_data.drop(columns=["CO_category"])
y = scaled_data["CO_category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

print("訓練集大小:", X_train.shape[0])
print("測試集大小:", X_test.shape[0])