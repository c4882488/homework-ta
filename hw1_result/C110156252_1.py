import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 讀取 CSV 文件
data = pd.read_csv('Child-Data.csv')

# 處理缺失值
missing_data = data.isnull().sum() 
num_col = data.select_dtypes(include=['number']) 
data[num_col.columns] = num_col.fillna(num_col.mean()).astype(int) 
missing_data = data.isnull().sum() 

# 輸出數值型和類別型敘述統計

categorical_stats = {} 
print("數值型敘述統計：")
print(data.describe())
print("\n類別型敘述統計：")
for column in data.select_dtypes(include=['object']).columns:
    print(data[column].value_counts())

#  特徵縮放
scaler = MinMaxScaler() 
scaled_features = scaler.fit_transform(num_col) 
data[num_col.columns] = scaled_features 

# 將處理後的數據保存到 CSV 文件中
data.to_csv('child_data_good.csv', index=False) 
  
# 類別型特徵轉換
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['jundice'] = label_encoder.fit_transform(data['jundice'])
data['austim'] = label_encoder.fit_transform(data['austim'])
data['used_app_before'] = label_encoder.fit_transform(data['used_app_before'])
data['Class/ASD'] = label_encoder.fit_transform(data['Class/ASD'])

# 將類別特徵進行獨熱編碼
data = pd.get_dummies(data, columns=['ethnicity', 'contry_of_res', 'relation'])



# 將處理後的數據保存到 CSV 文件中
data.to_csv('child_dataencoded.csv', index=False) 

# 訓練與測試分割
data = pd.read_csv('child_dataencoded.csv') 
X = data.drop(columns=['Class/ASD', 'result']) 
y = data[['Class/ASD', 'result']] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 輸出資料形狀
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
