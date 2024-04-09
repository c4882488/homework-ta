# C110156246 王皓 
# Hw01_資料預處理 
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
a = 1
'''
1. 處理missing data (刪除與填補資料) 
2. 每個特徵與Label的敘述統計 
3. 特徵縮放 
4. 類別特徵轉換 
5. 訓練與測試split data 
'''
# data
with open('D:/ALL/NKUST/機器學習/Hw/data/Autism-Child-Data.arff', 'r', encoding='utf-8') as f:
    data, meta = arff.loadarff(f)

data_str = []
for row in data:
    row_str = [value.decode('utf-8') if isinstance(value, bytes) else value for value in row]
    data_str.append(row_str)
    
columns = meta.names()
df = pd.DataFrame(data_str, columns=columns)
# print(df.head)
df.to_csv('D:/ALL/NKUST/機器學習/Hw/data/Autism-Data.csv', index=False)

# drop不要的欄位
data = pd.read_csv('D:/ALL/NKUST/機器學習/Hw/data/Autism-Data.csv')
data.drop(columns=['age_desc'], inplace=True)
for i in range(1,11):
    data.drop(columns=[f"A{i}_Score"], inplace=True)

# =================================================================
# 1. 處理 missing data
missing_data = data.isnull().sum()
# print(missing_data)
num_col = data.select_dtypes(include=['number'])
data[num_col.columns] = num_col.fillna(num_col.mean()).astype(int)
missing_data = data.isnull().sum()
# print(missing_data)

data.to_csv('D:/ALL/NKUST/機器學習/Hw/data/Autism_data_OK.csv', index=False)

# =================================================================
# 2. 每個特徵與 Label的敘述統計 
data = pd.read_csv('D:/ALL/NKUST/機器學習/Hw/data/Autism_data_OK.csv')

# 數值型敘述統計
num_describe = data.describe()
print(f"數值型敘述統計：\n{num_describe}\n")

# 類別型敘述統計
categorical_features = data.select_dtypes(include=['object'])
categorical_stats = {}
print(f"類別型敘述統計：")
for column in categorical_features.columns:
    categorical_stats[column] = data[column].value_counts()
for column, stats in categorical_stats.items():
    print(f"{column}:\n{stats}\n")
    a = 1

# =================================================================
# 3. 特徵縮放
num_features = data.select_dtypes(include=['number'])

# 歸一化
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(num_features)
data[num_features.columns] = scaled_features

data.to_csv('D:/ALL/NKUST/機器學習/Hw/data/Final_Autism_data.csv', index=False)

# =================================================================
# 4. 類別特徵轉換 
# 二元類別
label_encoder = LabelEncoder()
tmpList = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
for i in range(0, len(tmpList)):
    data[tmpList[i]] = label_encoder.fit_transform(data[tmpList[i]])

# 多元類別
columns_to_encode = ['ethnicity', 'contry_of_res', 'relation']
data = pd.get_dummies(data, columns=columns_to_encode)

data.to_csv('D:/ALL/NKUST/機器學習/Hw/data/Final_Autism_data_encoded.csv', index=False)

# =================================================================
# 5. 訓練與測試split data 
data = pd.read_csv('D:/ALL/NKUST/機器學習/Hw/data/Final_Autism_data_encoded.csv')

X = data.drop(columns=['Class/ASD', 'result'])
y = data[['Class/ASD', 'result']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
