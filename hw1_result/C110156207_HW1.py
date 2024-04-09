import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 讀取CSV檔案
data = pd.read_csv('HW1.csv')

# 1. 處理missing data
# 找出缺失值
missing_data = data.isnull().sum()
print("Missing Data:")
print(missing_data)

# 處理缺失值
data.dropna(inplace=True) #刪除

# 2. 每個特徵與Label的敘述統計
statistics = data.describe()
print("\nDescriptive Statistics:")
print(statistics)

# 3. 特徵縮放
# 切分特徵和標籤
X = data.drop('Y house price of unit area', axis=1)
y = data['Y house price of unit area']

# 初始化特徵縮放器
scaler = StandardScaler()

# 對特徵進行縮放
X_scaled = scaler.fit_transform(X)

# 4. 類別特徵轉換
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. 訓練與測試split data
# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 打印訓練集和測試集的大小
print("\nTrain/Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
