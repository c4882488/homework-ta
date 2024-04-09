import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 讀取 car.data 文件
df = pd.read_csv("car.data", header=None)

# 1. 處理缺失值
# 檢查是否有缺失值
missing_values = df.isnull().sum().sum()

if missing_values == 0:
    print("car.data 文件中沒有缺失值。")
else:
    print("car.data 文件中有缺失值，總共有 {} 個缺失值。".format(missing_values))

# 2. 每個特徵與 Label 的敘述統計
print("敘述統計：")
print(df.describe())

# 3. 特徵縮放
# 使用 LabelEncoder 將字符串特徵轉換為數值
label_encoder = LabelEncoder()
for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

# 使用標準化對特徵進行縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(df.columns[-1], axis=1))  # 使用 iloc 切片選取除了最後一列之外的所有特徵

# 將縮放後的特徵轉換為 DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# 將縮放後的特徵與標籤合併
df_scaled['class'] = df[df.columns[-1]]

# 4. 訓練與測試數據集分割
X = df_scaled.drop('class', axis=1)
y = df_scaled['class']

# 將數據集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印訓練集和測試集的形狀
print("訓練集特徵形狀：", X_train.shape)
print("訓練集標籤形狀：", y_train.shape)
print("測試集特徵形狀：", X_test.shape)
print("測試集標籤形狀：", y_test.shape)
