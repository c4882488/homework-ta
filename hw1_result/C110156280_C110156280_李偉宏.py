import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# 列印 CSV 檔案
with open('ForestFires.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = []
    for row in reader:
        data.append(row)
for row in data:
    print(row)

# 1. 檢查缺失資料
df = pd.read_csv('ForestFires.csv')
print(df.isnull().sum())
print('------------------------------------------------')

# 2. 每個特徵,Label敘述統計
print(df.describe())

print('------------------------------------------------')

# 3. 類別 -> code
def convert_area_description(area):
    if area == 0:
        return '無火災'
    elif area <= 5.99:
        return '小火災'
    elif area <= 25.99:
        return '中火災'
    elif area <= 1090.84:
        return '大火災'
def convert_area_category(area):
    if area == 0:
        return 0
    elif area <= 5.99:
        return 1
    elif area <= 25.99:
        return 2
    elif area <= 1090.84:
        return 3
df['area_description'] = df['area'].apply(convert_area_description)
df['area_category'] = df['area'].apply(convert_area_category)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)

print('------------------------------------------------')

# 4. 特徵縮放
# 將非數值類型的特徵進行獨熱編碼
df_encoded = pd.get_dummies(df, columns=['month', 'day'])
# 保存含有 'month' 和 'day' 列的数据作为特征
df_features = df_encoded.copy()
# 對數值類型的特徵進行標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded.drop(['area', 'area_description', 'area_category'], axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df_encoded.columns[:-3])
# 顯示標準化後的 DataFrame
print("標準化後的 DataFrame:")
print(df_scaled.head(200))

print('------------------------------------------------')


#5.Splint Data Traning/Testing
# 將特徵和目標變量分開
X = df_scaled.drop(['area','area_description', 'area_category'], axis=1)
y = df_scaled['area']

# 拆分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 打印訓練集和測試集的長度
print("訓練集:", len(X_train))
print("測試集:", len(X_test))

