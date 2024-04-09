import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 讀取資料
路徑 = 'C:/Users/88693/Desktop/ML/best_buy_laptops_2024.csv'
資料 = pd.read_csv(路徑)

print(資料.shape)
資料.head() 

# 1. 處理 missing data
# 選擇數值型的特徵並以平均值填補缺失值
缺失值數量= 資料.isnull().sum()
print('缺失值數量:')
print(缺失值數量)


數值特徵 = 資料.select_dtypes(include=['number'])
資料.fillna(數值特徵.mean(), inplace=True)

#  類別特徵轉換
label_encoder = LabelEncoder()
for col in 資料.columns:
    if 資料[col].dtype == 'object':
        資料[col] = label_encoder.fit_transform(資料[col])

#  特徵與標籤的分離
特徵 = 資料.drop(columns=['aggregateRating/ratingValue', 'aggregateRating/reviewCount', 'offers/price', 'depth', 'width'])
標籤 = 資料[['aggregateRating/ratingValue', 'aggregateRating/reviewCount', 'offers/price', 'depth', 'width']]

#特徵與Label的敘述統計
print("特徵描述統計:")
print(特徵.describe())
print("\n標籤描述統計:")
print(標籤.describe())

# 計算特徵的變異數
特徵變異數 = 特徵.var()
print("特徵的變異數:")
print(特徵變異數)

# 計算標籤的變異數
標籤變異數 = 標籤.var()
print("\n標籤的變異數:")
print(標籤變異數)


# 4. 特徵縮放
scaler = StandardScaler()
縮放特徵 = scaler.fit_transform(特徵)
print('縮放特徵')
print(縮放特徵)


# 5. 訓練與測試 split data
特徵訓練, 特徵測試, 標籤訓練, 標籤測試 = train_test_split(縮放特徵, 標籤, test_size=0.2, random_state=42)
print('訓練資料:')
print('特徵')
print(特徵訓練)
print('標籤')
print(標籤訓練)
print('測試資料')
print('特徵')
print(特徵測試)
print('標籤')
print(標籤測試)

