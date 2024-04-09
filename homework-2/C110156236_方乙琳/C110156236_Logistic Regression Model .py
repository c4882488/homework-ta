import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


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

# 特徵與標籤的分離
特徵 = 資料[['depth', 'width']]  # 選擇 'depth' 和 'width' 兩個特徵
標籤 = 資料['brand']  # 只選擇 'brand' 作為標籤

# 特徵縮放
scaler = StandardScaler()
縮放特徵 = scaler.fit_transform(特徵)

# 訓練與測試 split data
特徵訓練, 特徵測試, 標籤訓練, 標籤測試 = train_test_split(縮放特徵, 標籤, test_size=0.2, random_state=42)

# 創建 Logistic Regression 模型
model = LogisticRegression()

# 使用兩個特徵進行訓練
model.fit(特徵訓練, 標籤訓練)

# 繪製決策邊界
# 確定繪圖範圍
x_min, x_max = 特徵訓練[:, 0].min() - 1, 特徵訓練[:, 0].max() + 1
y_min, y_max = 特徵訓練[:, 1].min() - 1, 特徵訓練[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# 進行預測
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製訓練集數據點
plt.scatter(特徵訓練[:, 0], 特徵訓練[:, 1], c=標籤訓練, cmap=plt.cm.Paired, edgecolors='k')

# 繪製決策邊界
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# 加入標籤和標題
plt.xlabel('Depth')
plt.ylabel('Width')
plt.title('Logistic Regression Decision Boundary')

# 顯示圖形
plt.show()
