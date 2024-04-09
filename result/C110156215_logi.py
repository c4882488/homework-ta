import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.family'] = ['Microsoft YaHei']

# 從 CSV 檔案載入資料
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# 創建 Logistic Regression 模型
model = LogisticRegression()

# 將模型與訓練集進行訓練
model.fit(X_train, y_train)

# 繪製決策邊界
x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, marker='o', edgecolors='k')
plt.xlabel('火災範圍')
plt.ylabel('溫度')
plt.title('邏輯回歸模型')
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 讀取資料
data = pd.read_csv('data.csv')

# 處理missing data (用眾數填補資料)
data.fillna(data.mode().iloc[0], inplace=True)

# 只保留 'month' 和 'day' 兩個特徵
data = data[['area', 'temp', 'Y']]

# 類別特徵轉換
label_encoders = {}

for feature in ['area', 'temp']:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# 特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(['Y'], axis=1))  # Y是標籤，不進行縮放
scaled_data = pd.DataFrame(scaled_features, columns=['area', 'temp'])  # 重新構建DataFrame，僅包含 'month' 和 'day'

# 訓練與測試split data
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['Y'], test_size=0.2, random_state=42)

# 打印部分處理後的資料
print("\n經處理後的資料 (部分示例):")
print(X_train.head())

# 保存處理後的資料
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
