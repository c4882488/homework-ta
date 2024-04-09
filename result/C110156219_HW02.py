import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_csv("AirQualityUCI.csv", delimiter=";")

# 將-200替換成NaN
data.replace(-200, np.nan, inplace=True)

# 替換逗號為點
data.replace(',', '.', regex=True, inplace=True)

# 定義特徵和標籤
features = data.drop(columns=["CO(GT)", "Date", "Time"])  # 移除非數值的欄位
label = data["CO(GT)"]

# 處理missing data
imputer = SimpleImputer(strategy="mean")
imputed_features = imputer.fit_transform(features)

# 將numpy array轉換為DataFrame
imputed_features = pd.DataFrame(imputed_features, columns=features.columns)


# 將特徵和標籤重新連接
data = pd.concat([data[['Date', 'Time']], imputed_features, label], axis=1, ignore_index=True)

# 重新設置索引
data.columns = ['Date', 'Time'] + list(imputed_features.columns) + ['CO(GT)']

# 特徵縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(imputed_features)

# 訓練與測試split data
X_train, X_test, y_train, y_test = train_test_split(scaled_features, label, test_size=0.2, random_state=42)

# 建立Logistic Regression模型
log_reg = LogisticRegression()

# 只使用兩個特徵進行訓練
X_train_2d = X_train[:, :2]
log_reg.fit(X_train_2d, y_train)

# 繪製決策邊界
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, s=20, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
