import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv("./AirQualityUCI.csv", delimiter=";")

# 資料預處理
data.replace([-200, "-200,0"], float("nan"), inplace=True)
filled_data = data.fillna(data.median())
filled_data.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)

# 特徵縮放
filled_data = filled_data.replace(',', '.', regex=True)
for col in filled_data.columns[2:]:
    filled_data[col] = filled_data[col].astype(float)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(filled_data.iloc[:, 2:])
scaled_data = pd.DataFrame(scaled_features, columns=filled_data.columns[2:])
scaled_data['Date'] = filled_data['Date']
scaled_data['Time'] = filled_data['Time']

# 數值轉類別的特徵轉換 (CO(GT))


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

# 訓練與測試分割
X = scaled_data[['PT08.S1(CO)', 'PT08.S2(NMHC)']]
y = scaled_data['CO_category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=69)

# Logistic Regression
model = LogisticRegression(C=100, random_state=100,
                           solver='lbfgs', multi_class='ovr')
model.fit(X_train, y_train)


# 畫圖
plt.figure(figsize=(10, 6))
plt.scatter(X_test['PT08.S1(CO)'], X_test['PT08.S2(NMHC)'],
            c=y_test, cmap='viridis', edgecolors='k', label='Data')

x_min, x_max = X_test['PT08.S1(CO)'].min() - 1, X_test['PT08.S1(CO)'].max() + 1
y_min, y_max = X_test['PT08.S2(NMHC)'].min() - \
    1, X_test['PT08.S2(NMHC)'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis',
             levels=[-0.5, 0.5, 1.5, 2.5], extend='both')

plt.xlabel('PT08.S1(CO)')
plt.ylabel('PT08.S2(NMHC)')
plt.title('Logistic Regression Decision Boundary')

plt.colorbar(ticks=[0, 1, 2])
plt.legend()
plt.show()
