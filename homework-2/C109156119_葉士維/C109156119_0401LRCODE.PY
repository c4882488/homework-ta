import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
df = pd.read_csv('Sleep_Efficiency.csv')

# 特徵欄位為'REM sleep percentage'和'Deep sleep percentage'，標籤欄位為'Sleep efficiency'
# 先取得特徵和標籤資料
X = df[['REM sleep percentage', 'Deep sleep percentage']]
y = df['Sleep efficiency']

# 處理缺失值，這裡假設使用平均值填補缺失值
X.fillna(X.mean(), inplace=True)

# 使用OneHotEncoder對特徵進行編碼
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立Logistic Regression模型
model = LogisticRegression()

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# 繪製預測結果
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()
