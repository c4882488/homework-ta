import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 載入數據集
data = pd.read_csv('data.csv')

# 將類別變量轉換為虛擬變量
data = pd.get_dummies(data, columns=['City', 'Card Type', 'Exp Type'])

# 將性別（Gender）轉換為數值型特徵
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# 將數據集分為特徵變量和目標變量
X = data[['Amount', 'Exp Type_Bills']]  # 選擇兩個特徵進行訓練和可視化
y = data['Gender']

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Logistic Regression模型
model = LogisticRegression()

# 使用訓練集來訓練模型
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("準確率:", accuracy)


x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Amount')
plt.ylabel('Exp Type_Bills')
plt.title('Logistic Regression Decision Boundary')
plt.show()
