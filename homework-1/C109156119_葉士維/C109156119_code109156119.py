import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 讀取資料
data = pd.read_csv("Sleep_Efficiency.csv")
data = pd.get_dummies(data, columns=['Gender'])
data = pd.get_dummies(data, columns=['Smoking status'])
data['Sleep efficiency'] = pd.cut(data['Sleep efficiency'], bins=[0, 0.5, 1.0], labels=[0, 1])

#Label是睡眠效率(sleep efficinecy)，特徵有Age,Gender,Sleep durationREM ,sleep percentage ,
# Deep sleep percentage ,Light sleep percentage, Awakenings,Caffeine,consumption,
# Alcohol consumption,Smoking status,Exercise frequency

# 分割訓練集和測試集
X = data.drop('Sleep efficiency', axis=1)
y = data['Sleep efficiency']
print(y.dtypes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 填補缺失值
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 建立模型
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# 在測試集上評估模型
score = classifier.score(X_test, y_test)
print(f"Accuracy: {score}")

# 繪製分類圖
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.title("Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()