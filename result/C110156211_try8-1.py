import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# 读取数据
data = pd.read_csv('train.csv')

# 删除不需要的列
data.drop(['Name','PassengerId','Ticket','Cabin'], axis=1, inplace=True)

# 填充缺失值
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].value_counts().index[0])

# 独热编码
dumm = pd.get_dummies(data[['Sex','Embarked']], drop_first=True)
data = pd.concat([data, dumm], axis=1)
data.drop(['Sex','Embarked'], axis=1, inplace=True)

# 数据缩放
data['Age']=(data['Age']-data['Age'].min())/(data['Age'].max()-data['Age'].min())
data['Fare']=(data['Fare']-data['Fare'].min())/(data['Fare'].max()-data['Fare'].min())

# 划分训练集和测试集
X = data[['Age', 'Fare']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建并训练模型
LR = LogisticRegression()
LR.fit(X_train, y_train)

# 输出训练集和测试集准确率
print('训练集准确率:\n', LR.score(X_train, y_train))
print('验证集准确率:\n', LR.score(X_test, y_test))

# 绘制样本数据
plt.figure(figsize=(8, 6))
plt.scatter(X_train['Age'], X_train['Fare'], c=y_train, cmap=plt.cm.coolwarm, marker='o', s=50)

# 绘制决策边界
x_min, x_max = X_train['Age'].min() - 0.1, X_train['Age'].max() + 0.1
y_min, y_max = X_train['Fare'].min() - 0.1, X_train['Fare'].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = LR.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()