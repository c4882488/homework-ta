import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 读取数据集
df = pd.read_csv("C110156224_mushrooms.csv")

# 选择两个特征
features = ['cap-color', 'odor'] 

# 将特征编码为数值
label_encoders = {}
for feature in features:
    label_encoders[feature] = LabelEncoder()
    df[feature] = label_encoders[feature].fit_transform(df[feature])

# 将目标标签编码为数值
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# 划分特征和目标标签
X = df[features].values
y = df['class'].values

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建并训练Logistic Regression模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Cap Shape')
plt.ylabel('Cap Color')
plt.show()
