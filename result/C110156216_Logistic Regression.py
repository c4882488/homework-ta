import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('salary.csv')

features = ['education-num', 'hours-per-week']
label = 'salary'

data = data[[*features, label]]

le = LabelEncoder()
data[label] = le.fit_transform(data[label])

X_train, X_test, y_train, y_test = train_test_split(data[features], data[label], test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, marker='o', edgecolors='k', label='Train set')
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, marker='s', edgecolors='k', label='Test set')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
