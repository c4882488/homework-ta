import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('freedom_index.csv')

data['Overall Score Binary'] = np.where(data['Overall Score'] > 60.9, 1, 0)

X = data[['Fiscal Health', 'Tax Burden']]
y = data['Overall Score Binary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()

model.fit(X_scaled, y)

plt.figure(figsize=(8, 6))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', edgecolors='k', label='Data')

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')

plt.xlabel('Fiscal Health')
plt.ylabel('Tax Burden')
plt.title('Logistic Regression Decision Boundary with Binary Overall Score')
plt.legend()
plt.show()
