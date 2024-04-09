import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=3)

model = LogisticRegression().fit(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o', edgecolors='k')

x_values = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_values = -(model.coef_[0][0]*x_values + model.intercept_) / model.coef_[0][1]
plt.plot(x_values, y_values, color='g', linestyle='--', linewidth=2)

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.colorbar(label='Class')

# 顯示圖表
plt.show()
