from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Load the data
file_path = 'C:\\Users\\USER\\Desktop\\機器學習\\freedom_index.csv'  # 更新檔案路徑
data = pd.read_csv(file_path)

# Define the target variable
median_value = data['Financial Freedom'].median()
data['Financial_Freedom_High'] = (data['Financial Freedom'] > median_value).astype(int)

# Selecting features
X = data[['Property Rights', 'Government Integrity']]
y = data['Financial_Freedom_High']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardizing the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Training the Logistic Regression model
lr = LogisticRegression(C=100, random_state=1, solver='lbfgs')
lr.fit(X_train_std, y_train)

# Function to plot decision boundary
def plot_decision_boundary(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Class {cl}', edgecolor='black')

# Combine standardised training and test datasets for plotting
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Plotting the decision boundary
plt.figure(figsize=(10, 6))
plot_decision_boundary(X_combined_std, y_combined, classifier=lr)
plt.xlabel('Property Rights [standardized]')
plt.ylabel('Government Integrity [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
