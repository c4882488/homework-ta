import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('shopping.csv')
#1
print(df.isnull().sum())
#2
print(df.describe())
#3
conditions = [
    (df['Purchase Amount (USD)'] <= 39.00),
    (df['Purchase Amount (USD)'] > 39.00) & (df['Purchase Amount (USD)'] <= 80.00),
    (df['Purchase Amount (USD)'] > 81.00)
]
choices = ['低消費金額', '中消費金額', '高消費金額']

df['price_type'] = pd.np.select(conditions, choices, default='Unknown')

price = {
    '低消費金額': 1,
    '中消費金額': 2,
    '高消費金額': 3
}

df['price_type'] = df['price_type'].map(price)
inv_price_mapping = {v: k for k, v in price.items()}
print(inv_price_mapping)

print(df)
#4
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['Age', 'Review Rating', 'Previous Purchases']] = scaler.fit_transform(df[['Age', 'Review Rating', 'Previous Purchases']])
print(df)

le = LabelEncoder()
df['Purchase_Amount_Category'] = pd.cut(df['Purchase Amount (USD)'],
                                        bins=[-np.inf, 39, 80, np.inf],
                                        labels=['Low', 'Medium', 'High'])
df['Purchase_Amount_Category'] = le.fit_transform(df['Purchase_Amount_Category'])

# 選擇特徵和目標變量
X = df[['Age', 'Review Rating']].values
y = df['Purchase_Amount_Category'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# 標準化特徵
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 使用Perceptron模型
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

# 繪製決策邊界
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o')  # Extend markers if needed
    colors = ('red', 'blue', 'lightgreen')
    print("Number of unique classes:", len(np.unique(y)))
    print("Length of colors tuple:", len(colors))
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx % len(markers)],  # Use modulo to ensure marker cycling
                    label=cl, 
                    edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='None',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

# Call the function to see the printed output



X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

# 使用LogisticRegression模型
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('Age [standardized]')
plt.ylabel('Review Rating [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()