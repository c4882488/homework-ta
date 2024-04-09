import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
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
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

# 讀取數據集
file = 'Credit card transactions-India.csv'
data = pd.read_csv(file)
data = data.drop('index', axis=1)


#預處理
low_threshold = data['Amount'].quantile(0.33)
high_threshold = data['Amount'].quantile(0.66)

#將Amount分為三個類別
data['Amount_Category'] = pd.cut(data['Amount'], bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['Low', 'Medium', 'High'])
#檢查缺失值
print(data.isnull().sum())

#敘述統計
print("數據預處理前的敘述統計:")
features = ['City', 'Date', 'Card Type', 'Exp Type', 'Gender']
for feature in features:
    if data[feature].dtype == 'object' or feature == 'City':
        print(f"\n{feature}:")
        print(data[feature].describe())
    if feature == 'Date':
        print(f"\n{feature} range:")
        print(data[feature].min(), "to", data[feature].max())
print(data['Amount'].describe())
#標籤編碼
data = pd.get_dummies(data, columns=features)
le = LabelEncoder()
data['Amount_Category'] = le.fit_transform(data['Amount_Category'])


# 分割數據為訓練集和測試集
X = data.drop(['Amount','Amount_Category'], axis=1)
y = data['Amount_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3, random_state=50)

#特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 初始化並訓練邏輯回歸模型
lr = LogisticRegression(C=50.0, random_state=1, solver='lbfgs', max_iter=1000 , n_jobs=-1)
lr.fit(X_train_scaled, y_train)

# 計算準確度
y_pred = lr.predict(X_test_scaled)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

feature_importance = np.abs(lr.coef_[0])

#找到係數最大的兩個特徵
top_two_feat_indices = np.argsort(feature_importance)[-2:]
top_two_feat_names = X_train.columns[top_two_feat_indices]

print('最重要的兩個特徵是:', top_two_feat_names)


# 分割數據為訓練集和測試集
X_new = data[['City_Shrigonda, India', 'City_Panipat, India']]
y_new = data['Amount_Category']
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size=3, random_state=50)

#特徵縮放
scaler = StandardScaler()
X_new_train_scaled = scaler.fit_transform(X_new_train)
X_new_test_scaled = scaler.transform(X_new_test)


#訓練模型
lr = LogisticRegression(C=50.0, random_state=52, solver='lbfgs', max_iter=100 , n_jobs=-1)
lr.fit(X_new_train_scaled, y_new_train)

y_new_pred = lr.predict(X_new_test_scaled)
print("準確度: ", accuracy_score(y_new_test, y_new_pred))

X_combined_std = np.vstack((X_new_train_scaled, X_new_test_scaled))
y_combined = np.hstack((y_new_train, y_new_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr)
plt.xlabel('City_Shrigonda, India')
plt.ylabel('City_Panipat, India')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
