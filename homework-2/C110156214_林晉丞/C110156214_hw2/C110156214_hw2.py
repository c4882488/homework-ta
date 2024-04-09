import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# 加載資料集
csv_file_path2 = "mushrooms.csv"
df2 = pd.read_csv(csv_file_path2)

# 對非數值型的分類屬性進行獨熱編碼
df2_encoded = pd.get_dummies(df2)

# 特徵和標籤
X = df2_encoded.drop(columns=['class_p', 'class_e']).values
y = df2_encoded['class_p'].values

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徵縮放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用 PCA 對數據進行降維到二維
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 訓練邏輯回歸模型
lr_pca = LogisticRegression()
lr_pca.fit(X_train_pca, y_train)

# 使用訓練集和測試集的特徵值
X_combined_pca = np.vstack((X_train_pca, X_test_pca))
# 使用訓練集和測試集的標籤
y_combined = np.hstack((y_train, y_test))

# 定義繪製決策區域圖的函數
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # 繪製決策邊界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # 繪製等高線圖
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # 繪製樣本點
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=[cmap(idx)],
                    marker=markers[idx], label=cl)
    
    # 高亮顯示測試集樣本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='yellow', # 設置測試集樣本的顏色為黃色
                    edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test Set')

# 繪製決策區域圖
plt.figure(figsize=(10, 6))
plot_decision_regions(X_combined_pca, y_combined, classifier=lr_pca, test_idx=range(len(X_train_pca), len(X_combined_pca)))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper left')
plt.title('Decision Region Plot with PCA')
plt.tight_layout()
plt.show()
