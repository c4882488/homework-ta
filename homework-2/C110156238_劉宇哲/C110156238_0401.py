import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

df = pd.read_csv('CarS.csv') 

missing_values_count = df.isnull().sum()
print(missing_values_count)

df=df.drop(['Car_id','Customer Name','Dealer_Name','Phone'],axis=1)
numeric_stats = df.describe()
categorical_stats = df.describe(include=['O'])

print("\n數值型特徵或label的描述性統計信息:")
print(numeric_stats)
print("\n類別型特徵或label的描述性統計信息:")
print(categorical_stats)

features = df[['Annual Income']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

median_price = df['Price ($)'].median()
df['Price_Category'] = df['Price ($)'].apply(lambda x: 'High' if x > median_price else 'Low')

features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
print("\n特徵縮放")
print(features_scaled_df)

categorical_features = ['Body Style','Gender', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Dealer_No ', 'Dealer_Region']

df_encoded = pd.get_dummies(df, columns=categorical_features)
print("\n類別特徵轉換後")
print(df_encoded)

# 选择特征和标签
X = df_encoded[['Annual Income', 'Company_Ford']].values
y = df_encoded['Price_Category'].map({'High': 1, 'Low': 0}).values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print("訓練集特徵形狀:", X_train.shape)
print("測試集特徵形狀:", X_test.shape)
print("訓練集標籤形狀:", y_train.shape)
print("測試集標籤形狀:", y_test.shape)

# 标准化特征
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 创建逻辑回归模型并训练
lr = LogisticRegression(C=100.0, random_state=1,max_iter=1000)
lr.fit(X_train_std, y_train)

# 绘图函数
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'green')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 绘制决策
    x1_min, x1_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    x2_min, x2_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.41, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制所有样本
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')
        
# 绘制决策边界
plot_decision_regions(X_train_std, y_train, classifier=lr)

plt.xlabel('Annual Income [standardized]')
plt.ylabel('Company_Ford [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
