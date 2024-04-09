#!/usr/bin/env python
# coding: utf-8

# 建立一個Logistic Regression Model 為了繪圖方便, 只需用兩個特徵加入訓練即可

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder ,LabelEncoder ,MinMaxScaler ,StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import accuracy_score ,classification_report, confusion_matrix

# sns.set(style="whitegrid")
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號


# In[ ]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data

# In[ ]:


df = pd.read_csv('/content/drive/MyDrive/大三/下學期/機器學習應用/Datasets/california_housing_train/california_housing_train_original.csv')
df # 20640 rows × 10 columns


# # Processing Data

# In[ ]:


df.rename(columns={'housing_median_age' :'housing_age'}, inplace=True)
df.rename(columns={'median_income' :'income'}, inplace=True)
df.rename(columns={'median_house_value' :'house_value'}, inplace=True)

# 平均值填入total_bedrooms的空值
mean_total_bedrooms = df['total_bedrooms'].mean()
df['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

df.describe()


# In[ ]:


numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 數值做StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 類別做LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[ ]:


df.describe()


# Original values for column 'ocean_proximity': ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']
# 
# Encoded values for column 'ocean_proximity': [3 0 1 4 2]
# 
# ocean_proximity：房子相對海洋的位置（類別：「<1H 海洋:0」、「內陸:1」、「靠近海洋:4」、「靠近海灣:3」、「島嶼:2」）

# # Split Data

# In[ ]:


X = df[['income' ,'house_value']]
y = df['ocean_proximity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Class labels:', np.unique(y))
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# # Model
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# C:正則化參數
# 
# random_state：隨機種子
# 
# solver：最佳化問題中使用的演算法
# 
# multi_class：多類別分類

# In[ ]:


lr = LogisticRegression(C=100.0, random_state=42, solver='lbfgs', multi_class='ovr')
lr.fit(X_train, y_train)


# # Evaluate

# In[ ]:


# 訓練集上的模型評分
y_train_pred = lr.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
train_classification_report = classification_report(y_train, y_train_pred)

print("Training Set Evaluation:")
print("Accuracy:", train_accuracy)
# print("Confusion Matrix:")
# print(train_conf_matrix)
# print("Classification Report:")
# print(train_classification_report)
print("\n")

# 測試集上的模型評分
y_test_pred = lr.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
test_classification_report = classification_report(y_test, y_test_pred)

print("Testing Set Evaluation:")
print("Accuracy:", test_accuracy)
print("Confusion Matrix:")
print(test_conf_matrix)
print("Classification Report:")
print(test_classification_report)


# 

# # Plot

# In[ ]:


from matplotlib.colors import ListedColormap

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
              edgecolor='white',
              alpha=0.4,
              linewidth=1,
              marker='o',
              s=40,
              label='test set')


# In[ ]:


X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined, y=y_combined, classifier=lr, test_idx=range(len(X_train), len(X_combined)))
plt.xlabel('income')
plt.ylabel('house_value')
plt.legend(loc='upper left')

plt.show()


# In[ ]:




