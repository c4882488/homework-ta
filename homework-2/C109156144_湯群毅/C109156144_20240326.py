import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# 載入資料集
df = pd.read_csv('mushrooms.csv', encoding='utf-8')

# 查詢有無缺失值        
print(df.isnull().sum())

# 創建 LabelEncoder 對象
le = LabelEncoder()

# 將每個類別轉換為數值
for columns in df.columns:
    df[columns] = le.fit_transform(df[columns])

# 做 One-Hot Encoding
df = pd.get_dummies(df)
print(df)

# 把 Label 劃分出來
X = df.drop('class', axis=1)
y = df['class']

# 劃分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# 特徵縮放 常態化
mms = MinMaxScaler()
X_train_normal = mms.fit_transform(X_train)
X_test_normal = mms.fit_transform(X_test)

# 特徵縮放 標準化
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)



feature_stats = df.describe()
# 次數
counts = feature_stats.loc['count']
print("次數：\n", counts)

# 平均值
means = feature_stats.loc['mean']
print("平均值：\n", means)

# 中位數
medians = feature_stats.loc['50%']
print("中位數：\n", medians)

# 變異數
variances = feature_stats.loc['std'] ** 2
print("變異數：\n", variances)

# 標準差
std_devs = feature_stats.loc['std']
print("標準差：\n", std_devs)



# Logistic regression
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regulariztion effect
# stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))