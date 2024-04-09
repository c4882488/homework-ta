import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

# 讀取 CSV 檔案
with open('ForestFires.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = []
    for row in reader:
        data.append(row)
for row in data:
    print(row)

# 1. 檢查缺失資料
df = pd.read_csv('ForestFires.csv')
print(df.isnull().sum())
print('------------------------------------------------')

# 2. 每個特徵,Label敘述統計
print(df.describe())
print('------------------------------------------------')

# 3. 類別 -> code
def convert_area_description(area):
    if area == 0:
        return '無火災'
    elif area <= 5.99:
        return '小火災'
    elif area <= 25.99:
        return '中火災'
    elif area <= 1090.84:
        return '大火災'

def convert_area_category(area):
    if area == 0:
        return 0
    elif area <= 5.99:
        return 1
    elif area <= 25.99:
        return 2
    elif area <= 1090.84:
        return 3

df['area_description'] = df['area'].apply(convert_area_description)
df['area_category'] = df['area'].apply(convert_area_category)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)

print('------------------------------------------------')

# 4. 特徵縮放
# 將非數值類型的特徵進行獨熱編碼
df_encoded = pd.get_dummies(df, columns=['month', 'day'])
# 保存含有 'month' 和 'day' 列的数据作为特征
df_features = df_encoded.copy()
# 對數值類型的特徵進行標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded.drop(['area', 'area_description', 'area_category'], axis=1))
df_scaled = pd.DataFrame(scaled_features, columns=df_encoded.columns[:-3])
# 顯示標準化後的 DataFrame
print("標準化後的 DataFrame:")
print(df_scaled.head(200))

print('------------------------------------------------')

# 5. Split Data Traning/Testing
# 將特徵和目標變量分開
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)
X = df_scaled[['temp', 'ISI']]
y = df['fire']

# 拆分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("訓練集:", len(X_train))
print("測試集:", len(X_test))

print('------------------------------------------------')

# 6. 建立一個 Logistic Regression 模型
log_reg_model = LogisticRegression(C=0.1, random_state=1, solver='lbfgs', multi_class='ovr')

# 擬合模型
log_reg_model.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = log_reg_model.predict(X_test)

# 繪製訓練集和測試集的散點圖以及模型的決策邊界
plt.figure(figsize=(10, 6))

# Plot training points
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis', s=50, edgecolors='k', label='Training Set')

# Plot testing points
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='viridis', s=50, marker='x', edgecolors='k', label='Test Set')

# Plot yellow circles for testing points predicted as class 0 (no fire)
plt.scatter(X_test[y_pred == 0].iloc[:, 0], X_test[y_pred == 0].iloc[:, 1], c='yellow', s=100, edgecolors='black', label='no fire')

# Plot yellow crosses for testing points predicted as class 1 (fire)
plt.scatter(X_test[y_pred == 1].iloc[:, 0], X_test[y_pred == 1].iloc[:, 1], c='yellow', marker='x', s=100, edgecolors='black', label='fire')

# Plot decision boundary
plot_decision_regions(X_train.values, y_train.values, clf=log_reg_model, legend=2)

plt.xlabel(X.columns[0])  # 第一個特徵的名稱
plt.ylabel(X.columns[1])  # 第二個特徵的名稱
plt.title('Logistic Regression Decision Boundary')
plt.legend(loc='upper right')  # 設定圖例位置為右上角

plt.show()
