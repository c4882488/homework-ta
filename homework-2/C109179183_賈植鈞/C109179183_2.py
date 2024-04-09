import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

# 讀取CSV檔案
df = pd.read_csv('carsale_data.csv')

# 將 'Date' 欄位轉換為日期時間型態
df['Date'] = pd.to_datetime(df['Date'])

# 從 'Date' 欄位中提取年份和月份，並將其作為新的 'Year' 和 'Month' 欄位
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 檢查資料是否有缺失值
missing_values = df.isnull().sum()

# 對類別型資料進行獨熱編碼
df_encoded = pd.get_dummies(df, columns=['Gender', 'Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Dealer_No ', 'Body Style', 'Dealer_Region'])

# 進行特徵縮放的數值型特徵
numeric_features = ['Annual Income', 'Price ($)']

# 初始化標準化器
scaler = StandardScaler()

# 對數值型特徵進行標準化
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

# 選擇使用的特徵
X = df_encoded[['Annual Income', 'Transmission_Auto']].values

# 將 'Price ($)' 分成兩類：低於中位數和高於中位數
price_median = df['Price ($)'].median()
df_encoded['Price_Class'] = (df['Price ($)'] > price_median).astype(int)

# 以 'Price_Class' 為目標 y
y = df_encoded['Price_Class'].values

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Logistic Regression模型
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')

# 訓練Logistic Regression模型
lr.fit(X_train, y_train)

# 預測測試集的目標標籤
y_pred = lr.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)



# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  

# 繪製決策邊界
plot_decision_regions(X_train, y_train, clf=lr, legend=2)

# 添加軸標籤和標題
plt.xlabel('年收入 (標準化)')
plt.ylabel('Transmission_Auto')
plt.title('Logistic 迴歸的決策邊界')
plt.legend(loc='upper left')
plt.show()


