import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 讀取 car.data 文件
df = pd.read_csv("car.data", header=None)

# 將類別特徵轉換為數值
label_encoder = LabelEncoder()
for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

# 使用標準化對特徵進行縮放
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(df.columns[-1], axis=1))

# 將縮放後的特徵轉換為 DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

# 將縮放後的特徵與標籤合併
df_scaled['class'] = df[df.columns[-1]]

# 選擇兩個特徵
X = df_scaled.iloc[:, :2]  # 選擇前兩列作為特徵
y = df_scaled['class']

# 將數據集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立 Logistic Regression 模型
model = LogisticRegression()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print("模型準確率：", accuracy)

# 繪製決策邊界
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')

# 繪製決策邊界
coefficients = model.coef_[0]
intercept = model.intercept_
x_values = [-2, 2]
y_values = [-(coefficients[0] * x + intercept) / coefficients[1] for x in x_values]
plt.plot(x_values, y_values, color='red')

plt.title('Logistic Regression Decision Boundary')
plt.legend(loc='upper left')
plt.show()
