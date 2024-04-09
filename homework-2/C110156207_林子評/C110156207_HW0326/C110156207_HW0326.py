import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# 讀取CSV檔案
data = pd.read_csv('D:/#上課用/01課程資料/機器學習應用/HW0326/log.csv')

# 1. 處理missing data
# 找出缺失值
missing_data = data.isnull().sum()
print("Missing Data:")
print(missing_data)

# 處理缺失值
data.dropna(inplace=True) #刪除

# 2. 每個特徵與Label的敘述統計
statistics = data.describe()
print("\nDescriptive Statistics:")
print(statistics)

# 3. 特徵縮放
# 切分特徵和標籤
X = data.drop('Y house price of unit area', axis=1)
y = data['Y house price of unit area']

# 初始化特徵縮放器
scaler = StandardScaler()

# 對特徵進行縮放
X_scaled = scaler.fit_transform(X)

# 4. 類別特徵轉換
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. 訓練與測試split data
# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 打印訓練集和測試集的大小
print("\nTrain/Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 讓使用者選擇兩個特徵
print("\nAvailable Features:")
for i, feature in enumerate(X.columns):
    print(f"{i + 1}. {feature}")

selected_features = input("\nEnter the numbers of the two features you want to use (example：4,5): ").strip().split(',')
selected_features = [int(x) - 1 for x in selected_features]  # 將用戶輸入的特徵號轉換為索引

# 建立並訓練Logistic Regression模型
model = LogisticRegression()
model.fit(X_train[:, selected_features], y_train)  # 只使用選擇的兩個特徵進行訓練

# 繪製決策邊界
plt.figure(figsize=(10, 6))

# 繪製訓練集資料點
plt.scatter(X_train[:, selected_features[0]], X_train[:, selected_features[1]], c=y_train, cmap='coolwarm', edgecolors='k', label='Train Data')

# 繪製決策邊界
x_min, x_max = X_scaled[:, selected_features[0]].min() - 1, X_scaled[:, selected_features[0]].max() + 1
y_min, y_max = X_scaled[:, selected_features[1]].min() - 1, X_scaled[:, selected_features[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# 加上圖例和標題
plt.xlabel(X.columns[selected_features[0]] + ' (scaled)')
plt.ylabel(X.columns[selected_features[1]] + ' (scaled)')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
