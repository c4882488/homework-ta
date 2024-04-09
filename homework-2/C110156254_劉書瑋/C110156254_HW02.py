import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 載入 Pokemon.csv 數據集
pokemon_data = pd.read_csv('Pokemon.csv')

# 只保留 'Attack' 和 'Defense' 兩個特徵，以及 'Legendary' 作為標籤
data = pokemon_data[['Attack', 'Defense', 'Legendary']]

# 檢查數據集是否存在缺失值
print("缺失值概覽：\n", data.isnull().sum())

# 分割特徵和標籤
X = data.drop('Legendary', axis=1)
y = data['Legendary']

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練 Logistic Regression 模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測測試集數據
y_pred = model.predict(X_test)

# 計算模型準確率
accuracy = accuracy_score(y_test, y_pred)
print("模型準確率：", accuracy)

# 繪製混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
