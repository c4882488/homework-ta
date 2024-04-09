import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


url = pd.read_csv('SleepEfficiency.csv')
mms = MinMaxScaler()

df = pd.DataFrame(url)
print(df)
# 使用平均值填補空值
mean_fill1 = df["Awakenings"].mean()
mean_fill2 = df["Caffeine consumption"].mean()
mean_fill3 = df["Alcohol consumption"].mean()
mean_fill4 = df["Exercise frequency"].mean()
median_value = df["Exercise frequency"].median()
print("Exercise frequency 的中位數為："+ str(median_value))
print("Exercise frequency 的平均數為："+ str(mean_fill4))
df = df.fillna(mean_fill1)
df = df.fillna(mean_fill2)
df = df.fillna(mean_fill3)
df = df.fillna(mean_fill4)
print(df.isnull().sum())

# edible = df[df["class"] == "e"]
# # print(edible)
# poisonous = df[df["class"] == "p"]
# print(poisonous)
X = df.drop(labels=["Sleep efficiency"], axis=1)
# print(x)
y = df["Sleep efficiency"]
X = pd.get_dummies(df)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print(X_test_norm)
# 建立邏輯回歸模型
# 建立線性迴歸模型
model = LinearRegression()

# 訓練模型
model.fit(X_train_norm, y_train)

# 使用已經訓練好的模型對測試集進行預測
y_pred = model.predict(X_test_norm)

# 計算均方誤差
mse = mean_squared_error(y_test, y_pred)
print("均方誤差：", mse)
