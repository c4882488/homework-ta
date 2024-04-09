import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('forestfires.csv')

X = df[['FFMC', 'temp']]
y = df['area'].apply(lambda x: 1 if x > 0 else 0) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

weights = model.coef_[0]
bias = model.intercept_

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', edgecolor='k', s=20)

x_boundary = np.array([X_train[:, 0].min(), X_train[:, 0].max()])
y_boundary = -(bias + weights[0] * x_boundary) / weights[1]

plt.plot(x_boundary, y_boundary, color='red') 
plt.xlabel('FFMC', fontsize=14)
plt.ylabel('temp', fontsize=14)
plt.title('Logistic Regression', fontsize=16)
plt.show()
