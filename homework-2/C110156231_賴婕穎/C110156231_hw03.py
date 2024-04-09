import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import seaborn as sns

df=pd.read_csv("C:/Users/user/Desktop/機器學習/shopping_trends/hw/shopping_trends_updated.csv")

print(df.isnull().sum())

print(df.describe(include='all'))

size_mapping={'XL': 4, 'L': 3, 'M': 2, 'S': 1}
df['Size']=df['Size'].map(size_mapping)

# ohe1_data=pd.get_dummies(df[['Gender', 'Subscription Status', 'Discount Applied', 'Promo Code Used']])
# ohe1_data=ohe1_data.astype(int)
# print(ohe1_data)

# ohe2_data=pd.get_dummies(df[['Category', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']])
# ohe2_data=ohe2_data.astype(int)
# print(ohe2_data)

le=LabelEncoder()
features=['Gender', 'Subscription Status', 'Discount Applied', 'Promo Code Used', 'Category', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases', 'Item Purchased', 'Location', 'Color']  # 这里列出你所有非树执行的特征
for feature in features:
    df[feature]=le.fit_transform(df[feature])

X=df[['Discount Applied', 'Purchase Amount (USD)']].values
y=le.fit_transform(df['Subscription Status'])

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Class labels:', np.unique(y))
print('Labels count in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))

stdsc=StandardScaler()
stdsc.fit(X_train)
X_train_std=stdsc.transform(X_train)
X_test_std=stdsc.transform(X_test)

sh=LogisticRegression(C=1.0, random_state=42, solver='sag')
sh.fit(X_train_std, y_train)

X_combined_std=np.vstack((X_train_std, X_test_std))
y_combined=np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined, clf=sh)
plt.xlabel('Discount Applied [standardized]')
plt.ylabel('Purchase Amount (USD) [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# sh.predict_proba(X_test_std[:3, :])
# sh.predict_proba(X_test_std[:3, :]).sum(axis=1)
# sh.predict_proba(X_test_std[:3, :]).argmax(axis=1)
# sh.predict(X_test_std[:3, :])
# sh.predict(X_test_std[0, :].reshape(1, -1))

# correlation_matrix=df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap of Features')
# plt.show()