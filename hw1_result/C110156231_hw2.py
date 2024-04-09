import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

shopping_data=pd.read_csv("C:/Users/user/Desktop/機器學習/shopping_trends/shopping_trends_updated.csv")

columns=['Age','Purchase Amount (USD)','Review Rating','Previous Purchases']
df=shopping_data[columns]

print(shopping_data.isnull().sum())

# print(df.describe(include='all'))

for column in columns:
    print(f"\n-----------------------------{column}---------------------------------")

    countN=df[column].count()
    print(f"次數:{countN}")

    meanN=np.mean(df[column])
    print(f"平均數:{meanN}")

    medianN=np.median(df[column])
    print(f"中位數:{medianN}")

    varianceN=np.var(df[column])
    print(f"變異數:{varianceN}")

    stdevN=np.std(df[column])
    print(f"標準差:{stdevN}")


size_mapping={'XL':4, 'L':3, 'M':2, 'S':1}
shopping_data['Size']=shopping_data['Size'].map(size_mapping)
print(shopping_data['Size'])

ohe1_data=pd.get_dummies(shopping_data[['Gender', 'Subscription Status', 'Discount Applied', 'Promo Code Used']])
ohe1_data=ohe1_data.astype(int)
print(ohe1_data)

# ohe2_data=pd.get_dummies(shopping_data[['Category', 'Season', 'Subscription Status', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']])
# ohe2_data=ohe2_data.astype(int)
# print(ohe2_data)

X, y = shopping_data[['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']].values, shopping_data['Purchase Amount (USD)'].values
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print('\nX_train:',len(X_train))
print('X_test:',len(X_test))
print('y_train:',len(y_train))
print('y_test:',len(y_test)) 


stdsc=StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print('\nX_train_std:\n',X_train_std)
print('\nX_test_std:\n',X_test_std)
