import numpy as np 
import pandas as pd
from sklearn import datasets
from io import StringIO
from sklearn.preprocessing import MinMaxScaler

#1
df = pd.read_csv('freedom_index.csv')
df = df.drop(df.columns[0],axis=1)
df = df.dropna()
print(df)

#2

iris = datasets.load_iris()
feature_df = pd.DataFrame(iris['data'],columns=iris['feature_names'])

result = feature_df.describe()
print(result)


#3
scaler = MinMaxScaler()
features_to_scale = ['Overall Score', 'Property Rights', 'Government Integrity', 'Judicial Effectiveness', 
                     'Tax Burden', 'Government Spending', 'Fiscal Health', 'Business Freedom', 'Labor Freedom', 
                     'Monetary Freedom', 'Trade Freedom', 'Investment Freedom', 'Financial Freedom']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print(df)


#4
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names:"+str(iris['feature_names']))
y = pd.DataFrame(iris['target'],columns = ['target_names'])
data = pd.concat([x,y],axis=1)
print(data.head(3))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
StandardScaler_x = scaler.fit_transform(x)
StandardScaler_x_df = pd.DataFrame(StandardScaler_x,columns=iris['feature_names'])
print(StandardScaler_x_df)


#5
print(len(x))
from sklearn.model_selection import train_test_split
x = pd.DataFrame(iris['data'],columns=iris['feature_names'])
print("target_names:"+str(iris['feature_names']))
y = pd.DataFrame(iris['target'],columns=['target_names'])
data = pd.concat([x,y],axis=1)
print(data.head(3))

X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=419)
print(len(X_train),len(X_test))