import numpy as np
import pandas as pd
from sklearn import datasets
from io import StringIO

###1.預處理
df=pd.read_csv('housing.csv')
# print(df)

df.dropna()


###2.敘述統計

iris=datasets.load_iris()
feature_df=pd.DataFrame(iris['data'],columns=iris['feature_names'])

result=feature_df.describe()
print(result)

result.loc['var']=result.loc['std']*result.loc['std']


###3.特徵縮放
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
StandardScaler_x=scaler.fit_transform(x)
StandardScaler_x_df=pd.DataFrame(StandardScaler_x,columns=iris['feature_names'])

###4.類別資料處理

#有序類別
print(df['ocean_proximity'].unique())
ocean_proximity_mapping={
    'NEAR BAY':3,
    '<1H OCEAN':0,
    'NEAR OCEAN':4,
    'INLAND':1,
    'ISLAND':2,
}
df['ocean_proximity']=df['ocean_proximity'].map(ocean_proximity_mapping)

print(df)

##5.訓練測試
x=pd.DataFrame(iris['data'],columns=iris['feature_names'])
y=pd.DataFrame(iris['target'],columns=['target_names'])
data=pd.concat([x,y],axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=419)
print(len(X_train),len(X_test))