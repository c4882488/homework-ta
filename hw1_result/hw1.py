import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

csv_file_path = "OnlineNewsPopularity.csv"
csv_file_path2="mushrooms.csv"
df = pd.read_csv(csv_file_path)
df2=pd.read_csv(csv_file_path2)


#Q1-1 刪除遺漏
print(df.isnull().sum())#尋找遺漏值
df.dropna(axis=0)#刪除遺漏值(列)

#Q1-2 填補遺漏
imr = SimpleImputer(missing_values=np.nan, strategy='mean')#(平均插補)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(df)

#Q2   敘述統計
description=df.describe()
print(description)

#Q3 類別->code  此為第二筆資料集
class_mapping={label:idx for idx,label in enumerate(np.unique(df2["class"]))}
df2['class']=df2["class"].map(class_mapping)

print(df2.iloc[:,0]) 

#Q4 
X,y=df2.iloc[:,:1].values,df2.iloc[:,:1].values
X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

#Q5

mms=MinMaxScaler()
X_train_norm=mms.fit_transform(X_train)
X_test_norm=mms.transform(X_test)
