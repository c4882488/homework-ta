import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer

# read csv
data=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# 檢查是否有無缺失值(有的話將以平均值填補缺失值)
print('1.檢查是否有無缺失值(有的話將以平均值填補缺失值)')
print()
numValue=data.isnull().sum()
if numValue.any()==0:
    print(data.isnull().sum())
else:
    data.fillna(data.mean(),inplace=True)

print('='*160)
# 敘述統計
print('2.敘述統計')
print()
label_column='NObeyesdad'
description=data.groupby(label_column).describe()
print(description)

print('='*160)

# 特徵處理
categorical_features = data.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(), categorical_features.tolist())],
    remainder='passthrough')
transformed_data = preprocessor.fit_transform(data)

# 特徵縮放
print('3.特徵縮放')
print()
scaler=StandardScaler()
scaler_features=scaler.fit_transform(transformed_data)
print(scaler_features)

print('='*160)

# 類別特徵轉換
print('4.類別特徵轉換')
print()
label_encoder=LabelEncoder()
data[label_column]=label_encoder.fit_transform((data[label_column]))
print(data[label_column])

print('='*160)
# 訓練與測試split data 
print('5.訓練與測試split data ')
X=scaler_features
y=data[label_column]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=80)

print(f'X_train:{X_train.shape}')
print(f'X_test:{X_test.shape}')
print(f'y_train:{y_train.shape}')
print(f'y_test:{y_test.shape}')