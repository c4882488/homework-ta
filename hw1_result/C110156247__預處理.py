#!/usr/bin/env python
# coding: utf-8

# 請根據各自組別資料, 利用程式完成下列的要求:
# 1. 處理missing data (刪除與填補資料)
# 2. 每個特徵與Label的敘述統計
# 3. 特徵縮放
# 4. 類別特徵轉換
# 5. 訓練與測試split data

# # Load Data

# In[ ]:


get_ipython().system(' pip install -q kaggle')
from google.colab import files
files.upload()
# 選擇剛剛下載好的 kaggle.json 檔案


# In[ ]:


get_ipython().system(' mkdir ~/.kaggle')
get_ipython().system(' cp kaggle.json ~/.kaggle/')
get_ipython().system(' chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system(' kaggle datasets download -d camnugent/california-housing-prices')


# In[ ]:


get_ipython().system(' mkdir california-housing-prices')

# 將剛剛載下來的.zip壓縮檔解壓縮進資料夾裡
get_ipython().system(' unzip california-housing-prices.zip -d california-housing-prices')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import missingno as msno

from sklearn.preprocessing import OneHotEncoder ,LabelEncoder ,MinMaxScaler ,StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import joblib

sns.set(style="whitegrid")
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號


# # EDA

# In[ ]:


# df = pd.read_csv('/content/drive/MyDrive/大三/下學期/機器學習應用/Datasets/california_housing_train/california_housing_train_original.csv')
df = pd.read_csv('/content/california_housing_train_original.csv')
df # 20640 rows × 10 columns


# In[ ]:


df.info()


# # Processing data

# In[ ]:


df.rename(columns={'housing_median_age' :'housing_age'}, inplace=True)
df.rename(columns={'median_income' :'income'}, inplace=True)
df.rename(columns={'median_house_value' :'house_value'}, inplace=True)

df


# # Missing Value

# In[ ]:


msno.matrix(df)


# In[ ]:


df.isnull().sum()


# total_bedrooms有207筆的缺失數值，其餘欄位都沒有缺失值

# In[ ]:


df.describe()


# 依作業要求，這邊就只處理缺失值，異常值之後再做處理

# In[ ]:


mean_total_bedrooms = df['total_bedrooms'].mean()

# 平均值填入total_bedrooms的空值
df['total_bedrooms'].fillna(mean_total_bedrooms, inplace=True)

df.describe()


# In[ ]:


df.isnull().sum()


# # Statistics

# In[ ]:


df.describe(include=['object'])


# In[ ]:


df.describe()


# # Feature Engineering

# In[13]:


numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 數值做StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 類別做LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[ ]:


df


# In[ ]:


df.describe()


# 經過標準化的轉換後，可看到各欄位的均值趨近於0，方差為1
# 
# 

# In[ ]:


# use_columns = ['ocean_proximity']
# df_copy = df.copy()

# label_encoder = LabelEncoder()

# for col in use_columns:
#     df_copy[col + '_encoded'] = label_encoder.fit_transform(df[col])

#     original_values = df_copy[col].unique()
#     encoded_values = df_copy[col + '_encoded'].unique()
#     print(f"Original values for column '{col}': {original_values}")
#     print(f"Encoded values for column '{col}': {encoded_values}")
#     print()

# for col in use_columns:
#     df[col] = label_encoder.fit_transform(df[col])


# Original values for column 'ocean_proximity': ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']
# 
# Encoded values for column 'ocean_proximity': [3 0 1 4 2]

# ocean_proximity：房子相對海洋的位置（類別：「<1H 海洋:0」、「內陸:1」、「靠近海洋:4」、「靠近海灣:3」、「島嶼:2」）

# # Split data

# In[16]:


X = df.drop('house_value', axis=1)
y = df['house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[ ]:




