#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
from io import StringIO
import sys

#讀取excel資料檔
df = pd.read_excel("AirQualityUCI.xlsx")
df


# In[3]:


#檢查缺失值或遺漏值
df.isnull().sum()


# In[5]:


#1.處理missing data
#原本資料的缺失值就有-200了，為沒有偵測到的意思，這邊將其改為0
#將所有的-200替換為0
df.replace(-200, 0, inplace=True)
print(df)


# In[6]:


#2.每個特徵與Label的敘述統計
#使用describe()方法計算描述統計信息(包括平均數、中位數、標準差、最小值、最大值)
statistics = df.describe()

#打印结果
print(statistics)


# In[28]:


#3.類別特徵轉換
#定義一個函數，根據數值傳回對應的編碼
def map_CO_GT(value):
    if value <= 4:
        return 0
    elif value <= 8:
        return 1
    elif value <= 12:
        return 2
    else:
        return value #如果超出了指定範圍，則保持原始值不變

#使用apply()方法將CO(GT)列根據映射函數進行編碼，並儲存在新的列 'CO(GT)_code' 中
df['CO(GT)_code'] = df['CO(GT)'].apply(map_CO_GT)

#列印CO(GT)列的前30行編碼值
print(df['CO(GT)_code'].head(30))


# In[35]:


#找出濃度為1、2的行的CO(GT)列
CO_GT_1 = df.loc[df['CO(GT)_code'] == 1, 'CO(GT)']
CO_GT_2 = df.loc[df['CO(GT)_code'] == 2, 'CO(GT)']

#使用先前定義的函數將CO(GT)列的值根據編碼標準顯示為對應的濃度等級
CO_GT_1_mapped = CO_GT_1.map(map_CO_GT)
CO_GT_2_mapped = CO_GT_2.map(map_CO_GT)

#列印CO(GT)列濃度為1和2的行，依編碼標準顯示濃度等級
print(CO_GT_1_mapped)
print(CO_GT_2_mapped)


# In[47]:


#4.訓練與測試split data
from sklearn.model_selection import train_test_split
import pandas as pd  

# 讀取文件
df_air_quality = pd.read_excel('AirQualityUCI.xlsx') 

# 刪除"Date"和"Time"欄位
df_air_quality.drop(columns=['Date', 'Time'], inplace=True)

# 使用train_test_split函數將數據集分割為訓練集和測試集
x, y = df_air_quality.iloc[:, 1:].values, df_air_quality.iloc[:, 0].values  
#x包含了從資料中的第二列到最後一列的所有特徵數據，y包含了資料中的第一列的目標數據

x_train, x_test, y_train, y_test =    train_test_split(x, y,  
                     test_size=0.2,  # 測試集大小為整個數據集的20%
                     random_state=0,  # 隨機種子設置為0，確保結果的可重現性
                     stratify=None)  # 不再進行分層抽樣


# In[49]:


#5.特徵縮放
from sklearn.preprocessing import MinMaxScaler#導入 MinMaxScaler，用於最小-最大標準化

mms = MinMaxScaler()

#使用 fit_transform 方法對訓練集進行標準化，將其縮放到指定的範圍（默認是[0, 1]）
x_train_norm = mms.fit_transform(x_train)
#使用 transform 方法對測試集進行標準化，使用與訓練集相同的縮放因子
x_test_norm = mms.transform(x_test)


# In[50]:


from sklearn.preprocessing import StandardScaler#導入 StandardScaler，用於標準化

stdsc = StandardScaler()
#使用 fit_transform 方法對訓練集進行標準化，將其轉換為平均值為0，標準差為1的分佈
x_train_std = stdsc.fit_transform(x_train)
#使用 transform 方法對測試集進行標準化，使用與訓練集相同的平均值和標準差進行轉換
x_test_std = stdsc.transform(x_test)


# In[ ]:




