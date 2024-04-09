#!/usr/bin/env python
# coding: utf-8

# In[66]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


import pandas as pd

#匯入CSV檔

df = pd.read_csv('Sleep_Efficiency.csv')


# In[68]:


#讀取各欄缺失值
df.isnull().sum()


# In[69]:


# 將無用資料刪除
df.drop('ID', axis=1, inplace=True)
df.drop('Bedtime', axis=1, inplace=True)
df.drop('Wakeup time', axis=1, inplace=True)


# In[70]:


from sklearn.impute import SimpleImputer
import numpy as np

# 將男性，女性轉換為1和2

size_mapping = {'Female': 1,
                'Male': 2}

df['Gender'] = df['Gender'].map(size_mapping)

# 將是否有抽菸轉會為有抽菸為1，沒抽菸為0

size_mapping2 = {'Yes': 1,
                'No': 0}

# 將上述規則覆蓋到資料集
df['Smoking status'] = df['Smoking status'].map(size_mapping2)

# 將每列的缺失值以平均替代
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)


# In[71]:


import pandas as pd
# 計算每個特徵的中位數、平均數、標準差和四分位距
statistics = df.describe()


print("\n統計資料：")
print(statistics)


# In[72]:


# 年紀的分布圖

import matplotlib.pyplot as plt

# 抓取年齡
ages = df['Age'].astype(int)

# 畫年齡直方圖
plt.hist(ages, bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age')
plt.show()



# In[73]:


# 性別的圓餅圖

import matplotlib.pyplot as plt

# 統計各性別出現的次數
gender_counts = df['Gender'].value_counts()
gender_labels = {1: 'Female', 2: 'Male'}
labels = [gender_labels[label] for label in gender_counts.index]

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = 'red'
plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Gender')
plt.axis('equal') 
plt.show()


# In[74]:


# 年齡的分布圖

import matplotlib.pyplot as plt

Sleep_duration = df['Sleep duration'].astype(int)

# 畫年齡直方圖
plt.hist(Sleep_duration, bins=20, edgecolor='black')
plt.xlabel('Sleep duration')
plt.ylabel('Frequency')
plt.title('Sleep duration ')
plt.show()


# In[75]:


a = 0
b = 0
for i in df['Sleep efficiency']:
    if i >= 0.85:
        a+=1
    else:
        b+=1
print(a)

print(b)

import matplotlib.pyplot as plt

sizes = [a, b]
labels = ['good', 'ordinary']
colors = ['#ff9999', '#66b3ff']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = 'red'
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # 確保圓餅圖是正圓的
plt.title('Sleep Efficiency', color='red')  # 圖表標題，紅色字體
plt.show()


# In[76]:


# 眼動頻率的分布圖

import matplotlib.pyplot as plt

REM = df['REM sleep percentage'].astype(int)

# 畫眼動頻率直方圖
plt.hist(REM, bins=20, edgecolor='black')
plt.xlabel('REM sleep percentage')
plt.ylabel('Frequency')
plt.title('REM sleep percentage')
plt.show()


# In[77]:


# 深眠占比的分布圖

import matplotlib.pyplot as plt

DEEP = df['Deep sleep percentage'].astype(int)

# 深眠占比直方圖
plt.hist(DEEP, bins=20, edgecolor='black')
plt.xlabel('Deep sleep percentage')
plt.ylabel('Frequency')
plt.title('Deep sleep percentage')
plt.show()


# In[78]:


# 淺眠占比的分布圖

import matplotlib.pyplot as plt

Light = df['Light sleep percentage'].astype(int)

# 淺眠占比直方圖
plt.hist(Light, bins=20, edgecolor='black')
plt.xlabel('Light sleep percentage')
plt.ylabel('Frequency')
plt.title('Light sleep percentage')
plt.show()


# In[79]:


# 醒來次數的直方圖

import matplotlib.pyplot as plt
import pandas as pd

# 統計各醒來次數出現次數
awakenings_counts = df['Awakenings'].value_counts()

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.bar(awakenings_counts.index, awakenings_counts.values, color='skyblue')
plt.xlabel('Number of Awakenings')
plt.ylabel('Frequency')
plt.title('Awakenings')
plt.show()



# In[80]:


# 咖啡因攝入量的直方圖

import matplotlib.pyplot as plt
import pandas as pd

# 統計咖啡因攝入量
awakenings_counts = df['Caffeine consumption'].value_counts()
print(awakenings_counts)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.bar(awakenings_counts.index, awakenings_counts.values, color='skyblue')
plt.xlabel('Number of Caffeine consumption')
plt.ylabel('Frequency')
plt.title('Caffeine consumption')
plt.show()


# In[81]:


# 酒精攝入量（每週次數）的直方圖

import matplotlib.pyplot as plt
import pandas as pd

# 統計酒精攝入量（每週次數）
awakenings_counts = df['Alcohol consumption'].value_counts()
print(awakenings_counts)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.bar(awakenings_counts.index, awakenings_counts.values, color='skyblue')
plt.xlabel('Number of Alcohol consumption')
plt.ylabel('Frequency')
plt.title('Alcohol consumption')
plt.show()


# In[82]:


# 是否抽菸的圓餅圖

import matplotlib.pyplot as plt

# 統計抽菸人數
gender_counts = df['Smoking status'].value_counts()
gender_labels = {1: 'Yes', 0: 'No'}
labels = [gender_labels[label] for label in gender_counts.index]

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 16
plt.rcParams['text.color'] = 'red'
plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Smoking status')
plt.axis('equal') 
plt.show()


# In[83]:


# 運動頻率的直方圖

import matplotlib.pyplot as plt
import pandas as pd

# 統計
awakenings_counts = df['Exercise frequency'].value_counts()
print(awakenings_counts)

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.bar(awakenings_counts.index, awakenings_counts.values, color='skyblue')
plt.xlabel('Number of Exercise frequency')
plt.ylabel('Frequency')
plt.title('Exercise frequency')
plt.show()


# In[84]:


from sklearn.model_selection import train_test_split

# 將DF數據轉換成NumPy
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

# 隨機切割數據，測試集與訓練集比例為7:3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=None)


# In[85]:


# 標準化(使用平均與標準差)

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

