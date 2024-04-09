#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np 
import pandas as pd 
from io import StringIO
df = pd.read_csv("C:/Users/DELL/Downloads/productivity+prediction+of+garment+employees/garments_worker_productivity.csv")
df


# ## 1.空值資料處理(Missing Data)

# In[84]:


df.isnull().sum()


# In[85]:


df.dropna()


# In[86]:


df.dropna(axis=1)


# In[87]:


df.dropna(how='all')


# In[88]:


df.fillna(0)


# ## 2.敘述統計(Descripitive Statistics)

# In[90]:


df = pd.read_csv("garments_worker_productivity.csv")
description_statistics = df.describe()
print("敘述統計：")
description_statistics


# In[91]:


description_statistics


# In[92]:


description_statistics.loc["var"] = description_statistics.loc["std"]*description_statistics.loc["std"]
description_statistics


# ## 3.類別資料處理(Categorical Data)

# In[101]:


df_encoded = pd.DataFrame(
    [['Quarter1', 'sweing', 'Thursday', 8],
    ['Quarter2', 'finishing', 'Wednesday', 7],
    ['Quarter3', 'finishing', 'Saturday', 11],
    ['Quarter4', 'finishing', 'Monday', 4]]
)
df_encoded.columns = ['quarter', 'department', 'day', 'team']
df_encoded


# In[102]:


quarter_mapping = {'Quarter1': 1, 
                   'Quarter2': 2, 
                   'Quarter3': 3, 
                   'Quarter4': 4}
df_encoded['quarter'] = df_encoded['quarter'].map(quarter_mapping)
df_encoded


# In[103]:


pd.get_dummies(df_encoded['day'])


# In[104]:


onehot_encoding = pd.get_dummies(df_encoded['day'], prefix = 'day')


# In[105]:


df_encoded = df_encoded.drop('day', 1)
df_encoded


# In[106]:


pd.concat([onehot_encoding, df_encoded],axis=1)


# ## 4.特徵縮放(Feature Scaling)

# In[97]:


data = pd.DataFrame(columns=['team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers', 'actual_productivity'])
data = pd.concat([data, df], ignore_index=True)
data.head(3)


# In[107]:


from IPython.display import Math


# In[108]:


Math(r'x^{(i)}_{norm}=\frac{x^{(i)}-x_{min}}{x_{max}-x_{min}}')


# In[109]:


from sklearn.preprocessing import MinMaxScaler


# In[132]:


Math(r'x^{(i)}_{std}=\frac{x^{(i)}-\mu_{x}}{\sigma_{x}}')


# In[151]:


from sklearn.preprocessing import StandardScaler
date_column = df['date'] 
df_numeric = df.drop(columns=['date'])

df_encoded = pd.get_dummies(df_numeric, columns=['quarter', 'department', 'day'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded)
scaled_df = pd.DataFrame(scaled_features, columns=df_encoded.columns)

scaled_df['date'] = date_column

scaled_df.head()


# ## 5.資料分割(Split Data)

# In[154]:


data = pd.DataFrame(columns=['team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers', 'actual_productivity'])
data = pd.concat([data, df], ignore_index=True)
data.head()
print(data.to_string())


# In[155]:


len(data)


# In[158]:


from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(scaled_df, test_size=0.2, random_state=42)

print("訓練集大小:", X_train.shape)
print("測試集大小:", X_test.shape)

