#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# In[ ]:


df_raw = pd.read_csv('../input/train.csv')


# In[ ]:


df_raw.head()


# In[ ]:


col = df_raw.select_dtypes(include='object')


# In[ ]:


for c in col.columns:
    df_raw[c] = df_raw[c].astype('category').cat.as_ordered()
    


# In[ ]:


df_raw.dtypes


# In[ ]:


df_raw.isnull().sum().sort_index()/len(df_raw)


# In[ ]:


age = df_raw['Age'].dropna()


# In[ ]:


df_raw['Age'].fillna(np.median(age), inplace=True)


# In[ ]:


for c in col.columns:
    df_raw[c] = df_raw[c].cat.codes+1


# In[ ]:


df_raw.head()


# In[ ]:





# In[ ]:





# In[ ]:


def train_valid(df):
    msk = np.random.rand(len(df)) < 0.90
    res = [df[msk], df[~msk]]
    return res


# In[ ]:


df_raw_train, df_raw_valid = train_valid(df_raw)


# In[ ]:


print(df_raw_train.shape,
df_raw_valid.shape)


# In[ ]:


df_raw_train_y = df_raw_train['Survived']
df_raw_train.drop('Survived', axis=1, inplace=True)


# In[ ]:


df_raw_valid_y = df_raw_valid['Survived']
df_raw_valid.drop('Survived', axis=1, inplace=True)


# In[ ]:


m = RandomForestClassifier(n_jobs=-1, n_estimators=30, max_features='sqrt', oob_score=True)
m.fit(df_raw_train, df_raw_train_y)


# In[ ]:


m.score(df_raw_train, df_raw_train_y)


# In[ ]:


m.score(df_raw_valid, df_raw_valid_y)


# In[ ]:


m.oob_score_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




