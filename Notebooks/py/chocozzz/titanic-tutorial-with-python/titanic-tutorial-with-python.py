#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.tail()


# In[ ]:


df_train['Fare'].describe()


# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


data = pd.concat([df_train['Fare'], df_train['Embarked']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Embarked', y="Fare", data=data)


# In[ ]:


df_train[df_train['Fare']>500] 


# In[ ]:


#histogram
#missing_data = missing_data.head(20)

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Total Missing Value (%)", fontsize = 20)


# In[ ]:


(df_train.isnull().sum()/df_train.isnull().count())


# In[ ]:


adsf = df_train.isnull()


# In[ ]:


import missingno as msno
missingdata_df = df_train.columns[df_train.isnull().any()].tolist()
msno.heatmap(df_train[missingdata_df], figsize=(8,6))
plt.title("Correlation with Missing Values", fontsize = 20)


# In[ ]:


df_train['hasCabin'] = df_train['Cabin'].isnull().apply(lambda x: 0 if x == True else 1)
df_train['hasAge'] = df_train['Age'].isnull().apply(lambda x: 0 if x == True else 1)


# In[ ]:


df_train.head()


# In[ ]:


df_train.corr() 


# In[ ]:


data = pd.concat([df_train['Fare'], df_train['hasCabin']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='hasCabin', y="Fare", data=data)


# In[ ]:


from statsmodels.graphics.mosaicplot import mosaic
mosaic(df_train, ['hasCabin', 'Pclass'],gap=0.02)
plt.show()


# In[ ]:


df_train[df_train['Embarked'].isnull()] 


# In[ ]:


data = pd.concat([df_train['Fare'], df_train['Embarked']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='Embarked', y="Fare", data=data)


# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x="Embarked", y="Fare", hue="Pclass",
               data=df_train, palette="Set3")


# In[ ]:




