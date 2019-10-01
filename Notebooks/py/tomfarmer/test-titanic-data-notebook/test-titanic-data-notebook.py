#!/usr/bin/env python
# coding: utf-8

# Getting packages and the data

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns #enables heatmap for correlations
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

train =  pd.read_csv('../input/train.csv')
test =  pd.read_csv('../input/test.csv')

print(len(train), len(test)) #891 then 418
print(train.head())
#print(test.head())


# Test code

# In[ ]:


print(train.columns)
train['Fare'].describe()


# In[ ]:


plt.hist(train['Fare'], 100)


# 

# In[ ]:


corrmat = train.corr()
sns.heatmap(corrmat)


# In[ ]:


enc = OneHotEncoder()

encodedData = train

print(encodedData.Sex.unique())
print(encodedData.Pclass.unique())
print(encodedData.Embarked.unique())
print(encodedData.SibSp.unique())
#encodedData[['Pclass','Sex']]

enc.fit(encodedData[['Pclass','SibSp']])
encodedData = enc.transform(encodedData[['Pclass','SibSp']])
encodedData

