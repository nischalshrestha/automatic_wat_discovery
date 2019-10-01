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



df=pd.read_csv('../input/train.csv')
df.head()


# **Finding  if any Missing Values are present**

# In[ ]:


df.info()


# 1. **missing values in age,cabin column**
# * **There are 5 text columns**
# 

# In[ ]:


df['Age'].fillna(df['Age'].mean(),inplace=True)
df.info()


# In[ ]:


def convertSex(x):
    if x=='Male':
        return 0
    else: return 1
df['Sex']=df.Sex.apply(convertSex)
def convertEmbarked(x):
    if x=='C':
        return 0
    elif x=='S':
        return 1
    else:
        return 2
df['Embarked']=df['Embarked'].apply(convertEmbarked)
df.head()


# In[ ]:


features=['Pclass','Sex','Age','SibSp','Parch','Embarked']
x=df[features].values
y=df['Survived'].values
print(type(x))


# *  **Importing required libraries**

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[ ]:


clf=LogisticRegression()
clf.fit(x,y)


# **Test Data**

# In[ ]:


test=pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


test.info()


# In[ ]:


test['Age'].fillna(test['Age'].mean(),inplace=True)
test.info()


# In[ ]:


test['Embarked']=test['Embarked'].apply(convertEmbarked)
test['Sex']=test['Sex'].apply(convertSex)


# In[ ]:


testx=test[features].values
res=clf.predict(testx)


# In[ ]:


ansdic = {'PassengerId': test['PassengerId'],'Survived': res}
ans = pd.DataFrame(ansdic)
ans.head()


# In[ ]:


ans.to_csv('answer.csv',index=False)

