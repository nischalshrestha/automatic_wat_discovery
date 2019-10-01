#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.neighbors import KNeighborsClassifier
a=pd.read_csv('../input/train.csv')


# In[7]:


b=a.drop(['Pclass','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Name'],axis=1)


# In[22]:


b['Age']=b['Age'].fillna(b['Age'].mean())


# In[24]:


k=KNeighborsClassifier(n_neighbors=5)


# In[25]:


b1=b.Survived


# In[36]:


b=b.drop(['PassengerId','Survived'],axis=1)


# In[31]:


b['Sex']=b.Sex.replace(['male','female'],[1,0])


# In[37]:


s=k.fit(b,b1)


# In[59]:


d1=pd.read_csv('../input/test.csv')
d=pd.read_csv('../input/test.csv')


# In[39]:


d=d[b.columns]


# In[42]:


d.Sex=d.Sex.replace(['male','female'],[1,0])
d.Age=d.Age.fillna(d.Age.mean())


# In[44]:


result=s.predict(d)


# In[51]:


f=pd.DataFrame(result)


# In[52]:


f.index=d1.PassengerId


# In[55]:


f.columns=['Survived']
f.index.name='PassengerId'


# In[61]:


f.to_csv('result.csv')


# In[62]:





# In[ ]:




