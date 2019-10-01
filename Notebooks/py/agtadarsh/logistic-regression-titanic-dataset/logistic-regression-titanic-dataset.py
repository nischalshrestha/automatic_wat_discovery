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


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import math
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/train.csv')
dft = pd.read_csv('../input/test.csv')


# In[ ]:


df.head()


# In[ ]:


sns.countplot(x='Survived', data=df)


# In[ ]:


sns.countplot(x='SibSp',hue='Survived', data=df)


# In[ ]:


df.isnull()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


sns.heatmap(df.isnull(), yticklabels=False)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.Sex = pd.get_dummies(df['Sex'],drop_first=True)


# In[ ]:


df['Male']=pd.get_dummies(df['Sex'],drop_first=True)


# In[ ]:


df.head()


# In[ ]:


df.drop('Sex',axis=1, inplace=True)


# In[ ]:


df.drop(['Embarked'], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


reg = LogisticRegression()


# In[ ]:


x=df.drop(['Survived'], axis=1)
y=df.Survived
reg.fit(x, y)


# In[ ]:


dft = pd.read_csv('../input/test.csv')


# In[ ]:


passid=dft.PassengerId


# In[ ]:


dft.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)


# In[ ]:


dft.head()


# In[ ]:


dft['Male']=pd.get_dummies(dft['Sex'],drop_first=True)


# In[ ]:


dft.drop('Sex',axis=1, inplace=True)


# In[ ]:


dft.isnull().sum()


# In[ ]:


dft['Age'] = dft['Age'].fillna(dft['Age'].median())


# In[ ]:


dft['Fare'] = dft['Fare'].fillna(dft['Fare'].median())


# In[ ]:





# In[ ]:


predictions = reg.predict(dft)


# In[ ]:


s=({"PassengerId":passid,"Survived":predictions})
submit=pd.DataFrame(data=s)
submit.to_csv('titanic.csv',index=False)


# In[ ]:




