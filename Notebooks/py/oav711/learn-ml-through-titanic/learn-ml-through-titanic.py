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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


# Import test and train datasets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# View first lines of training data
df_train.head(n=4)


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


sns.countplot(x='Survived', data=df_train);


# In[ ]:


df_test.head()


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);


# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)


# In[ ]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# In[ ]:




