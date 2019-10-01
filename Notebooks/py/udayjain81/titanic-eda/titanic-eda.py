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


#Importing modules
import pandas as pd             #Data Manipulation
import matplotlib.pyplot as plt #Data Visualisation
import seaborn as sns           #Statistical Data views
from sklearn import tree        #Machine Learning

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()                       #invoke seaborn, changes visualisations to be base seaborn style



# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.head(n=4)


# In[ ]:


df_test.head(n=4)


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# In[ ]:


sns.countplot(x='Survived',data = df_train);
#suppress output by using ';' semicolon
#counterplot - barplot which tells us count


# In[ ]:


#df_test['Survived'] = 0
#df_test[['PassengerId','Survived']].to_csv('no_survivors.csv', index=False)


# In[ ]:


sns.countplot(x='Sex',data = df_train);


# In[ ]:


sns.catplot(x='Survived', col='Sex', kind = 'count', data= df_train);
#barplot
#factorplot renamed to catplot


# In[ ]:


df_train.groupby('Sex').Survived.sum()


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x:int(x))
df_test.head()


# In[ ]:


df_test[['PassengerId','Survived']].to_csv('women_survived.csv', index = False)


# In[ ]:




