#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:



# read train and test-data
train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")
combine = [train_df, test_df]

# print the column names
print(train_df.columns.values)
print(train_df.shape)


# In[3]:


# preview the data
train_df.head()


# In[7]:


train_df.info()


# In[8]:


train_df.describe()


# In[23]:


# describe also the string/object like columns
train_df.describe(include=['O'])


# In[46]:


train_df[['Survived', 'Name']].groupby('Survived').count()


# In[24]:


train_df[['Sex', 'Survived']].groupby('Sex').mean()


# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=10)


# In[49]:


train_df[['Name', 'Parch']]

