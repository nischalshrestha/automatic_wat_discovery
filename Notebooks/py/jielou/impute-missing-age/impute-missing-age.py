#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


titanic = pd.read_csv("../input/train.csv")


# In[4]:


titanic.head()


# ## Impute missing values
# 
# Here I categorize passengers according to Pclass, Sex, SibSp and Parch. 

# In[6]:


titanic.groupby(['Pclass','Sex','Parch','SibSp']).mean()['Age'].size


# There are 74 groups. For each group, average age is computed and then it is imputed for those whose age is missing.

# In[7]:


titanic["Age"] = titanic.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))


# In[8]:


titanic.Age.isnull().sum()


# There are still 8 passengers whose age is missing. Why?

# In[10]:


titanic[titanic.Age.isnull()]


# The first 7 people were expected to come from the same family because they all have the same family name. We will assign them into groups without considering their number of sibiling. 

# In[11]:


titanic["Age"] = titanic.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))


# In[12]:


titanic.Age.isnull().sum()


# Done!

# This is my first try to impute missing values. I know there are more fancy way to do it. I'll learn tomorrow!
