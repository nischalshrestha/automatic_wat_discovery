#!/usr/bin/env python
# coding: utf-8

# Prelude , A Datasaster for Newbie
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this case, I try to generate the analysis of what sorts of people were likely to survive. In particular, to apply the tools of machine learning to predict which passengers survived the tragedy.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd
import datetime, pytz
import io
import requests

import seaborn as sb
import matplotlib as mpl

import sklearn


# Data Collection

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print(train.columns.values)


# In[ ]:


print(test.columns.values)


# In[ ]:


# preview training data
train.head()


# In[ ]:


# preview test data
test.head()


# Survived passengers (%) by gender

# In[ ]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Survived passengers (%) by ticket class

# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Survived passengers (%) by number of siblings/spouses

# In[ ]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Survived passengers (%) by number of parents/children

# In[ ]:


train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Survived passengers (%) by combination of parents/children and siblings/spouses

# In[ ]:


train[['SibSp','Parch', 'Survived']].groupby(['Parch','SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Survived passengers (%) by port of embarkation
# 
# (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Survived passengers (%) by age

# In[ ]:


train['Age'].describe()


# In[ ]:


agecount = train[['Age', 'Survived']].groupby(['Age'],as_index=False).count()
sb.barplot(x='Age', y='Survived', data=agecount)


# In[ ]:




