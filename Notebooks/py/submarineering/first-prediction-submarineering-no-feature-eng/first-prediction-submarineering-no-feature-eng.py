#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[2]:


# Load datasets
X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')


# In[3]:


# Specifying the training labels
Y_train = X_train['Survived'].copy()


# In[4]:


# An overview.
X_test.head()


# In[5]:


# Dropping unnecessary columns for the first attempt from train DS
X_train.drop(['PassengerId','SibSp','Name', 'Ticket','Survived', 'Parch', 'Cabin', 'Embarked'], axis=1, inplace=True)


# In[6]:


# Dropping unnecessary columns for the first attempt from test DS
X_test.drop(['PassengerId','SibSp','Name', 'Ticket', 'Parch', 'Cabin', 'Embarked'], axis=1, inplace=True)


# In[7]:


# Categorizing the Sex column. This could be done using map function passing a dictionary too. 
X_train['Sex'].loc[X_train['Sex']=='female'] = 0

X_train['Sex'].loc[X_train['Sex']=='male'] = 1


# In[8]:


# Categorizing the Sex column. This could be done using map function passing a dictionary too. 
X_test['Sex'].loc[X_test['Sex']=='male'] = 1

X_test['Sex'].loc[X_test['Sex']=='female'] = 0


# In[9]:


# Cleaning data
X_train.dropna(axis=0, inplace=True)

Y_train = Y_train.loc[X_train.index]
X_test.fillna(30, inplace=True)


# In[10]:


#Converting to integer.
X_train['Age'] = X_train['Age'].astype(int)

X_test['Age'] = X_test['Age'].astype(int)


# In[11]:


#Converting to integer.
X_train['Fare'] = X_train['Fare'].astype(int)

X_test['Fare'] = X_test['Fare'].astype(int)


# In[12]:


#Cross validation for a first estimation of the classifier performance.
clf = LogisticRegression()
scores = cross_val_score(clf,X_train,Y_train, cv=3)
clf.fit(X_train,Y_train)
scores


# In[13]:


#Cross validation for a first estimation of the classifier performance.
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,X_train,Y_train, cv=3)
clf.fit(X_train,Y_train)
scores


# **What Next? ** 

# Try to make some features engineering to improve your accuracy. 
# I recommend : https://www.kaggle.com/startupsci/titanic-data-science-solutions

# Note: I have droped the 'PassengerId' also. 
