#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier # Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression.

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:


# remove Name, Ticket and Cabin
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


print(train.shape)


# In[ ]:


print(train.dtypes)


# In[ ]:


# Age and Embaked has null values
train.info()


# In[ ]:


train.nunique()


# In[ ]:


train.describe()


# In[ ]:


# Convert categorical variable into dummy/indicator variables
new_train = pd.get_dummies(train)
new_test = pd.get_dummies(test)
new_train.head()


# In[ ]:


# replace NULL values for mean Age
new_train['Age'].fillna(new_train['Age'].mean(), inplace=True)
new_test['Age'].fillna(new_test['Age'].mean(), inplace=True)
new_test['Fare'].fillna(new_test['Fare'].mean(), inplace=True)


# In[ ]:


# To separate data train to Survived 
x = new_train.drop('Survived', axis=1)
y = new_train['Survived']


# In[ ]:


tree = DecisionTreeClassifier(max_depth=10, random_state=0)
tree.fit(x, y)


# In[ ]:


tree.score(x, y)


# In[ ]:


submission = pd.DataFrame()
submission['PassengerId'] = new_test['PassengerId']
submission['Survived'] = tree.predict(new_test)


# In[ ]:


submission.to_csv('submission.csv', index=False)

