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


# Importing data

# In[ ]:


import pandas as pd
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


X = pd.read_csv("../input/train.csv")
X.describe()


# In[ ]:


y = X.pop("Survived")
y.head()


# In[ ]:


numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()


# In[ ]:


X["Age"].fillna(X.Age.mean(), inplace = True)
X.tail()


# In[ ]:


X[numeric_variables].head()


# In[ ]:


model = RandomForestClassifier(n_estimators = 100)
model.fit(X[numeric_variables], y)


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test[numeric_variables].head()


# In[ ]:


test['Age'].fillna(test.Age.mean(), inplace = True)


# In[ ]:


test = test[numeric_variables].fillna(test.mean()).copy()


# In[ ]:


y_pred = model.predict(test[numeric_variables])
y_pred


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test["PassengerId"],"Survived":y_pred})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


my_submission.head()


# In[ ]:




