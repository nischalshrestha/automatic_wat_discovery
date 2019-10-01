#!/usr/bin/env python
# coding: utf-8

# This is my first experimentation to analyze a data set on my own.

# In[20]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[21]:


titanic_test.head()


# In[22]:


#Preparing features for decision trees
titanic_train.Sex.unique()
d = {'male':1, 'female':0}
titanic_train['Sex'] = titanic_train['Sex'].map(d)
titanic_test['Sex'] = titanic_test['Sex'].map(d)
titanic_train['FamilySize'] = titanic_train['SibSp'] + titanic_train['Parch']
titanic_test['FamilySize'] = titanic_test['SibSp'] + titanic_test['Parch']
titanic_train.drop(['SibSp', 'Parch', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
titanic_test.drop(['SibSp', 'Parch', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)


# In[23]:


d = {'U':0,'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'T':7, 'G':8}
titanic_train['Cabin'] = titanic_train['Cabin'].fillna('Unknown')
titanic_train['Age'] = titanic_train['Age'].fillna(0)
titanic_test['Fare'] = titanic_test['Fare'].fillna(0)
titanic_test['Cabin'] = titanic_test['Cabin'].fillna('Unknown')
titanic_test['Age'] = titanic_test['Age'].fillna(0)

def diction_value(x, d):
    value = str(x)[0]
    return d[value]
titanic_train['Cabin'] = titanic_train['Cabin'].map(lambda x: diction_value(x, d)) 
titanic_test['Cabin'] = titanic_test['Cabin'].map(lambda x: diction_value(x, d)) 


# In[ ]:


titanic_train.head()


# In[ ]:


titanic_test.head()


# In[ ]:


train_features = list(titanic_train.columns[1:])
train_features


# In[ ]:


y = titanic_train["Survived"]
x = titanic_train[train_features]
x.columns[x.isnull().any()]
model = sm.Logit(y, x)
result = model.fit()


# In[ ]:


print(result.summary())


# In[ ]:


print(np.exp(result.params))


# In[ ]:


linearmodel  = linear_model.LogisticRegression(C=1e9)
result1 = linearmodel.fit(x, y)
scoremodel = linearmodel.score(x, y)
print(scoremodel)


# In[ ]:


#Calculate precision, recall, F1 score
y_pred = linearmodel.predict(x)
print(precision_score(y, y_pred))
print(recall_score(y, y_pred))
print(f1_score(y, y_pred))


# In[ ]:


#Predict the values on test set
predict_test = linearmodel.predict(titanic_test)
print(titanic_test)
print(predict_test)

