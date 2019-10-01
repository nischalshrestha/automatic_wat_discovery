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


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.info()


# In[ ]:


trainAge = train['Age'].fillna(train['Age'].mean())
train['Age'] = trainAge


# In[ ]:


trainEmbarked = train['Embarked'].fillna('S')
train['Embarked'] = trainEmbarked


# In[ ]:


embSeries = train['Embarked'].apply(lambda x : 1 if x == 'S' else (2 if x == 'Q' else 3))
train['EmbarkedInt'] = embSeries


# In[ ]:


train.head()


# In[ ]:


y_train = train['Survived']


# In[ ]:


mfSeries = train['Sex'].apply(lambda x : 1 if x == 'male' else 0)


# In[ ]:


train['M_F'] = mfSeries


# In[ ]:


train['AddPsng'] = train['SibSp'] + train['Parch']


# In[ ]:


train.groupby('Embarked').count()


# In[ ]:


test = pd.read_csv('../input/test.csv')
trainAge = test['Age'].fillna(test['Age'].mean())
test['Age'] = trainAge
trainFare = test['Fare'].fillna(test['Fare'].mean())
test['Fare'] = trainFare
mfSeries = test['Sex'].apply(lambda x : 1 if x == 'male' else 0)
test['M_F'] = mfSeries
test['AddPsng'] = test['SibSp'] + test['Parch']
test.info()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
y_train = train['Survived']
x_features = ['Pclass', 'Age' , 'M_F' , 'AddPsng' , 'Fare']
x_train = train[x_features]
clf = RandomForestClassifier()
clf.fit(x_train , y_train)
x_test = test[x_features]
pred = clf.predict(x_test)
#acc = accuracy_score(pred , y_train)
#print ("Accuacy Score : ",acc)


# In[ ]:


test_sub = test[['PassengerId']]


# In[ ]:


test_sub['Survived'] = pred
test_sub


# In[ ]:


test_sub.to_csv('Submission.csv', index = False)


# In[ ]:




