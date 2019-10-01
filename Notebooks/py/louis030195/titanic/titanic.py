#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns  

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Regex
import re

# Math
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
gender_submission = pd.read_csv("../input/gender_submission.csv")
print("Train ", train.head())
print("Test ", test.head())
print("Gender submission ", gender_submission.head())


# **Displaying columns from dataset**

# In[ ]:


print("Train columns ", train.columns)
print("Test columns ", test.columns)


# In[ ]:


for dataset in [train]:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for dataset in [test]:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train.info()
print('_'*40)
test.info()


# In[ ]:


print(train[['Cabin','Ticket', 'Fare', 'Embarked']].iloc[0:20])


# * We could use Name to check what is the origin of the person
# * The origin could be a factor in survival (discrimination)
# * But it would require some NLP ...
# * PassengerId, Ticket aren't very useful
# * Fare can be, maybe better position on the boat, better access to survival stuff ...
# * Cabin could be useful if we knew the distance between the cabin and the front of the boat
# * Because the boat is sinking from the front, more survival chances at the bottom i guess
# * Embarked ??

# In[ ]:


X_train = train.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin', 'Embarked'], axis=1)
X_test = test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Embarked'], axis=1)

y_train = train['Survived']


# We will try to convert cabin into number

# In[ ]:


"""
def cabinToValue(s):
    s = s.split()[0] # If multiple cabins just taking the first
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    # print(head, tail)
    return ord(head if head != "" else 0) + int(tail if tail != "" else 0)

print(cabinToValue(train['Cabin'].iloc[1]))
"""


# Now we are going to replace cabin string with its value that would show the distance with the front of the boat
# 
# We fill NaN with the "average position" in the boat, takes about 30 secs

# In[ ]:


"""
avg = 0
nonnan = 0
for i in range (0, len(train)):
    if not pd.isnull(X_train['Cabin'].iloc[i]):
        avg += cabinToValue(X_train['Cabin'].iloc[i])
        nonnan += 1
        X_train['Cabin'].iloc[i] = cabinToValue(X_train['Cabin'].iloc[i])
avg /= nonnan
X_train['Cabin'] = X_train['Cabin'].fillna(avg)

print(X_train['Cabin'].head())

print("-"*30)

avg = 0
nonnan = 0
for i in range (0, len(test)):
    if not pd.isnull(X_test['Cabin'].iloc[i]):
        avg += cabinToValue(X_test['Cabin'].iloc[i])
        nonnan += 1
        X_test['Cabin'].iloc[i] = cabinToValue(X_test['Cabin'].iloc[i])
avg /= nonnan
X_test['Cabin'] = X_test['Cabin'].fillna(avg)

print(X_test['Cabin'].head())
"""


# Age column is full of NaN values, we just fill with 0, but we could guess the age from other columns maybe
# 
# Fare seems to have also a NaN value

# In[ ]:


#complete missing age with median
X_train['Age'].fillna(X_train['Age'].median(), inplace = True)

#complete missing fare with median
X_train['Fare'].fillna(X_train['Fare'].median(), inplace = True)

#complete missing age with median
X_test['Age'].fillna(X_train['Age'].median(), inplace = True)

#complete missing fare with median
X_test['Fare'].fillna(X_train['Fare'].median(), inplace = True)


# Just checking if everything is okay (no NaN values)

# In[ ]:


X_train.info()
print('_'*40)
X_test.info()


# Time to apply logistic regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
result = pd.DataFrame()
result['Survived'] = logreg.predict(X_test)
acc_log = logreg.score(X_train, y_train) * 100
print(acc_log)


# Checking some samples ...

# In[ ]:


import random
rnd = random.randint(0,200)
for i in range(rnd, rnd+5):
    deadoralive = logreg.predict(X_test.iloc[i].values.reshape(1, -1))[0]
    
    print(X_test.iloc[i])
    print("Dead or alive ?", deadoralive)
    print("Correct" if train.iloc[i]['Survived'] == deadoralive else "Incorrect")
    print('-'*30)


# In[ ]:


print(test['PassengerId'].shape, result.shape)
result['PassengerId'] = test['PassengerId']

result = result[result.columns[::-1]] # Had to reverse columns ...

result.to_csv('result.csv', index=False, header=['PassengerId', 'Survived'])
print(pd.read_csv('result.csv'))


# In[ ]:




