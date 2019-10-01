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


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


#function for drawing a bar chart
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['survived','Dead']
    df.plot(kind='bar',stacked = True, figsize = (10,5))


# In[10]:


bar_chart('Sex')


# In[11]:


bar_chart('Pclass')


# In[12]:


bar_chart('Embarked')


# In[13]:


bar_chart('Parch')


# In[14]:


#combining train and test dataset

train_test_data = [train,test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)


# In[15]:


#map the title 
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[16]:


test.head()


# In[17]:


#deleting unnecessary data from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[18]:


#gender mapping
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[19]:


#filling missing age with median age for each title

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"),inplace = True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"),inplace = True)


# In[20]:


train.groupby("Title")["Age"].transform("median")


# In[21]:


#filling blank spaces by default value S
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[22]:


embark_mapping = {"S": 0, "Q": 2, "C": 1 }
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping)


# In[24]:


#droppping unnecessary data
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[25]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[26]:


train_data.drop('Cabin',axis = 1, inplace = True)
train_data.drop('Fare',axis = 1, inplace = True)


# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


# In[28]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[29]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[30]:


# kNN Score
round(np.mean(score)*100, 2)


# In[31]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[32]:


round(np.mean(score)*100,2)


# In[33]:


test.drop('Cabin',axis = 1,inplace = True)
test.drop('Fare',axis = 1,inplace = True)


# In[34]:


clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[35]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[36]:


submission = pd.read_csv('submission.csv')
submission.head(100)


# In[ ]:




