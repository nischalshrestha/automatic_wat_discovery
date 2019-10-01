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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import mpl_toolkits
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:


pd.isnull(train).sum()


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train)


# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=train)


# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
bins = [0, 5, 12, 18, 24, 35, 60, 100]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

sns.barplot(x="AgeGroup", y="Survived", data=train)


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train)


# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train)


# In[ ]:


train["Cabin"] = (train["Cabin"].notnull().astype('int'))
test["Cabin"] = (test["Cabin"].notnull().astype('int'))
sns.barplot(x="Cabin", y="Survived", data=train)


# In[ ]:


southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)
sns.barplot(x="Embarked", y="Survived", data = train)


# In[ ]:


sex_map = {'male':0, 'female':1}
train['Sex'] = train['Sex'].map(sex_map)
test['Sex'] = test['Sex'].map(sex_map)


# In[ ]:


train.isnull().sum()


# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test.isnull().sum()


# In[ ]:


# dropping features
label = train["Survived"]
PassengerId = test["PassengerId"]
train = train.drop(["Survived","PassengerId","Name","Ticket","Embarked","AgeGroup"], axis=1)
test = test.drop(["PassengerId","Name","Ticket","Embarked","AgeGroup"], axis=1)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(train,label)
gbc.score(train,label)
# train score


# In[ ]:


pred = gbc.predict(test)


# In[ ]:


pred_df = pd.DataFrame(pred, columns=['Survived'])
result = pd.concat([PassengerId,pred_df],axis =1)
# result


# In[ ]:


# result.set_index("PassengerId", inplace=True)
# print(result)
result.to_csv('result.csv', index=False)


# In[ ]:




