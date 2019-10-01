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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


x_train = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y_train = train['Survived']


# In[ ]:


sns.countplot(y_train)
y_train.value_counts()


# In[ ]:


x_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[ ]:


x_train['Sex'] = x_train['Sex'].replace(['male','female'],[1,0]) 


# In[ ]:


x_test['Sex'] = x_test['Sex'].replace(['male','female'],[1,0])


# In[ ]:


x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())


# In[ ]:


x_test['Age'] = x_test['Age'].fillna(x_train['Age'].mean())


# In[ ]:


x_train['Embarked'] = x_train['Embarked'].fillna('S')
x_test['Embarked'] = x_test['Embarked'].fillna('S')


# In[ ]:


x_train['Embarked'] = x_train['Embarked'].replace({'C','Q','S'},{0,1,2})
x_test['Embarked'] = x_test['Embarked'].replace({'C','Q','S'},{0,1,2})


# In[ ]:


x_train['Sex'] = x_train['Sex'].fillna(x_train['Sex'].mode())
x_test['Sex'] = x_test['Sex'].fillna(x_train['Sex'].mode())


# In[ ]:


x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].mean())


# In[ ]:


x_train.info()


# In[ ]:


X_train = np.array(x_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])
Y_train = np.array(y_train)


# In[ ]:


X_test = np.array(x_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']])


# In[ ]:


regr = RandomForestClassifier(max_depth=8,max_features=None, n_estimators=50, min_samples_split=8)


# In[ ]:


regr.fit(X_train, Y_train)


# In[ ]:


regr.score(X_train,Y_train)


# In[ ]:


x_test.info()


# In[ ]:


output = pd.DataFrame()
output['PassengerId'] = test['PassengerId']
output['Survived'] = regr.predict(x_test)
output.to_csv('output.csv',index=False)


# In[ ]:





# In[ ]:




