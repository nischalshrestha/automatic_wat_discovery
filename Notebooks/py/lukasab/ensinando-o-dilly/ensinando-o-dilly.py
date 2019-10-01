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


df_train = pd.read_csv('../input//train.csv')


# In[ ]:


df_train.tail()


# In[ ]:


df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)


# In[ ]:


df_train.info()


# In[ ]:


df_train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


df_train.info()


# In[ ]:


df_train.dropna(inplace=True)


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_train.drop(['Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_train['Sex'] = pd.get_dummies(df_train['Sex'], drop_first=True)


# In[ ]:


df_train.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


sns.pairplot(data=df_train, hue='Survived')


# In[ ]:


sns.heatmap(df_train.corr(), cmap='RdYlGn')


# In[ ]:


df_train.drop('Embarked', axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


X = df_train.drop(['Survived', 'PassengerId'], axis=1)
X.head()


# In[ ]:


Y = df_train['Survived']
Y.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, test_size = 0.25)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=200)


# In[ ]:


rfc.fit(X_train, Y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)
rfc_pred


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(Y_test, rfc_pred))


# In[ ]:


rfc_final = RandomForestClassifier(n_estimators=200)
rfc_final.fit(X,Y)


# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_test.info()


# In[ ]:


df_test['Sex'] = pd.get_dummies(df_test['Sex'], drop_first=True)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
passengerId_test = df_test['PassengerId'].copy()
df_test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1, inplace=True)


# In[ ]:


df_test.info()


# In[ ]:


rfc_final_pred =rfc_final.predict(df_test)


# In[ ]:


submission = pd.DataFrame({'PassengerId': passengerId_test ,'Survived': rfc_final_pred})


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




