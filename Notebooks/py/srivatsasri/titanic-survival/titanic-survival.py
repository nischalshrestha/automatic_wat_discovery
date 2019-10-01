#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


test_data.head()


# In[ ]:


train_data.Pclass.value_counts().plot(kind='bar')


# In[ ]:


train_data.isnull().any()


# In[ ]:


train = train_data.drop(['Name','Ticket','Cabin','Fare','PassengerId'],axis=1)


# In[ ]:


train.head(3)


# In[ ]:


train.Sex = train.Sex.map({'female':0,'male':1})


# In[ ]:


train.loc[train.Age.isnull(),'Age'] = train.Age.median()


# In[ ]:


train.loc[train.Embarked.isnull(),'Embarked'] = train.Embarked.mode()


# In[ ]:


train = pd.get_dummies(train, columns=['Pclass','Embarked'])


# In[ ]:


train.head(1)


# In[ ]:


train.isnull().any()


# In[ ]:


y = train['Survived']


# In[ ]:


y.__len__()


# In[ ]:


X = train.drop(['Survived'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


print(X_train.shape)


# In[ ]:


print(X_test.shape)


# In[ ]:





# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


kfold = KFold(n_splits=10, random_state=0)
dTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dTree.fit(X_train,y_train)
scoring = 'accuracy'
results = cross_val_score(dTree,X_train,y_train, cv=kfold, scoring=scoring)
acc_dt = results.mean()
dt_std = results.std()
acc_dt


# In[ ]:


y_pred = dTree.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


test_data.isnull().any()


# In[ ]:


test_data.loc[test_data.Age.isnull(),'Age'] = test_data.Age.median()


# In[ ]:


test = test_data.drop(['Name','Ticket','Cabin','Fare','PassengerId'],axis=1)


# In[ ]:


test.Sex = test.Sex.map({'female':0,'male':1})


# In[ ]:


test = pd.get_dummies(test, columns=['Pclass','Embarked'])


# In[ ]:


test.head(2)


# In[ ]:


prediction = dTree.predict(test)


# In[ ]:


prediction.__len__()


# In[ ]:


submission = pd.DataFrame({"PassengerId":test_data["PassengerId"],"Survived":prediction})
submission.to_csv("sample_submission.csv",index=False)  


# In[ ]:




