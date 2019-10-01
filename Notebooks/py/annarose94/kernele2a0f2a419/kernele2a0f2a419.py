#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pylab as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


testTitanic = pd.read_csv('../input/test.csv')
trainTitanic = pd.read_csv('../input/train.csv')


# In[ ]:


testTitanic.head()


# In[ ]:


trainTitanic.head()


# In[ ]:


trainTitanic.info()


# In[ ]:


trainTitanic.describe(include=['O'])


# In[ ]:


#Fill the null values in age


# In[ ]:


trainTitanic.loc[trainTitanic.Age.isnull(), 'Age'] = trainTitanic.groupby('Pclass')['Age'].transform('mean')
testTitanic.loc[testTitanic.Age.isnull(), 'Age'] = testTitanic.groupby('Pclass')['Age'].transform('mean')


# In[ ]:


trainTitanic = trainTitanic.drop('Cabin', axis=1)
testTitanic = testTitanic.drop('Cabin', axis=1)


# In[ ]:


trainTitanic = trainTitanic.drop(['PassengerId','Name','Ticket'], axis=1)
testTitanic    = testTitanic.drop(['Name','Ticket'], axis=1)


# In[ ]:


trainTitanic["Embarked"] = trainTitanic["Embarked"].fillna("S")
testTitanic["Embarked"] = testTitanic["Embarked"].fillna("S")


# In[ ]:


features = ['Fare', 'Pclass', 'Sex']


# In[ ]:


trainTitanic = pd.get_dummies(trainTitanic,columns = ['Pclass', 'Sex'], drop_first = True)


# In[ ]:


trainTitanic.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, t_test = train_test_split(trainTitanic.drop('Survived', axis = 1), trainTitanic['Survived'], test_size = 0.2)


# In[ ]:


#Logistic regression


# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[ ]:


trainTitanic.dtypes


# In[ ]:


model = RFE(logreg, 15)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


#K Nearest Neighbors


# In[ ]:


model2 = KNeighborsClassifier()


# In[ ]:


model2.fit(X_train, y_train)


# In[ ]:


predictions = model2.predict(X_test)


# In[ ]:


score2 = (y_test, predictions)


# In[ ]:


#Decision Tree


# In[ ]:


model3=DecisionTreeClassifier()


# In[ ]:


model3.fit(X_train, y_train)


# In[ ]:


predictions = model3.predict(X_test)


# In[ ]:


score3(y_test, predicitions)

