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


traindf = pd.read_csv('../input/train.csv',index_col = 0)


# In[ ]:


traindf.head()


# In[ ]:


traindf.info()


# In[ ]:


traindf['Age'] = traindf['Age'].fillna(traindf['Age'].mean())


# In[ ]:


traindf.info()


# In[ ]:


traindf = traindf.drop('Cabin',axis=1)


# In[ ]:


traindf.head()


# In[ ]:


male = pd.get_dummies(traindf['Sex'],drop_first = True)


# In[ ]:


embarked = pd.get_dummies(traindf['Embarked'],drop_first = True)


# In[ ]:


traindf['Male'] = male


# In[ ]:


traindf.head()


# In[ ]:


traindf = traindf.drop(['Sex','Embarked','Ticket','Name'],axis=1)


# In[ ]:


traindf.head()


# In[ ]:


traindf.info()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(traindf.drop('Survived',axis=1),traindf['Survived'])


# In[ ]:


testdf = pd.read_csv('../input/test.csv',index_col = 0)


# In[ ]:


testdf.info()


# In[ ]:


testdf['Age'] = testdf['Age'].fillna(testdf['Age'].mean())


# In[ ]:


testdf.info()


# In[ ]:


testdf['Fare'] = testdf['Fare'].fillna(testdf['Fare'].mean())


# In[ ]:


testdf.info()


# In[ ]:


testdf = testdf.drop('Cabin',axis=1)


# In[ ]:


male = pd.get_dummies(testdf['Sex'],drop_first = True)


# In[ ]:


embarked = pd.get_dummies(testdf['Embarked'],drop_first = True)


# In[ ]:


testdf['Male'] = male


# In[ ]:


testdf.head()


# In[ ]:


testdf = testdf.drop(['Sex','Embarked','Ticket','Name'],axis=1)


# In[ ]:


testdf.head()


# In[ ]:


pred = logreg.predict(testdf)


# In[ ]:


res = pd.read_csv('../input/gender_submission.csv',index_col = 0)


# In[ ]:


res.head()


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[ ]:


accuracy_score(res,pred)


# In[ ]:


confusion_matrix(res,pred)


# In[ ]:


res1 = res['Survived'].tolist()


# In[ ]:


accuracy_score(res1,pred)


# In[ ]:


print(res1)


# In[ ]:


print(pred)


# In[ ]:




