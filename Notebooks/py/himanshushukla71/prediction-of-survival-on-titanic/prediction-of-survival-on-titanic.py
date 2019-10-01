#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/train.csv")
data_test=pd.read_csv("../input/test.csv")


# In[ ]:


data.head(2)


# In[ ]:


data_test.head()


# In[ ]:


data.info()


# In[ ]:


col_delete=['PassengerId','Name','Ticket','Cabin','Embarked']
data.drop(col_delete,axis=1,inplace=True)
data_test.drop(col_delete,axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data_test.head()


# In[ ]:


data.isnull().sum()



# In[ ]:


data_test.isnull().sum()


# In[ ]:


data.Age.fillna(data.Age.mean(),inplace=True)
data_test.Age.fillna(data_test.Age.mean(),inplace=True)
data_test.Fare.fillna(data_test.Fare.mean(),inplace=True)


# In[ ]:


def f(s):
    if s=='male':
        return 0
    else:
        return 1
data['Sex']=data.Sex.apply(f)
data_test['Sex']=data_test.Sex.apply(f)


# In[ ]:


data.head()


# In[ ]:


x_variable=['Pclass','Sex','Age','Parch','Fare']
train_x=data[x_variable]
train_y=data.Survived
test_x=data_test[x_variable]


# In[ ]:


model=RandomForestClassifier()


# In[ ]:


model.fit(train_x,train_y)


# In[ ]:


model.score(train_x,train_y)


# In[ ]:


result=model.predict(test_x)
df=pd.DataFrame(result)
df.head()


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


df['PassengerId']=test['PassengerId']


# In[ ]:


df.head()


# In[ ]:


df.columns = ["Survived", "PassengerId"]


# In[ ]:


df.head()


# In[ ]:


df.to_csv('submission.csv', index=False)

