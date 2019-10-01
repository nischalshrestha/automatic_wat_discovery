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


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# To plot pretty figures
get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import mode
import string
import os

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train_test = [train, test]
train.shape


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


cols = ['Name','Ticket','Cabin']
train = train.drop(cols,axis=1)


# In[ ]:


train.head()


# In[ ]:


train = train.dropna()


# In[ ]:


ds = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
 ds.append(pd.get_dummies(train[col]))


# In[ ]:


titanic_ds = pd.concat(ds, axis=1)


# In[ ]:


train = pd.concat((train,titanic_ds),axis=1)


# In[ ]:


train = train.drop(['Pclass','Sex','Embarked'],axis=1)


# In[ ]:


train['Age'].fillna(train['Age'].median(), inplace=True)


# In[ ]:


X = train.values
y = train['Survived'].values


# In[ ]:


X = np.delete(X,1,axis=1)


# In[ ]:


from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.35,random_state=0)


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[ ]:


test['Survived'] = 0
test.loc[test['Sex'] == 'female','Survived'] = 1
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':test['Survived']
})


# In[ ]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)

