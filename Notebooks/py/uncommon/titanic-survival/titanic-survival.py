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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import seaborn as sns


# In[ ]:


import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[ ]:


get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


data.info(
)


# In[ ]:


# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[ ]:


df_train.shape
df_test.shape


# In[ ]:


df_test.head()


# In[ ]:


df_train.head()


# In[ ]:


print(df_train[df_train.Sex == 'female'].Name.sum())


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.count())


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.value_counts())


# In[ ]:


data = pd.get_dummies(data, columns=['Sex'],drop_first=True)


# In[ ]:


data.head(
)


# In[ ]:


data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]


# In[ ]:


data.head()


# In[ ]:


example_answer = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


example_answer.shape


# In[ ]:


data.info()


# In[ ]:


data_train = data.iloc[:891]


# In[ ]:


data_test = data.iloc[891:]


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# In[ ]:


data_test.shape


# In[ ]:


data_train.shape


# In[ ]:


#transformng data from Data frame to Array 

X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred


# In[ ]:


my_prediction = df_test[['PassengerId', 'Survived']]


# In[ ]:


my_prediction.head()


# In[ ]:


my_prediction.to_csv('my_prediction.csv', index = False)


# In[ ]:




