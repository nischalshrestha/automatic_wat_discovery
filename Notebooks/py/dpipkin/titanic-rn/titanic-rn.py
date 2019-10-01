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


df = pd.read_csv('../input/train.csv', index_col='PassengerId')


# In[ ]:


df.sample(10)


# In[ ]:


df['Survived'].sum() / len(df)


# In[ ]:


1 - _


# In[ ]:


len(df[(df['Sex'] == 'male') & (df['Survived'] == 1)]) /   len(df[df['Sex'] == 'male'])


# In[ ]:


len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)]) /   len(df[df['Sex'] == 'female'])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'])


# In[ ]:


x_train.head()


# In[ ]:


x_train = pd.get_dummies(x_train, columns=['Sex'])


# In[ ]:


x_train = x_train[['Sex_female','Sex_male']]


# In[ ]:


x_train.sample(10)


# In[ ]:


clf.fit(x_train, y_train)


# In[ ]:


x_test = pd.get_dummies(x_test, columns=['Sex'])[['Sex_female', 'Sex_male']]


# In[ ]:


y_pred = clf.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


test_data = pd.read_csv('../input/test.csv', index_col='PassengerId')


# In[ ]:


x = pd.get_dummies(test_data, columns=['Sex'])[['Sex_female', 'Sex_male']]


# In[ ]:


y = clf.predict(x)


# In[ ]:


y = pd.Series(y, index=x.index)


# In[ ]:


y.head()


# In[ ]:


y.name = 'Survived'


# In[ ]:


pd.read_csv('../input/gender_submission.csv').sample(10)


# In[ ]:


y.to_csv('random_forest_gender.csv', header=True)

