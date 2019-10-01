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


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


train_df= pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# In[ ]:


train_df.head()


# In[ ]:


train_df.isnull().values.any()


# In[ ]:


train_df.isnull().sum().sum()


# In[ ]:


rf = RandomForestClassifier()


# In[ ]:


def prepare_x(data, process_y=False):
    drop_cols = ['Name', 'Ticket', 'Cabin']
    new_data = data
    new_data.Age.fillna(new_data.Age.mean(), inplace=True)
    new_data = new_data.drop(drop_cols, axis=1)
    sexes = pd.get_dummies(new_data.Sex, prefix='sex')
    new_data = pd.concat([new_data, sexes], axis=1)
    new_data.drop('Sex', inplace=True, axis=1)
    emb = pd.get_dummies(new_data.Embarked, prefix='embarked')
    new_data = pd.concat([new_data, emb], axis=1)
    new_data.drop('Embarked', inplace=True, axis=1)
    new_data.fillna(0, inplace=True)
    
    if process_y:
        return new_data.drop('Survived', axis=1), new_data.Survived
    else:
        return new_data


# In[ ]:


X, y = prepare_x(train_df, process_y=True)


# In[ ]:


parameters = {
    'max_depth': [4,8,12],
    'n_estimators': [50, 200, 400],
    'max_features': ['log2', 'sqrt', 'auto'],
    'n_jobs': [-1]
}


# In[ ]:


grid = GridSearchCV(rf, parameters)


# In[ ]:


grid.fit(X, y)


# In[ ]:


best_est = grid.best_estimator_


# In[ ]:


X_test = prepare_x(test_df)


# In[ ]:


best_est.score(X, y)


# In[ ]:


predictions = best_est.predict(X_test)


# In[ ]:


submissions = pd.DataFrame(predictions, index=X_test.PassengerId)


# In[ ]:


submissions.to_csv('submission.csv', index=True)


# In[ ]:




