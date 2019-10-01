#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

import os
print(os.listdir("../input"))
PATH = '../input'
# Any results you write to the current directory are saved as output.


# **First I import the necessary data**

# In[ ]:


train_data = pd.read_csv(os.path.join(PATH,'train.csv'))
test_data = pd.read_csv(os.path.join(PATH, 'test.csv'))
example = pd.read_csv('../input/gender_submission.csv')


# **The 'Cabin' column has null values in it and contains non-categorical strings so I did not want to use it in the regression. However, it may contain useful information so I just determined whether someone had a cabin or not in the 'is_cabin' column.**

# In[ ]:


train_data['is_cabin']=train_data['Cabin'].notna().astype(int)
train_data.head()
test_data['is_cabin']=test_data['Cabin'].notna().astype(int)


# In[ ]:


train_data.head()


# Here I create a 'ticket_num' column to extract the numerical value from the 'Ticket' column. 

# In[ ]:


def extract_int(entry):
    for s in entry.split():
        if s.isdigit():
            return int(s)
train_data['ticket_num']=list(map(extract_int, train_data['Ticket']))
test_data['ticket_num']=list(map(extract_int, test_data['Ticket']))


# In[ ]:


# to fill in any missing numerical values
from sklearn.impute import SimpleImputer


# **Assigned the train and test values. Used one hot encoding since 'age' and 'embarked' are categorical values.**

# In[ ]:


train_X = train_data.drop(['Survived','PassengerId','Name','Ticket','Cabin'], axis=1)
train_y = train_data['Survived']

test_X = test_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

#one hot encoding
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)


# **Check to see that there are equal numbers of columns in both test and train.**

# In[ ]:


print(train_X.columns)
print(test_X.columns)


#  ****Initiate the model. Here I use XGBoost and set the 'objective' parameter to 'binary:logistic' since our only possible outcomes are 1 or 0. The model will output the probability that a particular passenger survived(1).****

# In[ ]:


my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(objective='binary:logistic'))
my_pipeline.fit(train_X, train_y)


# **Output the predictions and convert those probabilities to integers 1 or 0, indicating survivorship.**

# In[ ]:


predictions = my_pipeline.predict(test_X)
new_predictions = [int(round(x)) for x in predictions]


# In[ ]:


example


# In[ ]:


test_raw = pd.read_csv('../input/test.csv')


# In[ ]:


pd.DataFrame({'PassengerId':test_raw['PassengerId'], 'Survived':new_predictions}).to_csv('submission.csv', index=False)


# In[ ]:




