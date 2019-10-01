#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading data
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
PassengerId = data_test['PassengerId'].values


# In[ ]:


#We don't need some columns
data_train.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1, inplace = True)
data_test.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1, inplace = True)

one_hot_encoded_training_predictors = pd.get_dummies(data_train)
one_hot_encoded_test_predictors = pd.get_dummies(data_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
                 


# In[ ]:


train_Y = final_train.Survived
train_X = final_train.drop("Survived", axis = 1)
test_X = final_test.drop("Survived", axis = 1)


# In[ ]:


my_pipeline = make_pipeline(Imputer(), XGBClassifier(n_estimators=1000))
my_pipeline.fit(train_X, train_Y)
predictions = my_pipeline.predict(test_X)

my_pipeline = make_pipeline(Imputer(), svm.SVC())
my_pipeline.fit(train_X, train_Y)
predictions = predictions + my_pipeline.predict(test_X)

my_pipeline = make_pipeline(Imputer(), RandomForestClassifier())
my_pipeline.fit(train_X, train_Y)
predictions = predictions + my_pipeline.predict(test_X)

final_predictions = []
for i in predictions:
    if (i > 1):
        final_predictions.append(1)
    else:
        final_predictions.append(0)
df = {'PassengerId' : PassengerId,'Survived' : final_predictions}
submission = pd.DataFrame(df)
submission.to_csv('titanic.csv', index = False)


