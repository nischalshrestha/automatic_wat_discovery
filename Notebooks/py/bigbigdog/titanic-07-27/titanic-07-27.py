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
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


filename_train = '../input/train.csv'
with open(filename_train) as f:
    reader_train = pd.read_csv(f)
# print(reader_train.columns)
train_set_X = reader_train[['Pclass', 'SibSp', 'Parch', 'Fare']]
train_set_y = reader_train[['Survived']]
# print(train_set_y.describe())
train_X, val_X, train_y, val_y = train_test_split(train_set_X, train_set_y, random_state = 0)


# In[ ]:


filename_test = '../input/test.csv'
with open(filename_test) as f:
    reader_test = pd.read_csv(f)
test_X = reader_test[['Pclass', 'SibSp', 'Parch', 'Fare']]

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(test_X)

test_X_imputed = pd.DataFrame(data_with_imputed_values)


# In[ ]:


model = DecisionTreeRegressor(max_leaf_nodes=25)
model.fit(train_X, train_y)
pred_y = model.predict(val_X)
count = 0
for i in range(len(pred_y)):
    pred_y[i] = round(pred_y[i])
    if pred_y[i] == val_y.values[i]:
        count += 1
print(count / len(pred_y))


# In[ ]:


model_final = DecisionTreeRegressor(max_leaf_nodes=25)
model_final.fit(train_set_X, train_set_y)
pred_y = model_final.predict(test_X_imputed)
for i in range(len(pred_y)):
    pred_y[i] = round(pred_y[i])
# print(pred_y)
p_y = np.zeros(len(pred_y), dtype = np.int32)
for i in range(len(pred_y)):
    p_y[i] = int(pred_y[i])
# print(p_y)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId' : reader_test.PassengerId, 'Survived' : p_y})
my_submission.to_csv('submission.csv', index=False)
# my_submission = pd.DataFrame({'Id': test_X.Id, 'SalePrice': predicted_prices})


# In[ ]:




