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


model = DecisionTreeRegressor(max_leaf_nodes=25)
model.fit(train_X, train_y)
pred_y = model.predict(val_X)
count = 0
# print(train_y)
for i in range(len(pred_y)):
    pred_y[i] = round(pred_y[i])
    if pred_y[i] == val_y.values[i][0]:
        count += 1
print(count/len(pred_y))
# mean_absolute_error(pred_y, train_y)
# print(train_X.head())
# print(train_y.head())


# In[ ]:


filename_test = '../input/test.csv'
with open(filename_test) as f:
    reader_test = pd.read_csv(f)
test_X = reader_test[['Pclass', 'SibSp', 'Parch', 'Fare']]

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(test_X)

test_X_imputed = pd.DataFrame(data_with_imputed_values)
print(test_X_imputed.describe())
print(reader_test.columns)


# In[ ]:


def get_dtf_rate(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    count = 0
    for i in range(len(preds_val)):
        preds_val[i] = round(preds_val[i])
        if preds_val[i] == targ_val.values[i][0]:
            count += 1
    # mae = mean_absolute_error(targ_val, preds_val)
    return(count / len(preds_val))


# In[ ]:


for max_leaf_nodes in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]:
    my_mae = get_dtf_rate(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Accurate Rate:  %f" %(max_leaf_nodes, my_mae))


# In[ ]:


rfr_model = RandomForestRegressor()
rfr_model.fit(train_X, train_y)
pred_y = rfr_model.predict(val_X)
count = 0
# print(train_y)
for i in range(len(pred_y)):
    pred_y[i] = round(pred_y[i])
    if pred_y[i] == val_y.values[i][0]:
        count += 1
print(count/len(pred_y))


# In[ ]:


model_final = RandomForestRegressor()
model_final.fit(train_set_X, train_set_y)
pred_y = model_final.predict(test_X_imputed)
for i in range(len(pred_y)):
    pred_y[i] = round(pred_y[i])
print(pred_y)
p_y = np.zeros(len(pred_y), dtype = np.int32)
for i in range(len(pred_y)):
    p_y[i] = int(pred_y[i])
print(p_y)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId' : reader_test.PassengerId, 'Survived' : p_y})
my_submission.to_csv('submission.csv', index=False)
# my_submission = pd.DataFrame({'Id': test_X.Id, 'SalePrice': predicted_prices})


# In[ ]:


print(int(1.0))


# In[ ]:




