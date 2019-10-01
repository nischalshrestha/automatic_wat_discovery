#!/usr/bin/env python
# coding: utf-8

# # Super Simple Titanic XGBoost
# Solve the Titanic challenge with XGBoost and Python only using three features: Sex, Pclass, Embarked
# 
# Basically trying to do the same as [this notebook](https://www.kaggle.com/pliptor/minimalistic-xgb) but using Python.
# **Edit:** Updated the notebook to include label encoding and automatic parameter tuning. With lots of help (i.e., copy and paste) from [this notebook using LightGBM](https://www.kaggle.com/pliptor/minimalistic-titanic-in-python-lightgbm).
# 
# I can't figure out why the result is so bad though...

# In[36]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline


feature_names = ['Sex','Embarked','Pclass']

# load data sets 
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId'] + feature_names)
test  = pd.read_csv('../input/test.csv', usecols =['PassengerId'] + feature_names )

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()


# In[37]:


# fix missing embarked values
print('Number of missing Embarked values ',comb['Embarked'].isnull().sum())
comb['Embarked'] = comb['Embarked'].fillna('S')

# use label encoding for categorical data (embarked and sex)
le = LabelEncoder()
comb.Embarked = le.fit_transform(comb.Embarked)
comb.Sex      = le.fit_transform(comb.Sex)
comb.head()


# In[38]:


# split the data back into train and test
df_train = comb.loc[comb['Survived'].isin([np.nan]) == False]
df_test  = comb.loc[comb['Survived'].isin([np.nan]) == True]

print(df_train.shape)
print(df_train.head())
print(df_test.shape)
print(df_test.head())


# In[39]:


# XGBoost model + parameter tuning with GridSearchCV
xgb = XGBRegressor()
param_grid = {'n_estimators': [10, 20, 30, 50, 100], 'max_depth': [2,3,4], 'colsample_bytree': [0.2, 0.8, 1], 'subsample': [0.2, 0.8, 1]} 
grs = GridSearchCV(xgb, param_grid=param_grid, cv = 10, n_jobs=4, return_train_score = False)
grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))

print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data: {0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))
# TODO: why is this so bad?


# In[40]:


train_X, val_X, train_y, val_y = train_test_split(df_train[feature_names], df_train["Survived"] ,random_state = 0)
xgb2 = XGBRegressor()
xgb2.fit(train_X, train_y)
pred = xgb2.predict(val_X).astype("int")
accuracy_score(val_y, pred)

