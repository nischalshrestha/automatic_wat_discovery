#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read training data
titanic_train_path = "../input/train.csv"
df_titanic_train = pd.read_csv(titanic_train_path)
df_titanic_train.head()


# In[ ]:


# Explore data
df_titanic_train.info()


# In[ ]:


df_titanic_train.describe()


# In[ ]:


df_titanic_train.describe(include="O")


# In[ ]:


df_titanic_train.columns


# In[ ]:


# Define target variable
y = df_titanic_train.Survived


# In[ ]:


# Define features

titanic_features = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']
X = df_titanic_train[titanic_features]
X.head()


# In[ ]:


# Handling categorical feature
X_one_hot = pd.get_dummies(X)
X_one_hot.head()


# In[ ]:


# Handling missing values
missing_cols = [col for col in X_one_hot.columns if X_one_hot[col].isnull().any()]
missing_cols


# In[ ]:


imputer = SimpleImputer()
X_one_hot_imputed = imputer.fit_transform(X_one_hot)


# In[ ]:


# Split the data into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_one_hot_imputed, y, test_size = 0.2, random_state= 1)


# In[ ]:


# Xgboost tunning
# xgb_param_grid = { 'n_estimators':[50,100,150,200],
#                    'max_depth':[2,3,4,5,6,7,8,9],
#                    'min_child_weight':[2,3,4,5],
#                    'colsample_bytree':[0.2,0.6,0.8],
#                    'colsample_bylevel':[0.2,0.6,0.8],
#                    'learning_rate': [0.1, 0.05, 0.02]
#                  }


# In[ ]:


# # Define the model
# model = XGBClassifier()
# # Parameter tuning
# clf = GridSearchCV(model, param_grid= xgb_param_grid, n_jobs= -1, scoring='accuracy', verbose= 2, cv=5)
# # Fit model
# clf.fit(X_train, y_train)


# In[ ]:


# clf.best_estimator_


# In[ ]:


# clf.best_params_


# In[ ]:


# clf.best_score_


# In[ ]:


# # Predict on validation set
# y_pred_val = clf.predict(X_val)
# # Check model performance on validation set
# accuracy_score(y_val, y_pred_val)


# In[ ]:


# Train the model with all training data
final_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
       colsample_bytree=0.8, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=4, min_child_weight=2, missing=None, n_estimators=50,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
final_model.fit(X_one_hot_imputed, y, early_stopping_rounds=5, eval_set=[(X_val, y_val)], verbose= False)


# In[ ]:


# Read test data
titanic_test_path = "../input/test.csv"
df_titanic_test = pd.read_csv(titanic_test_path)
df_titanic_test.head()


# In[ ]:


# Align training and test set
X_test = df_titanic_test[titanic_features]
X_oh_test = pd.get_dummies(X_test)
X_oh_impute_test = imputer.transform(X_oh_test)


# In[ ]:


# Prediction on test set
y_pred_test = final_model.predict(X_oh_impute_test)


# In[ ]:


# Create output
output = pd.DataFrame({"PassengerId": df_titanic_test.PassengerId,
                      "Survived": y_pred_test})
output.to_csv("submission.csv", index= False)

