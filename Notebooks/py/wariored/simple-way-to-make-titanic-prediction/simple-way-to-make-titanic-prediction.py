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


# # 1- Read csv files with pandas

# Read test and train files and explore them

# In[ ]:


test = pd.read_csv('../input/test.csv', header=0)
train = pd.read_csv('../input/train.csv', header=0)
gender = pd.read_csv('../input/gender_submission.csv', header=0)


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


# print both train and test shape
print(train.shape, test.shape)


# In[ ]:


# check if there are np.nan values in train columns
train.isnull().sum()


# In[ ]:


# show description
train.describe()


# In[ ]:


# show all culumns data types and there values lenght
train.info()


# In[ ]:


# We need to clean Sex, Embarked and Survived. Let's look at their values
print(train.Sex.unique())
print(train.Embarked.unique())
print(train.Survived.unique())


# # 2- Cleaning Data

# In[ ]:


# this help us to not directly modify our initial dataframes
train_data = train.copy()
test_data = test.copy()


# In[ ]:


# let's replace Sex values on both train_data and test_data 
train_data.loc[train_data.Sex == "male", 'Sex'] = 1
train_data.loc[train_data.Sex == "female", 'Sex'] = 2
test_data.loc[test_data.Sex == "male", 'Sex'] = 1
test_data.loc[test_data.Sex == "female", 'Sex'] = 2


# In[ ]:


# print changes
print(train_data.Sex.head(3))
print(test_data.Sex.head(3))


# In[ ]:


# replace missings values by the column mean
train_data.Age.fillna(train_data.Age.mean(), inplace=True)
test_data.Age.fillna(test_data.Age.mean(), inplace=True)
# there is no missing values for Fare Column in train_data
# only test_data is in concern
test_data.Fare.fillna(test_data.Fare.mean(), inplace=True)


# In[ ]:


# replace Embarked missings by the most frequent value
embarked_val = {'S': 1, 'C': 2, 'Q': 3}
train_data.Embarked.fillna(train_data.Embarked.value_counts().idxmax(), inplace=True)
test_data.Embarked.fillna(test_data.Embarked.value_counts().idxmax(), inplace=True)
train_data.Embarked = train_data.Embarked.replace(embarked_val)
test_data.Embarked = test_data.Embarked.replace(embarked_val)


# In[ ]:


# look at changes
train_data.Embarked.unique()


# In[ ]:


# drop useless Columns, these columns won't affect our prediction
train_data.drop('Name', axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)
train_data.drop('PassengerId', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)
test_data.drop('PassengerId', axis=1, inplace=True)


# Look at entries of  cleaned data

# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# # 3- Prediction

# In[ ]:


# make values readable for Scikit-Learn
y_train = train_data.Survived.values
X_train = train_data.drop(['Survived'], axis=1).values
X_test = test_data.values


# In[ ]:


# import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
reg = LogisticRegression()
# Fit training data
reg.fit(X_train, y_train)


# In[ ]:


# Predict test data
result = reg.predict(X_test)
print(len(result))
print(result[:20])


# In[ ]:


survived = pd.DataFrame(test.PassengerId)
survived['Survived'] = result
survived.head()


# In[ ]:


survived.tail()


# In[ ]:


# check my score
accuracy = accuracy_score(y_true=gender.Survived.values, y_pred=result)
print(accuracy)


# In[ ]:


# save result in CSV file
survived.to_csv('submission.csv', index = False)

