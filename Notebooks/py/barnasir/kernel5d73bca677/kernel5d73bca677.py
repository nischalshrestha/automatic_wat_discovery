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


test_file = pd.read_csv("../input/test.csv")


# In[ ]:


test_shape = test_file.shape


# In[ ]:


print(test_shape)


# In[ ]:


train_file = pd.read_csv("../input/train.csv")


# In[ ]:


train_shape = train_file.shape


# In[ ]:


print(train_shape)


# In[ ]:


train_file.head(10)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# Prediction of survival on the basis of gender.
sex_pivot = train_file.pivot_table(index="Sex", values="Survived")


# In[ ]:


sex_pivot


# In[ ]:


sex_pivot.plot.bar()


# In[ ]:


# Prediction of survival on the basis of Pclass.
pclass_pivot = train_file.pivot_table(index="Pclass", values="Survived")


# In[ ]:


pclass_pivot


# In[ ]:


survived_df = train_file[train_file["Survived"]==1]


# In[ ]:


died_df = train_file[train_file["Survived"]==0]


# In[ ]:


# Change the continuous data into categorical data
def classify_age_categories(df, boundary_points, label_age_category):
    df["Age"]=df["Age"].fillna(-0.5)
    df["age_categories"] = pd.cut(df["Age"], boundary_points, labels=label_age_category)
    return df


# In[ ]:


boundary_points = [-1,0, 5, 13, 19, 40, 65, 100]


# In[ ]:


label_age_category = ["Missing", "Infant", "Child", "Teenager", "Grown Man", "Adult", "Senior"]


# In[ ]:


train = classify_age_categories(train_file, boundary_points, label_age_category)
test = classify_age_categories(test_file, boundary_points, label_age_category)


# In[ ]:


categorical_age_pivot = train.pivot_table(index="age_categories", values="Survived")


# In[ ]:


categorical_age_pivot


# In[ ]:


df = train
dummies = pd.get_dummies(df["Pclass"], prefix="Pclass")
dummies.head()


# In[ ]:


# create dummy variables for the sex, age categories and pclass
def create_dummy_variables(df,col_name):
    dummies = pd.get_dummies(df[col_name],prefix=col_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[ ]:


train = create_dummy_variables(train,"Pclass")
test = create_dummy_variables(test,"Pclass")
train = create_dummy_variables(train,"Sex")
test = create_dummy_variables(test,"Sex")
train = create_dummy_variables(train,"age_categories")
test = create_dummy_variables(test,"age_categories")


# In[ ]:


train.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


lr = LR()


# In[ ]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'age_categories_Missing','age_categories_Infant',
       'age_categories_Child', 'age_categories_Teenager',
       'age_categories_Grown Man', 'age_categories_Adult',
       'age_categories_Senior']


# In[ ]:


kaggle_test = test


# In[ ]:


col_to_train = train[columns]


# In[ ]:


col_to_predict = train['Survived']


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(col_to_train, col_to_predict, test_size=0.2,random_state=0)


# In[ ]:


lr.fit(train_X, train_y)


# In[ ]:


predictions = lr.predict(test_X)


# In[ ]:


accuracy = accuracy_score(test_y, predictions)


# In[ ]:


accuracy


# In[ ]:


conf_matrix = confusion_matrix(test_y, predictions)
pd.DataFrame(conf_matrix, columns=['Survived', 'Died'], index=[['Survived', 'Died']])


# In[ ]:


scores = cross_val_score(lr, col_to_train, col_to_predict, cv=10)


# In[ ]:


np.mean(scores)


# In[ ]:


kaggle_test.head()


# In[ ]:


predict_lr = LR()
predict_lr.fit(col_to_train, col_to_predict)
kaggle_test_predictions = predict_lr.predict(kaggle_test[columns])
kaggle_test_predictions


# In[ ]:


predicted = {"PassengerId":kaggle_test["PassengerId"], "Survived":kaggle_test_predictions}


# In[ ]:


predicted_df = pd.DataFrame(predicted)


# In[ ]:


predicted_df.to_csv('titanic.csv', index=False)


# In[ ]:




