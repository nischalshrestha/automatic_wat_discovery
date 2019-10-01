#!/usr/bin/env python
# coding: utf-8

# Dataset summary:
# 
# * **Survived** 0 = No, 1 = Yes 
# * **pclass** Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# * **Sex**
# * **Age**: Age in years
# * **Sibsp**: # of siblings / spouses aboard the Titanic 
# * **Parch** # of parents / children aboard the Titanic
# * **Ticket** Ticket number 
# * **Fare** Passenger fare
# * **Cabin** number embarked 
# * **Port** of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Notes:
# 
# * pclass: A proxy for socio-economic status (SES)
# * * 1st = Upper
# * * 2nd = Middle
# * * 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way...
# * Sibling = brother, sister, stepbrother, stepsister
# * Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * parch: The dataset defines family relations in this way...
# * Parent = mother, father
# * Child = daughter, son, stepdaughter, stepson
# * Some children travelled only with a nanny, therefore parch=0 for them.

# Initially, we get a general dataset structure: shape, columns, numerical and categorical columns.
# It is worth to highlight the amount of NaN values. Dataset shape rows decrease from 891 to 183! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

train = pd.read_csv("../input/train.csv")
train.drop("PassengerId", axis=1, inplace=True)
train_na = train.dropna(axis=0, inplace=False)

print ("Training Columns\n", list(train.columns))
print ("Training set shape\n", train.shape)
print ("Training set shape without NaN values\n", train_na.shape)

numerical_columns = ["Age", "Fare"]
string_columns = ["Name", "Ticket", "Cabin"]
categorical_columns = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Survived"]
label = "Survived"

survived = train[train.Survived == 1]
notSurvived = train[train.Survived == 0]

# Any results you write to the current directory are saved as output.


# In[ ]:


# describing numerical values
train.drop(categorical_columns, axis=1).describe(include=[np.number])


# In[ ]:


# describing categorical columns
train.drop(numerical_columns, axis=1).describe()


# A heatmap graph is generated to spot relation between dataset columns.

# In[ ]:


# using a heat map to get a general vision of a dataset
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corrmat, vmax=0.8, square=True)


# Age is the main continuous column on the dataset, thus Age x Survived and Age x not survived is showed on left and right, repespectively.

# In[ ]:


# survived age distribution
f, axes = plt.subplots(1, 2, figsize=(12, 5))
g = sns.distplot(survived.Age.dropna(), bins=10, ax=axes[0])
g = sns.distplot(notSurvived.Age.dropna(), bins=10, ax=axes[1], color="red")


# Histogram counting genre for both Survived and not survived.

# In[ ]:


# sex x survived
g = sns.factorplot(x="Sex", col="Survived", data=train.dropna(), kind="count", size=5, aspect=1)


# Histogram counting Pclass for both Survived and not survived.

# In[ ]:


# Pclass x survived
sns.factorplot(x="Pclass", col="Survived", data=train.dropna(), kind="count", size=5, aspect=1)


# Histogram counting Parch for both Survived and not survived.

# In[ ]:


# Parch x survived
sns.factorplot(x="Parch", col="Survived", data=train.dropna(), kind="count", size=5, aspect=1)


# After all, some conclusion can be made.
# 
# 1. Age distribution graph shows people between 20 and 40 years old are more likely to survive. Also, babies had a good chance to survived.
# 2. Genre histogram shows female are more likely to survived (women and babies first?)
# 3. Class 1 tickets owners are more likely to survived. Did they have access to boats first?
# 4. Finally, passengers with no parent aboard were more likely to survived. They were by their own.
# 
