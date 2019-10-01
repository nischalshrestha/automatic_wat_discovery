#!/usr/bin/env python
# coding: utf-8

# My First submission - Titanic using python

# In[2]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#Get Data and some transformations
titanic_df = pd.read_csv('../input/titanic/train.csv')
titanic_df.head(),titanic_df.columns.values

def sex_convert(x):
    if x=='female':
        return 1
    if x=='male':
        return 0


titanic_df['Sex'] = titanic_df['Sex'].apply(sex_convert)
titanic_df['Age'] = titanic_df['Age'].fillna((titanic_df['Age'].mean()))


# In[11]:


# Test and Train split and some cleanup
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(titanic_df, test_size = 0.2)
test_df = test_df.drop(['Name'], axis=1)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['SibSp'], axis=1)
train_df = train_df.drop(['SibSp'], axis=1)
test_df = test_df.drop(['Parch'], axis=1)
train_df = train_df.drop(['Parch'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Fare'], axis=1)
train_df = train_df.drop(['Fare'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Embarked'], axis=1)
train_df = train_df.drop(['Embarked'], axis=1)
train = pd.DataFrame(train_df)
test = pd.DataFrame(test_df)
print(train.columns.values,test.columns.values)


# In[12]:


train.describe()


# In[21]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.copy()
X_test = X_test.drop("Survived", axis=1)
X_train.shape, Y_train.shape, X_test.shape
print(X_test)


# In[16]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[8]:


print(Y_pred)


# In[34]:


submission = pd.DataFrame({
 'PassengerId' : X_test['PassengerId'],
  'Survived' : Y_pred
 }).astype(int) 
submission.to_csv("Kaggle.csv",index=False)

