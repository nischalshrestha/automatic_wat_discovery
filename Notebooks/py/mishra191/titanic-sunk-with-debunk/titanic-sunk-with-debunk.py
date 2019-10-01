#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 

# First import the train and test data

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# Review of the test and train data

# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# Now we analyse the survival rate using Passenger Class and sex

# In[ ]:


sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train_data)


# We can see from above plot that Female survived higher than male. Morover the person with higher class was able to survive more than person with lower class

# **Find NAN values**

# Now first we need to convert those feature whose datatype is in string into int. Because machine learning can not operate on string

# In[ ]:


#First need to combine the test and train data
train_data.isnull().any()


# In[ ]:


test_data.isnull().any()


# Replace NAN values with mean value of column

# In[ ]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())


# In[ ]:


train_data.isnull().any()


# In[ ]:


test_data.isnull().any()


# In[ ]:


train_data['Cabin'].fillna('U', inplace=True)
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])
train_data['Cabin'].unique()


# In[ ]:


test_data['Cabin'].fillna('U', inplace=True)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])
test_data['Cabin'].unique()


# In[ ]:


replacement = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

train_data['Cabin'] = train_data['Cabin'].apply(lambda x: replacement.get(x))
train_data['Cabin'] = StandardScaler().fit_transform(train_data['Cabin'].values.reshape(-1, 1))
train_data.head()['Cabin']

test_data['Cabin'] = test_data['Cabin'].apply(lambda x: replacement.get(x))
test_data['Cabin'] = StandardScaler().fit_transform(test_data['Cabin'].values.reshape(-1, 1))
test_data.head()['Cabin']


# In[ ]:


#train_data['Cabin'].head()
test_data['Cabin'].head()


# In[ ]:


train_data.head()


# Apply the same logic of Cabin column to Embarked Column

# In[ ]:


train_data['Embarked'].fillna('N', inplace=True)
train_data['Embarked'] = train_data['Embarked'].apply(lambda x: x[0])
train_data['Embarked'].unique()

test_data['Embarked'].fillna('N', inplace=True)
test_data['Embarked'] = test_data['Embarked'].apply(lambda x: x[0])
test_data['Embarked'].unique()


# In[ ]:


replacement = {
    'S': 0,
    'C': 1,
    'Q': 2,
    'N': 3
}

train_data['Embarked'] = train_data['Embarked'].apply(lambda x: replacement.get(x))
train_data['Embarked'] = StandardScaler().fit_transform(train_data['Embarked'].values.reshape(-1, 1))
train_data.head()['Embarked']

test_data['Embarked'] = test_data['Embarked'].apply(lambda x: replacement.get(x))
test_data['Embarked'] = StandardScaler().fit_transform(test_data['Embarked'].values.reshape(-1, 1))
test_data.head()['Embarked']


# Fare column (filling the NaN values with the mean of the other columns.

# In[ ]:


train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())


# In[ ]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())


# In[ ]:


train_data.head()


# In[ ]:


combined = [train_data, test_data]
for dataset in combined:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_data.head()


# In[ ]:


test_data.head()


# New Feature:- family size

# In[ ]:


def process_family_train():
    
    # introducing a new feature : the size of families (including the passenger)
    train_data['FamilySize'] = train_data['Parch'] + train_data['SibSp'] + 1
    
    # introducing other features based on the family size
    train_data['Singleton'] = train_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    train_data['SmallFamily'] = train_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    train_data['LargeFamily'] = train_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    return train_data


# In[ ]:


train_data = process_family_train()
train_data.head()


# In[ ]:


def process_family_test():
    
    # introducing a new feature : the size of families (including the passenger)
    test_data['FamilySize'] = test_data['Parch'] + test_data['SibSp'] + 1
    
    # introducing other features based on the family size
    test_data['Singleton'] = test_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    test_data['SmallFamily'] = test_data['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    test_data['LargeFamily'] = test_data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    return test_data


# In[ ]:


test_data = process_family_test()
test_data.head()


# In[ ]:


test_data_Id = test_data["PassengerId"]

train_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
test_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

train_data.head()
test_data.head()


# In[ ]:


X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]


# In[ ]:


Logistic Regression


# In[ ]:


regr = LogisticRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(test_data)
acc_log = round(regr.score(X_train, Y_train) * 100, 2)
acc_log


# Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test_data)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# Submission

# In[ ]:


subm = pd.DataFrame({
        "PassengerId": test_data_Id,
        "Survived": Y_pred
    })
subm.to_csv('subm.csv', index=False)

