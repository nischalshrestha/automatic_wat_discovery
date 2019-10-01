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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Any results you write to the current directory are saved as output.


# In[ ]:


#played with gender submission file
gender_data = pd.read_csv('../input/gender_submission.csv')
gender_data.head()


# In[ ]:


#get info about the data
gender_data.info()


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
#show first 5 rows
train_data.head()


# In[ ]:


#some sample queries that didn't work
#train_data['Name'].value_counts() - useless because unique
#train_data['Ticket'].value_counts() - highest # is 7


# In[ ]:


#value count to see how many there are of each (can check if important values are balanced or not)
train_data['Pclass'].value_counts()


# In[ ]:


train_data['Pclass'].describe()


# In[ ]:


train_data['Sex'].value_counts()


# In[ ]:


train_data['Sex'].describe()


# In[ ]:


t = train_data['Age'].value_counts()
#get the top 30 values by count
t.nlargest(30)


# In[ ]:


train_data['SibSp'].describe()


# In[ ]:


sib = train_data['SibSp'].value_counts()


# In[ ]:


train_data['Parch'].describe()


# In[ ]:


train_data['Parch'].value_counts()


# In[ ]:


train_data['Ticket'].describe()


# In[ ]:


ticket = train_data['Ticket'].value_counts()
ticket.nlargest(30)


# In[ ]:


train_data['Fare'].describe()


# In[ ]:


fare = train_data['Fare'].value_counts()
fare.nlargest(25)


# In[ ]:


train_data['Cabin'].describe()


# In[ ]:


cabin = train_data['Cabin'].value_counts()
cabin.nlargest(20)


# In[ ]:


train_data['Embarked'].value_counts()


# In[ ]:


train_data['Embarked'].describe()


# In[ ]:


train_data['Survived'].mean()   


# In[ ]:


train_data = train_data.drop(['Cabin', 'Fare'], axis=1)


# In[ ]:


train_data = train_data.drop(['Ticket'], axis=1)


# In[ ]:


train_data = train_data.drop(['PassengerId'], axis=1)


# In[ ]:


train_data = train_data.drop(['Embarked'], axis=1)


# In[ ]:


train_data = train_data.drop(['Name'], axis=1)


# In[ ]:


train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_data.dropna(how='any', inplace=True)


# In[ ]:


train_data.describe()


# In[ ]:


X_train = train_data.dropna()


# In[ ]:


X_train.describe()


# In[ ]:


X_train = train_data.drop(['Survived'], axis=1)


# In[ ]:


#map male to 0
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0


# In[ ]:


#map female to 1
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1


# In[ ]:


train_data.head()


# In[ ]:


#set Y_train to the labels
Y_train = train_data["Survived"]


# In[ ]:


print(X_train)


# In[ ]:


X_train.describe()


# In[ ]:


print(Y_train)


# In[ ]:


#train_data = train_data.drop(['Survived'], axis =1)
#train_data = train_data.drop(['Survived'], axis=1)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data.head()


# In[ ]:





# In[ ]:


test_data  = test_data.drop(['Name'], axis=1)


# In[ ]:


test_data  = test_data.drop(['Ticket'], axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data = test_data.drop(['Cabin'], axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data = test_data.drop(['Embarked'], axis=1)


# In[ ]:


test_data = test_data.drop(['Fare'], axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0


# In[ ]:


test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1


# In[ ]:


test_data.describe()


# In[ ]:


test_data['Age']= test_data['Age'].fillna('30.2')


# In[ ]:


test_data.head()


# In[ ]:


X_train.loc[X_train['Sex'] == 'male', 'Sex'] = 0


# In[ ]:


X_train.loc[X_train['Sex'] == 'female', 'Sex'] = 1


# In[ ]:


X_train.head()


# In[ ]:





# In[ ]:





# In[ ]:


X_train.shape, Y_train.shape, test_data.shape


# In[ ]:


print(test_data.to_string())



# In[ ]:


test_data = test_data.drop(['PassengerId'], axis =1)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(test_data)
print(Y_pred)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

