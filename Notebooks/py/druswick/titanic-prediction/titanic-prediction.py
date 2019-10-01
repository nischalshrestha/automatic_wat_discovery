#!/usr/bin/env python
# coding: utf-8

# Titanic Survival Model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')


# Drop columns that logically shouldn't matter or are mostly null:

# In[ ]:


train.drop(['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket', 'Fare'], axis=1, inplace=True)


# Convert sex into a dummy varible so we can regress on it:

# In[ ]:


train['Male'] = pd.get_dummies(train['Sex'])['male']
train.drop('Sex', axis=1, inplace=True)


# Let's create another dummy that indicates whether they were a child or not:

# In[ ]:


train['Child'] = train.apply(lambda row: row['Age'] <= 12.0, axis=1)
train.drop('Age', inplace=True, axis=1)


# Instead of regressing on family sizes, let's create a dummy indicating whether the person had any family

# In[ ]:


train['SibsSp'] = train.apply(lambda row: row['SibSp'] > 0, axis=1)
train['Parch'] = train.apply(lambda row: row['Parch'] > 0, axis=1)


# Separate y values:

# In[ ]:


labels = train['Survived']
train.drop('Survived', axis=1, inplace=True)


# Import various models to try out:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold


# Do a KFold validation for each of our candidate models, then print the best:

# In[ ]:


splits = KFold(n_splits=5, shuffle=True)
for model in [RandomForestClassifier(), SVC(), DecisionTreeClassifier(), GaussianNB()]:
    print(model)
    print(np.mean([model.fit(train.iloc[tr], labels.iloc[tr]).score(train.iloc[te], labels.iloc[te]) for tr, te in splits.split(train, labels)]))


# It definitely looks like Support Vectors are performing the best here, so that is the model we will use for our predictions on the test data.

# Performing data massaging on the test data:

# In[ ]:


test = pd.read_csv('../input/test.csv')
test.drop(['Name', 'Cabin', 'Embarked', 'Ticket', 'Fare'], axis=1, inplace=True)
test['Male'] = pd.get_dummies(test['Sex'])['male']
test.drop('Sex', axis=1, inplace=True)
test['Child'] = test.apply(lambda row: row['Age'] <= 12.0, axis=1)
test.drop('Age', inplace=True, axis=1)
test['SibsSp'] = test.apply(lambda row: row['SibSp'] > 0, axis=1)
test['Parch'] = test.apply(lambda row: row['Parch'] > 0, axis=1)


# Predict the survival values and write them to a csv file:

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test['PassengerId'],
        "Survived": SVC().fit(train, labels).predict(test.drop('PassengerId', axis=1))
    })
submission.to_csv('titanic.csv', index=False)

