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


# First things first, we need to read in the input, and figure out which columns are good to use. We can drop the name, the ticket, the cabin, and the passengerID because these are all strings, and it is hard to parse these into integers
# 
# We can map the Sex and Embarked to integers because they only take one of three values, so we are using the `map` method as shown below.

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df = train_df.drop('Name', 1)
train_df = train_df.drop('Ticket', 1)
train_df = train_df.drop('Cabin', 1)
train_df = train_df.drop('PassengerId', 1)
train_df = train_df.fillna(train_df.mean())

train_df['Sex'] = train_df['Sex'].map(lambda x: 0 if x == 'male' else 1)
train_df['Embarked'] = train_df['Embarked'].map(lambda x: 0 if x == 'S' else 1 if x == 'C' else 2)

train_df.head(10)


# Now we need to split the dataset into the features (*X*), and the label (*Y*). This is done like so:

# In[ ]:


X = train_df.drop('Survived', 1)
Y = train_df['Survived']

X.head()


# In order to evaluate the performance of the model, let us KFold the dataset.

# In[ ]:


from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)
cv = kf.split(train_df)

values = []

for train_ind, test_ind in cv:
    train_X = list(X.values[train_ind])
    train_Y = list(Y.values[train_ind])
    test_X = list(X.values[test_ind])
    test_Y = list(Y.values[test_ind])
    values += [[train_X, train_Y, test_X, test_Y]]


# Now comes the machine learning. This block runs each bit of training and testing data through the classifier and measures its performance. We can tune the depth, and see that the performance of the tree changes a little

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

depth = 5

for train_X, train_Y, test_X, test_Y in values:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    
#     print(confusion_matrix(pred_Y, test_Y))
#     print(classification_report(pred_Y, test_Y))
    print(accuracy_score(pred_Y, test_Y))


# Since we are happy with the depth, now let's just train the tree on the entire dataset

# In[ ]:


clf = DecisionTreeClassifier(max_depth=depth)
clf.fit(X, Y)


# Now lets do the same for the test set

# In[ ]:


test_df = pd.read_csv('../input/test.csv')

passenger_id = test_df['PassengerId']

test_df = test_df.drop('Name', 1)
test_df = test_df.drop('Ticket', 1)
test_df = test_df.drop('Cabin', 1)
test_df = test_df.drop('PassengerId', 1)
test_df = test_df.fillna(test_df.mean())

test_df['Sex'] = test_df['Sex'].map(lambda x: 0 if x == 'male' else 1)
test_df['Embarked'] = test_df['Embarked'].map(lambda x: 0 if x == 'S' else 1 if x == 'C' else 2)

test_df.head()


# Here we're making predictions on the test set, and writing them to the file `titanic.csv`. We can submit this file to the competition in the main kernel page!

# In[ ]:


pred_Y = clf.predict(test_df)
submission = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": pred_Y
    })
submission.to_csv('titanic.csv', index=False)

