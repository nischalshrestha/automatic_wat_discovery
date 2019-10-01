#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv')
examples = len(train.index)
print('in train.csv' , examples, 'elements')
header_list = list(train.columns)
print('Header in train.csv', header_list )
train['Sex'] = train.Sex.map({'male': 0, 'female': 1})


# In[ ]:


train = train.drop(['Ticket','Cabin','Embarked'], axis=1)
train = train.dropna()
median_age = train['Age'].dropna().median()

if len(train.Age[ train.Age.isnull() ]) > 0:

    train.loc[ (train.Age.isnull()), 'Age'] = median_age
    
header_list = list(train.columns)
print('Header in train.csv', header_list )
print(train)


# In[ ]:


feature_labels = [ 'Pclass',  'Sex', 'Parch','SibSp','Age']
idx = train[feature_labels].dropna().index
X = train.loc[idx, feature_labels].values
y = train.Survived.loc[idx]    


# In[ ]:


test = pd.read_csv('../input/test.csv')

test['Sex'] = test.Sex.map({'male': 0, 'female': 1})
test = test.dropna()

median_age_2 = test['Age'].dropna().median()
if len(test.Age[ test.Age.isnull() ]) > 0:
    train.loc[ (test.Age.isnull()), 'Age'] = median_age_2
    
print(test)


# In[ ]:


clf = LogisticRegression()
clf.fit(X, y)
yhats = {}
yhats['yhat_logistic_0'] = clf.predict(test[feature_labels].values)
print("Training Score: ", clf.score(X, y))

