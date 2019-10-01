#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# IMPORT
import pandas as pd
from pandas import Series, DataFrame

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC


# In[ ]:


# Load data
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


# Drop something that seems not such useful.
train.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'], inplace=True)
test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Fill NaN number.
train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Embarked'] = test['Embarked'].fillna('S')
test['Age'] = test['Age'].fillna(test['Age'].mean())


# In[ ]:


# Get some qualitative information.
# Sex
train_sex_pos = train['Sex'][train['Survived'] == 1].value_counts()
train_sex_neg = train['Sex'][train['Survived'] == 0].value_counts()
df_sex = DataFrame({'Survived': train_sex_pos, 'Not survived': train_sex_neg})
df_sex.plot(kind='bar', stacked=True)

# Pclass
train_cls_pos = train['Pclass'][train['Survived'] == 1].value_counts()
train_cls_neg = train['Pclass'][train['Survived'] == 0].value_counts()
df_cls = DataFrame({'Survived': train_cls_pos, 'Not survived': train_cls_neg})
df_cls.plot(kind='bar', stacked=True)

# SibSp
train_sib_pos = train['SibSp'][train['Survived'] == 1].value_counts()
train_sib_neg = train['SibSp'][train['Survived'] == 0].value_counts()
df_sib = DataFrame({'Survived': train_sib_pos, 'Not survived': train_sib_neg})
df_sib.plot(kind='bar', stacked=True)

# Age
train_age_pos = train['Age'][train['Survived'] == 1].value_counts()
train_age_neg = train['Age'][train['Survived'] == 0].value_counts()
df_age = DataFrame({'Survived': train_age_pos, 'Not survived': train_age_neg})
df_age.plot(kind='bar', stacked=True)


# In[ ]:


# Strings to data.
train['Sex'] = train['Sex'].apply(lambda sex: 1 if sex == 'male' else 0)
train['Embarked'] = train['Embarked'].apply(lambda e: 'SCQ'.find(e))

test['Sex'] = test['Sex'].apply(lambda sex: 1 if sex == 'male' else 0)
test['Embarked'] = test['Embarked'].apply(lambda e: 'SCQ'.find(e))

train.head()


# In[ ]:


# Data to matrix.
X = train.drop(columns=['Survived'], inplace=False).as_matrix()
y = train['Survived'].as_matrix()

test_passengerId = test['PassengerId']
test.drop(columns=['PassengerId'], inplace=True)
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
Xval = test.as_matrix()

# Just Logistic it.
lr = LogisticRegression(C=1, penalty='l1', tol=1e-5)
lr.fit(X, y)
y_lr = lr.predict(Xval)

# And SVM.
svc = SVC()
svc.fit(X, y)
y_svc = svc.predict(Xval)


# In[ ]:


submission1 = DataFrame({'PassengerId': test_passengerId, 'Survived': y_lr})
submission2 = DataFrame({'PassengerId': test_passengerId, 'Survived': y_svc})
submission1.to_csv('titanic.csv', index=False)

