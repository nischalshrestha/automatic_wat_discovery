#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic(u'matplotlib inline')

from sklearn import tree


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_id = train_df[['PassengerId']]

train_id.head()


# In[ ]:


train_df = train_df.drop('PassengerId', axis=1)


# In[ ]:


train_df.isnull().sum()


# In[ ]:


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())


# In[ ]:


test_id = test_df['PassengerId']

test_df = test_df.drop('PassengerId', axis=1)


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df = train_df.drop('Cabin', axis=1)
test_df = test_df.drop('Cabin', axis=1)


# In[ ]:


test_df['Age'] = test_df['Age'].fillna(train_df['Age'].median())

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())


# In[ ]:


train_df['Embarked'] = train_df['Embarked'].fillna(train_df.mode(axis=0)['Embarked'].iloc[0])


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df = train_df.drop('Ticket', axis=1)
test_df = test_df.drop('Ticket', axis=1)


# In[ ]:


X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df


# In[ ]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[ ]:


X_train = X_train.drop('Name', axis=1)
X_test = X_test.drop('Name', axis=1)


# In[ ]:


X_train.loc[X_train['Sex'] == 'female', 'Sex'] = 0
X_train.loc[X_train['Sex'] == 'male', 'Sex'] = 1
X_test.loc[X_test['Sex'] == 'female', 'Sex'] = 0
X_test.loc[X_test['Sex'] == 'male', 'Sex'] = 1
X_train.loc[X_train['Embarked'] == 'S', 'Embarked'] = 1
X_train.loc[X_train['Embarked'] == 'C', 'Embarked'] = 2
X_train.loc[X_train['Embarked'] == 'Q', 'Embarked'] = 3
X_test.loc[X_test['Embarked'] == 'S', 'Embarked'] = 1
X_test.loc[X_test['Embarked'] == 'C', 'Embarked'] = 2
X_test.loc[X_test['Embarked'] == 'Q', 'Embarked'] = 3


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


X_train = pd.get_dummies(X_train, columns=['Pclass', 'Embarked'])

X_test = pd.get_dummies(X_test, columns=['Pclass', 'Embarked'])


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
cross_val_score(clf, X_train, Y_train, cv=10)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
cross_val_score(clf, X_train, Y_train, cv=10)


# In[ ]:


from sklearn.svm import LinearSVC
clf = LinearSVC()
cross_val_score(clf, X_train, Y_train, cv=10)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
cross_val_score(clf, X_train, Y_train)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
cross_val_score(clf, X_train, Y_train, cv=10)


# In[ ]:


clf = GradientBoostingClassifier()
clf.fit(X_train, Y_train)
p = clf.predict(X_test)


# In[ ]:


p.shape


# In[ ]:


test_id.shape


# In[ ]:


result = pd.DataFrame(test_id)


# In[ ]:


result['Survived'] = p


# In[ ]:


result.to_csv('submission.csv', index=False)


# In[ ]:


clf = AdaBoostClassifier()
clf.fit(X_train, Y_train)
p = clf.predict(X_test)


# In[ ]:


result['Survived'] = p
result.to_csv('adaboostresult.csv', index=False)


# In[ ]:




