#!/usr/bin/env python
# coding: utf-8

# In[227]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[228]:


train = pd.read_csv('../input/train.csv', index_col='PassengerId')
test = pd.read_csv('../input/test.csv', index_col='PassengerId')


# In[229]:


train.head()


# In[230]:


test.head()


# In[231]:


dataset = train.append(test)


# In[232]:


train['Embarked'].unique()


# In[233]:


sex_map = {'female': 1, 'male': 0}
embark_map = {None: 0, 'S': 1, 'C': 2, 'Q': 3}
title_map = {
    'Col': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Don': 0,
    'Lady': 0,
    'Countess': 0,
    'Mme': 0,
    'Mr': 1,
    'Sir': 1,
    'Miss': 2,
    'Ms': 2,
    'Mrs': 3,
    'Master': 4,
}

for t in [train, test]:
    t['Age'] = t['Age'].fillna(dataset['Age'].median())
    t['Sex'] = t['Sex'].map(sex_map)
    t['Embarked'] = t['Embarked'].map(embark_map)
    t['title'] = t['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    t['title'] = t['title'].map(title_map).fillna(0)
    t['Fare'] = t['Fare'].fillna(dataset['Fare'].median())


# In[234]:


train.head()


# In[235]:


s = StandardScaler()
for c in ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'title', 'Embarked'):
    train[c] = s.fit_transform(train[c].values.reshape(-1, 1))
    test[c] = s.fit_transform(test[c].values.reshape(-1, 1))


# In[236]:


train.head()


# In[237]:


print(train.isnull().sum())
print(test.isnull().sum())


# In[239]:


X_train = train.drop(['Name', 'Survived', 'Ticket', 'Cabin'], axis=1)
y_train = train['Survived']

X_test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# In[240]:


X_train.head()


# In[241]:


X_test.head()


# In[253]:


from sklearn.linear_model import LogisticRegressionCV
m = LogisticRegressionCV()
m.fit(X_train, y_train)
print([s.mean() for s in m.scores_[1]])
y_test = m.predict(X_test)

submission = X_test.copy()
submission['Survived'] = y_test
submission.head()

submission.to_csv('submission.csv', columns=['Survived'])


# In[254]:


from sklearn.svm import SVC
m = SVC()
m.fit(X_train, y_train)
print(m.score(X_train, y_train))
y_test = m.predict(X_test)

submission = X_test.copy()
submission['Survived'] = y_test
submission.head()

submission.to_csv('submission.csv', columns=['Survived'])


# In[ ]:





# In[ ]:




