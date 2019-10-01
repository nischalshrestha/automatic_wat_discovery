#!/usr/bin/env python
# coding: utf-8

# I will use Scikit-Learn's Random Forest.

# In[ ]:


import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


features = ['Sex', 'Pclass', 'Fare', 'Age', 'Embarked', 'SibSp', 'Parch']
data = train[features]
target = train['Survived']
X = test[features]


# In[ ]:


data.info()
X.info()


# In[ ]:


data['Embarked'].fillna('S', inplace=True)
X['Embarked'].fillna('S', inplace=True)


# In[ ]:


data['Age'].fillna(data['Age'].mean(), inplace=True)
X['Age'].fillna(X['Age'].mean(), inplace=True)


# In[ ]:


data['Fare'].fillna(data['Fare'].mean(), inplace=True)
X['Fare'].fillna(X['Fare'].mean(), inplace=True)


# In[ ]:


data.info()
X.info()


# In[ ]:


dict_vec = DictVectorizer(sparse = False)
data = dict_vec.fit_transform(data.to_dict(orient = 'record'))
dict_vec.feature_names_
X = dict_vec.fit_transform(X.to_dict(orient = 'record'))


# In[ ]:


model = RandomForestClassifier()
model.fit(data, target)
y = model.predict(X)


# In[ ]:


rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y})
rfc_submission.to_csv('rfc_submission.csv', index = False)

