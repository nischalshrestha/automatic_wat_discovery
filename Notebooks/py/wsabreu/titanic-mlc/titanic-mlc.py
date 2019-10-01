#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#reading test and train sets
train = pd.read_csv ('../input/titanic-mlc-csv/test.csv')
test = pd.read_csv ('../input/titanic-mlc-csv/test.csv')


# In[ ]:


#Train Set five first rows
train.head()


# In[ ]:


#drop from set name, ticket and cabin
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[ ]:


#creating new dataframe with one-hot encoding
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)


# In[ ]:


new_data_test.head()


# In[ ]:


new_data_train.head()


# In[ ]:


#Amount of Null values at Train set
new_data_train.isnull().sum().sort_values(ascending=False).head(10)


# In[ ]:


#fullfil null values
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)


# In[ ]:


#Amount of Null values at Test set
new_data_test.isnull().sum().sort_values(ascending=False).head(10)


# In[ ]:


#fullfil null values at Test Set
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)


# In[ ]:


#Sorting features and target to model design
x = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']


# In[ ]:


#create model
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x, y)


# In[ ]:


#checking train set score
tree.score(x, y)


# In[ ]:


submission = pd.DataFrame()
submission['PassengerId'] = new_data_test ['PassengerId']
submission['Survived'] = tree.predict(new_data_test)


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




