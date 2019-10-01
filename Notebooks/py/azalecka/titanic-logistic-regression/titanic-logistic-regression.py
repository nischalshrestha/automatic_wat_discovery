#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# ### Loading dataset

# In[ ]:


# loading data
data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


# how many observations and columns
data.shape


# ### Data review

# In[ ]:


print('Data shape: ', data.shape)


# In[ ]:


print('Data type: ', data.dtypes)


# In[ ]:


data.describe()


# ### Missing data

# In[ ]:


data.isnull().sum()


# In[ ]:


data.isnull().mean()


# In[ ]:


# Sex, creating a new column male
data['Male'] = ((data['Sex'] == 'male') + 0)


# In[ ]:


# Age (null values replaced with mean)
age_mean = data['Age'].mean()
data['Age'].replace(np.nan, age_mean, inplace=True)


# In[ ]:


# Embarked (B28 has null, but B20 and B22 are in S, so we assume also B28 is S)
data['Embarked'].replace(np.nan, 'S', inplace=True)
data_embarked = pd.get_dummies(data['Embarked'])
data_embarked.columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
data = data.join(data_embarked)


# In[ ]:


del_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
for i in del_columns:
    del data[i]


# In[ ]:


y = data['Survived']
X = data.copy()
del X['Survived']


# In[ ]:


# scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[ ]:


# split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=123,
                                                    stratify=y)


# In[ ]:


# logistic regression
penalty    = ['l1','l2']
C_range    = 2. ** np.arange(-10, 0, step=1)
parameters = [{'C': C_range, 'penalty': penalty}]

grid = GridSearchCV(LogisticRegression(), parameters, cv=5)
grid.fit(X_train, y_train)

bestC = grid.best_params_['C']
bestP = grid.best_params_['penalty']
print ("The best parameters are: cost=", bestC , " and penalty=", bestP, "\n")

print("Accuracy: {0:.3f}".format(accuracy_score(grid.predict(X_test), y_test)))


# In[ ]:


# loading original test file
original_test = pd.read_csv('../input/test.csv')
PassengerId = original_test['PassengerId']


# In[ ]:


original_test.shape


# In[ ]:


original_test.columns


# In[ ]:


# preparing test dataset to be predicted
age_mean = original_test['Age'].mean()
original_test['Age'].replace(np.nan, age_mean, inplace=True)
fare_mean = original_test['Fare'].mean()
original_test['Fare'].replace(np.nan, fare_mean, inplace=True)
original_test['Male'] = ((original_test['Sex'] == 'male') + 0)
original_data_embarked = pd.get_dummies(original_test['Embarked'])
original_data_embarked.columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
original_test = original_test.join(original_data_embarked)


# In[ ]:


original_del_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
for i in original_del_columns:
    del original_test[i]


# In[ ]:


original_test.isnull().sum()


# In[ ]:


scaler = StandardScaler()
scaler.fit(original_test)
original_test = scaler.transform(original_test)


# In[ ]:


survived = grid.predict(original_test)
final = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': survived })
final.to_csv('final.csv', index=False)

