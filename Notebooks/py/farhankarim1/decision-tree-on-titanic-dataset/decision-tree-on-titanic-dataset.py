#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()

# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


data.info()


# In[ ]:


# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())


# In[ ]:


#encode sex column
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[ ]:


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


#split back to train-test
data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[ ]:


X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('predictions.csv', index=False)


# In[ ]:




