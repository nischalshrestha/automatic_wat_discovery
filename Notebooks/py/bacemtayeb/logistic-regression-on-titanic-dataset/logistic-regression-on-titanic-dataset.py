#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Linear Algebra
import numpy as np
# Loading + Cleaning
import pandas as pd
pd.set_option('max_rows',5)
# Visualization
import seaborn as sns
# ML
from sklearn import linear_model


# In[ ]:


# We start by loading the data
train_d = pd.read_csv('../input/train.csv')
train_d = pd.DataFrame(train_d)
test_d = pd.read_csv('../input/test.csv')
test_d = pd.DataFrame(test_d)


# In[ ]:


train_d.head()
# We will only need Sex , Age, Survived columns
train_d = train_d.drop(columns={'Name','Ticket','Fare','Cabin','Embarked','Parch','SibSp'})


# In[ ]:


train_d.isnull().sum()
# The age columns is missing 177 values let's fill them with the mean
train_d.Age = train_d.Age.fillna(train_d.Age.mean())


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=train_d,palette='winter')


# In[ ]:


train_features = train_d.drop(['Survived','Pclass','PassengerId'],axis=1)


# In[ ]:


train_features['S'] = train_d['Sex'].map({'male':1,'female':0})


# In[ ]:


train_features =  train_features.drop(columns={'Sex'})


# In[ ]:


x_features = train_features.as_matrix()
y_label = train_d['Survived'].as_matrix()


# In[ ]:


# Load the model
lg = linear_model.LogisticRegression(C=1)
lg.fit(x_features,y_label)


# In[ ]:


# Get the features
test_features = test_d.drop(columns={'PassengerId','Pclass','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'})
test_features


# In[ ]:


test_features['Age'] = test_d['Age'].fillna(test_d.Age.mean())
test_features


# In[ ]:


# Map categorical data to numerical
test_features['S'] = test_features.Sex.map({'male':1,'female':0})


# In[ ]:


test_features = test_features.drop(columns={'Sex'})


# In[ ]:


test_features = test_features.as_matrix()


# In[ ]:


pred = lg.predict(test_features)
Survived = pd.DataFrame(data=pred,columns=['Survived'])
Survived.to_csv('test.csv')

