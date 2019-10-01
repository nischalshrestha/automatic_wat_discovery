#!/usr/bin/env python
# coding: utf-8

#  # Sup, this is my first entry to the Kaggle competitions!
# 
# I opted for setting this kernel public, as I'm trying to finally practice all the theory I have learned until now with some hands-on. I know there are lack of explanations throughout the analysis but I think those analysis I've made are pretty basics and intuitive, since there is many other well explained works around this dataset, so I focused on coding and submiting my first results. Please, feel free to give me some feedback about my work, I'll be very happy to learn more, or simply let me know If I'm doing it right too!
# 
# Now, I'll try to enhance this result forking this notebook! :)

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import make_scorer, accuracy_score

get_ipython().magic(u'matplotlib inline')

df_train = pd.read_csv("../input/train.csv")
df_train.head()


# In[2]:


df_train.info()


# In[3]:


plt.figure(figsize=(10,7))
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False)


# In[4]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
df_missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df_missing.head()


# In[5]:


plt.figure(figsize=(10,7))
sns.boxplot('Pclass', 'Age', data=df_train)


# In[6]:


def train_imputer(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        age = df_train[df_train['Pclass']==pclass]['Age'].median()
        
    return age

df_train['Age'] = df_train[['Age', 'Pclass']].apply(train_imputer, axis=1)


# In[7]:


plt.figure(figsize=(14,7))
sns.set_style('whitegrid')
sns.kdeplot(df_train[df_train['Survived']==0]['Age'], shade=True, legend=False)
sns.kdeplot(df_train[df_train['Survived']==1]['Age'], shade=True, legend=False)


# In[8]:


plt.figure(figsize=(10,7))
sns.barplot('Pclass', 'Survived', data=df_train)


# In[9]:


df_train.drop('Cabin', axis=1, inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].value_counts().index[0], inplace=True)


# In[10]:


df_train.isnull().any()


# In[11]:


plt.figure(figsize=(10,7))
sns.barplot('Survived', 'Fare', data=df_train)


# In[12]:


plt.figure(figsize=(10,7))
sns.violinplot('Sex', 'Age', data=df_train, hue='Survived', split=True)


# In[13]:


plt.figure(figsize=(10,7))
sns.barplot('Sex', 'Survived', data=df_train)


# In[14]:


df_train.drop('PassengerId Name Ticket'.split(), axis=1, inplace=True)
df_train['Sex'] = df_train['Sex'].map({'male': 1, 'female': 0})


# In[15]:


embarked = pd.get_dummies(df_train['Embarked'], prefix='Embarked', drop_first=True)
df_train = pd.concat([df_train, embarked], axis=1)
df_train.drop('Embarked', axis=1, inplace=True)
df_train.head()


# In[16]:


# Checking results for cross validation.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

logistic_regression_pipeline = make_pipeline(LogisticRegression())
gradient_boosting_pipeline = make_pipeline(GradientBoostingClassifier())

scores = cross_val_score(logistic_regression_pipeline, df_train.drop('Survived', axis=1), df_train['Survived'], scoring='accuracy', cv=3)
print('Logistic Regression Score: %2f' % (scores.mean()))

scores = cross_val_score(gradient_boosting_pipeline, df_train.drop('Survived', axis=1), df_train['Survived'], scoring='accuracy', cv=3)
print('Gradient Boosting Score: %2f' % (scores.mean()))


# In[18]:


# Checking results for train test split.

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

train_X = df_train.drop('Survived', axis=1)
train_y = df_train['Survived']

train_X, test_X, train_y, test_y = train_test_split(train_X.as_matrix(), train_y.as_matrix(), test_size=0.25)

gradient_boosting_pipeline.fit(train_X, train_y)
predictions = gradient_boosting_pipeline.predict(test_X)
print(classification_report(test_y, predictions))


# # From here, I'll do pretty the same for test data

# In[19]:


df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[20]:


total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
df_missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df_missing.head()


# In[21]:


def test_fare_imputer(cols):
    fare = cols[0]
    pclass = cols[1]
    
    if pd.isnull(fare):
        fare = df_test[df_test['Pclass']==pclass]['Fare'].median()
        
    return fare

df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(test_fare_imputer, axis=1)


# In[22]:


def test_age_imputer(cols):
    age = cols[0]
    pclass = cols[1]
    
    if pd.isnull(age):
        age = df_test[df_test['Pclass']==pclass]['Age'].median()
        
    return age

df_test['Age'] = df_test[['Age', 'Pclass']].apply(test_age_imputer, axis=1)


# In[23]:


df_test.drop('Cabin', axis=1, inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].value_counts().index[0], inplace=True)
df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 0})

embarked = pd.get_dummies(df_test['Embarked'], prefix='Embarked', drop_first=True)
df_test = pd.concat([df_test, embarked], axis=1)

df_test.drop('Name Ticket Embarked'.split(), axis=1, inplace=True)
df_test.head()


# # Predicting my final results on the test data with the already fitted model

# In[25]:


predictions = gradient_boosting_pipeline.predict(df_test.drop('PassengerId', axis=1))

my_submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})
my_submission.to_csv('titanic_first_try.csv', index=False)


# In[ ]:




