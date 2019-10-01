#!/usr/bin/env python
# coding: utf-8

# ## Import libraries and get data
# Solution from Kaggle Titanic 
# 
# **Data inofrmation:**
# - Survived: Outcome of survival (0 = No; 1 = Yes)
# - Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# - Name: Name of passenger
# - Sex: Sex of the passenger
# - Age: Age of the passenger (Some entries contain NaN)
# - SibSp: Number of siblings and spouses of the passenger aboard
# - Parch: Number of parents and children of the passenger aboard
# - Ticket: Ticket number of the passenger
# - Fare: Fare paid by the passenger
# - Cabin Cabin number of the passenger (Some entries contain NaN)
# - Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[6]:


#import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Get data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# 
# 
# ## Analyse data
# Visualize the first 5 rows with the head() function.  
# 

# In[7]:


# First 5 rows
train.head()


# 
# 
# ## Removing unused data
# Removing the "Name", "Ticket" and "Cabin" from datasets (training and tests)

# In[8]:


train.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
test.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
train.head()


# ## Generate one-hot (dummies) variables from categorical data
# Using the 'get_dummies' function from Pandas to gerenate the one-hot encoders

# In[9]:


one_hot_train = pd.get_dummies(train)
one_hot_test = pd.get_dummies(test)

# First five rows from train dataset
one_hot_train.head()


# In[10]:


# First five rows from test dataset
one_hot_test.head()


# 
# 
# ## Check and dealing wiht null values

# In[11]:


# Visualize the null values (train)
one_hot_train.isnull().sum().sort_values(ascending=False)


# In[12]:


# Fill the null Age values with the mean of all ages
one_hot_train['Age'].fillna(one_hot_train['Age'].mean(), inplace=True)
one_hot_test['Age'].fillna(one_hot_test['Age'].mean(), inplace=True)
one_hot_train.isnull().sum()


# In[13]:


# Visualize the null values (test)
one_hot_test.isnull().sum().sort_values(ascending=False)


# In[14]:


# Fill the null Fare values with the mean of all Fares
one_hot_test['Fare'].fillna(one_hot_test['Fare'].mean(), inplace=True)
one_hot_test.isnull().sum().sort_values(ascending=False)


# 
# 
# ## Modeling
# We are going to split the data into features and targer, create the model and verify the the score

# In[16]:


# Creating the feature and the target
feature = one_hot_train.drop('Survived', axis=1)
target = one_hot_train['Survived']

# Model creation
rf = RandomForestClassifier(random_state=1, criterion='gini', max_depth=10, n_estimators=50, n_jobs=-1)
rf.fit(feature, target)


# In[17]:


# Verifying score
rf.score(feature, target)


# 
# 
# ## Generate the CSV file with the results
# We will use the Pandas to generate the CSV file with the results to be able to submit to Kaggle

# In[19]:


# Generate a DataFrame with Padas with 'PassengerId' and 'Survived' colunms
submission = pd.DataFrame()
submission['PassengerId'] = one_hot_test['PassengerId']
submission['Survived'] = rf.predict(one_hot_test)

# Generate the CSV file with 'to_csv' from Pandas
submission.to_csv('submission.csv', index=False)


# In[ ]:




