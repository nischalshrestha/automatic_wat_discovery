#!/usr/bin/env python
# coding: utf-8

# In[210]:


# Imports
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import os


# In[211]:


# Load dataset
train_ds = pd.read_csv('../input/train.csv')
test_ds = pd.read_csv('../input/test.csv')


# In[212]:


train_X = train_ds.drop(columns='Survived')
train_Y = train_ds['Survived']
test_X = test_ds

train_X.head()


# In[213]:


def get_missing_val_percents(df) :
    missing_vals_percents = df.isnull().sum() / train_X.shape[0]
    return missing_vals_percents


# In[214]:


# check the percentage of missing values in each column of train and test dataframes
train_missing_percents = get_missing_val_percents(train_X)
test_missing_percents = get_missing_val_percents(test_X)

print('Train:\n', train_missing_percents * 100)
print('Test:\n', test_missing_percents * 100)


# In[215]:


# Since Cabin has 77% missing values in train set and 36% missing values in test set, we'll drop it
train_X = train_X.drop(columns='Cabin')
test_X = test_X.drop(columns='Cabin')

train_X.head()


# In[216]:


# We'll Impute the Age

# For the train set
train_X.loc[train_X['Age'].isnull(), 'Age'] = train_X['Age'].mean()
train_X['Age'].isnull().sum()


# In[217]:


# Now for the test set
test_X.loc[test_X['Age'].isnull(), 'Age'] = test_X['Age'].mean()
test_X['Age'].isnull().sum()


# In[218]:


# set null embarked values to S in train set
train_X.loc[train_X['Embarked'].isnull(), 'Embarked'] = 'S'
train_X['Embarked'].isnull().sum()


# In[219]:


# Impute Fare in test set
test_X.loc[test_X['Fare'].isnull(), 'Fare'] = test_X['Fare'].mean()
test_X['Fare'].isnull().sum()


# In[220]:


# Now check if all missing values are handled
train_X.isnull().sum()


# In[221]:


test_X.isnull().sum()


# In[222]:


# Drop Ticket
train_X = train_X.drop(columns='Ticket')
test_X = test_X.drop(columns='Ticket')

train_X.head()


# In[223]:


# Add SibSp and Parch to create a new column FamilySize
train_X['FamilySize'] = train_X['SibSp'] + train_X['Parch']
test_X['FamilySize'] = test_X['SibSp'] + test_X['Parch']

train_X.head()


# In[224]:


train_X = train_X.drop(columns=['SibSp', 'Parch'])
test_X = test_X.drop(columns=['SibSp', 'Parch'])

train_X.head()


# In[225]:


# convert Sex into dummies
train_X = pd.get_dummies(train_X, columns=['Sex'])
test_X = pd.get_dummies(test_X, columns=['Sex'])

test_X.head()


# In[226]:


# drop Sex_female as the sex can be identified by only Sex_male
train_X = train_X.drop(columns='Sex_female')
test_X = test_X.drop(columns='Sex_female')

train_X.head()


# In[227]:


# drop PassengerId
# No worries, the PassengerId is saved in train_ds
train_X = train_X.drop(columns='PassengerId')
test_X = test_X.drop(columns='PassengerId')

train_X.head()


# In[228]:


# drop Name
train_X = train_X.drop(columns='Name')
test_X = test_X.drop(columns='Name')

train_X.head()


# In[229]:


# convert Embarked to integer values
train_X['Embarked'] = pd.Categorical(train_X['Embarked'], categories=['S', 'C', 'Q']).codes
test_X['Embarked'] = pd.Categorical(test_X['Embarked'], categories=['S', 'C', 'Q']).codes

train_X['Embarked'].head()


# In[230]:


# Scale Age and Fare to range 0,1 
scaler = MinMaxScaler()
train_X['Age'] = scaler.fit_transform(train_X['Age'].values.reshape(-1, 1))
test_X['Age'] = scaler.fit_transform(test_X['Age'].values.reshape(-1,1))

train_X['Fare'] = scaler.fit_transform(train_X['Fare'].values.reshape(-1, 1))
test_X['Fare'] = scaler.fit_transform(test_X['Fare'].values.reshape(-1,1))

test_X.head()


# In[231]:


# All set, now create and train the classifier
clf = SVC(kernel='rbf')
clf.fit(train_X, train_Y)
clf.score(train_X, train_Y)


# In[232]:


# make predictions and save them into the submission file
predictions = clf.predict(test_X)
submission = pd.DataFrame()
submission['PassengerId'] = test_ds['PassengerId']
submission['Survived'] = predictions
submission.head()

submission.to_csv('submission_file.csv', index=False)
print(os.listdir('./'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




