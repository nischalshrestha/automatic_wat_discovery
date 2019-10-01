#!/usr/bin/env python
# coding: utf-8

# # Titanic
# ***
# ### Using Random Forest Classifier
# ***

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Import modules

# In[ ]:


import matplotlib.pyplot as plt #for plotting
get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


# In[ ]:


def read_data(path):
    '''read data from path and return a pandas dataframe'''
    df = pd.read_csv(path)
    return df

train_path = '../input/train.csv'
test_path = '../input/test.csv'

train = read_data(train_path)
test = read_data(test_path)


# In[ ]:


train.head()


# ### Data Exploration

# In[ ]:


train.describe().T


# Age column has some missing value. About 75% of survivived are female for the descriptive statistics. The average fare is ``$32 `` and about  ``75%`` of the passenger pay around ``$31`` with a maximun fare of ``$512.32.``

# In[ ]:


test.describe().T


# Age and Fare also has a missing values in the test datasets

# In[ ]:


train.info()


# * Numeric columns : PassengerId, Survived,Pclass,Age,SibSp,Partch,Fare
# * Categorical columns : Name,Sex,Ticket,Cabin,Embarked

# In[ ]:


test.info()


# In[ ]:


train.columns


# In[ ]:


def map_sex(df):
    '''map female to 1 and male to 0 and return the dataframe'''
    df['Sex'] = df['Sex'].map({'female': 1, 'male' : 0}).astype(int)
    return df
train = map_sex(train)
test = map_sex(test)


# In[ ]:


train.head()


# In[ ]:


print('The most occurence values in train : {}'.format(train['Embarked'].mode()))
print('The most occurrence values in test : {}'.format(test['Embarked'].mode()))
print('Train : {}'.format(train['Embarked'].unique()))
print('Test : {}'.format(test['Embarked'].unique()))


# We use ``S`` to fill the missing values in the Embarked colums

# In[ ]:


mos_freq = train['Embarked'].mode()
def fill_missing_emberked(df):
    '''fill the values values in embarked column with the mode'''
    df['Embarked'] = df['Embarked'].fillna(mos_freq)
    return df

train = fill_missing_emberked(train)
test = fill_missing_emberked(test)
    


# In[ ]:


#map embarked to numeric representation
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train['Age'].hist(bins=50) #check the Age duistribution


# In[ ]:


round(train['Age'].mean())


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


from pandas.plotting import scatter_matrix


# In[ ]:


scatter_matrix(train, alpha=0.2, figsize=(6, 6), diagonal='kde')


# In[ ]:



train['Age'].fillna(round(train['Age'].mean()), inplace=True)
test['Age'].fillna(round(test['Age'].mean()), inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train['Embarked'].isnull().sum()


# In[ ]:


train['Embarked'].fillna(0, inplace=True)
test['Embarked'].fillna(0, inplace=True)


# In[ ]:


feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked']


# In[ ]:


train_df = train[feature_columns]


# In[ ]:


train_df.head()


# In[ ]:


scatter_matrix(train_df,alpha=0.2,diagonal='kde')


# In[ ]:


test_df = test[feature_columns]


# In[ ]:


test_df.head()


# In[ ]:


X_train = train[feature_columns]
y_train = train['Survived']
X_test = test[feature_columns]


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# Split train datasets

# In[ ]:


X_train1,X_val, y_train1,y_val = train_test_split(X_train,y_train, random_state=42)


# In[ ]:


print(X_train1.shape, y_train1.shape)
print(X_val.shape, y_val.shape)


# ### Model

# In[ ]:


#baseline model to check the oob_score
rdf = RandomForestClassifier(oob_score=True, random_state=42,n_estimators=20,max_features = 2,
                             n_jobs =-1,max_depth=5)


# In[ ]:


#we use all the test data set to check the oob score
rdf.fit(X_train,y_train)
print(rdf.oob_score_)


# In[ ]:


cross_val = cross_val_score(rdf,X_train,y_train, scoring='accuracy')
cross_val.mean()


# In[ ]:


from sklearn.metrics import accuracy_score

depth_option = [1,2,3,4,5,6,7,8,9,10]
train_score =[]
val_score = []
for depth in depth_option:
    model = RandomForestClassifier(n_estimators=100,max_depth=depth)
    model.fit(X_train1,y_train1)
    y_train_pred = model.predict(X_train1)
    y_val_pred = model.predict(X_val)
    train_score.append(accuracy_score(y_train1,y_train_pred))
    val_score.append(accuracy_score(y_val,y_val_pred))
    print('max_depth : {}'.format(depth))
    print('train score : {}'.format(accuracy_score(y_train1,y_train_pred)))
    print('test score : {}'.format(accuracy_score(y_val,y_val_pred)))
    
plt.plot(depth_option,train_score, label='train score')
plt.plot(depth_option,val_score, label='val score')
plt.xlabel('max_depth option')
plt.ylabel('accuracy score')
plt.legend()
plt.show()




# In[ ]:


max_features_option = [1,2,3,4,5,6,7]
train_score =[]
val_score = []
for features in max_features_option:
    model = RandomForestClassifier(n_estimators=100,max_depth=3, max_features=features)
    model.fit(X_train1,y_train1)
    y_train_pred = model.predict(X_train1)
    y_val_pred = model.predict(X_val)
    train_score.append(accuracy_score(y_train1,y_train_pred))
    val_score.append(accuracy_score(y_val,y_val_pred))
    print('number of features per tree : {}'.format(features))
    print('train score : {}'.format(accuracy_score(y_train1,y_train_pred)))
    print('test score : {}'.format(accuracy_score(y_val,y_val_pred)))
    
plt.plot(max_features_option,train_score, label='train score')
plt.plot(max_features_option,val_score, label='val score')
plt.xlabel('max_features option')
plt.ylabel('accuracy score')
plt.legend()
plt.show()



# In[ ]:


model = RandomForestClassifier(n_estimators=200,max_depth=9, max_features= 2)

model.fit(X_train1,y_train1)
y_pred = model.predict(X_train1)
y_val_pred = model.predict(X_val)
print('Train Score : {}'.format(accuracy_score(y_train1,y_pred)))
print('validation Score : {}'.format(accuracy_score(y_val,y_val_pred)))


# In[ ]:


cross_val_score = cross_val_score(model,X_train,y_train, scoring='accuracy')
cross_val_score.mean()


# In[ ]:




