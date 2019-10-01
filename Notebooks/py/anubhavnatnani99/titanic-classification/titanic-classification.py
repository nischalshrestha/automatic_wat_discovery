#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_y=train['Survived'].values


# In[ ]:


#visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#columns to be removed as they do no affect the performance and are assumed to be random
#passengerID,Ticket,Cabin(As it has many null values)
#cleaning the data
train_x = train.copy()
train_x=train_x.drop(columns=['Survived'])
data_clean = [train_x,test]
print(train.info())
train.sample(10)#pick 10 random samples


# In[ ]:


#DATA cleaning
print(train_x.isnull().sum())
print('/'*12)
print(test.isnull().sum())


# In[ ]:


train_x.describe(include="all")


# In[ ]:


#Age has many missing values both in the test and train set 
for dataset in data_clean:
    #fix the value in that place
    dataset["Age"].fillna(dataset["Age"].median(),inplace=True)
    #we will add mode to the embarked 
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    #we will do similar with fare
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

#train_x.reset_index(drop=True)
#we have to reset our index as passengerid is the index
to_drop = ['PassengerId','Cabin', 'Ticket']
train_x.drop(to_drop, axis=1, inplace = True)
#inplace helps do this change in the original copy 
#by default it is false


# Doing the feature engg and data prediction

# In[ ]:


print(train_x.isnull().sum())
print('-'*10)
print(test.isnull().sum())


# In[ ]:


train_x.head()


# In[ ]:


for dataset in data_clean:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
    dataset['IsAlone']=1
    #dataset['FareBin'] = pd.qcut(dataset['Fare'], 4,duplicates="drop")
    #dataset['AgeBin'] = pd.qcut(dataset['Age'].astype(int),5,duplicates="drop")


# In[ ]:


train_x.head()


# In[ ]:


train_x = train_x.drop(columns=['Name'])


# In[ ]:


train_x.head()


# In[ ]:


#encoding the some other info
def gencov(a):
    if(a=='male'):
        return 1
    else:
        return 0
def embcov(a):
    if(a=='C'):
        return 1
    elif(a=='S'):
        return 2
    else:
        return 3

    
"""
for dataset in data_clean:
    dataset['Sex'] = dataset['Sex'].apply(lambda x:1 if x=="male" else 0)
    dataset['Embarked'] = dataset['Embarked'].apply(lambda x:1 if x=="C" else(2 if x=="S" else 0))
""" 


# In[ ]:


pid = test['PassengerId'].values
test=test.drop(columns=['Name','Ticket','Cabin','PassengerId'])
test.head()


# In[ ]:


#applying the encoding
test.info()


# In[ ]:


train_x['Sex'] = train_x['Sex'].map(gencov)
train_x['Embarked'] = train_x['Embarked'].map(embcov)


# In[ ]:


test['Sex'] = test['Sex'].map(gencov)
test['Embarked'] = test['Embarked'].map(embcov)


# In[ ]:


train_x.head()


# In[ ]:


test.head()


# Doing prediction by modeling 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm


# In[ ]:


lr = LogisticRegression()
lr.fit(train_x,train_y)
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(train_x,train_y)
scores = cross_val_score(lr, train_x, train_y, cv=5)
scores2 = cross_val_score(clf,train_x,train_y,cv=5)


# In[ ]:


print(scores)
print(scores2)


# In[ ]:


"""
clf = svm.SVC(kernel='poly', C=1)
scores = cross_val_score(clf, train_x, train_y, cv=5)
print(scores)
"""


# In[ ]:


test_Y =lr.predict(test)
test = pd.DataFrame( { 'PassengerId': pid , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )


# In[ ]:




