#!/usr/bin/env python
# coding: utf-8

# Hi All, 
# 
# This is my solution of Titanic Survival prediction problem 

# In[ ]:


#Importing Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization 
import seaborn as sns # Awesome Visualization library 
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Checking Training Dataset
train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


# Checking Test Dataset
test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


# Some Data Visulalization for finding null values in Training Dataset. We see null vaules in Age & Cabin
sns.heatmap(train.isnull())


# In[ ]:


# Looking for null values in Test Dataset as well 
sns.heatmap(test.isnull())
# Null values found in Age & Cabin Column


# In[ ]:


# Continuing with explorartory data analysis 
sns.countplot(x='Survived',data=train, hue='Sex')


# Above graph indicates that women have high chances of survival from the disaster

# In[ ]:


# Survival chances with PClass
sns.countplot(x='Pclass',data=train, hue='Sex')


# In[ ]:


# Above figure indicates maximum casualties occur in the 3rd Passenger class 


# In[ ]:


sns.countplot(x='Age',data=train, hue='Sex')


# In[ ]:


sum(train['Age'].isnull()) # For 177 ppl age data is missing 
sum(train['Cabin'].isnull())#687 cabin data is missing 
sum(train['Embarked'].isnull()) # 2 passenger data is missing


# In[ ]:


# Cleaing up the data before running prediction algorithms


# Average age class wise

# In[ ]:


# Average age of 1st Class passengers -38 years
#train[train['Pclass']==1]['Age'].mean()

# Average age of 1st Class passengers -29 years
#train[train['Pclass']==2]['Age'].mean()

# Average age of 1st Class passengers -25 years
#train[train['Pclass']==3]['Age'].mean()


# In[ ]:


# Function for filling missing age data 
def fill_age(cols):
    age = cols[0]
    pc = cols[1]
    if pd.isnull(age):
        if pc==1:
            return 38
        elif pc==2:
            return 29
        else:
            return 25
    else:
        return age
    


# In[ ]:


# Filling up missing age data 
train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


# drop Cabin column as there are lot of missing values
train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.dropna(inplace=True)
#Now drop all missing values - 2 Embarked Values 


# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


#Create dummy variables for Pclass , Sex & Embarked columns
sex =pd.get_dummies(train['Sex'],drop_first=True)
embarked = pd.get_dummies(train['Embarked'],drop_first=True)
pclass = pd.get_dummies(train['Pclass'],drop_first=True)



# In[ ]:


sex.head()


# In[ ]:


embarked.head()


# In[ ]:


pclass.head()


# In[ ]:


train = pd.concat([train,sex,embarked,pclass],axis=1)


# In[ ]:


train.head()


# In[ ]:


# Defining the Trainging Set for Algorithm testing 


# In[ ]:


X_train=train.drop(['Name','Sex','Ticket','Embarked','PassengerId','Pclass','Survived'],axis=1)


# In[ ]:


X_train.head()


# In[ ]:


# Defining the y_train
y_train=train['Survived']


# In[ ]:


y_train.head()


# In[ ]:


# Using Logistic Regression Approach on our Dataset
#from sklearn.linear_model import LogisticRegression


# In[ ]:


#logmodel = LogisticRegression()
#logmodel.fit(X_train,y_train)


# In[ ]:


#Defining & Cleaining X_test Data


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


# Filling up missing age data 
test['Age'] = test[['Age','Pclass']].apply(fill_age,axis=1)


# In[ ]:


# drop Cabin column as there are lot of missing values
test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(test.isnull())


# In[ ]:


#Create dummy variables for Pclass , Sex & Embarked columns
sex =pd.get_dummies(test['Sex'],drop_first=True)
embarked = pd.get_dummies(test['Embarked'],drop_first=True)
pclass = pd.get_dummies(test['Pclass'],drop_first=True)


# In[ ]:


test = pd.concat([test,sex,embarked,pclass],axis=1)


# In[ ]:


test.head()


# In[ ]:


X_test=test.drop(['Name','Sex','Ticket','Embarked','PassengerId','Pclass'],axis=1)


# In[ ]:


X_test.head() 


# In[ ]:


# Predicting Survival from our X_test Dataset


# In[ ]:


X_test.isnull().any()


# In[ ]:


X_test['Fare'].fillna(value=35,inplace=True)


# In[ ]:


X_test.isnull().any()


# In[ ]:


# Predicting Survival from our X_test Dataset using Logistic Regression


# In[ ]:


#y_predict = logmodel.predict(X_test)


# In[ ]:


# Using Random Forest Approach to predict results 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
y_predict = rfc.predict(X_test)



# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('randonforest_titanic.csv', index=False)

