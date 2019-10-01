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


# # **Reading Data into Pandas DataFrames**

# In[ ]:


train_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")
train_set.head()


# # **Exploratory Data Analysis**
# 
# Investigating details about the Data
# 

# In[ ]:


print(train_set['Sex'].value_counts())
print(train_set['Embarked'].value_counts())
print(train_set.isnull().values.any())
print(train_set.isnull().sum().sum())
print(train_set.describe())


# # **Selecting required features from training dataset**

# In[ ]:


train_set.drop(['PassengerId','Name','Cabin','Ticket'],axis=1 ,inplace=True)
test_set.drop(['PassengerId','Name','Cabin','Ticket'],axis=1, inplace=True)
print(train_set.head())
print(test_set.head())


# # **Encoding Categorial Data**

# In[ ]:



train_set = pd.get_dummies(data= train_set , dummy_na = True,columns =['Sex' , 'Embarked'])
test_set = pd.get_dummies(data= test_set , dummy_na = True,columns =['Sex' , 'Embarked'])
train_set.drop('Sex_nan',axis=1,inplace=True)
test_set.drop('Sex_nan',axis=1,inplace=True)
print(train_set.head())
print(test_set.head())


# # **Impute missing values by mean on train and test set**

# In[ ]:



train_set.fillna(train_set.mean(),inplace=True)
train_set.isnull().values.any()
test_set.fillna(train_set.mean(),inplace=True)
#Checking for nan values
test_set.isnull().values.any()


# # ** Selecting Features and target**

# In[ ]:



X = train_set.iloc[:,1:13].values
y = train_set.iloc[:,0].values
X_test = test_set.iloc[:,:].values


# # Validating Model for Parameter tuning

# In[ ]:



from sklearn.model_selection import train_test_split
X_train , X_validate , y_train , y_validate = train_test_split(X,y,test_size=0.18,random_state=42)


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_validate = sc_X.transform(X_validate)


# # Now Appling Various ML Models For Classification 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=5,random_state=42,warm_start=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_validate)


# # Metrics of the Model

# In[ ]:


from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(y_validate,y_pred)
print(cnf)
#Out of 161 validation set 130(84+46) predictions are right
acu = (130/161)*100
print(acu)


# # Now Traning Model on Total Dataset with Hyperparameters decided by Validation model
# # Testing  Model on test data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,min_samples_split=30,min_samples_leaf=5,random_state=42,warm_start=True)
clf.fit(X,y)


# # Predicting the survial on test set

# In[ ]:


y_predict = clf.predict(X_test)


# # Creating Submission File and Analyzing it

# In[ ]:


sub = pd.read_csv('../input/gender_submission.csv')
sub['Survived']=y_predict
sub.to_csv('submissions.csv',index=False)
final = pd.read_csv('submissions.csv')
print(final['Survived'].value_counts())

