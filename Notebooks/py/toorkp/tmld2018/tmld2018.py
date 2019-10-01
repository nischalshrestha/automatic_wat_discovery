#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[3]:


x = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/gender_submission.csv')
x.head()


# In[4]:


# looking for null values in training set
x.isnull().sum()


# In[5]:


avgAge = x.Age.mean()
x.Age = x.Age.fillna(value = avgAge)
x.Age.isnull().sum()
x.Embarked.isnull().sum()
x.dropna(inplace = True)
x.isnull().sum()
# dropping the columns
#x_drop = x.drop(['PassengerId','Name','Ticket','Cabin'],1)
#x_drop.head()


# In[6]:


x = pd.get_dummies(data = x, columns = ['Sex','Pclass','Embarked'])
x.head()
#x = x.drop(['PassengerId','Name','Ticket','Cabin'])
#x.head()


# In[7]:


x = x.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)#ropna(inplace = True)
x.head()


# In[8]:


X = x.iloc[:,1:].values
y = x.iloc[:,0].values


# In[ ]:


'''
rfc = RandomForestClassifier
model = rfc(n_estimators = 100)
model.fit(X,y)
'''


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2, random_state = 22)


# In[10]:


#lr = LogisticRegression()
rfc = RandomForestClassifier
model = rfc(n_estimators = 100)
model_rfc = model.fit(X_train,y_train)


# In[12]:


#y_pred = lr.predict(X_val)
y_pred = model_rfc.predict(X_val)


# In[13]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_val, y_pred)
confusion_matrix

#print(classification_report(y_val, y_pred))


# In[14]:


from sklearn import metrics 
from sklearn.metrics import classification_report
print(classification_report(y_val,y_pred))


# In[15]:


Train_Accuracy = accuracy_score(y_val, model_rfc.predict(X_val))
Train_Accuracy


# In[16]:


test.head()
test.isnull().sum()


# In[17]:


avgAge_test = test.Age.mean()
test.Age = test.Age.fillna(value = avgAge_test)
avgFare = test.Fare.mean()
test.Fare = test.Fare.fillna(value = avgFare)
test.Age.isnull().sum()
test.Fare.isnull().sum()


# In[18]:


test_drop = test.drop(['PassengerId','Name','Ticket','Cabin'],axis =1)
test_drop.head()


# In[19]:


test_dummy = pd.get_dummies(data = test_drop, columns = ['Sex','Pclass','Embarked'])
test_dummy.head()


# In[20]:


y_test = model_rfc.predict(test_dummy)


# In[21]:


test_Accuracy = accuracy_score(y_test, model_rfc.predict(test_dummy))
test_Accuracy


# In[22]:


#sub = pd.to_csv(test['PassengerId'],y_test)
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()


# In[27]:


final = pd.DataFrame()
#final = pd.DataFrame(['PassengerId','Survived' == y_test])
final['PassengerId'] = test.PassengerId
final['Survived'] = y_test
final.head()


# In[29]:


final.to_csv('sub.csv', index = False)
sub = pd.read_csv('sub.csv')
sub


# In[ ]:




