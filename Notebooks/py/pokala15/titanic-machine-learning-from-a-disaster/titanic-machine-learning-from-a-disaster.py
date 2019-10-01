#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#importing the dataset
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
y_test=pd.read_csv("../input/gender_submission.csv")


# In[ ]:


#top five elements of traning dataset
train.head()


# In[ ]:


#droping the columns which are not required for the prediction
train=train.drop(['PassengerId','Name','Ticket'],axis=1)
train.head()


# In[ ]:


#one hot encoding for coloumn Sex
d_t=pd.get_dummies(train['Sex'])
d_t
d_t.drop(['male'],axis=1)
train=train.join(d_t)


# In[ ]:


train=train.drop(['Sex'],axis=1)


# In[ ]:


train.head()


# In[ ]:


#filling the missing values in the specified coloumns
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Cabin'].fillna(train['Cabin'].mode(),inplace=True)


# In[ ]:


train=train.drop(['male'],axis=1)
train.head()


# In[ ]:


#one hot encoding for Embarked coloumn
e_t=pd.get_dummies(train['Embarked'])
e_t=e_t.drop(['S'],axis=1)
e_t
train=train.drop(['Embarked'],axis=1)
train=train.join(e_t)


# In[ ]:


train=train.drop(['Cabin'],axis=1)


# In[ ]:


train.head()


# In[ ]:


#converting data frame to numpy array
x=train.iloc[:,1:9].values
y=train.iloc[:,0:1].values


# In[ ]:


#creating logistic regression model
from sklearn.linear_model import LogisticRegression
le=LogisticRegression()
le.fit(x,y)


# In[ ]:


test.head()


# In[ ]:


#droping the coloumns which are not required
test=test.drop(['Name','PassengerId','Ticket'],axis=1)


# In[ ]:


test.head()


# In[ ]:


test=test.drop(['Cabin'],axis=1)
test.head()


# In[ ]:


#one hot encoding for the specified coloumns
st=pd.get_dummies(test['Sex'])
st=st.drop(['male'],axis=1)
test=test.drop(['Sex'],axis=1)
test=test.join(st)
test.head()


# In[ ]:


et=pd.get_dummies(test['Embarked'])
et=et.drop(['S'],axis=1)
test=test.drop(['Embarked'],axis=1)
test=test.join(et)
test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


#filling the missing values
test['Age'].fillna(test['Age'].mean(),inplace=True)


# In[ ]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# In[ ]:


test.isnull().sum()


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


#conveting test data frame to numpy array
x_test=test.iloc[:,:].values


# In[ ]:


#converting datatype float to int
test['Age']=test['Age'].astype(int)


# In[ ]:


test['Fare']=test['Fare'].astype(int)


# In[ ]:


#predicting output for the given test set
y_pred=le.predict(x_test)


# In[ ]:


y_t=y_test.iloc[:,1:2].values


# In[ ]:


#checking for accuracy
from sklearn.metrics import confusion_matrix
ob=confusion_matrix(y_t,y_pred)
ob


# In[ ]:


#creating model with SVC
from sklearn.svm import SVC 


# In[ ]:


s=SVC(kernel='linear',random_state=0)
s.fit(x,y)


# In[ ]:


#predicting the output using the SVC model
y_p=s.predict(x_test)


# In[ ]:


#accuracy for the model obtained from SVC
from sklearn.metrics import confusion_matrix
ob=confusion_matrix(y_t,y_p)
ob

