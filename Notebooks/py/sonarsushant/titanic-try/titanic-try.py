#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train=pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False)


# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),cbar=False,yticklabels=False)


# In[ ]:


sns.countplot(x='Survived',data=train,hue='Sex')


# In[ ]:


sns.countplot(x='Survived',data=train,hue='Pclass')


# In[ ]:


sns.distplot(train['Age'],kde=False,bins=30)


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


sns.countplot(x='Embarked',data=train)


# In[ ]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train=pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


X_train=train.drop('Survived',axis=1)
y_train=train['Survived']


# In[ ]:


X_train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:





# In[ ]:


test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data['Age']=test_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


test_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


test_data.dropna(inplace=True)


# In[ ]:


sex=pd.get_dummies(test_data['Sex'],drop_first=True)
embark=pd.get_dummies(test_data['Embarked'],drop_first=True)


# In[ ]:


test_data=pd.concat([test_data,sex,embark],axis=1)


# In[ ]:


test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


X_test=test_data.drop('PassengerId',axis=1)


# In[ ]:


predictions=logmodel.predict(X_test)


# In[ ]:


test_data['Survived']=predictions


# In[ ]:




