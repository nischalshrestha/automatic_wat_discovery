#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train=pd.read_csv("../input/train.csv")


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


train.Cabin.fillna(train.Cabin.value_counts().idxmax(),inplace=True)


# In[ ]:


train.Age.fillna(train.Age.value_counts().idxmax(),inplace=True)


# In[ ]:


train.Embarked.fillna(train.Embarked.value_counts().idxmax(),inplace=True)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.Cabin.fillna(test.Cabin.value_counts().idxmax(),inplace=True)


# In[ ]:


test.Age.fillna(test.Age.value_counts().idxmax(),inplace=True)


# In[ ]:


test.Fare.fillna(test.Fare.value_counts().idxmax(),inplace=True)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le=LabelEncoder()


# In[ ]:


intcols=train.select_dtypes(include=['int64'])
intcols1=intcols.apply(le.fit_transform)
floatcols=train.select_dtypes(include=['float64'])


# In[ ]:


objectcols=train.select_dtypes(include=['object'])
objectcols1=objectcols.apply(le.fit_transform)


# In[ ]:


train1=pd.concat([objectcols1,intcols1,floatcols],axis=1)


# In[ ]:


train1.dtypes


# In[ ]:


intcols=test.select_dtypes(include=['int64'])
objectcols=test.select_dtypes(include=['object'])
floatcols=test.select_dtypes(include=['float64'])
intcols1=intcols.apply(le.fit_transform)
objectcols1=objectcols.apply(le.fit_transform)


# In[ ]:


test1=pd.concat([intcols1,floatcols,objectcols1],axis=1)


# In[ ]:


test1.dtypes


# In[ ]:


y=train1.Survived
x=train1.drop(['Survived','PassengerId'],axis=1)
xtest=test1.drop('PassengerId',axis=1)


# In[ ]:


x.shape


# In[ ]:


xtest.shape


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbc=GradientBoostingClassifier(n_estimators=2000)


# In[ ]:


gbcmodel=gbc.fit(x,y)


# In[ ]:


gbcmodel.score(x,y)


# In[ ]:


predict=gbcmodel.predict(xtest)


# In[ ]:


predict


# In[ ]:


submission = pd.DataFrame(data={'PassengerId': (np.arange(len(predict)) + 1), 'Survived': predict})
submission.to_csv('gender_submission.csv', index=False)
submission.tail()  


# In[ ]:




