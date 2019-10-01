#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
very simple implementation with XGBoost

"""


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Load data
train = pd.read_csv("../input/train.csv",header=0)
test = pd.read_csv("../input/test.csv",header=0)


# In[35]:


#Fill the missing values 
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(train['Fare'].median())


# In[36]:


train['Embarked'] = train['Embarked'].fillna('S')# S is the most common symbol
test['Embarked'] = test['Embarked'].fillna('S')


# In[7]:


#Calc number of family
train['family'] = train['SibSp'] + train['Parch']
test['family'] = test['SibSp'] + test['Parch']


# In[16]:





# In[ ]:





# In[26]:


#Convert categorical variable into dummy/indicator variables
def add_dummy(df):
    df['Pclass'] = df['Pclass'].astype(np.str)
    temp = pd.get_dummies(df[['Sex','Embarked','Pclass']], drop_first = False)
    temp['PassengerId'] = df['PassengerId']
    return pd.merge(df, temp)
train = add_dummy(train)
test = add_dummy(test)


# In[28]:


#Drop unnecessary feature
def get_feature_mat(df):
    temp = df.drop(columns=['PassengerId','Name','Sex','SibSp','Parch','Ticket','Embarked','Age','Cabin','Pclass'])
    try:
        temp = temp.drop(columns=['Survived'])
    except:
        pass
    print (temp)
    return temp.as_matrix()
x_train = get_feature_mat(train)
x_test = get_feature_mat(test)


# In[29]:


y_train = train['Survived'].as_matrix()


# In[30]:


#Fit and Predict
from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
xgb.fit(x_train, y_train)
y_test_xgb = xgb.predict(x_test)


# In[31]:


#Generate timestamp
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
ts = datetime.now(JST).strftime('%y%m%d%H%M')


# In[32]:


#Save
test["Survived"] = y_test_xgb.astype(np.int)
test[["PassengerId","Survived"]].to_csv(('submit_'+ts+'.csv'),index=False)


# In[34]:


test["Survived"].head()

