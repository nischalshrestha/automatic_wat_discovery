#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ### Loading Datasets

# In[7]:


import os
# Checking that you are in the same directory as the required files...
os.path.realpath('.')
# Importing datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[8]:


train.head()


# In[9]:


test.head()


# ### See if you can kaggle it

# In[10]:


# Finding importance of data...
sns.boxplot(x='Survived', y='Pclass', data=train, palette='hls')


# In[11]:


sns.boxplot(x='Survived', y='SibSp', data=train, palette='hls')


# In[12]:


sns.boxplot(x='Survived', y='Fare', data=train, palette='hls')


# In[13]:


sns.boxplot(x='Survived', y='Age', data=train, palette='hls')


# In[14]:


train_data = train.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'], 1)
train_data['Sex'].replace(['female','male'],[0,1],inplace=True)
train_data.head()


# In[15]:


# From data-mania.com to imputate the missing values for age
def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[18]:


train_data['Age'] = train_data[['Age', 'Pclass']].apply(age_approx, axis=1)
test_data = test.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'], 1)
test_data['Age'] = test_data[['Age', 'Pclass']].apply(age_approx, axis=1)
test_data['Fare']  = test_data[['Fare', 'Pclass']].apply(age_approx, axis=1)


# In[19]:


X = train_data.loc[:,('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare')].values
y = train_data.loc[:,'Survived'].values
LogReg = LogisticRegression()
LogReg.fit(X, y)
train_data.head()


# In[20]:


test_data['Sex'].replace(['female','male'],[0,1],inplace=True)


# In[21]:


y_pred = LogReg.predict(test_data)


# In[22]:


df = pd.DataFrame({ 'PassengerId': test['PassengerId'].values,
                            'Survived': y_pred})


# In[23]:


df.to_csv("submission.csv", index=False)


# In[ ]:




