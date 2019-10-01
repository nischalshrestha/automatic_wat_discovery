#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# Read in data sets and explore
# 

# In[19]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[15]:


train.head()


# In[16]:


train.describe()


# In[17]:


train.info()


# # check for missing data

# In[22]:



#increase figure size
sns.set_style('whitegrid')
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[23]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)


# Strategy:
# 
# Toss the cabin columnn,
# 
# create linear regression model to guess age (exclude survived column)
# 
# Toss row with missing embarked data in training data
# 
# set missinf fare data in test to median

# In[24]:


train.drop('Cabin',axis=1,inplace=True)#axis=1 drops a col. default drops rows.
test.drop('Cabin',axis=1,inplace=True)


# In[25]:


#also drop text columns name and ticket
train.drop(['Name','Ticket'],axis=1,inplace=True)#axis=1 drops a col. default drops rows.
test.drop(['Name','Ticket'],axis=1,inplace=True)


# ## Change text columns to dummy variable columns

# In[26]:


#drop first to prevent perfectly correlated columns 

Embarked=pd.get_dummies(train['Embarked'],drop_first=True)
Sex=pd.get_dummies(train['Sex'],drop_first=True)
train=pd.concat([train,Embarked,Sex],axis=1)

Embarked=pd.get_dummies(test['Embarked'],drop_first=True)
Sex=pd.get_dummies(test['Sex'],drop_first=True)
test=pd.concat([test,Embarked,Sex],axis=1)


# In[27]:


train.head()


# In[28]:


train.drop(['Sex','Embarked'],axis=1,inplace=True)
test.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[29]:


cols=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Q', 'S', 'male']
trainAGE=pd.concat([test,train[cols]])


# In[31]:


def fixnan(x):
    if np.isnan(x):
        return 14.45
    return x

trainAGE['Fare']=trainAGE['Fare'].apply(fixnan)


# In[33]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X=trainAGE[['Pclass',  'SibSp', 'Parch', 'Fare', 'Q', 'S', 'male']]
y=trainAGE['Age']
index=[np.invert(trainAGE['Age'].isnull())][0]
lm.fit(X[index],y[index])
pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[34]:


import math
def applypredictions(age,cols):
    if math.isnan(age):
        z=lm.predict(cols.values.reshape(1, -1))
        if z > 0:
            return z
        return 1.0
    return age
pred_cols=['Pclass',  'SibSp', 'Parch', 'Fare', 'Q', 'S', 'male']
train['Age']=train.apply(lambda row: applypredictions(row['Age'], row[pred_cols]), axis=1)
test['Age']=test.apply(lambda row: applypredictions(row['Age'], row[pred_cols]), axis=1)


# In[35]:


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[36]:


#remember to fix the nan fare in the actual test data.....zzzz
def fixnan(x):
    if np.isnan(x):
        return 14.45
    return x

test['Fare']=test['Fare'].apply(fixnan)

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)


# # Time to run the actual logistic regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()


# In[37]:


train.head()


# In[38]:


test.head()


# In[ ]:




