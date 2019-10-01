#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset

# In this notebook,I am just trying to implement Logistic Regression.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head()


# Finding the null values

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,
            cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x="Survived",hue="Pclass",data=train)


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[ ]:


train.info()


# In[ ]:


sns.countplot(x='SibSp',data=train)


# In[ ]:


train['Fare'].hist(bins=40,figsize=(12,6))


# ## Data cleaning process starts here

# By using boxplot we know the average,mean and median

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y="Age",data=train)


# Implement a function to fill the null value in the age

# In[ ]:


def impute_age(cols):
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


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Age value has been filled.See the below diagram

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.dropna(inplace=True)


# All the null values has been cleared

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# ## Developin Logistic Regression Model starts here 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# Lets make a submission
# 
# 

# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test.head()


# In[ ]:


test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test.dropna(inplace=True)
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)
test.head()


# In[ ]:


predictions = logmodel.predict(test)


# In[ ]:


output = pd.Series(predictions)
output.to_csv("output.csv")


# In[ ]:




