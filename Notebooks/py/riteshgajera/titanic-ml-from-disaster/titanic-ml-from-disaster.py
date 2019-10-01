#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.info()


# In[ ]:


train.head()


# # Visualization Data

# In[ ]:


# Check Missing Data
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Survived', data=train)


# In[ ]:


sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')


# In[ ]:


sns.countplot(x='Survived', data=train, hue='Pclass')


# In[ ]:


sns.distplot(train['Age'].dropna(), kde=False, bins=30)


# In[ ]:


train['Age'].plot.hist(bins=30)


# In[ ]:


sns.countplot(x='SibSp', data=train)


# In[ ]:


train['Fare'].hist(bins=40, figsize=(10,4))


# # Cleaning Data

# In[ ]:


import cufflinks as cf


# In[ ]:


cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist', bins=50)


# In[ ]:


sns.boxplot(x='Pclass', y='Age', data=train)


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


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='BuPu')


# In[ ]:


train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='Greens')


# In[ ]:


train.dropna(inplace=True)


# # Categorical Data

# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[ ]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[ ]:


train = pd.concat([train,sex,embark], axis=1)


# In[ ]:


train.head()


# In[ ]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.drop(['PassengerId'], axis=1, inplace=True)


# In[ ]:


train.head()


# # Train Model

# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


print(confusion_matrix(y_test,predictions))

