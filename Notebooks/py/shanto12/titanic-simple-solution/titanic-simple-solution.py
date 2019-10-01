#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data_set = [train, test]


# In[ ]:


train.head()


# In[ ]:


train.isnull()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Survived', data=train, hue='Sex', palette = 'RdBu_r')


# In[ ]:


sns.countplot(x='Survived', data=train, hue='Pclass', palette = 'magma')


# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False, bins=30)


# In[ ]:


train['Age'].plot.hist(bins=35, ec='white')


# In[ ]:


train.info()


# In[ ]:


sns.countplot(train['SibSp'], data=train)


# In[ ]:


train['Fare'].hist(bins=40, figsize = (10, 4))


# Clearning Data

# In[ ]:


plt.figure(figsize=(15, 10))
sns.boxplot(x='Pclass', y = 'Age', data =train)


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

    
for df in data_set:    
    df['Age'] = df[['Age','Pclass']].apply(impute_age, axis=1)
    


# In[ ]:


plt.title('train')
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


for df in data_set:   
    df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


columns = ['Fare', 'Embarked']       
for df in data_set:
    for col in columns:
        df[col].fillna(value=df[col].value_counts().index[0], inplace=True)


# In[ ]:



test.isnull().sum()


# In[ ]:


train.isnull().sum()


# In[ ]:


# for df in data_set:   
#     df.dropna(inplace=True)


# In[ ]:


train.isnull().sum()


# In[ ]:


categorical_cols = ['Sex','Embarked','Pclass']
new_columns = []
for index, df in enumerate(data_set):
    new_columns.clear()
    for col in categorical_cols:
        new_columns.append(pd.get_dummies(df[col], drop_first=True))
    new_columns.append(df)
    df = pd.concat(new_columns, axis =1)
    df.drop(categorical_cols, axis=1, inplace=True, errors=False)
    data_set[index] =df


# In[ ]:


train, test = data_set


# In[ ]:


# sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[ ]:


# embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[ ]:


# pclass = pd.get_dummies(train['Pclass'], drop_first=True)


# In[ ]:


# train = pd.concat([train, sex, embark, pclass], axis =1)
train.head()


# In[ ]:


test_passengerIds = test['PassengerId'].copy()
for df in data_set:
    df.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True, errors=False)


# train.drop(['Sex','PassengerId', 'Embarked','Name','Ticket', 'Pclass'], axis=1, inplace=True, errors=False)


# In[ ]:


train.tail()


# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X, y)


# In[ ]:


y_pred = logmodel.predict(test)


# In[ ]:


pd.DataFrame({'PassengerId': test_passengerIds, 'Survived': y_pred}).to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:




