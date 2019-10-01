#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from IPython.display import display

print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_sub = pd.read_csv('../input/gender_submission.csv')
sample_sub.head()


# In[ ]:


print(train.columns)
print('--'*46)
display(train.describe())
print('--'*46)
display(train.head())


# In[ ]:


train['Title'] = train['Name'].map(lambda x : x[x.find(",")+1:x.find(".")])
test['Title'] = test['Name'].map(lambda x : x[x.find(",")+1:x.find(".")])
display(train['Title'].unique())
print('--'*46)
display(test['Title'].unique())


# In[ ]:


def title_mapping(x):
    x = x.strip()
    if x=='Dona' or x=='Lady' or x=='the Countess' or x=='Capt' or x=='Col' or     x=='Don' or x=='Dr' or x=='Major' or x=='Rev'or x=='Sir' or x=='Jonkheer':
        return 1
    elif x=='Mlle' or x=='Ms' or x=='Miss':
        return 2
    elif x=='Mrs' or x=='Mme':
        return 3
    elif x=='Mr':
        return 4
    else:
        return 5
        
train['TitleEn'] = train['Title'].map(lambda x : title_mapping(x))
test['TitleEn']  = test['Title'].map(lambda x : title_mapping(x))


# In[ ]:


print(train['TitleEn'].unique())
print('--'*46)
print(test['TitleEn'].unique())


# In[ ]:


print(train.isnull().sum())
print('--'*46)
print(test.isnull().sum())


# In[ ]:


train['SibSp'].unique()


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

cm = confusion_matrix(train['SibSp'], train['Survived'])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train['Survived'].unique(), yticklabels=train['SibSp'].unique())


# In[ ]:


cm = confusion_matrix(train['Parch'], train['Survived'])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train['Survived'].unique(), yticklabels=train['Parch'].unique())


# In[ ]:


train['Age'].fillna((train['Age'].median()), inplace=True)
test['Age'].fillna((test['Age'].median()), inplace=True)

test['Fare'].fillna((test['Fare'].median()), inplace=True)

train['Embarked'].fillna('S', inplace=True)


# In[ ]:


train['Sex'] = train['Sex'].map(lambda x : 1 if x=='male' else 2)
test['Sex'] = test['Sex'].map(lambda x : 1 if x=='male' else 2)


# In[ ]:


train['Fsize'] = train['SibSp'] + train['Parch']
test['Fsize'] = test['SibSp'] + test['Parch']


# In[ ]:


display(train['Age'].hist())
display(test['Age'].hist())
print(train['Age'].max(), train['Age'].min())
print(test['Age'].max(), test['Age'].min())
train['AgeEn'] = train['Age'].map(lambda x : 1 if x>=0 and x<20 
                                 else 2 if x>=20 and x<30
                                 else 3 if x>=30 and x<50
                                 else 4)
test['AgeEn'] = test['Age'].map(lambda x : 1 if x>=0 and x<20 
                                 else 2 if x>=20 and x<30
                                 else 3 if x>=30 and x<50
                                 else 4)
print(train['Age'].mean(), train['Age'].std())
print(test['Age'].mean(), test['Age'].std())


# In[ ]:


train['Fsize'].hist()
test['Fsize'].hist()


# In[ ]:


train['Fare'].hist()
test['Fare'].hist()


# In[ ]:


print(train['Embarked'].unique())
print('--'*46)
print(test['Embarked'].unique())
train['Embarked'] = train['Embarked'].map(lambda x : 1 if x=='S' else 2 if x=='C' else 3)
test['Embarked'] = test['Embarked'].map(lambda x : 1 if x=='S' else 2 if x=='C' else 3)


# In[ ]:


drop_features = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Pclass', 'Title', 'Age']
train.drop(train[drop_features], axis=1, inplace=True)
test.drop(test[drop_features], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


features = ['AgeEn', 'Sex', 'Fsize', 'Fare', 'TitleEn', 'Embarked']
label = 'Survived'
train.head()


# In[ ]:


X = train[features]
y = train[label]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# In[ ]:


import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_absolute_error

pred_val = model.predict(X_val)
print(mean_absolute_error(y_val, pred_val))


# In[ ]:


pred = model.predict(test[features])


# In[ ]:


sub = sample_sub
sub[label] = pred
sub.to_csv('submission.csv', index=False)


# In[ ]:




