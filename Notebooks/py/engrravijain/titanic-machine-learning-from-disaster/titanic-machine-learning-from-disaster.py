#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


# Load the training and testing data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = [train, test]


# In[ ]:


train.head()


# In[ ]:


for data in combine:
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
print  (pd.crosstab(train['Title'], train['Survived']))
print ('\n\n')
print  (pd.crosstab(train['Title'], train['Sex']))
data.head()


# In[ ]:


for data in combine:
    data['Title'] = data['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Rare')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


map_title = {"Master": 1, 'Miss': 2, "Mr": 3, "Mrs": 4, "Rare": 5}
for data in combine:
    data['Title'] = data['Title'].map(map_title)
    data['Title'] = data['Title'].fillna(0)
test.head()


# In[ ]:


train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]


# In[ ]:


# age column
for data in combine:
    data['Sex'] = data['Sex'].map({"male": 1, "female": 0}).astype(int)
train.head()


# In[ ]:


age = np.zeros([2,3])
for data in combine:
    for i in range (0,2):
        for j in range (0,3):
            guess_df = data[(data['Sex']==i) & (data['Pclass']==j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            age[i,j] = int(age_guess/0.5 + 0.5) * 0.5
    for i in range (0,2):
        for j in range (0,3):
            data.loc[(data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1), 'Age'] = age[i,j]
    data['Age'] = data['Age'].astype(int)
train.head()


# In[ ]:


# create age bands to see correlation between who survived
train['AgeGroup'] = pd.cut(train['Age'], 5)
train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True)


# In[ ]:


for data in combine:    
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']
train.head()


# In[ ]:


train = train.drop(['AgeGroup'], axis=1)
combine = [train, test]
train.head(40)


# create a new feature called family size using Parch and SibSp.

# In[ ]:


for data in combine:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=True)


# lets create a IsAlone feature from the FamilySize feature

# In[ ]:


for data in combine:
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]
train.head()


# In[ ]:


train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]
train.head()


# create a artificial feature using Age and Pclass

# In[ ]:


for data in combine:
    data['Age*Pclass'] = data.Age * data.Pclass


# In[ ]:


# most frequently embarked port
freq_port = train.Embarked.dropna().mode()[0]
# fill missing values by freq_port
for data in combine:
    data['Embarked'] = data['Embarked'].fillna(freq_port)


# In[ ]:


for data in combine:
    data['Embarked'] = data['Embarked'].map({"S": 0, "C": 1, "Q": 2}).astype(int)
train.head()


# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


for data in combine:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]
    
train.head(10)


# In[ ]:


test['Title'] = test['Title'].astype(int)
test = test.drop(['PassengerId'], axis=1)
test.head(5)


# # Model

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


train = train.values
x_train = train[:,1:]
y_train = train[:,:1]
x_train.shape, y_train.shape


# In[ ]:


test = test.values
x_test = test
x_test.shape


# In[ ]:


model = Sequential()
model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
model.add(Dense(9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.summary()


# In[ ]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 16, epochs = 150)


# In[ ]:


y_pred = model.predict(x_test)
y_final = (y_pred > 0.5).astype(int).reshape(x_test.shape[0])


# In[ ]:


df_test = pd.read_csv('../input/test.csv')
output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
output.to_csv('prediction.csv', index=False)

