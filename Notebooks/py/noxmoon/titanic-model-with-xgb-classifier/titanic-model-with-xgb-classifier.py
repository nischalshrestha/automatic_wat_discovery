#!/usr/bin/env python
# coding: utf-8

# In[176]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[177]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
data = pd.concat([train_data,test_data],axis=0,ignore_index=True)
print('train shape ', train_data.shape, '; test shape ',test_data.shape, '; all data',data.shape)


# In[178]:


data.head()


# In[179]:


data.describe()


# In[180]:


data.describe(include=['O'])


# In[181]:


data_pruned= pd.concat([data.Survived,data.Pclass],axis=1)
data_pruned.head()


# In[182]:


FamilySize = data.SibSp + data.Parch
data_pruned['FamilySize'] = FamilySize


# In[183]:


Sex = data.Sex.map({'male':1,'female':0})
data_pruned['Sex'] = Sex.values
data_pruned.head()


# In[184]:


Embarked = data.Embarked
data_pruned['Embarked'] = Embarked.values


# In[185]:


data.groupby(['Embarked','Pclass']).Fare.mean()


# In[186]:


Title = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
Title.iloc[1305]
Title = Title.map({'Capt':0,'Col':0,'Major':0,'Rev':0,
           'Countess':1,'Don':1,'Dona':1,'Dr':1,'Jonkheer':1,'Lady':1,'Sir':1,
           'Miss':2,'Mlle':2,
           'Mme':3,'Mrs':3,'Ms':3,
           'Mr':4,'Master':5})
data_pruned['Title']=Title.values
data_pruned.groupby('Title').Survived.value_counts()


# In[187]:


data_pruned['Age']=data.Age.values
guess_age = data_pruned.groupby(['Pclass','Title']).Age.median()
#print(guess_age)
for i in data_pruned.Pclass.unique():
    for j in data_pruned.Title.unique():
        if (i,j) in guess_age.index:
            data_pruned.loc[(data.Age.isnull()) & (data_pruned.Pclass == i) & (data_pruned.Title==j), 'Age'] = guess_age[i,j]

#print(data_pruned.loc[data.Age.isnull()])
data_pruned.loc[ data_pruned['Age'] <= 8, 'Age'] = 0
data_pruned.loc[(data_pruned['Age'] > 8)  & (data_pruned['Age'] <= 20), 'Age'] = 1
data_pruned.loc[(data_pruned['Age'] > 20) & (data_pruned['Age'] <= 40), 'Age'] = 2
data_pruned.loc[(data_pruned['Age'] > 40) & (data_pruned['Age'] <= 60), 'Age'] = 4
data_pruned.loc[ data_pruned['Age'] > 60, 'Age'] = 5
print(data_pruned.groupby('Survived').Age.value_counts().sort_index())


# In[188]:


hasCabin = data.Cabin.notna().astype('int')
data_pruned['hasCabin'] = hasCabin.values
data_pruned.groupby('Survived').hasCabin.value_counts()


# In[189]:


data_pruned['Fare']=data.Fare.values
guess_Fare = data_pruned.groupby(['Pclass']).Fare.median()
#print(guess_Fare)
for i in data_pruned.Pclass.unique():
    data_pruned.loc[data.Fare.isnull() & data_pruned.Pclass == i, 'Fare'] = guess_Fare[i]
#print(data_pruned[data.Fare.isnull()])
data_pruned.loc[ data_pruned['Fare'] <= 8, 'Fare'] = 0
data_pruned.loc[(data_pruned['Fare'] > 8)  & (data_pruned['Fare'] <= 20), 'Fare'] = 1
data_pruned.loc[(data_pruned['Fare'] > 20) & (data_pruned['Fare'] <= 50), 'Fare'] = 2
data_pruned.loc[(data_pruned['Fare'] > 50) & (data_pruned['Fare'] <= 100), 'Fare'] = 3
data_pruned.loc[ data_pruned['Fare'] > 100, 'Fare'] = 4
print(data_pruned.groupby('Survived').Fare.value_counts().sort_index())


# In[190]:


isalone = ((data.Parch + data.SibSp)==0).astype('int')
isalone.value_counts()
data_pruned['Alone'] = isalone
data_pruned.head()


# In[191]:


data_pruned['Embarked'] = Embarked.values
data_pruned['Title']=Title.values
Embarked_ctg = pd.get_dummies(data.Embarked,prefix='Embarked')
Title_ctg = pd.get_dummies(Title,prefix='Title')
data_pruned.drop(['Embarked','Title'],axis=1,inplace=True)
data_pruned = pd.concat([data_pruned,Embarked_ctg,Title_ctg],axis=1)
data_pruned.head()


# In[192]:


data_pruned.info()


# In[193]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, GridSearchCV


# In[194]:


train = data_pruned.iloc[0:891]
test = data_pruned.iloc[891:1309]
train_X = train.loc[:,'Pclass':]
train_Y = train.loc[:,'Survived']
test_X = test.loc[:,'Pclass':]
test_Y = test.loc[:,'Survived']
print('shapes: ',train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)


# In[195]:


model = LogisticRegression()
model.fit(train_X, train_Y)
pred_Y = model.predict(test_X)
acc_log = round(model.score(train_X, train_Y) * 100, 2)
acc_log


# In[196]:


model = SVC()
model.fit(train_X, train_Y)
pred_Y = model.predict(test_X)
acc_log = round(model.score(train_X, train_Y) * 100, 2)
acc_log


# In[197]:


model = RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_Y)
pred_Y = model.predict(test_X)
acc_random_forest = round(model.score(train_X, train_Y) * 100, 2)
acc_random_forest


# In[198]:


cv_split = 4
model = RandomForestClassifier()
cv = cross_validate(model, train_X, train_Y, cv=cv_split, return_train_score=True)
model.fit(train_X, train_Y)
print('initial params', model.get_params())
print('cv train score ',cv['train_score'].mean(),'; cv test score ',cv['test_score'].mean())

param_grid={'n_estimators':[5,10,15,20,40],
           'max_leaf_nodes':[5,10,20,None],
           'max_depth':[4,8,16,None],
           'criterion':['gini','entropy']}
tune_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv_split, scoring = 'roc_auc')
tune_model.fit(train_X, train_Y)
cv = cross_validate(tune_model, train_X, train_Y, cv=cv_split, return_train_score=True)

print('final params ', tune_model.best_params_)
print('cv train score ',cv['train_score'].mean(),'; cv test score ',cv['test_score'].mean())

pred_Y = tune_model.predict(test_X)
train_score=round(tune_model.score(train_X, train_Y) * 100, 2)
print('score for all training data',train_score)


# In[199]:


#print(test_data.columns.values)
#print(test_data.PassengerId.values)
submission2 = pd.DataFrame({'PassengerId':test_data.PassengerId.values,'Survived':pred_Y.astype(int)})
submission2.to_csv('../working/submission.csv', index=False)

