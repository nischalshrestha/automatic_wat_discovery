#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set() #setting seaborn default for plot


# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))


# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('Pclass')


# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# In[ ]:


bar_chart('Embarked')


# In[ ]:


## Applying Feature engg

train_test_data = [train,test] #combining train and test data

sex_mapping = {'male':0,'female':1}

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


bar_chart('Sex')


# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)


# In[ ]:


train['Title'].value_counts()


# In[ ]:


title_mapping = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Dr':3,'Rev':3,'Mlle':3,'Major':3,'Col':3,'Countess':3,'Lady':3,'Mme':3,'Ms':3,'Jonkheer':3,'Sir':3,'Capt':3,'Don':3,'Dona':3} 

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


bar_chart('Title')


# In[ ]:


train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace = True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace = True)


# In[ ]:


facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()

plt.show()


# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26) , 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36) , 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62) , 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4
    


# In[ ]:


bar_chart('Age')


# In[ ]:


Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['Pclass1','Pclass2','Pclass3']
df.plot(kind='bar',stacked=True,figsize = (10,5))


# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


embarked_mapping = {'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace = True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace = True)


# In[ ]:


facet = sns.FacetGrid(train,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade = True)
facet.set(xlim = (0,train['Fare'].max()))

facet.add_legend()
plt.show()


# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17,'Fare'] = 0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30),'Fare'] = 1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare'] = 2,
    dataset.loc[dataset['Fare']>100,'Fare'] = 3


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


cabin_mapping = {'A':0.0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace = True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace = True)


# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}

for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


features_drop = ['Name','Ticket','SibSp','Parch']
train = train.drop(features_drop,axis = 1)
test = test.drop(features_drop,axis = 1)
train = train.drop(['PassengerId'],axis = 1)


# In[ ]:


train_data = train.drop(['Survived'],axis=1)
target = train['Survived']


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits = 10,shuffle = True,random_state = 0)


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=13)
score = cross_val_score(clf,train_data,target,cv = k_fold,n_jobs=1,scoring='accuracy')


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


clf = DecisionTreeClassifier()
score = cross_val_score(clf,train_data,target,cv = k_fold,n_jobs=1,scoring='accuracy')
round(np.mean(score)*100,2)


# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
score = cross_val_score(clf,train_data,target,cv = k_fold,n_jobs=1,scoring='accuracy')
round(np.mean(score)*100,2)


# In[ ]:


clf = GaussianNB()
score = cross_val_score(clf,train_data,target,cv = k_fold,n_jobs=1,scoring='accuracy')
round(np.mean(score)*100,2)


# In[ ]:


clf = SVC()
score = cross_val_score(clf,train_data,target,cv = k_fold,n_jobs=1,scoring='accuracy')
round(np.mean(score)*100,2)


# In[ ]:


clf = SVC()
clf.fit(train_data,target)

test_data = test.drop('PassengerId',axis = 1).copy()
prediction = clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})
submission.to_csv('submit.csv',index=False)


# In[ ]:


#submit = pd.read_csv('submit.csv')
#submit.head()


# In[ ]:




