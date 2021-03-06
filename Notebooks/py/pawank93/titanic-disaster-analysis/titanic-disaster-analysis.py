#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#load packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

#load training data from file
train = pd.read_csv('../input/train.csv')


# In[ ]:


#load test data from file
test = pd.read_csv('../input/test.csv')


# In[ ]:


#Publish first 5 row's of training data
train.head()


# In[ ]:


#Publish first 5 row's of test data
test.head()


# In[ ]:


#publish row's and column's for training data
train.shape


# In[ ]:


#publish row's and column's for test data
test.shape


# In[ ]:


#detal info of elements for training data
train.info()


# In[ ]:


#detal info of elements for test data
test.info()


# In[ ]:


#total null elements in training data
train.isnull().sum()


# In[ ]:


#total null elements in test data
test.isnull().sum()


# In[ ]:


#load libraries 
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()


# In[ ]:


#bar_chart func defination based on survival
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['survived', 'dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))


# In[ ]:


#plot bar_chart w.r.t 'Sex'
bar_chart('Sex')


# In[ ]:


#plot bar_chart w.r.t 'Pclass'
bar_chart('Pclass')


# In[ ]:


#plot bar_chart w.r.t 'Embarked'
bar_chart('Embarked')


# In[ ]:


#Combine training and test data
train_test_data = [train,test]


# In[ ]:


#extract more relvent info from 'Name' feature
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand = False)


# In[ ]:


#Count title and it's elements for training data 
train['Title'].value_counts()


# In[ ]:


#Count title and it's elements for test data 
test['Title'].value_counts()


# In[ ]:


#Converting numerical title to categorial title as Mr, Miss and Mrs and more in numbers as compared to other title's
title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master": 3, "Dr": 3,"Rev":3, "Mlle":3, "Major":3, "Col":3,"Lady":3, "Don":3, "Countess": 3, "Mme":3, "Jonkheer":3,"Ms":3, "Capt":3, "Sir":3, "Dona": 3}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#plot bar_chart w.r.t 'Title'
bar_chart('Title')


# In[ ]:


#drop feature Name as we have extaracted the required info
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#Converting/mappimg sex in numerical form
sex_mapping = {"male":0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


#plot bar_chart w.r.t 'Sex'
bar_chart("Sex")


# In[ ]:


train.head()


# In[ ]:


test.head(20)


# In[ ]:


train.isnull().sum()


# In[ ]:


#we have Age = 177, Cabin = 687 and Embarked = 2 as missing fields we need to fill them.
#we can fill this missing fields with average values
#for Age:
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)


# In[ ]:


train.isnull().sum()


# In[ ]:


#same for test data:
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace = True)


# In[ ]:


test.isnull().sum()


# In[ ]:


#plot graph for training data for age and Survived
facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.show()


# In[ ]:


#Binning/Converting numerical age to categorical age
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[ ]:


train.head()


# In[ ]:


bar_chart('Age')


# In[ ]:


Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar',stacked=True,figsize=(10, 5))


# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


embarked_mapping = {"S": 0, "C":1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked']. map(embarked_mapping)


# In[ ]:


train.head()


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)


# In[ ]:


test.info()


# In[ ]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot, 'Fare', shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()
plt.show()


# In[ ]:


#Binning/Converting numerical age to categorical age
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


train.head()


# In[ ]:


train.Cabin.value_counts()


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[ ]:


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2.0, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace = True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace = True)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot, 'FamilySize', shade = True)
facet.set(xlim = (0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[ ]:


train.head()


# In[ ]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)


# In[ ]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# In[ ]:


#testing
clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId",axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId" : test["PassengerId"],
    "Survived" : prediction
})

submission.to_csv('submission.csv', index=False)


# In[ ]:




