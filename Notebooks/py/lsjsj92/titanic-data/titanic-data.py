#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#titanic data


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.head(2))


# In[ ]:


print(train.shape)


# In[ ]:


print(train.columns)


# In[ ]:


print(train.isnull().sum())


# In[ ]:


print(test.isnull().sum())


# - survived : 0 = No, 1 = Yes
# - pclass : Ticket class1 = 1st, 2=2nd
# - parch : of parents / children aboard the Titanic
# - ticket : ticket number
# - cabin : cagin number

# see the test data too

# In[ ]:


test.head()


# In[ ]:


print(train.shape)
print(test.shape)


# train and test shape is different! why?

# In[ ]:


print(train.info())


# see the age and cabin.  it have bifferent number compare with others.
# 
# they have NaN values.
# 
# after later i will fill the value in NaN

# In[ ]:


print(test.info())


# train has 'survived' column. but test has no column 'survived'. because we have to anticipate that.

# let`s visualize the data.
# 
# ** survived **

# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('Pclass')


# - the chart confirms female more likely survived than male
# - the chart confirms 1st class more likely survived than other classes

# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# # let`s doing feature engineering
# 
# 
# feature engineering is the process of using domain knowledge of data to create features(feature vectors) that make machine learning algorithms work
# 
# feature(속성) to vector : ex ) string -> number, NaN -> value...
# 
# 
# 
# 

# # Name
# 
# get 'Mr', 'Mrs' ,'miss' text

# In[ ]:


train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-za-z]+)\.', expand=False)


# In[ ]:


train['Title'].value_counts()

print(train.head())


# In[ ]:


title_mapping = {
    "Mr":0,
    "Miss":1,
    "Mrs" : 2,
    "Master":3,
    "Dr":3,
    "Rev":3,
    "Major":3,
    "Mlle":3,
    "Col":3,
    "Capt":3,
    "Mme":3,
    "Don":3,
    "Lady":3,
    "Jonkheer":3,
    "Countess":3,
    "Sir":3,
    "Ms":3,
}


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
train.head()


# In[ ]:


test.head()


# In[ ]:


bar_chart('Title')


# # sex
# 
# mail : 0, female : 1
# 

# In[ ]:


sex_mapping = {
    "male" : 0,
    "female" : 1
}

for dataset in train_test_data:
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)
    
print(dataset["Sex"].head())


# In[ ]:


bar_chart('Sex')


# ** now i do fill the value instead of NaN **
# 
# likely mean value.
# 

# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


gra = sns.FacetGrid(train, hue="Survived", aspect=4)
gra.map(sns.kdeplot, 'Age', shade=True)
gra.set(xlim=(0, train['Age'].max()))
gra.add_legend()

plt.show()


# In[ ]:


gra = sns.FacetGrid(train, hue="Survived", aspect=4)
gra.map(sns.kdeplot, 'Age', shade=True)
gra.set(xlim=(0, train['Age'].max()))
gra.add_legend()

plt.xlim(10, 30) #age 10 ~ 30

plt.show()


# binning the age
# 
# Numerical age to catecorical 
# 
# child : 0
# young : 1
# adult : 2
# mid-age : 3
# senior : 4

# In[ ]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age' ] = 0
    dataset.loc[ (dataset['Age'] > 16 ) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[ (dataset['Age'] > 26 ) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[ (dataset['Age'] > 36 )& (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4


# In[ ]:


print(train.head())


# In[ ]:


bar_chart('Age')


# fill NaN value in embarked

# In[ ]:


Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True)


# more than 50% of 1st class are from 'S'
# almost from 'Q' people in 3rd class
# 
# and 'S' is so many count in Embarked. so i will fill value 'S' in NaN value.
# 

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


embarked_marking = {
    'S' : 0,
    'C' : 1,
    'Q' : 2
}

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_marking)


# In[ ]:


print(train.isnull().sum())


# and fill Cabin NaN data.
# 
# but, fist fill binning data in Fare

# In[ ]:


print(train["Fare"].head())
print(train['Fare'].isnull().sum())


# In[ ]:


gra = sns.FacetGrid(train, hue='Survived', aspect=4)
gra.map(sns.kdeplot, 'Fare', shade=True)
gra.set(xlim=(0, train['Fare'].max()))
gra.add_legend()

plt.show()


# In[ ]:


print(set(train["Survived"].values))

'''
for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[ (dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[ (dataset['Fare'] > 30 ) & (dataset['Fare'] <= 100 ), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 100] = 3'''


# In[ ]:


print(set(train["Survived"].values))


# In[ ]:


train.head(20)


# In[ ]:


print(set(train["Survived"].values))

print(train[train["Survived"] == 3])


# and fill Cabin NaN data
# 
# 

# In[ ]:


print(train['Cabin'].value_counts())


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1] #get a first char


# In[ ]:


Pclass1 = train[train['Pclass'] == 1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,6))


# In[ ]:


cabin_mapping = {
    "A" : 0,
    "B" : 0.4,
    "C" : 0.8,
    "D" : 1.2,
    "E" : 1.6,
    "F" : 2,
    "G" : 2.4,
    "T" : 2.8
}
#소숫점을 사용한 이유는 유클리디언 거리로 인해 값이 커지면 머신러닝 입장에선 더 중요한 값이라고 생각할 수도 있기 때문.
#그래서 이것을 어느정도 ㄱ길이가 일정한 것으로 변환. 이거를 feature scaling이라고 한다.
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[ ]:


train['Cabin'].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test['Cabin'].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# ** set a family size **
# 

# In[ ]:


train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


gra = sns.FacetGrid(train, hue='Survived', aspect=4)
gra.map(sns.kdeplot, 'family_size', shade=True)
gra.set(xlim=(0, train['family_size'].max()))
gra.add_legend()
plt.xlim(0)


# In[ ]:


family_mapping = {
    1 : 0,
    2 : 0.4,
    3 : 0.8,
    4 : 1.2,
    5 : 1.6,
    6 : 2,
    7 : 2.4,
    8 : 2.8,
    9 : 3.2,
    10 : 3.6,
    11 : 4
}


for dataset in train_test_data:
    dataset['family_size'] = dataset['family_size'].map(family_mapping)


# In[ ]:


print(train.head())


# and drop columns

# In[ ]:


drop_columns = ['Ticket', 'SibSp', 'Parch', 'Name']
train = train.drop(drop_columns, axis=1)
test = test.drop(drop_columns, axis=1)

print(train.columns)
print(test.columns)

train = train.drop(['PassengerId'], axis=1)
#test = test.drop(['PassengerId'], axis=1)
print(train.shape)
print(test.shape)


# In[ ]:


train_data = train.drop('Survived', axis=1)
print(train_data.head())


# In[ ]:


target_data = train['Survived']
print(target_data.head())


# In[ ]:


print(set(target_data.values))

print(set(train["Survived"].values))


# # finally we classifiy data by machine learning

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[ ]:


print(train.info())


# ** cross validation **

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score

k_fold = KFold(n_splits=10, shuffle = True, random_state=0)


# ** KNN **

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 9)
scoring = 'accuracy'


# In[ ]:


score = cross_val_score(clf, train_data, target_data, cv=k_fold, n_jobs=1, scoring = scoring)
print(score)


# ** knn score **

# In[ ]:


round(np.mean(score)*100, 2)


# ** Decision Tree **

# In[ ]:


clf = DecisionTreeClassifier()
scoreing = 'accuracy'

score = cross_val_score(clf, train_data, target_data, cv=k_fold, n_jobs=5, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ** random forest **

# In[ ]:


clf = RandomForestClassifier(n_estimators = 50)
scoreing = 'accuracy'

score = cross_val_score(clf, train_data, target_data, cv=k_fold, n_jobs=5, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ** navie bayes **

# In[ ]:


clf = GaussianNB()
scoreing = 'accuracy'

score = cross_val_score(clf, train_data, target_data, cv=k_fold, n_jobs=5, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ** SVM **

# In[ ]:


clf = SVC(C=10)
scoreing = 'accuracy'

score = cross_val_score(clf, train_data, target_data, cv=k_fold, n_jobs=3, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ** test **

# In[ ]:


clf = SVC(C=10)
clf.fit(train_data, target_data)


# In[ ]:


print(test.head())
print(test.isnull().sum())
test = test.dropna(how='any')
print(test.isnull().sum())


# In[ ]:


test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)
print(prediction)


# ** if you want submit as csv file **

# In[ ]:


# if you want submission as csv

submission = pd.DataFrame({
 "PassengerId" : test["PassengerId"],
 "Survived" : prediction

})

submission.to_csv('submission.csv', index=False)


# In[ ]:


sub = pd.read_csv('submission.csv')
sub.head()


# and submit this file!

# In[ ]:




