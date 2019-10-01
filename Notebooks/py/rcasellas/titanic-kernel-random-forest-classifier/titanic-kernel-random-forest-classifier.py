#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = [train, test]


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


excl = ['PassengerId', 'Survived', 'Ticket', 'Cabin', 'Name']
cols = [c for c in train.columns if c not in excl]


# In[ ]:


train['Name']


# In[ ]:


train[cols].head()


# In[ ]:


train['Pclass'][train['Pclass'] == 3].count()


# In[ ]:


sns.countplot(train['Pclass'])


# In[ ]:


sns.countplot(train['Deck'])


# In[ ]:


train.isnull().sum()


# In[ ]:


sns.countplot(train['Embarked'])


# In[ ]:


sns.distplot(train['Fare'].dropna())


# In[ ]:


sns.countplot(train['Parch'].dropna())


# In[ ]:


sns.countplot(train['SibSp'])


# In[ ]:


sns.distplot(train['Age'].dropna())


# In[ ]:


sns.countplot(train['Sex'])


# In[ ]:


sns.countplot(train['Survived'])


# In[ ]:


for df in combine:
    df['child'] = float('NaN')
    df["child"][df["Age"] < 18] = 1
    df["child"][df["Age"] >=18] = 0


# In[ ]:


train["Survived"][train["child"] == 1].value_counts(normalize = True)


# In[ ]:


train["Survived"][train["child"] == 0].value_counts(normalize = True)


# In[ ]:


for df in combine:
    # Convert the male and female groups to integer form
    df["Sex"][df["Sex"] == "male"] = 0
    df["Sex"][df["Sex"]== "female"] = 1


# In[ ]:


grid = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


for df in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) &                                   (df['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    df['Age'] = df['Age'].astype(int)


# In[ ]:


#Method for finding substrings
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan


# In[ ]:


#Map titles
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
for df in combine:
    df['Title'] = df['Name'].astype(str).map(lambda x: substrings_in_string(x, title_list))


# In[ ]:


#Replace rare titles
for df in combine:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


#Change title to numnerics
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for df in combine:
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)


# In[ ]:


sns.countplot(train['Title'])


# In[ ]:


#Map cabins
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
for df in combine:
    df['Deck'] = df['Cabin'].astype(str).map(lambda x: substrings_in_string(x, cabin_list))


# In[ ]:


# Convert the Deck classes to integer form
for df in combine:
    df["Deck"][df["Deck"] == "A"] = 1
    df["Deck"][df["Deck"] == "B"] = 2
    df["Deck"][df["Deck"] == "C"] = 3
    df["Deck"][df["Deck"] == "D"] = 4
    df["Deck"][df["Deck"] == "E"] = 5
    df["Deck"][df["Deck"] == "F"] = 6
    df["Deck"][df["Deck"] == "G"] = 7
    df["Deck"][df["Deck"] == "T"] = 8


# In[ ]:


# Impute the Embarked variable
for df in combine:
    df["Deck"] = df["Deck"].fillna(0)


# In[ ]:


#Create family size feature
for df in combine:
    df['Family_size'] = df['SibSp']+df['Parch']+1


# In[ ]:


#Create fare per person
for df in combine:
    df['Fare_Per_Person']=df['Fare']/(df['Family_size']+1)


# In[ ]:


#Create isAlone feature
for df in combine:
    df['isAlone']=0
    df.loc[df['Family_size']==1, 'isAlone'] = 1


# In[ ]:


train[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean()


# In[ ]:


test[cols].head()


# In[ ]:


null_counts = test[cols].isnull().sum()/len(test[cols])


# In[ ]:


test[cols] = test[cols].fillna(0)


# In[ ]:


test[cols].head()


# In[ ]:


plt.figure(figsize=(16,8))
plt.xticks(np.arange(len(null_counts)), null_counts.index, rotation='vertical')
plt.ylabel('fraction of rows with missing data')
plt.bar(np.arange(len(null_counts)),null_counts)


# In[ ]:


for df in combine:
    # Impute the Embarked variable
    df["Embarked"] = df["Embarked"].fillna("S")

    # Convert the Embarked classes to integer form
    df["Embarked"][df["Embarked"] == "S"] = 0
    df["Embarked"][df["Embarked"] == "C"] = 1
    df["Embarked"][df["Embarked"] == "Q"] = 2


# In[ ]:


target = train["Survived"].values


# In[ ]:


features = train[cols].values


# In[ ]:


train[cols].head()


# In[ ]:


logr = LogisticRegression()
logr.fit(features, target)


# In[ ]:


logr.score(features, target)


# In[ ]:


coeff_df = pd.DataFrame(train[cols].columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logr.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


rfr = RandomForestClassifier(n_estimators=100, random_state=10, verbose=0)


# In[ ]:


rfmod = rfr.fit(features, target)


# In[ ]:


rfmod.score(features, target)


# In[ ]:


etc = ExtraTreesClassifier(n_estimators=100, max_depth=4, n_jobs=-1, random_state=1, verbose=0)
etcmod = etc.fit(features, target)


# In[ ]:


fi = etcmod.feature_importances_


# In[ ]:


importances = pd.DataFrame(fi, columns = ['importance'])
importances['feature'] = cols


# In[ ]:


importances.sort_values(by='importance', ascending=False)


# In[ ]:


test_features = test[cols].values


# In[ ]:


pred = etcmod.predict(test_features)


# In[ ]:


PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])


# In[ ]:


my_solution.to_csv("extraTrees.csv", index_label = ["PassengerId"])

