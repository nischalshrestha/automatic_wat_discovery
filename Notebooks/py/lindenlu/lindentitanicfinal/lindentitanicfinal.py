#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# In[ ]:


print (train_df.columns.values)
train_df.head()


# In[ ]:


train_df.info()
#train_df[['Name', 'Sex']].info()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe(include=['O'])


# In[ ]:


train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['Embarked', 'Sex', 'Name']].groupby(['Embarked', 'Sex']).count()


# In[ ]:


train_df[['Embarked', 'Pclass', 'Name']].groupby(['Embarked', 'Pclass']).count()


# In[ ]:


d=pd.Series('r', index=['a', 'b', 'c', 'd', 'e'])
print (d)
import math
math.isnan((d[:1]+d[-1:])[0])


# In[ ]:


d=train_df.assign(sibpar=train_df['SibSp']+train_df['Parch'])
d.head()


# In[ ]:


#first row, returns a pandas.Series
train_df.loc[0]
train_df.iloc[0]
#first row returns a one-row dataframe
train_df.loc[[0]]
train_df.iloc[[0]]
#first 5 rows
train_df.iloc[0:5]
#train_df.loc[0:5]
#Name column
train_df.loc[:, 'Name']
#First row name
train_df.loc[0, 'Name']
#Names that don't have Embarked
train_df.loc[train_df.Embarked.isnull(), 'Name']


# In[ ]:


train_df.T


# In[ ]:


print (train_df.index.values)
train_df.columns


# In[ ]:


train_df['Name']
train_df.loc[[0, 2]]


# In[ ]:


#filter certain rows
survived_df = train_df[train_df['Survived']==1]
survived_df = train_df.query('Survived==1')
survived_df
train_df[train_df['Embarked'].isnull()]
train_df.loc[train_df['Survived']==1]
train_df[train_df.Survived==1]
train_df[train_df['Embarked'].isnull()]


# In[ ]:


name_sex_df = train_df[['Name', 'Sex']]
name_sex_df.head()


# In[ ]:


import pandas as pd

train_df = pd.read_csv('../input/train.csv')
#information
train_df.info()
#columns
train_df.columns
train_df.columns.values
#rows
train_df.index
#preview first 10 rows
train_df.head(10)
#preview last 10 rows
train_df.tail(10)
#overview description of dataframe
#on a mixed datatype df, default describe will ristrict summary to include only numerical columns
#if non are, only category columns
#only number fields
train_df.describe()
train_df.describe(include=['number'])
#object fields
train_df.describe(include=['object'])
train_df.describe(include=['O'])
#all fields
train_df.describe(include='all')
#filter rows
#survived rows
survived_df = train_df[train_df['Survived'] == 1]
survived_df = train_df.query('Survived==1')
survived_df.head()
#certern rows by index
train_df.loc[0]
#row 0 and 2
train_df.loc[[0, 2]]
#selected columns
name_sex_df = train_df[['Name', 'Sex']]
name_sex_df.describe()


# In[ ]:


#Analyse by pivoting features
#value frequency
print (train_df['Pclass'].value_counts())
#pclass and survived
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#Sex and Survived
print (train_df['Sex'].value_counts())
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[ ]:


#class+sex survival rate, if rich people less gentleman
train_df[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex']).mean()


# In[ ]:


train_df[['Embarked','Pclass', 'Sex', 'Survived']].groupby(['Embarked', 'Pclass','Sex']).count()


# In[ ]:


#embarked+sex
train_df[['Embarked','Pclass', 'Sex', 'Survived']].groupby(['Embarked', 'Pclass','Sex']).mean()
#train_df.info()


# In[ ]:


print (train_df['SibSp'].value_counts())
train_df[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


print(train_df['Parch'].value_counts())
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


#data visualization
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
#correlating numerical features
#A histogram is a graphical representation of the distribution of numerical data.
#It is an estimate of the probability distribution of a continuous variable (quantitative variable)
#row and col determine the shape of grid
#g=sns.FacetGrid(train_df, col='Survived', size=3, aspect=1.5)
g=sns.FacetGrid(train_df, row='Sex', col='Survived', size=3, aspect=1.5)
sns.set(style="ticks", color_codes=True)
#g.map(plt.hist, 'Age', bins=20)
bins = np.arange(0,80,10)
#g.add_legend()
g.map(plt.hist, 'Age', bins=bins, color='g')


# In[ ]:


#data visualization, better do it in notebook
#correlating numerical features
#col='Survived', grid has 1 row and 2 columns, with Survived=0 and 1,
#size: height of each facet in inchs, aspect:width ratio
g=sns.FacetGrid(train_df, col='Survived', size=3, aspect=1.5)
#col='Survived', row='Sex', grid has 2 rows and 2 coumns, row1: male-0, male-1, row2: female-0, female-1
#g=sns.FacetGrid(train_df, row='Sex', col='Survived', size=3, aspect=1.5)
#3 rows(Embarked), 2 columns(Sex), and different color for Survived=0/1
#g=sns.FacetGrid(train_df, row='Embarked', col='Sex', hue='Survived', size=3, aspect=1.5)
#Set aesthetic parameters, optional and not necessary here
sns.set(style="ticks", color_codes=True)
#plot histogram, bins is number of bars across all 'Age'
g.map(plt.hist, 'Age', bins=20)
#customized bins, for Age 0-80, each bin with width 10, color='b'-Blue, 'r'-Red, 'y'-Yellow, 'g'-Green
bins = np.arange(0,80,10)
g.map(plt.hist, 'Age', bins=bins, color='b')


# In[ ]:


g=sns.FacetGrid(train_df, row='Embarked', col='Sex', hue='Survived', size=3, aspect=1.5)
sns.set(style="ticks", color_codes=True)
g.add_legend()
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


#correlating ordinal features
grid=sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.7)
grid.map(plt.hist, 'Age', alpha=0.9, bins=20)
grid.add_legend()


# In[ ]:


#pointplot, Show point estimates and confidence intervals using scatter plot glyphs.
grid=sns.FacetGrid(train_df, row='Embarked', size=3, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', pallete='deep')
#add legend, in this case, female/male color
grid.add_legend()


# In[ ]:


#Categorical plot
grid=sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', pallete='deep')
grid.add_legend()


# In[ ]:


train_df_C = train_df[train_df['Embarked']=='C']
train_df_C.head()
grid = sns.FacetGrid(train_df_C, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
grid.add_legend()


# In[ ]:


train_df[['Embarked', 'Pclass', 'Survived']].groupby(['Embarked', 'Pclass']).mean()


# In[ ]:


#Categorical plot
#barplot, show point estimates and confidence intervals as rectangular bars.
grid = sns.FacetGrid(train_df, row="Embarked", col='Survived')
#ci=None, no confidence interval, alpha, darkness of bar
#grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5)


# In[ ]:


#Analyse Title
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#cross tabulation, 
pd.crosstab(train_df['Title'], train_df['Sex'])
#pd.crosstab(train_df['Sex'], train_df['Survived'])


# In[ ]:


#group titles
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


#print (train_df[['Title', 'Survived']].groupby('Title').mean())
#count of each title
print (train_df['Title'].value_counts())
#survived(0/1) count of each title
pd.crosstab(train_df['Title'], train_df['Survived'])


# In[ ]:


test_df.info()


# In[ ]:


#convert title to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()


# In[ ]:


#drop columns, axis=1 denotes column, default axis=0 denotes row
train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
train_df.head()
combine=[train_df, test_df]
print (train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# In[ ]:


print (train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df.head(10)


# In[ ]:


combine[0].info()


# In[ ]:


train_df.head(10)


# In[ ]:


#convert categorical features to numerical
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1}).astype(int)
train_df.info()
train_df.head()


# In[ ]:


#Completing numerical continuous feature, Age
guess_age = np.zeros((2,3))
guess_age
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()
            guess_age[i,j] = guess_df.median()
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1), 'Age']=guess_age[i,j]
train_df.head(10) 
train_df.info()


# In[ ]:


#Complete missing data, Age
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            #use median age of people with same sex and pclass value
            guess_df = dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()
            guess_ages[i,j] = guess_df.median()       
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1), 'Age'] = guess_ages[i,j]
        dataset['Age'] = dataset['Age'].astype(int)
print (guess_ages)            
train_df.head(10)


# In[ ]:


#train_df[['AgeBand', 'Survived']].groupby(['AgeBand']).mean().sort_values(by='AgeBand', ascending=True)
#band Age and determine correlation with Survived
for dataset in combine:
    dataset['AgeBand'] = pd.cut(dataset['Age'], 5)


# In[ ]:


#as_index=False makes AgeBand a column so that sort_values(by='AgeBand') works
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


#replace Age with ordinals
for dataset in combine:
    dataset.loc[(dataset['Age']<=16), 'Age'] = 0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32), 'Age'] = 1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48), 'Age'] = 2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64), 'Age'] = 3
    dataset.loc[(dataset['Age']>64), 'Age'] = 4
train_df.head(10)


# In[ ]:


#drop AgeBand
train_df = train_df.drop(['AgeBand'], axis=1)
test_df = test_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
test_df.head()


# In[ ]:


train_df.head(20)


# In[ ]:


train_df[['Embarked', 'Survived']].groupby(['Embarked']).count()
freq_port = train_df.Embarked.dropna().mode()
freq_port


# In[ ]:


train_df.info()
train_df[['Embarked', 'Survived']].groupby(['Embarked']).count()
#complete Embarked
train_df.loc[train_df['Embarked'].isnull(), 'Embarked']=train_df.Embarked.dropna().mode()[0]


# In[ ]:


#convert Embarked to ordinals
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)
train_df.head()


# In[ ]:


#create FamilySize based on SibSp and Parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df.head(10)


# In[ ]:


train_df[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[['FamilySize', 'Survived']].groupby('FamilySize').count()


# In[ ]:


#create isAlone based on FamilySize
for dataset in combine:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'isAlone'] = 1
train_df[['isAlone', 'Survived']].groupby('isAlone', as_index=False).mean()


# In[ ]:


#drop SibSp, Parch
train_df = train_df.drop(['SibSp', 'Parch'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[ ]:


#adjust Fare based on FamilySize
train_df['Fare'] = train_df['Fare']/train_df['FamilySize']
train_df.head(10)


# In[ ]:


test_df['Fare'] = test_df['Fare']/test_df['FamilySize']
test_df.head()
test_df.loc[test_df.Fare.isnull()]


# In[ ]:


test_df.loc[[152]]


# In[ ]:


test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass==3), 'Fare']
pf=test_df[['Pclass', 'Fare']].groupby('Pclass', as_index=False).median()
pf


# In[ ]:


pf.loc[pf.Pclass==3, 'Fare']


# In[ ]:


test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass==3), 'Fare'] = 7.75
test_df.loc[[152]]


# In[ ]:


#convert Fare to FareBand
train_df['FareBand'] = pd.qcut(train_df['Fare'], 6)
train_df[['FareBand', 'Survived']].groupby('FareBand').mean()


# In[ ]:


combine[1].info()


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]


# In[ ]:


#drop FamilySize
train_df = train_df.drop(['FamilySize'], axis=1)
test_df = test_df.drop(['FamilySize'], axis=1)
combine = [train_df, test_df]


# In[ ]:


print (train_df.head())
print (test_df.head())


# In[ ]:


# Model, Predict and Solve
x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
x_test = test_df.drop('PassengerId', axis=1).copy()
x_train.shape, y_train.shape, x_test.shape


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train)*100, 2)
acc_random_forest


# In[ ]:


#submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred
    })
submission.head(20)
submission.to_csv('linden_Titanic_submission.csv', index=False)


# In[ ]:


result_df = pd.read_csv('linden_Titanic_submission.csv')
result_df.head()

