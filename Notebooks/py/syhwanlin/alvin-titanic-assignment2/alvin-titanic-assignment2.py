#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns


# ### Load Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submit = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train_Y = train.loc[:, 'Survived']
Data = train.append(test)


# ### Missing Handling

# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


Data.reset_index(inplace=True, drop=True)
Data


# ### Visualization

# In[ ]:


sns.set()
sns.countplot(train_Y)


# In[ ]:


sns.countplot(train['Sex'], hue=train['Survived'])


# In[ ]:


sns.countplot(train['Pclass'], hue=train['Survived'])


# In[ ]:


sns.countplot(train['Embarked'], hue=train['Survived'])


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(sns.distplot,'Age', kde=False)


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(sns.distplot,'Parch', kde=False)


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)


# In[ ]:


Data['Family_Size'] = Data['Parch'] + Data['SibSp']
Data.head()


# In[ ]:


train['Family_Size'] = train['Parch'] + train['SibSp']
g = sns.FacetGrid(train, col='Survived')
g.map(sns.distplot, 'Family_Size', kde=False)


# ### Feature Engineering

# In[ ]:


Data['Name'].head()


# In[ ]:


Data['Name'].str.split(", ", expand=True).head(3)


# In[ ]:


Data['Title1'] = Data['Name'].str.split(", ", expand=True)[1]
Data['Title1'].head(3)


# In[ ]:


Data['Title1'] = Data['Title1'].str.split(".", expand=True)[0]
Data['Title1'].head(3)


# In[ ]:


Data['Title1'].unique()


# In[ ]:


pd.crosstab(Data['Sex'], Data['Title1']).style.background_gradient(cmap='ocean_r')


# In[ ]:


train['Title1'] = train['Name'].str.split(", ", expand=True)[1]
train['Title1'].head(3)
train['Title1'] = train['Title1'].str.split(".", expand=True)[0]
train['Title1'].head(3)


# In[ ]:


pd.crosstab(train['Survived'], train['Title1']).style.background_gradient(cmap='ocean_r')


# In[ ]:


Data.groupby(['Title1'])['Age'].mean()


# In[ ]:


Data['Title2'] = Data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess',
                                         'Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                                        ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr',
                                         'Mr','Mr','Mr','Mr','Mrs'])


# In[ ]:


Data['Title2'].unique()


# In[ ]:


pd.crosstab(Data['Sex'], Data['Title2']).style.background_gradient(cmap='ocean_r')


# In[ ]:


train['Title2'] = train['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess',
                                           'Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                                          ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr',
                                           'Mr','Mr','Mr','Mr','Mrs'])
pd.crosstab(train['Survived'], train['Title2']).style.background_gradient(cmap='ocean_r')


# In[ ]:


Data.groupby(['Title2'])['Age'].mean()


# In[ ]:


list(Data.groupby(['Title2', 'Pclass'])['Age'].mean().iteritems())[:3]


# In[ ]:


Data.info()


# In[ ]:


Data['Ticket_info'] = Data['Ticket'].apply(lambda x: x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
Data['Ticket_info'].unique()


# In[ ]:


train['Ticket_info'] = train['Ticket'].apply(lambda x: x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
train['Ticket_info'].unique()


# In[ ]:


sns.countplot(train['Ticket_info'], hue=train['Survived'])


# ### Missing Value

# In[ ]:


Data['Embarked'] = Data['Embarked'].fillna('S')


# In[ ]:


Data.info()


# In[ ]:


Data['Fare'] = Data['Fare'].fillna(Data['Fare'].mean())


# In[ ]:


Data.info()


# In[ ]:


Data['Cabin'] = Data['Cabin'].apply(lambda x: str(x)[0] if not pd.isnull(x) else 'NoCabin')


# In[ ]:


Data['Cabin'].unique()


# In[ ]:


Data['Sex'] = Data['Sex'].astype('category').cat.codes
Data['Embarked'] = Data['Embarked'].astype('category').cat.codes
Data['Pclass'] = Data['Pclass'].astype('category').cat.codes
Data['Title1'] = Data['Title1'].astype('category').cat.codes
Data['Title2'] = Data['Title2'].astype('category').cat.codes
Data['Cabin'] = Data['Cabin'].astype('category').cat.codes
Data['Ticket_info'] = Data['Ticket_info'].astype('category').cat.codes


# In[ ]:


dataAgeNull = Data[Data["Age"].isnull()]
dataAgeNotNull = Data[Data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))                     
                     ]
rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(X= dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues
Data = dataAgeNull.append(dataAgeNotNull)
Data.reset_index(inplace=True, drop=True)


# In[ ]:


dataTrain = Data[pd.notnull(Data['Survived'])].sort_values(by=["PassengerId"])
dataTest = Data[~pd.notnull(Data['Survived'])].sort_values(by=["PassengerId"])


# In[ ]:


dataTrain.columns


# In[ ]:


dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=12345,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)


# In[ ]:


pd.concat((pd.DataFrame(dataTrain.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# In[ ]:


rf_res =  rf.predict(dataTest)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)


# In[ ]:




