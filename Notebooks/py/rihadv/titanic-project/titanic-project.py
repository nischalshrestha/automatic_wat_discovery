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


import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()


# **List of all FUNCTIONS**

# In[ ]:


def male_female_child( passenger):
    # Take the Age and Sex
    age,sex = passenger # no need any more
    # Compare the age, otherwise leave the sex
    if passenger['Age']<16:
        return 'child'
    else:
        return passenger['Sex']
train[['Age','Sex']].apply(male_female_child, axis=1).head()
train['Passenger']=train[['Age','Sex']].apply(male_female_child, axis=1)
print(train.head())


# Function is used for plotting. bar_char function is used for investigating each features of datasets

# In[ ]:


alpha_color=0.5
bins=[0,10,20,30,40,50,60,70,80]
train['AgeBin']=pd.cut(train['Age'],bins)

#bar_char function
def bar_char(feature):
    survived=train[train['Survived']==1][feature].value_counts(normalize=True).sort_index()
    dead=train[train['Survived']==0][feature].value_counts(normalize=True).sort_index()
    data=pd.DataFrame([survived, dead])
    data.index=['Survived', 'Dead']
    data.plot(kind='bar',alpha=alpha_color)
    
#facet_grid function
def facet_grid(feature,a, b):
    facet=sns.FacetGrid(train, hue='Survived', aspect=4)
    facet.map(sns.kdeplot, feature, shade=True)
    facet.set(xlim=(0, train[feature].max()))
    facet.add_legend()
    plt.xlim(a,b)
    plt.show()
    
#fillna one feature based on mean of other feature
def fill_na(groupbyfeature, meanfeature):
    train[meanfeature]=train.fillna(train.groupby(groupbyfeature)[meanfeature].transform('mean'), inplace=True)
    test[meanfeature]=test.fillna(train.groupby(groupbyfeature)[meanfeature].transform('mean'), inplace=True)
    
# drop column --> Works along each row --> axis =1
#work along each row: axis=1, work along each column axis=0
def drop_feature(dataframe,feature):
    dataframe=dataframe.drop(feature,axis=1)
    return dataframe


# Feature Engineering is the process of using domain knowledge of data to create features ( featers vectors) that make machine learning algorithms work
# Each column is a feature,  
# Change text into value, such as Male, Female into 0 or 1

# In[ ]:


#MAIN STEPS FOR PRE-PROCESSING DATA
# Firstly, Handle missing value, and fill in them
#Secondly, Normalize data: categories,...(formatted as number)
train_test_data=[train, test]
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
#change title into group
title_mapping={"Mr":0, "Miss":1, "Mrs":2,
               "Master":3, "Dr":3, "Rev":3,
               "Col":3, "Major":3, "Mile":3,
               "Countess":3, "Ms":3, "Lady":3,
               "Jonkheer":3, "Don":3, "Dona":3, "Mme":3, "Capt":3, "Sir":3}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
    
#Sex feature: change male and female into value 0 and 1
sex_mapping={'male':0, 'female':1}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)
    
#Age: some age is missing --> use Title's meadian age for missing Age
# train['Age']=train.fillna(train.groupby('Title')['Age'].transform("median"), inplace=True)
# test['Age']=train.fillna(test.groupby('Title')['Age'].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16, 'Age']=0,
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26), 'Age']=1,
    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36), 'Age']=2,
    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62), 'Age']=3,
    dataset.loc[dataset['Age']>62, 'Age']=4
    
# Embarked feature
Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()
df=pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index=('1st class','2nd class', '3rd class')
df.plot(kind='bar', stacked=True)
#S is the most --> Filling S for missing values
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    
embarked_mapping={'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].map(embarked_mapping)
    
#Fare feature: Filling missing fares with mean of fare, grouped by Pclass
# train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace=True)
# test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)
for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[dataset['Fare']>100, 'Fare']=3
    
#Cabin feature
# fill_na('Pclass', 'Cabin')
cabin_mapping={'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F': 2, 'G': 2.4, 'T':2.8}
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)
    
#Family size
train['FamilySize']=train['SibSp']+train['Parch']+1
test['FamilySize']=test['SibSp']+test['Parch']+1
family_mapping={1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_data:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)


# In[ ]:


# train.drop('Name', axis=1, inplace=True)
# test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


drop_feature(train,['Parch', 'Ticket', 'PassengerId', 'Survived','SibSp','AgeBin','Name'])


# In[ ]:


train.groupby('Title')['Age'].transform('mean')


# In[ ]:


#Age: some age is missing --> use Title's meadian age for missing Age
# train['Age']=train.fillna(train.groupby('Title')['Age'].transform("median"), inplace=True)
# test['Age']=train.fillna(test.groupby('Title')['Age'].transform("median"), inplace=True)
# for dataset in train_test_data:
#     dataset.loc[dataset['Age']<=16, 'Age']=0,
#     dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26), 'Age']=1,
#     dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36), 'Age']=2,
#     dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62), 'Age']=3,
#     dataset.loc[dataset['Age']>62, 'Age']=4


# In[ ]:





# In[ ]:





# In[ ]:


facet=sns.FacetGrid(train, hue='Survived', aspect=4)
facet.map(sns.kdeplot,'Fare', shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()
plt.show()


# In[ ]:


df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                   columns=list('ABCD'))
df


# In[ ]:


df.groupby('A')['B'].transform('median')


# In[ ]:


df.fillna(df.groupby('A')['B'].mean(), inplace=True) 


# In[ ]:


df


# In[ ]:


df.groupby('A')['B'].mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




