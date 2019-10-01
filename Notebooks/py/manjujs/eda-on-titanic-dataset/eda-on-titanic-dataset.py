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


#Here I am performing an EDA  of the titanic dataset.Mainly trying to figure out the factors that influenced 
#the survival rate by drawing different plots and queries.The main details given in the description are:
#1.On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 
#1502 out of 2224 passengers and crew. Translated 32% survival rate.
#2.Although there was some element of luck involved in surviving the sinking, some groups of people were 
#more likely to survive than others, such as women, children, and the upper-class.


# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
titanic=pd.concat([train_df,test_df],axis=0,sort=False)
#we will combine both train and test data test to do eda.


# In[ ]:


titanic.head()


# In[ ]:


titanic.info()


# In[ ]:


#It is observed from the count that there are null values in Survived,Age,Cabin,Fare and embarked.


# In[ ]:


titanic.describe(include='all')


# In[ ]:


#There are 4 categorical columns:embarked,survived,sex,Pclass.Ticket is a mix of numeric and 
#alphanumeric data types. Cabin is alphanumeric.Numerical:AGE,FARE(continuous),SibSp,Parch(discrete.)


# **Derived columns**
# #We can derive a new column showing total family members using SibSp,Parch.

# In[ ]:


titanic['Family']=titanic['SibSp']+titanic['Parch']


# In[ ]:


titanic[['Family','SibSp','Parch']].head()


# **Data Imputation**

# In[ ]:


pd.isnull(titanic).sum()


# In[ ]:


#There are lot of null values in Survived,Age,cabin.Let us impute the values for the age column.


# In[ ]:


titanic[['Age','Pclass']].groupby('Pclass').mean()


# In[ ]:


#From the figure it is observed that the mean age of class 1 is 39,class 2 is 29 and class 3 is 25 approximately.


# In[ ]:


def calc_age(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29 
        else:
            return 25
    else:
        return Age


# In[ ]:


titanic['Age']=titanic[['Age','Pclass']].apply(calc_age,axis=1)


# **Frequency Analysis of the categorical columns**

# In[ ]:


titanic['Survived'].value_counts()


# In[ ]:


#OUt of 891 people in the sample,we know that 342 has survived.That is a survival rate of 38%.


# In[ ]:


labels={1:'First class',
       2:'Second class',
       3:'Third class'}
titanic['Pclass']=titanic['Pclass'].replace(labels)
titanic['Pclass'].value_counts()


# In[ ]:


pc=(titanic['Pclass'].value_counts()/titanic.shape[0])*100


# In[ ]:


pc.plot.bar(color='steelblue',figsize=(12,5))


# In[ ]:


#Most people were travelling in third class.


# In[ ]:


titanic['Sex'].value_counts()


# In[ ]:


sc=(titanic['Sex'].value_counts()/titanic.shape[0])


# In[ ]:


sc.plot.bar(color='red')


# In[ ]:


#Nearly 65% of the passengers were male.


# In[ ]:


titanic['Embarked'].value_counts()


# In[ ]:


ec=(titanic['Embarked'].value_counts()/titanic.shape[0])


# In[ ]:


ec.plot.bar(color='green')


# In[ ]:


#Nearly 70% of the passengers embarked from 'S'


# In[ ]:


age_bins = [18,30, 60, 90]
labels = {0: 'kids',
          1: 'youth',
          2: 'elders',
          3: 'senior citizen'}
titanic['Age_bin'] = titanic['Age'].apply(lambda v: np.digitize(v, bins=age_bins))
titanic['Age_bin'] = titanic['Age_bin'].replace(labels)
titanic['Age_bin'].value_counts()


# **Numerical vs Categorical**

# In[ ]:


#Age vs sex
titanic.boxplot('Age',by='Sex',figsize=(10,5),rot=10)


# In[ ]:


titanic[['Age','Sex']].groupby('Sex').mean()


# In[ ]:


#We can see that the average age of both male and female passengers lie between 28-30.There are a number 
#of ouliers representing elder citizens.


# In[ ]:


#Age vs class
titanic.boxplot('Age',by='Pclass',figsize=(10,5))


# In[ ]:


titanic[['Age','Pclass']].groupby('Pclass').mean()


# In[ ]:


#Fare vs class
titanic.boxplot('Fare',by='Pclass',figsize=(10,5))


# In[ ]:


titanic[['Fare','Pclass']].groupby('Pclass').mean()


# In[ ]:


#First class tickets are very costly compared to third class as expected.


# In[ ]:


#Fare vs Agebin
titanic.boxplot('Fare',by='Age_bin',figsize=(10,5),rot=10)


# In[ ]:


titanic[['Fare','Age_bin']].groupby('Age_bin').mean()


# In[ ]:


#Fare is mainly dependent on the class than the age.


# **Numerical vs Numerical**

# In[ ]:


titanic[['Age','Fare']].corr()


# In[ ]:


sns.set(rc={'figure.figsize':(10,5)})
p = sns.heatmap(titanic[['Age','Fare']].corr(), cmap='Blues')


# In[ ]:


#Correlation does not exist between age and fare.


# **Categorical vs categorical**

# In[ ]:


#Survived vs Sex
obs = titanic.groupby(['Survived', 'Sex']).size()
obs.name = 'Freq'
obs = obs.reset_index()
obs = obs.pivot_table(index='Survived', columns='Sex',
                values='Freq')
sns.heatmap(obs, cmap='CMRmap_r')


# In[ ]:


titanic[['Survived','Sex']].groupby('Sex').mean()


# In[ ]:


#The survival rate is high in females and very low in males.


# In[ ]:


#Survived vs Age
obs = titanic.groupby(['Survived', 'Age_bin']).size()
obs.name = 'Freq'
obs = obs.reset_index()
obs = obs.pivot_table(index='Survived', columns='Age_bin',
                values='Freq')
sns.heatmap(obs, cmap='CMRmap_r')


# In[ ]:


titanic[['Survived','Age_bin']].groupby('Age_bin').mean().reset_index().sort_values(by='Survived',ascending=False)


# In[ ]:


#Survival rate is highest in kids and least in senior citizens.


# In[ ]:


##Survived vs Class
obs = titanic.groupby(['Survived', 'Pclass']).size()
obs.name = 'Freq'
obs = obs.reset_index()
obs = obs.pivot_table(index='Survived', columns='Pclass',
                values='Freq')
sns.heatmap(obs, cmap='CMRmap_r')


# In[ ]:


titanic[['Survived','Pclass']].groupby('Pclass').mean().reset_index().sort_values(by='Survived',ascending=False)


# In[ ]:


#First class people has the highest survival rate.

