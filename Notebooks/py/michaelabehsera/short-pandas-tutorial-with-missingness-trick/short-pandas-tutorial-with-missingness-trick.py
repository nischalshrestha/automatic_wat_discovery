#!/usr/bin/env python
# coding: utf-8

# ## Short Pandas Tutorial with an Awesome Trick to improve Titanic Score

# I've been studying data science for the past two months and just wanted to post the main pandas functions that I currently use on a regular basis. Including some neat tricks that helped me improve my score with the Titanic dataset (indicating missingness below). I also wanted to get everyone's feedback on any suggestions, tips, and tricks with Pandas that you guys use regularly. Thank you!

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import warnings


# In[ ]:


train = pd.read_csv('../input/titanic-dataset/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


x = train[['Pclass', 'Name', 'Sex', 'Age', 'Cabin', 'Embarked']]
y = train['Survived']


# In[ ]:


train.head()


# In[ ]:


#Checking the tail of the dataset / You can add a number in the function to increase the amount of rows you could see
train.tail(10)


# In[ ]:


#To check how many columns and rows
train.shape


# In[ ]:


#To check the mean, standard deviation and other stats of your columns
train.describe()


# In[ ]:


train.describe(include=['O']) #for categorical data


# In[ ]:


#To check total null values in a set
train.isnull().sum()


# In[ ]:


#To check for unique values in a vector
train.Embarked.unique()


# In[ ]:


type(x)


# In[ ]:


#Craft new columns based on missinginess (Vastly improves your score with the Titanic dataset)
train['Has_Cabin'] = x['Has_Cabin'] = np.where(train['Cabin'].isnull(), 0, 1)


# In[ ]:


x.head()


# In[ ]:


#Calculating the mean
x['Age'].mean()


# In[ ]:


#Calculating median
x['Age'].median()


# In[ ]:


#Viewing corralation in the data
x.corr()


# In[ ]:


#Describe
x.describe()


# In[ ]:


#Imputation for categorical variables
x = pd.get_dummies(data=x, columns=['Sex'])


# In[ ]:


x.head()


# In[ ]:


#Getting a benchmark score for your dependent variable
train.Survived.value_counts() / train.Survived.count() * 100


# In[ ]:


#Filling a missing value based on the most popular categorical variable
x['Embarked'].fillna('Q', inplace=True)


# In[ ]:


x.isnull().sum()


# In[ ]:


#Fill missing values with mean or median
x['Age'].fillna(x['Age'].mean(), inplace=True)


# In[ ]:


x.isnull().sum()


# In[ ]:


#Drop a column from dataframe
x.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


x.head()


# In[ ]:




