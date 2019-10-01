#!/usr/bin/env python
# coding: utf-8

# ### Importing required packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings("ignore")

import os
os.listdir("../input")


# ### Loading dataset

# In[ ]:


data = pd.read_csv("../input/train.csv")
data.head()


# ### Data types

# In[ ]:


print(data.dtypes)


# We can see that numbers are represted as int or float in this dataset and data type conversion is not needed here.

# ### Proportion of target (Survived)

# In[ ]:


print("Total count:", len(data))
print()
print(round((data.Survived.value_counts()/len(data)) * 100,2))


# This dataset has a decent proportion of target class and it is not skewed to any one.

# ### Unique Values

# In[ ]:


def get_uniquevals(df):
    print("-"*40)
    for col in df.columns:
        if len(df[col].unique()) <= 10:
            print("{} - Unqiue Values:".format(df[col].name))
            print(df[col].unique())
            print()
            print("{} - # of occurences of each values:".format(df[col].name))
            print(df[col].value_counts())
        else:
            print("{} has {} unqiue values:".format(df[col].name,len(df[col].unique())))
        print("-"*40)


# In[ ]:


get_uniquevals(data)


# Pclass, Sex, SibSp, Parch and Embarked are having **less distinct values** and they can be converted to numeric values 

# ### Null Values

# In[ ]:


def getnullcounts(df):
    print("-"*20)
    non_nullcols = []
    for col in df.columns:
        if df[col].isna().sum() > 0:
            print("{} : {}".format(df[col].name, df[col].isna().sum()))
        else:
            non_nullcols.append(df[col].name)
    print("-"*20)
    print('Non-null features:\n',', '.join(non_nullcols))
    print("-"*20)


# In[ ]:


getnullcounts(data)


# Age & Embarked has null values which should be imputed before prediction

# ### Feature Elimination

# In[ ]:


def feature_elimination(df):
    print('Features to be considered for elimiation:')
    for col in df.columns:
        if len(df[col].unique()) == (len(df)) and df[col].dtype != 'object':
            print(df[col].name)
        if len(df[col].unique()) > (len(df)*0.50) and df[col].dtype == 'object':
            print(df[col].name)


# In[ ]:


feature_elimination(data)


# Note: These are suggestions as the number of distinct values are high. Care should be taken before elimination.
# for example "Name" can be used to create a feature called "title" which can be used for prediction

# ### Visual Exploration

# In[ ]:


f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x='Survived', y="Age",  data=data);


# Very less chances for age > 60+ to survive (except some outliers)

# In[ ]:


f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x="Sex", y="Age", hue="Survived", data=data);


# Gives clarity to the above finding that Male has very less chances for age > 60+

# In[ ]:


f, ax = plt.subplots(figsize=(7,3))
sns.barplot(x='Sex', y="Survived",  data=data);


# Number of female passengers survived is more than male passengers

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=data);


# Passenger Class 1 has high survival rate

# In[ ]:


sns.barplot(x="Pclass", y="Survived",hue="Sex", data=data);


# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=data);


# More the siblings less the survival chance

# In[ ]:


sns.barplot(x="Parch", y="Survived", data=data);


# In[ ]:


data["family"] = data["SibSp"] + data["Parch"]
data["occumpanied"] = data["family"].apply(lambda x: 0 if x == 0 else 1)
sns.barplot(x="Survived", y="occumpanied", data=data);


# Those who are occumpanied by a family member (elder or siblings) had high survival rate

# In[ ]:


sns.distplot(data['Age'].dropna());


# Passengers aged between 18-38 had high survival rate compared to others

# In[ ]:


survived = data.loc[data['Survived']==1,"Age"].dropna()
sns.distplot(survived)
plt.title("Survived");


# In[ ]:


not_survived = data.loc[data['Survived']==0,"Age"].dropna()
sns.distplot(not_survived)
plt.title("Not Survived");


# Infants had high survival rate and elderly passengers above 65+ were less likely to survive

# In[ ]:


sns.pairplot(data.dropna());

