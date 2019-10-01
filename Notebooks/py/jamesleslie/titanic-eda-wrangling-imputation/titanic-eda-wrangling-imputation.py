#!/usr/bin/env python
# coding: utf-8

# # Titanic challenge part 1
# In this notebook, we will be covering all of the steps required to wrangle the Titanic data set into a format that is suitable for machine learning.   
# We will do each of the following:
#   - impute missing values
#   - create new features (feature engineering)
#   
# [**Part 2**](https://www.kaggle.com/jamesleslie/titanic-random-forest-grid-search) of this challenge involves fitting and tuning a **random forest** to make predictions.

# # Table of Contents:
# 
# - **1. [Load Packages and Data](#loading)**
# - **2. [Imputation](#impute-missing)**
#   - **2.1. [Age](#age)**
#   - **2.1. [Fare](#fare)**
#   - **2.1. [Embarked](#embarked)**
# - **3. [Feature engineering](#feature-engineering)**

# <a id="loading"></a>
# # 1. Load packages and data
# First step, as always, is to import the necessary Python packages and load the input data as a Pandas dataframe.
# 
# I chose to combine the train and test set into one. Since we will have to impute some missing age and fare values, I prefer to do this across the entire dataset, rather than separately across train and test sets. 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

get_ipython().magic(u'matplotlib inline')
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='muted',
        rc={'figure.figsize': (12,8)})

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("../input"))


# In[ ]:


# Load data as Pandas dataframe
train = pd.read_csv('../input/train.csv', )
test = pd.read_csv('../input/test.csv')
df = pd.concat([train, test], axis=0, sort=True)


# In[ ]:


df.head()


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

        
display_all(df.describe(include='all').T)


# In[ ]:


df['Survived'].value_counts()


# <a id="impute-missing"></a>
# # 2. Imputation 
# We can see above that there are a few columns with missing values. The `Cabin` column is missing over 1000 values, so we won't use that for predictions, but the `Age`, `Embarked` and `Fare` columns are all complete enough that we can fill in the missing values through imputation.   
# <a id="age"></a>
# ## 2.1. Impute missing age values
# A simple option for the missing age values is to use the median age value. Let's go a little further and use each passenger's *Title* to estimate their age. E.g. if a passenger has the title of *Dr*, I will give them the median age value for all other passengers with the same title.

# ### Extract title from name
# We can use a regular expression to extract the title from the `Name` column. We will do this by finding the adjacent letters that are immediately followed by a full stop.
# 

# In[ ]:


# create new Title column
df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)


# In[ ]:


df.head()


# ### Use only the most common titles
# Let's take a look at the unique titles across all passengers:

# In[ ]:


df['Title'].value_counts()


# As we can see above, there are quite a few different titles. However, many of these titles are just French versions of the more common English titles, e.g. Mme = Madame = Mrs.   
# 
# We will use the six most common titles, replacing all other titles with the most appropriate of these six.

# In[ ]:


# replace rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
df.replace({'Title': mapping}, inplace=True)


# In[ ]:


# confirm that we are left with just six values
df['Title'].value_counts()


# ### Use median of title group
# Now, for each missing age value, we will impute the age using the median age for all people with the same title.

# In[ ]:


# impute missing Age values using median of Title groups
title_ages = dict(df.groupby('Title')['Age'].median())

# create a column of the average ages
df['age_med'] = df['Title'].apply(lambda x: title_ages[x])

# replace all missing ages with the value in this column
df['Age'].fillna(df['age_med'], inplace=True, )
del df['age_med']


# We can visualize the median ages for each title group. Below, we see that each title has a distinctly different median age. 
# > **Note**: There is no risk in doing this after imputation, as the median of an age group has not been affected by our actions.

# In[ ]:


sns.barplot(x='Title', y='Age', data=df, estimator=np.median, ci=None, palette='Blues_d')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.countplot(x='Title', data=df, palette='hls', hue='Survived')
plt.xticks(rotation=45)
plt.show()


# <a id="fare"></a>
# ## 2.2. Impute missing fare values
# For the single missing fare value, I also use the median fare value for the passenger's class.   
# 
# > Perhaps you could come up with a cooler way of visualising the relationship between the price a passenger paid for their ticket and their chances of survival?

# In[ ]:


sns.swarmplot(x='Sex', y='Fare', hue='Survived', data=df)
plt.show()


# In[ ]:


# impute missing Fare values using median of Pclass groups
class_fares = dict(df.groupby('Pclass')['Fare'].median())

# create a column of the average fares
df['fare_med'] = df['Pclass'].apply(lambda x: class_fares[x])

# replace all missing fares with the value in this column
df['Fare'].fillna(df['fare_med'], inplace=True, )
del df['fare_med']


# <a id="embarked"></a>
# ## 2.3. Impute missing "embarked" value
# There are also just two missing values in the `Embarked` column. Here we will just use the Pandas 'backfill' method.
# 

# In[ ]:


sns.catplot(x='Embarked', y='Survived', data=df,
            kind='bar', palette='muted', ci=None)
plt.show()


# In[ ]:


df['Embarked'].fillna(method='backfill', inplace=True)


# <a id="feature-engineering"></a>
# # 3. Add family size column
# We can use the two variables of **Parch** and **SibSp** to create a new variable called **Family_Size**. This is simply done by adding `Parch` and `SibSp` together.

# In[ ]:


# create Family_Size column (Parch +)
df['Family_Size'] = df['Parch'] + df['SibSp']


# In[ ]:


display_all(df.describe(include='all').T)


# # 4. Save cleaned version
# Finally, let's save our cleaned data set so we can use it in other notebooks.

# In[ ]:


train = df[pd.notnull(df['Survived'])]
test = df[pd.isnull(df['Survived'])]


# In[ ]:


train.to_csv('train_clean.csv', index=False)
test.to_csv('test_clean.csv', index=False)

