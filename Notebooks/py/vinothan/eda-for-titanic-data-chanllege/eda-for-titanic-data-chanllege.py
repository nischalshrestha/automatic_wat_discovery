#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Introducation
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## Data Description
# 
# PassengerId which is just a running index and the indication whether this passenger survived (1) or not (0) we have the following information for each person:
# 
# Pclass is the Ticket-class: first (1), second (2), and third (3) class tickets were used. This is an ordinal integer feature.
# 
# Name is the name of the passenger. The names also contain titles and some persons might share the same surname; indicating family relations. We know that some titles can indicate a certain age group. For instance Master is a boy while Mr is a man. This feature is a character string of variable length but similar format.
# 
# Sex is an indicator whether the passenger was female or male. This is a categorical text string feature.
# 
# Age is the integer age of the passenger. There are NaN values in this column.
# 
# SibSp is another ordinal integer feature describing the number of siblings or spouses travelling with each passenger.
# 
# Parch is another ordinal integer features that gives the number of parents or children travelling with each passenger.
# 
# Ticket is a character string of variable length that gives the ticket number.
# 
# Fare is a float feature showing how much each passenger paid for their rather memorable journey.
# 
# Cabin gives the cabin number of each passenger. There are NaN in this column. This is another string feature.
# 
# Embarked shows the port of embarkation as a categorical character value.

# In[ ]:


train_df=pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df=pd.read_csv("../input/test.csv")


# In[ ]:


test_df.head(5)


#  ### Number of records and Features in the datasets
#  ** Let Examine the  Train  and Test DataSet**

# In[ ]:


print('---'*40)
print("The number of Features in  train dataset :",train_df.shape[1])
print("The number of Rows in Train dataset :",train_df.shape[0])
print('---'*40)
print('-----Test Dataset------------------------------')
print("The number of Features in  test dataset :",test_df.shape[1])
print("The number of Rows in  Test dataset :",test_df.shape[0])


# ## Identifying Numerical and Categorical Features
# ###  Function for  find out Numerical and categeical Variables

# In[ ]:


def type_features(data):
    categorical_features = data.select_dtypes(include = ["object"]).columns
    numerical_features = data.select_dtypes(exclude = ["object"]).columns
    print( "categorical_features :",categorical_features)
    print('-----'*40)
    print("numerical_features:",numerical_features)


# In[ ]:


print('Train_dataset')
print('````'*40)
type_features(train_df)


# In[ ]:


print('Test_dataset')
print('````'*40)
type_features(test_df)


# ## identifying the missing values

# In[ ]:


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms


# In[ ]:


missingdata(train_df)


# In[ ]:


print('------------------------------Test_Dataset--------------------------------------------')
missingdata(test_df)


# ## Checking the Imbalance of Target Variable

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(8,5))
train_df.Survived.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Distribution of target variable')
sns.countplot('Survived',data=train_df,ax=ax[1])
ax[1].set_title('Count of Survived VS Not Survived Passengers')
plt.show() 


# **It is evident that  many Passenger are unable to survived for the accident  i.e Only 38.4% of the total passenger are able to survived.
# We need to drill down more to get better insights from the data and see which categories of the passenger are  able to survied or not for the accident.**
# 
# We will try to check the survied  and dead rate by using the different features of the dataset. 
# Some of the features being sex,age,Pclass,etc. First let us understand the different types of features.

# ## Types Of Features
# 
# ### Categorical Features:
# A categorical variable is one that has two or more categories and each value in that feature can be categorised by them.For example, 
#  Sex  is a categorical variable having Two categories. Now we cannot sort or give any ordering to such variables. They are also known as Nominal Variables.
# 
# Categorical Features in the dataset: Sex
# 
# ### Analysing The Features
# ***SEX is a Categorical Feature***

# In[ ]:


def group_by(df,t1='',t2=''):
    a1=df.groupby([t1,t2])[t2].count()
    return a1


# In[ ]:


def plot_re(df,t1='',t2=''):
    f,ax=plt.subplots(1,2,figsize=(15,5))
    df[[t1,t2]].groupby([t1]).count().plot.bar(ax=ax[0])
    ax[0].set_title('count of passenger Based on  '+ t1)
    sns.countplot(t1,hue=t2,data=df,ax=ax[1])
    ax[1].set_title(t1 + ': Survived vs dead')
    a=plt.show()
    return a


# In[ ]:


train_df.columns


# In[ ]:


plot_re(train_df,'Sex','Survived')


# In[ ]:


group_by(train_df,'Sex','Survived')


# *** by looking above the given plot and group by fuction its clear proof that the female has more count of survived rate compare to male***

# ### Now create bin for age so that we can know which age of passenger count has maximum survived rate

# In[ ]:


train_df['Age_bin'] = pd.cut(train_df['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
train_df['Age_bin'].head()


# In[ ]:


group_by(train_df,'Age_bin','Survived')


# In[ ]:


plot_re(train_df,'Age_bin','Survived')


# In[ ]:


facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()


# ###  based on above given plot its clear the passenger  in between 0 to 18 has high count of survival and age between 20 to 40 also has high count of survival rate compare to other. 

# ## Analysis Name column
# Creating new feature extracting from existing
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.
# 
# Observations.
# 
# When we plot Title, Age, and Survived, we note the following observations.
# 
# Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# Decision.
# 
# We decide to retain the new Title feature for model training.
# 

# In[ ]:



train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train_df['Title'].value_counts()


# ### We can replace many titles with a more common name or classify them as Rare

# In[ ]:


train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


plot_re(train_df,'Title','Survived')


# #### Ordinal Features:
# An ordinal variable is similar to categorical values, but the difference between them is that we can have relative ordering or sorting between the values. For eg: If we have a feature like Height with values Tall, Medium, Short, then Height is a ordinal variable. Here we can have a relative sort in the variable.
# 
# Ordinal Features in the dataset:Pclass and Embarked

# In[ ]:


def or_plot(df,t1='',t2=''):
    f,ax=plt.subplots(1,2,figsize=(10,6))
    df[t1].value_counts().plot.bar(ax=ax[0],color='Green')
    ax[0].set_title('Number Of Passenger By '+t1)
    ax[0].set_xlabel("Score of :"+t1)
    ax[0].set_ylabel('Count')
    sns.countplot(t1,hue=t2,data=train_df,ax=ax[1],palette="spring")
    ax[1].set_title(t1+':Survived vs Dead')
    a=plt.show()
    return a


# In[ ]:


or_plot(train_df,'Pclass','Survived')


# In[ ]:


or_plot(train_df,'Embarked','Survived')


# ### based on above given plot it clear the the Pclass 1 has the high rate of survival and in Embarked c has high rate of survival compare with other.

# ## now we analysis with base Parch(Parents travel with passenger) SibSp(siblings travel with the passenger)

# In[ ]:


or_plot(train_df,'Parch','Survived')


# In[ ]:


or_plot(train_df,'SibSp','Survived')


# #### by look at the above given plot it clear that passenger travel with 1 siblings has high count of survival rate.

# In[ ]:


# Create new feature FamilySize as a combination of SibSp and Parch
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1


# In[ ]:


train_df['FamilySize'].value_counts()


# In[ ]:


or_plot(train_df,'FamilySize','Survived')


# In[ ]:


# Create new feature IsAlone from FamilySize
train_df['Alone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'Alone'] = 1


# In[ ]:


train_df.Alone.value_counts()


# In[ ]:


or_plot(train_df,'Alone','Survived')


# Correlating numerical and ordinal features
# We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# ### if look at the above given plot its is clear that the pclass 1 has high rate of survived and Pclass 3 has higest rate of dead.

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# 
# ### Correlation Between The Features

# In[ ]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(10,10))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':16}
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train_df)


# In[ ]:


# most correlated features
corrmat = train_df.corr()
top_corr_features = corrmat.index[abs(corrmat["Survived"])>=0.05]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="Oranges")


# ### Pairplots
# 
# Finally let us generate some pairplots to observe the distribution of data from one feature to the other. Once again we use Seaborn to help us.

# In[ ]:


g = sns.pairplot(train_df[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


# those are few EAD i have done please vote for me which help my movitvation to increase to do a lot of work if there any imporvment can be done means please say in comments
# References
# This notebook has been created based on great work done solving the Titanic competition and other sources.
# 
# A journey through Titanic
# Getting Started with Pandas: Kaggle's Titanic Competition
# Titanic Best Working Classifier

# In[ ]:




