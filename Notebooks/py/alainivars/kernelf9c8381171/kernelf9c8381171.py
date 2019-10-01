#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Alain Ivars](http://highfeature.com). Source and license info is on [GitHub]
# (https://github.com/alainivars/data-science-notebooks)

# # Kaggle Competition: Titanic: Machine Learning from Disaster

# ## Introduction

# * First a big Thanks to [Harrison Sentdex](http://pythonprogramming.net) for the Free Machine learning course online.
# * An other very important thing for me; if you know some way to get the same result of a block, dont hesitate to let me know, by submit a poll request on Github, any share is very wellcome, Thanks.

# * Competition Site
# * Competition Description
# * More informations
# * Setup Imports and Variables
# * Explore the Data
# * Feature: Passenger Classes
# * Feature: Sex
# * Feature: Embarked
# * Feature: Age
# * Feature: Family Size
# * Final Data Preparation for Machine Learning
# * Data Wrangling Summary
# * Random Forest: Training
# * Random Forest: Predicting
# * Random Forest: Prepare for Kaggle Submission
# * Support Vector Machine: Training
# * Support Vector Machine: Predicting

# ## Competition Site

# More information about the chalenge [here!](https://www.kaggle.com/c/titanic)

# ## Competition Description

# ![alt text](http://free.bridal-shower-themes.com/img/p/r/printable-pictures-of-the-titanic_1.jpg)

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## More informations

# For more informations about the chalenge, [its evaluation](https://www.kaggle.com/c/titanic#evaluation) and download go [here!](https://www.kaggle.com/c/titanic/data)

# ## Setup Imports and Variables

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def figure_size_small():
    """Set the size of matplotlib figures"""
    plt.rc('figure', figsize=(10, 5))

def figure_size_big():
    """Set the global default size of matplotlib figures"""
    plt.rc('figure', figsize=(15, 10))

# Set the global default size of matplotlib figures
figure_size_small()

# graph type
kind_type = ['bar', 'barh', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie']


# ## Explore the Data to find a strategy

# ### Load data and view some statistics and certainly also missing data

# In[ ]:


'''Read the data:'''
df_train = pd.read_csv('../input/train.csv')


# Data are now loaded in pandas DataFrame. First see what kind of data we have

# In[ ]:


'''View the data types of each column:'''
df_train.dtypes


# Well numeric in big part, but some in other format and from what I know of "deep learning", it like  only digital data. If you know some liking no-digital format link it me please.
# What are inside?

# In[ ]:


df_train.count()


# Data missing for :
# 

# In[ ]:


(df_train.count()['PassengerId'] - df_train.count()['Age'], "Age", 
df_train.count()['PassengerId'] - df_train.count()['Cabin'], "Cabin", 
df_train.count()['PassengerId'] - df_train.count()['Embarked'], "Embarked")


# Some missing data certainly visible in the head

# In[ ]:


'''Head data (first 5 records)'''
df_train.head()


# Yes in 'Cabin', now we look in the tail

# In[ ]:


'''tail data (last 5 records)'''
df_train.tail()


# And here in 'Age' also.
# 
# Well, some label (field) will not be useful to try to discover who will survive, like:
# - PassagerId
# - Name
# 
# and certainly 
# - Embarked
# - Cabin : because right now we dont have the cabin map of the boat
# 
# For all other, we should verify they coerences

# ### Replace Nan and empty value with some more usefull value for statistic

# In[ ]:


'''replace Age = Nan by, Age = 130'''
df_train['Age'] = df_train['Age'].fillna(np.float64(130))
'''replace Cabin empty by, Cabin = Alpha'''
df_train['Cabin'] = df_train['Cabin'].fillna('Alpha')


# In[ ]:


df_train.count()


# ### Now let go to verify some intuitions

# #### Survived by Class

# In[ ]:


crosstab_pclass_survived = pd.crosstab(df_train['Pclass'], df_train['Survived'], normalize='index')
crosstab_pclass_survived


# In[ ]:


figure_size_small()
kind=kind_type[0]
crosstab_pclass_survived.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Class')
plt.xlabel('Class')
plt.ylabel('Survival Rate')
plt.show()


# Obviously, the best class to survive is in order: first, second and third.

# #### Survived by Cabin

# Let us now see if 'Cabin' with the data we have (no plan of cabits on the steamer), teaches us something exploitable:

# In[ ]:


kind=kind_type[0]
df_crosstab = pd.crosstab(df_train['Cabin'], df_train['Survived'])
df_crosstab
# df_crosstab.plot(kind=kind, stacked=True)
# plt.show()


# Nothing is obvious, except the 'cabin' Alpha (passenger without cabin) which at a general rate of survival of 2 out of 5.

# #### Age by Class

# In[ ]:


kind=kind_type[3]
class_list = sorted(df_train['Pclass'].unique())
for pclass in class_list:
    df_train.Age[df_train.Pclass == pclass].plot(kind=kind)
plt.title('Age Density Plot by Passenger Class')
plt.xlabel('Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')
plt.show()


# Also obvious, third class is the younger, follow by second and first class are the older

# #### Survived by Class and Age

# In[ ]:


kind=kind_type[4]
df_train_norm1 = pd.crosstab([df_train['Age'],df_train['Pclass']], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Age/Class')
plt.xlabel('Age/Class')
plt.ylabel('Survival Rate')
plt.show()


# Here nothing clear, let's try another point of view.

# In[ ]:


#kind_type = ['bar', 'barh', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie']
kind=kind_type[4]
df_train_norm1 = pd.crosstab([df_train['Pclass'],df_train['Age']], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Class/Age')
plt.xlabel('Class/Age')
plt.ylabel('Survival Rate')
plt.show()


# Not bad, but nothing very clear, let's try another point of view.

# #### Survived by Gender, Class and Age

# In[ ]:


kind=kind_type[4]
df_train_norm1 = pd.crosstab([df_train['Sex'],df_train['Pclass'],df_train['Age']], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=True)
plt.title('Survival Rate by Sex/Class/Age')
plt.xlabel('Sex/Class/Age')
plt.ylabel('Survival Rate')
plt.show()


# The graph is much clearer, it was better to be young, female and first class to survive.
# 
# Let's try to confirm this with other graphs

# In[ ]:


ranges_age = [
    0, 5, 10, 15,   # childs
    25, 35, 45, 55,  # Adult
    65, 70, 80, 120,  # Older
    600]  # unknow Age
group_by_age = pd.cut(df_train["Age"], ranges_age)
#group_by_age


# In[ ]:


kind=kind_type[0]
df_train_norm1 = pd.crosstab([df_train['Pclass'],df_train['Sex'],group_by_age], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=False)
plt.title('Survival Rate by Class/Sex/Age')
plt.xlabel('Sex/Class/Age')
plt.ylabel('Survival Rate')
plt.show()


# In[ ]:


figure_size_big()
kind=kind_type[0]
df_train_norm1 = pd.crosstab([df_train['Sex'],df_train['Pclass'],group_by_age], df_train['Survived'], normalize='index')
df_train_norm1.plot(kind=kind, stacked=False)
plt.title('Survival Rate by Sex/Class/Age')
plt.xlabel('Sex/Class/Age')
plt.ylabel('Survival Rate')
plt.show()


# This is now confirming, with one exception however the third class males have a higher survival rate than those of 2nd class. There is certainly a lack of data sources such as bridges and access plans to better understand this exception.

# #### Draft unsorted of analyses

# In[ ]:


pd.crosstab([df_train['Sex'],df_train['Pclass'],group_by_age], df_train['Survived'], normalize='index')


# In[ ]:


ranges_age = [
    0, 5, 10, 15,   # childs
    25, 35, 45, 55,  # Adult
    65, 70, 80, 120,  # Older
    600]  # unknow Age
group_by_age = pd.cut(df_train["Age"], ranges_age)
# group_by_age
# age_grouping = df_train.groupby(group_by_age).mean()
# age_grouping['Survived'].plot.bar()
# plt.show()
data_fit = pd.crosstab(
    index=[
        df_train['Sex'],
        df_train['Pclass'],
        group_by_age
    ],
    columns=df_train['Survived'],
    rownames=['Sex', 'Pclass', 'Age'],
    colnames=['Survived'],
    normalize='index'
)
print(data_fit)


# In[ ]:


df_train['Survived'].value_counts()


# In[ ]:


total = df_train['Survived'].value_counts()
dead = total[0] / (total[0] + total[1])
survived = total[1] / (total[0] + total[1])
survived, dead


# In[ ]:




