#!/usr/bin/env python
# coding: utf-8

# <h1> INTRODUCTION </h1>
# Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning of 15 April 1912, after colliding with an iceberg during her maiden voyage from Southampton to New York City.
# 
# The Titanic: Machine Learning from Disaster dataset consists of passenger information like their gender and class. Using the given training and test set, the goal is to predict if a passenger survives. 
# 
# The findings will be able to predict the survival of the people who were aboard the titanic and for future accidents which might occur on the same scenario.

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk # for machine learning
import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
import seaborn as sns

import matplotlib.pyplot as plt
import collections as co
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# <h1> Training Data Set </h1>
# We will use this for building the machine learning models, to provide outcome for each passengers.

# In[22]:


train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )


# <h1> Test Data Set</h1>
# This data will be used by the model to predict whether they survived or not.

# In[23]:


test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
test_data


# <h1> Describing the Data Set </h1>
# 

# In[24]:


test_data.describe()


# In[25]:


train_data.describe()


# <h1> Modeling the data </h1>
# <h3> Distribution of each category </h3>

# In[26]:


fig1, ax = plt.subplots()
ax.pie(train_data.Survived.value_counts(),explode = (0.1, 0.1), labels=(0, 1), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

survivalRate = plt.title("Distribution of Survival, (1 = Survived)")
plt.show()


# In[27]:


fig1, ax = plt.subplots()
ax.pie(train_data.Pclass.value_counts(), explode = (0.1, 0.1, 0.1), labels=('Class 3', 'Class 1', 'Class 2'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

classes = plt.title("Distribution of Classes, (Pclass)")
plt.show()


# In[28]:


sns.kdeplot(train_data.Age.dropna(), shade=True, label='Age')
plt.axvline(train_data.Age.dropna().median(), label='Median', ls='dashed')
plt.axvline(train_data.Age.dropna().mean(), label='Mean', ls='dashed')
plt.legend()
ageDistribution = plt.title("Distribution of Age, (Age)")


# In[29]:


fig1, ax = plt.subplots()
ax.pie(train_data.Sex.value_counts(), explode=[0.1, 0.1],  labels=('male', 'female'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

genderPie = plt.title("Distribution of Gender, (Sex)")
plt.show()


# In[30]:


fig1, ax = plt.subplots()
ax.pie(train_data.Embarked.value_counts(), explode=[0.1, 0.1, 0.1],  labels=('S', 'C', 'Q'), autopct='%1.1f%%',
        shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

embarked = plt.title("Distribution of Embarked, (Embarked)")
plt.show()


# In[31]:


sns.kdeplot(train_data.Fare[train_data.Pclass==1].apply(lambda x: 80 if x>80 else x), shade=True, label='1st Class')
sns.kdeplot(train_data.Fare[train_data.Pclass==2].apply(lambda x: 80 if x>80 else x), shade=True, label='2nd Class')
sns.kdeplot(train_data.Fare[train_data.Pclass==3].apply(lambda x: 80 if x>80 else x), shade=True, label='3rd Class')
plt.axvline(train_data.Fare.median(), label='Median', ls='dashed')
plt.axvline(train_data.Fare.mean(), label='Mean', ls='dashed')
plt.legend()
fare = plt.title("Fare Distribution by Class, (Fare)")


# In[32]:


sns.kdeplot(train_data.Age[train_data.Sex=='male'].dropna(), shade=True, label='Male')
sns.kdeplot(train_data.Age[train_data.Sex=='female'].dropna(), shade=True, label='Female')
plt.legend()
agesex = plt.title("Age Distribution by Sex")


# In[33]:


train_data.SibSp.value_counts().plot(kind='bar', color = 'blue', alpha=0.55)
train_data.Parch.value_counts().plot(kind='bar', color = 'yellow', alpha=0.55)
plt.legend(labels = ('No. Of Siblings and Spouses (SibSp)', 'No. of Parents and Children (Parch)'))
siblings = plt.title("Number of Siblings, Spouses, Parents and Children")


# In[34]:


first = plt.subplot2grid((20,30),(0,6),rowspan=10,colspan=3)
train_data.Survived[train_data.Pclass==1].value_counts().sort_index().plot(color = 'red', kind='bar', alpha=0.85, label='1st Class')
plt.title("Survival in 1st Class")

second = plt.subplot2grid((20,30),(0,14),rowspan=10,colspan=3)
train_data.Survived[train_data.Pclass==2].value_counts().sort_index().plot(color = 'blue', kind='bar', alpha=0.85, label='2nd Class')
plt.title("Survival in 2nd Class")

third = plt.subplot2grid((20,30),(0,22),rowspan=10,colspan=3)              
train_data.Survived[train_data.Pclass==3].value_counts().sort_index().plot(color = 'green', kind='bar', alpha=0.85, label='3rd Class')
plt.title("Survival in 3rd Class")
plt.show()


# In[35]:


plt.bar(np.array([0,1])-0.25, train_data.Survived[train_data.Sex=='male'].value_counts().sort_index(), width=0.25, label='Male',alpha=0.85)
plt.bar(np.array([0,1]), train_data.Survived[train_data.Sex=='female'].value_counts().sort_index(), width=0.25, label='Female',alpha=0.85)
plt.xticks(np.arange(0, 2, 1))
plt.legend()
sexSurvived = plt.title("Survival By Gender (Sex, Survived = 1)")


# In[36]:


sns.kdeplot(train_data.Age[train_data.Survived==0].dropna(), shade=True, label='Died')
sns.kdeplot(train_data.Age[train_data.Survived==1].dropna(), shade=True, label='Survived')
plt.legend()
ageSurvival = plt.title("Survival By Age")


# In[37]:


embarkedC = plt.subplot2grid((20,30),(0,6),rowspan=10,colspan=5)
plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==0], color = 'red', label='')
plt.title("Survival By Embarked (C)")

embarkedQ = plt.subplot2grid((20,30),(0,16),rowspan=10,colspan=5)
plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==1], color = 'green',label='')
plt.title("Survival By Embarked (Q)")

embarkedS = plt.subplot2grid((20,30),(0,26),rowspan=10,colspan=5) 
plt.hist(train_data.Survived[pd.Categorical(train_data.Embarked).codes==2], color = 'yellow', label='')
embarkedS = plt.title("Survival By Embarked (S)")



# In[38]:


train_data['FamilySize'] = train_data.SibSp + train_data.Parch;
sns.kdeplot(train_data.FamilySize[train_data.Survived==0],shade=True,label='Died');
sns.kdeplot(train_data.FamilySize[train_data.Survived==1],shade=True,label='Survived');
plt.title('Survival By Family Size');
plt.legend();
plt.show();


# In[75]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

train_data['Title'] = train_data["Name"].apply(get_title);
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona":9, "Lady": 9, "Countess": 9, "Jonkheer": 9, "Sir": 7, "Capt": 7, "Ms": 2}
train_data["TitleCat"] = train_data.loc[:,'Title'].map(title_mapping);

f, ax1 = plt.subplots(1, figsize=(10,5))

# Set the bar width
bar_width = 0.75

bar_l = [i+1 for i in range(len(train_data.TitleCat[(train_data.Survived==1)].value_counts()))]
tick_pos = [i+(bar_width/2) for i in bar_l]

ax1.bar(bar_l,
        train_data.TitleCat[(train_data.Survived==0)].value_counts(),
        width=bar_width,
        label='Pre Score',
        alpha=0.5,
        color='#F4561D')

ax1.bar(bar_l,
        train_data.TitleCat[(train_data.Survived==1)].value_counts(),
        width=bar_width,
        label='Mid Score',
        alpha=0.5,
        color='#F1911E')

# set the x ticks with names

# Set the label and legends
ax1.set_ylabel("Total Score")
ax1.set_xlabel("Test Subject")
plt.legend(loc='upper right')

# Set a buffer around the edge
plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

print("LEGEND")
print("[1] - Mr")
print("[2] - Miss, Ms")
print("[3] - Master")
print("[4] - Doctor")
print("[5] - Reverent")
print("[6] - Major")
print("[7] - Colonel, Don, Sir, Captain")
print("[8] - Mme, Mlle")
print("[9] - Dona, Lady, Countess, Jonkheer")
plt.show();


# In[ ]:




