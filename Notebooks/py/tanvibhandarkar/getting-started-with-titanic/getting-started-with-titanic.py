#!/usr/bin/env python
# coding: utf-8

# # Introduction to the notebook
# 
# Talk the truth and shit.

# ### Contents :
# 1. Data Loading
# 2. Data Exploration
# 3. Feature Enfineering
# 4. Applying Machine Learning
# 5. Selecting the best-fitted model
# 6. Submitting the file

# ## DATA LOADING
# *     Loding modules
# *    Loding Data
# *  Understanding the Data

# ## 1. Loading modules

# In[ ]:


# pandas, numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame

# matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## 2. Loading Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_test_data = [train, test]


# ## 3. Understanding Data

# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


print ('TRAINING DATA\n')
train.info()
print ("----------------------------------------\n")
print ('TESTING DATA\n')
test.info()


# Here we can evaluate that training data has missing values for *Age, Cabin, Embarked*  and testing data has missing values for *Age, Fare, Cabin, Embarked*.

# In[ ]:


train.describe(include=['O'])


# ## DATA EXPLORATION
# 
# Here we will see the relationship of various features with the *Survival*, as that is what we have to predict ultimately.
# * Pclass vs. Survival
# * Sex vs. Survival
# * Age vs. Survival
# * Embarked vs. Survival
# * Fare vs. Survival
# * Parch vs. Survival
# * SibSp vs. Survival

# In[ ]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]


# In[ ]:


sns.countplot(x='Survived',data=train)


# In[ ]:


print ("Survived: %i (%.1f%%)" %(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)" %(len(not_survived), float(len(not_survived))/len(train)*100.0))


# Nearly two-third of the population on Titanic has not survived.

# ### Pclass vs. Survival

# In[ ]:


train.Pclass.value_counts()


# In[ ]:


train.groupby('Pclass').Survived.value_counts()


# In[ ]:


train.groupby('Pclass').Survived.mean()


# In[ ]:


sns.factorplot(x="Pclass", y="Survived", data=train,size=5, kind="bar", palette="BuPu", aspect=1.3)


# **Observation** : People in First-Class had maximum chances of survival than second and third.

# ### Sex vs. Survival

# In[ ]:


train.Sex.value_counts()


# In[ ]:


train.groupby('Sex').Survived.value_counts()


# In[ ]:


train.groupby('Sex').Survived.mean()


# In[ ]:


sns.factorplot(x='Sex',y='Survived',data=train, size=5, palette='RdBu_r', ci=None, kind='bar', aspect=1.3)


# **Observation** : Female Passengers had greater chance of survival.

# ### Age vs. Survival

# In[ ]:


# Filling NaN values
for dataset in train_test_data:
    avg = dataset['Age'].mean()
    std = dataset['Age'].std()
    null_count = dataset['Age'].isnull().sum()
    random = np.random.randint(avg-std, avg+std, size=null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = random
    dataset['Age'] = dataset['Age'].astype(int)
    
train['FinalAge'] = pd.cut(train['Age'], 5)
print (train[['FinalAge', 'Survived']].groupby(['FinalAge'], as_index=False).mean())


# In[ ]:


age_survival = sns.FacetGrid(train, hue="Survived",aspect=4)
age_survival.map(sns.kdeplot,'Age',shade= True)
age_survival.set(xlim=(0, train['Age'].max()))
age_survival.add_legend()


# In[ ]:


sns.set_color_codes("deep")
fig , (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(17,5))
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, palette={0: "b", 1: "r"},split=True, ax=ax1)
sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train,palette={0: "b", 1: "r"}, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train,palette={0: "b", 1: "r"}, split=True, ax=ax3)


# **Observation** : From the available data, Children and middle-aged had greater chance of survival.

# ### Embarked vs. Survival

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


# As there are only 2 missing values, we will fill those by most occuring "S"
train['Embarked'] = train['Embarked'].fillna('S')
train.Embarked.value_counts()


# In[ ]:


train.groupby('Embarked').Survived.value_counts()


# In[ ]:


train.groupby('Embarked').Survived.mean()


# In[ ]:


sns.factorplot(x='Embarked', y='Survived', data=train, size=4, aspect=2.5)


# **Observation** : Those who embarked from C, survived the most.

# ### Fare vs. Survival 

# In[ ]:


# As there is one missing value in test data, fill it with the median.
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Convert the Fare to integer values
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

# Compute the Fare for Survived and Not Survived
fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived = train["Fare"][train["Survived"] == 1]

# Group up the Fare values
train['FinalFare'] = pd.qcut(train['Fare'], 4)
print (train[['FinalFare', 'Survived']].groupby(['FinalFare'], as_index=False).mean())


# In[ ]:


sns.factorplot(x="Survived", y="Fare", data=train,size=5, kind="bar", ci=None, palette="Set3", aspect=1.3)


# In[ ]:


train["Fare"][train["Survived"] == 1].plot(kind='hist', alpha=0.6, figsize=(15,3),bins=100, xlim=(0,60))
train["Fare"][train["Survived"] == 0].plot(kind='hist', alpha=0.4, figsize=(15,3),bins=100, xlim=(0,60), title='Fare of Survived(Blue) and Not Survived(Green)')


# **Observation**: There are more number of passengers with cheaper fare but their *Survival* rate is low.

# ### Parch vs. Survival

# In[ ]:


train.Parch.value_counts()


# In[ ]:


train.groupby('Parch').Survived.value_counts()


# In[ ]:


train.groupby('Parch').Survived.mean()


# In[ ]:


sns.barplot(x='Parch',y='Survived', data=train, ci=None)


# ### SibSp vs. Survival

# In[ ]:


train.SibSp.value_counts()


# In[ ]:


train.groupby('SibSp').Survived.value_counts()


# In[ ]:


train.groupby('SibSp').Survived.mean()


# In[ ]:


sns.barplot(x='SibSp', y='Survived', data=train, ci=None)


# In[ ]:


plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), square=True, annot=True, center=0)


# ## Feature Engineering

# In[ ]:


train.dtypes.index


# ### 1. PassengerId
# 
# It is not required in training dataset.  

# In[ ]:


del train['PassengerId']
train.head()


# ### 2.  Pclass

# ### 3. Name

# ### 4. Sex

# ### 5. Age

# ### 6. SibSp & Parch

# ###7

# Consider Parents Children(Parch) & Sibling Spouse (SibSp) as Family. Adding this will give *Family*.
