#!/usr/bin/env python
# coding: utf-8

# 

# ##1. Prepare

# ###1. 1 Import Libraries
# Import the python libraries used.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
sns.set(style='whitegrid', context='notebook', palette='Set2')
sns.despine()


# ### 1. 2 Load Data
# Loading the train set and test set , and combining them also.
# 
# The train set column excluding the **Survived** will has same columns with test set

# In[ ]:


# get train & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# combine train and test set
combine = pd.concat([test,train.drop('Survived',1)])
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
dataset
print("train matrix shape:",train.shape)
print("test matrix shape:",test.shape)
print("combine matrix shape:",combine.shape)


# ##2. Data understanding
# 

# ###2. 1 Start from Panorama
# Let's have a whole cognition and it would be a good understand with the data at the begining.

# ####2. 1. 1 Data Summary
# We will to know:
# 
#  - combine set has 11 columns totally excluding Survived column
#  - Age and Cabin column have more null value than others, while Fare column just one
#  - numerical columns : PassengerId, Pclass, Age, SibSp, Parch, Fare, Survived
#  - alphabetic string columns: Name, Sex, Ticket, Cabin, Embarked
# 

# In[ ]:


combine.info()
print('-'*40)
train.info()
print('-'*40)
test.info()


# ####2. 1. 2 More Describe

# Have a look at the value distribution of data set and the data value

# In[ ]:


combine.head(10)


# In[ ]:


combine.describe()


# In[ ]:


combine.describe()


# In[ ]:


combine.describe(include=['O'])


# ####2. 1. 3 The Unique Value of Per Column
# 

# In[ ]:


# To see the value of each column which without duplicated

max_num_of_val = 50
for column in combine:
    uni_val = combine[column].unique()
    print(column, ":")
    print(uni_val[:(len(uni_val)%max_num_of_val)], "\n")


# ####2. 1. 4 Clean Outlier
# After having a look at the overview, we would to find the outlier of data, and drop some that is too abnormal
# 

# In[ ]:





# In[ ]:


def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   


# ##2. 2 Check Feature One by One
# 
#  - We will move to understand the data of each column feature, the main idea will focus on the relation between feature with the feature **Survived**
#  - PassengerId is just a order number, and just feel free to pass it
# 

# ####2. 2. 1 Pclass Feature
# We get that the survived ratio among Pclass is  Pclass1 >  Pclass2 > Pclass3

# In[ ]:


# pclass feature survived ratio
pclass_survived = train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass', y='Survived', data=pclass_survived)


# ####2. 2. 2 Name Feature
# 
#  - the value of name is too dispersed to explore
#  
#  - **convert  and talk about later**
# 
# 

# In[ ]:


# pclass feature top 20 survived ratio
train[["Name", "Survived"]].groupby(['Name'], as_index=False).mean().sort_values(by='Survived', ascending=False).head(20)


# ####2. 2. 3 Sex Feature
# Easy to see female has higher Survived ratio.

# In[ ]:


# sex feature survived ratio
sex_survived = train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Sex', y='Survived', data=sex_survived)


# ####2. 2. 4 Age Feature
# 
#  - Like Name Feature, Age's unique item too many so need to be converted
#  - The elder and younger would has much change to live

# In[ ]:


# Age feature survived ratio
age_survived = train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# plot Age Survived Ratio distribution
fig, axis1 = plt.subplots(1,1,figsize=(18,8))
age_survived["Age"] = age_survived["Age"].astype(int)
sns.barplot(x='Age', y='Survived', data=age_survived)


# ####2. 2. 5 SibSp Feature
#  - Many SibSp equals 1 or 0, especially in male people , SibSp equals 0
#  - SibSp equals 0 has much none survived in male set
#  - **convert  and talk about later**
# 
# 

# In[ ]:


# SibSp feature survived ratio
sibsp_survived=train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='SibSp', y='Survived', data=sibsp_survived)


# It seems like SibSp equles1 or 2 has much higher survived ratio, let's see the plot with sex feature.
# 

# In[ ]:


# SibSp count plot with Sex
g = sns.FacetGrid(train, col="Sex", hue="Survived")
g.map(sns.countplot, "SibSp", alpha=.5)
g.add_legend()


# ####2. 2. 6  Parch Feature
# 
#  - Parch Feature performing  so like SibSp in the plot 
#  - Parch equals 0 has much none survived in male set
#  - Parch Feature is very similar like SibSp Feature
#  - **convert  and talk about later**
# 
# 

# In[ ]:


# Parch feature survived ratio
parch_survived = train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Parch', y='Survived', data=parch_survived)


# Like SibSp Feature, we compare the count of Parch with Sex Feature

# In[ ]:


# Parch count plot with Sex
g = sns.FacetGrid(train, col="Sex", hue="Survived")
g.map(sns.countplot, "Parch", alpha=.5)
g.add_legend()


# ####2. 2. 7 Fare Feature
# 
#  - In male set, high Fare mostly have not survived
#  - While female set  mostly the survived ratio is not influenced by the Fare Feature
# 
#  

# In[ ]:


# Fare stripplot with Sex and Pcalss
g = sns.FacetGrid(train, col="Sex", hue="Survived")
g.map(sns.stripplot, "Fare", "Pclass", alpha=.5)
g.add_legend()


# In[ ]:


# Fare feature survived ratio
fare_survived = train[["Fare", "Survived"]].groupby(["Fare"], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.lmplot(x="Fare", y="Survived", data=fare_survived)


# ####2. 2. 8 Embarked Feature

# In[ ]:


# Cabin feature survived ratio
embarked_survived = train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False).head(20)
sns.barplot(x='Embarked', y='Survived', data=embarked_survived)


# In[ ]:


# Parch count plot with Sex
g = sns.FacetGrid(train, col="Sex", hue="Survived")
g.map(sns.countplot, "Embarked", order=["C", "Q", "S"], alpha=.5)
g.add_legend()


# ##3. Feature Discussion
# we will copy a train data set to try convert and explore, and it will give some idea to the next section.

# In[ ]:


# copy train set
train_cp = train.copy()


# ###3. 1 Extract Title from Name

# In[ ]:


# extract title from name, and show title with sex by crosstable 

train_cp['Title'] = train_cp.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_cp['Title'], train_cp['Sex'])


# In[ ]:


# translate some title and show title survived ratio
train_cp['Title'] = train_cp['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_cp['Title'] = train_cp['Title'].replace('Mlle', 'Miss')
train_cp['Title'] = train_cp['Title'].replace('Ms', 'Miss')
train_cp['Title'] = train_cp['Title'].replace('Mme', 'Mrs')
    
# title survived ratio    
title_survived = train_cp[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.barplot(x='Title', y='Survived', data=title_survived)


# ###3. 2 Sum SibSp and Parch as FamilySize

# In[ ]:


# sum sibsp and parch as familysize and show survived ratio

# plus self so add 1 in the familysize
train_cp['FamilySize'] = train_cp['SibSp'] + train_cp['Parch'] + 1

fam_survived = train_cp[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
sns.barplot(x='FamilySize', y='Survived', data=fam_survived)


# In[ ]:


# FamilySize count plot with Sex and Pclass
g = sns.FacetGrid(train_cp, row="Sex", col="Pclass", hue="Survived")
g.map(sns.countplot, "FamilySize", order=[0,1,2,3,4,5,6], alpha=.5)
g.add_legend()


# ##4. Feature Conversion

# In[ ]:


# Pclass to dummies and merge to data set
train =pd.get_dummies(train, columns=['Pclass'], prefix='Pclass')
test =pd.get_dummies(test, columns=['Pclass'], prefix='Pclass')

# 


# In[ ]:


train


# In[ ]:





# In[ ]:




