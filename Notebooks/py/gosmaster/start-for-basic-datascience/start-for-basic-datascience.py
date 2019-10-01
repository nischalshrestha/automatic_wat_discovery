#!/usr/bin/env python
# coding: utf-8

# 

# #start from novice
# the start map:
# 1.data preprocessing( cleaning/transformation/reduction/discretization/text cleaning)
# -try to explore data:
# ++input: raw data
# ++output: good structure data
# 2.learning
# 3.evaluation
# #PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked

# 

# import pandas as pd
# # visualization
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# #1.preprocessing
# titanic_raw_train= pd.read_csv("../input/train.csv")
# #titanic_raw_test= pd.read_csv("../input/test.csv")
# #titanic_raw_train.head()
# #titanic_raw_train.tail()
# #titanic_raw_train.index
# #titanic_raw_train.describe()
# #describe() show that Age values are 714, it have some missing data?
# #titanic_raw_train.index.values
# #titanic_raw_train.sort_values(by='Age')
# #clean data func
# def cleandata(rawdata):
#     cleandata= rawdata.head()
#     return cleandata
# #tempdata= cleandata(titanic_raw_train)
# #tempdata
# titanic_raw_train_copy= titanic_raw_train
# #titanic_raw_train_cut_age_na= titanic_raw_train_copy['Age'].dropna()
# #titanic_raw_train_cut_age_na.head()
# #titanic_raw_train_cut_age= titanic_raw_train_copy.drop(titanic_raw_train_copy[titanic_raw_train_copy['Age'] =='NaN'])
# #titanic_raw_train_cut_age
# #titanic_raw_train_copy.duplicated()
# #titanic_raw_train_copy.drop_duplicates()
# #titanic_raw_train_copy.columns.values
# #titanic_raw_train_copy.info()
# #titanic_raw_train_copy.info()
# #g = sns.FacetGrid(titanic_raw_train_copy, col='Survived')
# #g.map(plt.hist, 'Age', bins=20)
# titanic_raw_train_copy.shape

# data exploratiion

# In[ ]:


import pandas as pd
titanic_raw_train= pd.read_csv("../input/train.csv")
titanic_raw_train.head()


# In[ ]:


titanic_raw_train.shape


# In[ ]:


import numpy as np
titanic_raw_train.describe()


# In[ ]:


##bayes theorem
n_rows= titanic_raw_train.shape[0]
#(titanic_raw_train['Survived']==1).sum()
p_survived= (titanic_raw_train.Survived==1).sum() / n_rows
p_not_survived = 1 - p_survived
#p_survived
#titanic_raw_train.Sex.isnull().sum()
p_male= (titanic_raw_train.Sex=="male").sum() / n_rows
p_female= 1 - p_male


# In[ ]:


## do the survival rate have a certain gender affect??? --> P(Survived|Female)=P(Female and Survived)/P(Female)
n_women= (titanic_raw_train.Sex=="female").sum()
survived_women=titanic_raw_train[(titanic_raw_train.Sex=="female") & (titanic_raw_train.Survived==1)].shape[0]
p_survived_women= survived_women / n_women
#p_survived_given_women= p_survived_women / p_female
p_survived_women


# In[ ]:


#

