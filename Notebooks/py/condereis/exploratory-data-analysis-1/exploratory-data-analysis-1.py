#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# # Exploratory Data Analysis
# 
# ## 1 - Setup
# 
# ### 1.1 - Import Packages

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'pylab inline')


# ### 1.2 - Load Data
# 
# #### VARIABLE DESCRIPTIONS:
# * survival:        Survival(0 = No; 1 = Yes)
# * pclass:          Passenger Class(1 = 1st; 2 = 2nd; 3 = 3rd)
# * name:            Name
# * sex:             Sex
# * age:             Age
# * sibsp:           Number of Siblings/Spouses Aboard
# * parch:           Number of Parents/Children Aboard
# * ticket:          Ticket Number
# * fare:            Passenger Fare
# * cabin:           Cabin
# * embarked:        Port of Embarkation(C = Cherbourg; Q = Queenstown; S = Southampton)

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
full_data = pd.concat([train_data, test_data]).reset_index(drop=True)
full_data['Pclass'] = full_data['Pclass'].astype('category')
full_data['Embarked'] = full_data['Embarked'].astype('category')
full_data['Sex'] = full_data['Sex'].astype('category')
full_data.tail()


# ### 1.2 - Variables Statistical Description
# 
# Generate various summary statistics, excluding NaN values. It's possible to notice that there are a lot of missing values for cabin and age and also a few for embarked and fare.

# In[ ]:


# Statistical description of the data
full_data.describe()


# In[ ]:


# Count how many NaN values there are in each column
len(full_data) - full_data.count()


# In[ ]:


# Passengers with missing values for Embarked and Fare.
full_data[full_data.drop(['Age','Cabin','Survived'], axis=1).isnull().any(axis=1)]


# ## 2 - Exploratory Data Analysis
# 
# ### 2.1 - Women and Children First
# 
# Let's first of all analyse the most obvious relationship, between sex, age and survival rate. The fist graph shows the strong relation between sex and survival rate. The relation is so strong that Kaggle sugest a simple model that returns 1 if the passenger is a female and 0 if it's a male. This model has a 76.6% accuracy.

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=full_data)


# When we divide the data by age groups, it's possible to notice, that the discrepancy between the survival rate of males and females almost desapers. The last plot shows the influence of both "Sex" and "Age" on the survival rate.

# In[ ]:


age_df = full_data[['Age','Survived', 'Sex']].copy()
age_df.loc[age_df.Age<15,'AgeGroup'] = 'Children'
age_df.loc[age_df.Age>=15,'AgeGroup'] = 'Adult'
sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=age_df)


# In[ ]:


sns.swarmplot(x='Age',y='Sex',hue='Survived',data=full_data)


# ### 2.2 - Do rich people survive?
# 
# As we could guess, the two histograms below confirm that people that payed more expencive fares, are more likely to survive.

# In[ ]:


p = plt.hist([full_data[(full_data.Survived==1)&(full_data.Fare<30)].Fare, 
              full_data[(full_data.Survived==0)&(full_data.Fare<30)].Fare], histtype='bar', stacked=True, bins=10)


# In[ ]:


p = plt.hist([full_data[(full_data.Survived==1)&(full_data.Fare>30)].Fare, 
              full_data[(full_data.Survived==0)&(full_data.Fare>30)].Fare], histtype='bar', stacked=True, bins=10)


# The following plot divides the data into 2 income groups. As we can see the chances of survival for both males and females increase if they had payed an expensive fare.

# In[ ]:


money_df = full_data[['Fare','Survived', 'Sex','Pclass']].copy()
money_df.loc[money_df.Fare>30,'FareLabel'] = 'Expensive'
money_df.loc[money_df.Fare<30,'FareLabel'] = 'Cheap'
sns.barplot(x='FareLabel', y='Survived', hue='Sex', data=money_df)


# But the plot below shows an even stronger relationship between the social class and the survival rate of passengers. It's possible to notice that meles traveling 1st class have a survival rate almost 2x larger them those traveling 2nd and 3rd class. For females traveling both, 1st and 2nd classes, have a survival rate almost 2x larger them those traveling 3rd class.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', hue='Sex', data=money_df)


# ### 2.3 - Family that travels together sinks together?
# 
# The plots below show that the chances of a female passenger surviving does not change if there are up to 3 relatives on board. For more them 3 relatives the chances drop dramatically. The chances for male passangers, however, increase as the number of relatives on board increses (also up to 3). For more them 3 relatives the chances of survival also drop dramatically.

# In[ ]:


family_df = full_data[['SibSp','Parch','Survived', 'Sex']].copy()
family_df.loc[:,'FamilySize'] =  family_df['SibSp'] + family_df['Parch'] +1
sns.barplot(x='FamilySize', y='Survived', hue='Sex', data=family_df)


# In[ ]:


family_df.loc[family_df.FamilySize==1,'FamilyLabel'] = 'Single'
family_df.loc[family_df.FamilySize==2,'FamilyLabel'] = 'Couple'
family_df.loc[(family_df.FamilySize>2)&(family_df.FamilySize<=4),'FamilyLabel'] = 'Small'
family_df.loc[family_df.FamilySize>4,'FamilyLabel'] = 'Big'
sns.barplot(x='FamilyLabel', y='Survived', hue='Sex', data=family_df, order=['Single', 'Couple', 'Small', 'Big'])


# The intend of this notebook is only make a superficial analysis of the most relevant features of the dataset. I have created another notebook to dig deeper on feature and model engineering. See you there ;)
