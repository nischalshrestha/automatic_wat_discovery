#!/usr/bin/env python
# coding: utf-8

# # **Competition Description**
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# In this article I have more focussed on exploratory data analysis.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


numerical_features = data.select_dtypes(include=[np.number])
numerical_features.columns


# # Univariate Analysis
#  - Univariate Analysis is plays vital role to understand each features values statistics (mean, median, standard deviation), pattern, misisng values, outliers presence. It is always recommended to undergo Univariate analysis.

# In[ ]:


data.describe()


# In[ ]:


data.drop('PassengerId', axis = 1).hist(figsize=(30,20), layout=(4,3))
plt.plot()


# In[ ]:


skew_values = skew(data[numerical_features.columns], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features.columns), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1,) 


# For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to 0, statistically speaking.
# 
# - Fare, parch and SibSp are high positively skewed
# - Pclass is negatively skewed
# 
# ##### Feature Type Classification
# - Numerical - Age, Fare, SibSp, Parch
# - Categorical - Survival, Pclass, Sex, Parch, Cabin, Embarked
#     - Nominal - Sex, Cabin, Embarked
#     - Ordinal - Pclass
#     - Interval - 

# In[ ]:


missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = missing_values/len(data)
combine_data = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values count', 'Percentage'])
pd.pivot_table(combine_data, index=combine_data.index,margins=True ) 


# # Bivariate Analysis
# - Understanding one features stats wiht respect to other features

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(data.drop('PassengerId', axis = 1).corr(), square=True, annot=True, vmax= 1,robust=True, yticklabels=1)
plt.show()


# - Survived is good related with Pclass and Fare
# - Pclass is enough correlated with Fare
# - Age with SbSP and Pclass
# - SbSP with Parch

# # Outliers detection and removal

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(data = data.drop('PassengerId', axis = 1))
plt.show()


# In[ ]:


# Let's see survival and Fare relation
var = 'Fare'
plt.scatter(x = data[var], y = data['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()


# - Scatter plot is good way to visualise data sparsity on general.
# - Fare vs.  Survived shows few points are loosly available
# - Let's plot box plot to see outliers in detail

# In[ ]:


plt.figure(figsize=(20,5))
sns.boxplot(x =data[var])
plt.show()


# In[ ]:


data.drop(data[data['Fare']> 100].index,  inplace= True)


# # Missing Values Treatment
# 
# - Cabin has >70 % missing values, hence, it's better to drop this column than fill up
# - Age has -37 % correlation with Pclass, while Pclass has -34% correlation with Survived
# - As Age and Pclass is correlated, it is better to remove one of them
# - AS Age has just -0.077% correlated with Survived, remove Age 
# 
# - Hence, in this problem missing values are handled by dropping columns (Age and Cabin) and droping rows

# In[ ]:


data.drop(['Age', 'Cabin'], axis = 1, inplace=True)


# In[ ]:


data.dropna(inplace=True)


# # Categorical Features Encoding
# - Name, Sex, Ticket, Embarked are categorical features
# - Name and Ticket are not relevant features (drop them)
# - Encode Sex and Embarked into numbers using inbuilt functions
# - Using dummy encoding is best option here

# In[ ]:


categorical_features = data.select_dtypes(include=[np.object])
categorical_features.columns


# In[ ]:


print('Sex has {} unique values: {}'.format(len(data.Sex.unique()),data.Sex.unique()))
print('Embarked has {} unique values: {}'.format(len(data.Embarked.unique()),data.Embarked.unique()))


# In[ ]:


data.drop(['Name', 'Ticket'], axis = 1, inplace=True)


# In[ ]:


data  = pd.get_dummies(data)


# In[ ]:


data.head()


# - **This is cleaned data to be used for pre-processing, modeling and evaluation.**

# 
