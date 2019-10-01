#!/usr/bin/env python
# coding: utf-8

# As a beginner, I always felt daunted by the number of features available. So I usually rush to build a end-to-end project to make a result  without doing much, if any, exploratory data analysis. Luckily, this dataset has a reasonablely less features so that I can do one.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # EDA with bar plot

# A glimpse at features

# In[ ]:


train.isnull().sum().plot(kind = 'barh')


# In[ ]:


train.head()


# Check the proportion of survivor for each sex

# In[ ]:


grouped = train.Survived.groupby(train.Sex)

sex_stats = pd.DataFrame({'total number':grouped.size(), 'number of survivor':grouped.sum()})
sex_stats.plot.bar()


# Seems like  female are more likely to survive. The reason behind this is respectable.

# Now, look at the influence of age.<br>
# First of all, the distribution of age

# In[ ]:


train.Age.plot.hist(bins = 50)


# In[ ]:


age_group = pd.cut(train.Age, bins = 8)

age_grouped = train.Survived.groupby(age_group)

age_stats = pd.DataFrame({'total number':age_grouped.size(), 'number of survivor':age_grouped.sum()})
age_stats.plot.bar()


# Middle-aged(20-40) are **less** to survive

# Check if 'pclass' has any influence upon 'survive'

# In[ ]:


class_grouped = train.Survived.groupby(train.Pclass)

class_stats = pd.DataFrame({'total number':class_grouped.size(), 'number of survivor':class_grouped.sum()})
class_stats.plot.bar()


# They do! Money really CAN buy life!!

# Fare must be corelated with Pclass. Whatever, let's check it.

# In[ ]:


train[['Pclass', 'Fare']].corr()


# Yes, they are corelated, but in a somewhat counter-intuitive way---negtive corelation. There must be something odd here. When we look at the variable notes, we will find that it is caused by the encoding of Pclass--The higher the class, the lower they get encoded.

# Let's see whether Fare could be a good sign for survival

# In[ ]:


train.Fare.plot.hist()


# Cut Fare column into 2 bins:low(<100) and high(>100)

# In[ ]:


train['Fare_class'] = train.Fare.apply(lambda x: 'low' if x < 100 else 'high')
fare_class_grouped = train.Survived.groupby(train.Fare_class)

f_stats = pd.DataFrame({'total number':fare_class_grouped.size(), 'number of survivor':fare_class_grouped.sum()})
f_stats.plot.bar()


# Fare class being 'high' seems to be a strong signal for survival. 

# Now, let's look at the number of relatives, which, intuitively, could be a moderate sign for survival. Since relatives help each other, at least more than strangers do.

# In[ ]:


train['n_relatives'] = train.SibSp + train.Parch


# In[ ]:


n_map = {0:'None', 1:'Fair', 2:'Fair', 3:'Fair'}
train['relatives'] = train.n_relatives.apply(lambda x:n_map.get(x, 'Many'))


# In[ ]:


n_relatives_grouped = train.Survived.groupby(train.relatives)

r_stats = pd.DataFrame({'total number':n_relatives_grouped.size(), 'number of survivor':n_relatives_grouped.sum()})
r_stats.plot.bar()


# Seems like when the number of relatives is 1 or 2, the passagers are more likely to survive.

# What about number of parents or children? Hard to say, since having children also means some children having parent(s). A good sign for crossing with Age attribute. Let's find it out.

# In[ ]:


train['parent_or_children_onboard'] = train.Parch.apply(lambda x: 'Yes' if x > 0 else 'No')
n_pc_grouped = train.Survived.groupby(train.parent_or_children_onboard)

r_stats = pd.DataFrame({'total number':n_pc_grouped.size(), 'number of survivor':n_pc_grouped.sum()})
r_stats.plot.bar()


# There seems to be some relationship between them, but not very strong.<br>

# Next step, check feature combination of age and parch, such as Young child(age < 13?) with parents(Parch != 0), middle-aged with Parch !=0 and old-aged with Parch != 0.<br>
# 

# TO BE CONTINUED.(Not coming up with a good solution right now)
