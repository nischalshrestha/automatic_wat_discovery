#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering for Titanic Dataset

# This notebook does the same feature engineering as Trevor Stephens did [here](http://trevorstephens.com/kaggle-titanic-tutorial/r-part-4-feature-engineering/) before. The difference is that Trevor Stephens uses R and this notebook uses python pandas library.

# ## Part 1: Label Encoding

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing as prep


# In[ ]:


get_ipython().magic(u'ls ../input')


# In[ ]:


# read csv files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


# add test df a survived cloumn

test_df['Survived'] = 0


# In[ ]:


test_df.head()


# In[ ]:


# concat train and test sets

concat = train_df.append(test_df, ignore_index=True)
concat.head()


# In[ ]:


print(train_df.shape)
print(test_df.shape)
print(concat.shape)


# In[ ]:


concat.Sex.unique()


# In[ ]:


# label encoder to transform categorical string data to integers
le = prep.LabelEncoder()


# In[ ]:


le.fit(concat.Sex)
le.classes_


# In[ ]:


Sex_le = le.transform(concat.Sex)

Sex_le[0:10]


# In[ ]:


concat_le = concat.copy()

concat_le.head()


# In[ ]:


concat_le.Sex = Sex_le

concat_le.head()


# In[ ]:


concat_le.dtypes


# In[ ]:


print(concat.Survived.unique())
print(concat.Pclass.unique())
print(concat.Sex.unique())
print(concat.SibSp.unique())
print(concat.Parch.unique())
print(concat.Embarked.unique())


# In[ ]:


# remove nans and fill with '0's
embarked = concat['Embarked'].fillna('0')
embarked.unique()


# In[ ]:


# label encode embarked
le.fit(embarked)
embarked = le.transform(embarked)
embarked[:10]


# In[ ]:


concat_le.Embarked = embarked


# In[ ]:


# check
concat_le.head(10)


# In[ ]:


# split train and test sets
train_le = concat_le.iloc[:891].copy()
test_le = concat_le.iloc[891:].copy()


# In[ ]:


# And save
get_ipython().magic(u'mkdir -p data')
train_le.to_csv('./data/train_le.csv', index=False)
test_le.to_csv('./data/test_le.csv', index=False)


# In[ ]:


get_ipython().magic(u'ls data')


# ## Part 2: Further Feature Engineering

# In[ ]:


train = pd.read_csv('./data/train_le.csv')
test = pd.read_csv('./data/test_le.csv')


# In[ ]:


# concat dfs again
concat = train.append(test)


# In[ ]:


# check numbers
concat.shape


# In[ ]:


train.shape[0] + test.shape[0]


# ### Feature engineer names

# In[ ]:


NameSplit = concat.Name.str.split('[,.]')


# In[ ]:


NameSplit.head()


# In[ ]:


titles = [str.strip(name[1]) for name in NameSplit.values]
titles[:10]


# In[ ]:


# New feature
concat['Title'] = titles


# In[ ]:


concat.Title.unique()


# In[ ]:


# redundancy: combine Mademoiselle and Madame into a single type
concat.Title.values[concat.Title.isin(['Mme', 'Mlle'])] = 'Mlle'


# In[ ]:


# keep reducing the number of factor levels
concat.Title.values[concat.Title.isin(['Capt', 'Don', 'Major', 'Sir'])] = 'Sir'
concat.Title.values[concat.Title.isin(['Dona', 'Lady', 'the Countess', 'Jonkheer'])] = 'Lady'


# In[ ]:


# label encode new feature too
le.fit(concat.Title)
le.classes_


# In[ ]:


concat.Title = le.transform(concat.Title)


# In[ ]:


concat.head(10)


# ### New features family size and family id

# In[ ]:


# new feature family size
concat['FamilySize'] = concat.SibSp.values + concat.Parch.values + 1


# In[ ]:


concat.head(10)


# New feature `FamilyID`, extract family information from surnames and family size information. Members of a family should have both the same surname and family size.

# In[ ]:


surnames = [str.strip(name[0]) for name in NameSplit.values]
surnames[:10]


# In[ ]:


concat['Surname'] = surnames
concat['FamilyID'] = concat.Surname.str.cat(concat.FamilySize.astype(str), sep='')
concat.head(10)


# In[ ]:


# mark any family id as small if family size is less than or equal to 2
concat.FamilyID.values[concat.FamilySize.values <= 2] = 'Small'


# In[ ]:


concat.head(10)


# In[ ]:


# check the frequency of family ids
concat.FamilyID.value_counts()


# Too many family ids with few family members, maybe some families had different last names or something else. Let's clean this too.

# In[ ]:


freq = list(dict(zip(concat.FamilyID.value_counts().index.tolist(), concat.FamilyID.value_counts().values)).items())
type(freq)


# In[ ]:


freq = np.array(freq)
freq[:10]


# In[ ]:


freq.shape


# In[ ]:


# select the family ids with frequency of 2 or less
freq[freq[:,1].astype(int) <= 2].shape


# In[ ]:


freq = freq[freq[:,1].astype(int) <= 2]


# In[ ]:


# assign 'Small' for those
concat.FamilyID.values[concat.FamilyID.isin(freq[:,0])] = 'Small'


# In[ ]:


concat.FamilyID.value_counts()


# In[ ]:


# label encoding for family id
le.fit(concat.FamilyID)
concat.FamilyID = le.transform(concat.FamilyID)
concat.FamilyID.unique()


# In[ ]:


# choose usefull features
concat_reduce = concat[[
    'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
    'Parch', 'Fare', 'Title', 'Embarked', 'FamilySize',
    'FamilyID', 'Survived']]
concat_reduce.head()


# In[ ]:


# split
train_final = concat_reduce.iloc[:891].copy()
test_final = concat_reduce.iloc[891:].copy()


# In[ ]:


# save
train_final.to_csv('./data/train_final.csv', index=False)
test_final.to_csv('./data/test_final.csv', index=False)


# In[ ]:


get_ipython().magic(u'ls data')


# In[ ]:




