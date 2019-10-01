#!/usr/bin/env python
# coding: utf-8

# # Lets get started!
# 
# This python library is very useful for exploratory data analysis (EDA). It can be used for loading, subsetting and wrangling data. I'll start with the import statement.

# In[ ]:


import pandas as pd


# Now lets load some data into a dataframe.

# In[ ]:


trainData = pd.read_csv('../input/train.csv')


# I have loaded train data from titanic competition of Kaggle. Lets check what all columns are there.

# In[ ]:


trainData.columns


# Have a glimpse at the data itself.

# In[ ]:


trainData.head()

I notice some nan values in column 'Cabin'. Lets see what percentage of data is missing from each column.
# In[ ]:


trainData.isnull().any()


# In[ ]:


trainData.isnull().sum()


# In[ ]:


missingDataPer = (trainData.isnull().sum()/trainData.shape[0])*100
print(missingDataPer)


# Here, I see that data is missing from three columns - Age, Cabin and Embarked. So now what can I do to rectify this situation? 
# 1. I can remove all the rows with missing data.
# 2. I can remove the whole columns with missing data.
# 3. I can impute values in place of missing data.
# 
# For 'embarked' missing data, I will remove the rows. For 'cabin', I will remove the column. And for 'age', I will imput the average value into missing value cells. 

# In[ ]:


trainData = trainData[trainData.Embarked.notnull()]

missingDataPer = (trainData.isnull().sum()/trainData.shape[0])*100
print(missingDataPer)


# In[ ]:


trainData = trainData.drop('Cabin',axis = 1)

trainData.columns


# In[ ]:


trainData = trainData.fillna(trainData.mean())

trainData.Age.hist(bins=20)


# In[ ]:


trainData[trainData.Survived == 1].Age.hist()


# As I am done handling the missing values, lets plot some graphs!

# In[ ]:


trainData.Pclass.hist()


# In[ ]:


trainData[trainData.Survived == 1].Pclass.hist()


# In[ ]:


trainData.Sex.hist()


# In[ ]:


trainData[trainData.Survived == 1].Sex.hist()


# In[ ]:


trainData.dtypes


# In[ ]:




