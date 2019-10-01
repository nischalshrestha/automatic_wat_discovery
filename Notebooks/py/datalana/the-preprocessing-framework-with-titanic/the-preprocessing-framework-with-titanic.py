#!/usr/bin/env python
# coding: utf-8

# # The preprocessing framework with Titanic

# *by Lana Samoilova (April, 2018)*

# Hi there,
# 
# I want to introduce you my thoughts about what and how to clean in data sets.   
# (no modeling part in this notebook, data cleansing only)     
# 
# I'm trying to create full routine framework to make that job easier.     
# For beginners espesially.   
# 
# Hope, it will help.   
# 
# 
# TABLE OF CONTENTS:     
# [1. Imports](#1)  
# [2. First meeting with Data (load file and understand the goal)](#2)  
# [3. Duplicate values](#3)  
# [4. Outliers](#4)  
# [5. Data types (deep understanding the data)](#5)  
# [6. Missing values](#6)  
# [7. Feature engineering (and feature selection)](#7)  
# [8. Encoding](#8)  
# [9. Final check and final thoughts](#9)  
# 
# I am new here and hoping to learn a lot, so any feedback is really welcome!

# # Imports 

# The simplest part, isn't it?    
# But do not forget that it can make you job a little bit easier if you do some customizing here.

# In[2]:


import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format # to format large numbers

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
palette = 'cubehelix' # good for bw printer

# show plots in Jupyter Notebook
get_ipython().magic(u'matplotlib inline')


# # First meeting with Data (load file and understand the goal) 

# Titanic.     
# Who doesn't know about this shipwrecks?    
# But behind of the story and movie there are people.    
# 
# So, my first suggestion is: always to remember that data is not only data, it's real live and real lifes.
# Let's dive into...

# In[3]:


raw_train = pd.read_csv('../input/train.csv')
raw_train.head()


# > *** Our goal is to predict survival for people in the test data set using train data as a base for machine learning***

# In[4]:


raw_test = pd.read_csv('../input/test.csv')
raw_test.head()


# As we can see, the data sets are contain the same info, except 'Survived' column.   
# 
# I'll save this column as 'target' and I'll concatinate data sets to exclude extra job during a data cleansing (I'll separate it back at the end of process)

# In[5]:


target = raw_train['Survived']
target.head()


# In[6]:


df = pd.concat(objs=[raw_train, raw_test], axis=0).reset_index(drop=True)
del df['Survived']
df.head()


# In[7]:


print('The shape of working data set is:', df.shape, sep='\n')


# # Duplicate values 

# If any duplicate values in data sets?

# In[8]:


df.duplicated().sum()


# # Outliers 

# If any outliers in data sets?   
# This question is not so simple to answer as previous.  

# > MOST COMMON CAUSES OF OUTLIERS ON A DATA SET:
# 1. Data entry errors
# 2. Measurement errors
# 3. Experimental errors
# 4. Data computation errors
# 5. Unusual data (not error actually)

# There are a lot methods of outliers detection (Z-score, Probabilistic, Linear Reg., etc.)  
# It Titanic case we can simply visualize the data to find it.

# In[9]:


sns.PairGrid(df).map(plt.scatter)
plt.show()


# We can definitely say that there are outliers in 'Fare' column and in 'SibSp'.   

# In[10]:


# Outliers in 'Fare'
df[(df.Fare > 300)]


# Was that mother, son and assistants?   

# In[11]:


# Outliers in 'SibSp'
df[(df.SibSp > 7)]


# In[12]:


# Let's get more information:
df[df['Name'].str.contains('Sage,')]


# So big family and nobody was survived...

# Usually, outliers were dropped.  
# But I don't do it, because I already know that it will be hidden by binning and feature engineering.

# # Data types (deep understanding the data) 

# It wasn't very honest to tell what I'm going to do without explaining why, was it?  
# Sorry for that.
# I'll fix it now by explaining my way to understanding the data.

# In[13]:


# I've wrote this simple function to print all data types of data set by column names
def print_dtypes(df):
    """
    :param df: data frame name
    :return: print lists with column's names by data type
    """
    cat, fl, integ, time = [], [], [], []
    cat = df.dtypes[df.dtypes == 'object'].index.tolist()
    fl = df.dtypes[df.dtypes == 'float64'].index.tolist()
    integ = df.dtypes[df.dtypes == 'int64'].index.tolist()
    time = df.dtypes[df.dtypes == 'datetime64'].index.tolist()
    print('Categorical columns:', cat, '', 'Numerical columns:', 'Floats:', fl, 
          'Integers:', integ, '', 'Time series:', time, sep='\n')


# In[14]:


print_dtypes(df)


# 'Integers' is the simpliest type, because it can be numbers only, no missing values etc.

# In[15]:


for i in ['Parch', 'PassengerId', 'Pclass', 'SibSp']:
    print(df[i].unique())


# **'PassengerId'** is random unique identifier with we'll need to output data frame, but not needed to modeling.

# In[16]:


id = raw_test['PassengerId']
id.shape


# In[17]:


df.drop(['PassengerId'], axis=1)
df.shape


# In[18]:


# and again...
df = df.drop(['PassengerId'], axis=1)
df.shape


# So simple mistake, but so important to result...   
# I advise to everybody to check any important step - it's much faster than looking for mistakes at the end of work

# **'Pclass'** is identifier of ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) => ordinal value (qualitative, not quantitative)
# At this step we will do nothing with this.

# **'Parch' and 'SibSp'**  
#     'SibSp' - number of siblings / spouses aboard the Titanic = discrete quantitative  
#     'Parch' -  # of parents / children aboard the Titanic = discrete quantitative  
# 
# I'll use this couple for the feature engineering...

# In[19]:


for i in ['Age', 'Fare']:
    print(df[i].describe(), '\n')


# It's 1309 rows in our data set == both columns have missing values   
# **Age** - passenger's age in years  
# **Fare** - passenger fare  
# Both columns I'm going to bin

# In[20]:


for i in ['Cabin', 'Embarked', 'Name', 'Sex', 'Ticket']:
    print(i, 'has', df[i].nunique(), 'unique values')


# **'Cabin'** - cabin number    
# **'Embarked'** - port of embarkation    
# **'Name'** - passenger name    
# **'Sex'** - passenger sex    
# **'Ticket'** - ticket number
# 
# Do I need 'Cabin' and 'Ticket' for modeling? It depends of how many missing values here.

# # Missing values 

# > There are 2 strategies to work with missing values:
# 1. Drop it
# 2. Compute and impute

# In[21]:


#
df.isnull().sum().sort_values(ascending=False)


# In[22]:


print('\n', round(df['Cabin'].isnull().sum() * 100 / len(df.index)), '% of Cabin values are missing')


# In[23]:


# So, we can drop it surely
df = df.drop(['Cabin'], axis=1)


# I'm not such kind of person who like drop missing values even if it's 2 rows only - I want to fill it with most used value.   
# I can do it by 2 ways: 
# 
# ** 1 way: use Imputer**  
# The code would be:  
#     from sklearn.preprocessing import Imputer  
#     imputer = Imputer()  
#     imputed_data = imputer.fit_transform(df)  
# Why 'would'? Because Imputer doesn't work with categorical values = we can't use it without encoding  
#   
# ** 2nd way: hand made imputing **  

# In[24]:


df = df.fillna({'Age': df['Age'].mode()[0],
                'Embarked': df['Embarked'].mode()[0],
                'Fare': df['Fare'].mode()[0]
               })


# In[25]:


# and check, and check
print('\n', 'There are', df.isnull().any().sum(), 'missing values in data frame now.')


# But what about 'Ticket' - column has no missing values, but 929 unique values only.  
# Let's try to understand more during feature engineering

# # Feature engineering (and feature selection, and...)

# In[26]:


# take a look to the first 13 group tickets
df.Ticket.value_counts()[:13]


# We can recognize the first value immediately - CA. 2343 is ticket of unlucky Sage family.

# In[27]:


# How many group tickets do we have?
(df.Ticket.value_counts() > 1).value_counts()


# So, 216 tickets were group tickets.   
# And... does it mean that Fare values for this tickets are wrong and I should divide every value by numbers of person? Shoul we play here more?  
# What do you think?  

# **'Parch' and 'SibSp'**  - what statistically important meanings are hidden here?  
# I'm going to create 'IsAlone' column and collect family size information as 'FSize'.

# In[28]:


df['FSize'] = (df['SibSp'] + df['Parch'] + 1).astype(int)
df.FSize.unique()


# In[29]:


df['IsAlone'] = 1  # as 1 == yes, is alone
df['IsAlone'].loc[df['FSize'] > 1] = 0 
df.IsAlone.unique()


# In[30]:


# I think, we don't need SibSp and Parch for modeling any more, so...
df = df.drop(['SibSp', 'Parch'], axis=1)


# Now I'm going to cut continuous quantitatives - **'Age' and 'Fare'**:

# In[31]:


df.Age.describe()


# In[32]:


# I divide max by std and so decide about bins amount
# more infomatio about pd.cut function here - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
labels = [0, 1, 2, 3, 4, 5]
df['BinnedAge'] = pd.cut(df['Age'], bins=6, labels=labels, include_lowest=True)
df['Age'] = df['BinnedAge'].astype(int)
print('\n', 'Age uniques:', df.Age.unique())


# In[33]:


# Do the same for 'Fare':
df.Fare.describe()


# In[34]:


labels = [0, 1, 2, 3, 4, 5]
df['BinnedFare'] = pd.qcut(df['Fare'], 6, labels=labels)
df['Fare'] = df['BinnedFare'].astype(int)
print('\n', 'Fare uniques:', df.Fare.unique())


# In[35]:


# delete unneсeыsary columns
df = df.drop(['BinnedAge', 'BinnedFare'], axis=1)


# **'Name'**  
# From this column we can extract information about passenger title:

# In[36]:


# extracting (title information goes after comma and before dot)
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# checking 
df['Title'].value_counts()


# In[37]:


# Let's create 'other' for rare values:
titles = (df['Title'].value_counts() < 10)
df['Title'] = df['Title'].apply(lambda x: 'other' if titles.loc[x] == True else x)
df['Title'].value_counts()


# In[38]:


# We don't need 'Name' any more
df = df.drop(['Name'], axis=1)


# In[39]:


# Let's check how is the data frame now:
df.head()


# That 'Ticket'!

# # Encoding 

# What do you decide about 'Ticket'?  
# 
# I still not sure for deleting...
# Let's label it by Label Encoder (convert text values to numbers)

# In[40]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Ticket'] = le.fit_transform(df['Ticket'])
df['Ticket'].unique()


# Why we need to encode all qualitative data to numbers?  
# We do it for modeling (some model work with numerical values only)
# 
# I prefer to do it "by hand" in such simple data sets:

# In[41]:


df.Title.unique()


# In[42]:


df['Title'] = df.Title.map({'Mr': 0,  'Mrs': 1,  'Miss': 2,  'Master': 3,  'other': 4}).astype(int)


# In[43]:


df.Embarked.unique()


# In[44]:


df['Embarked'] = df.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[45]:


df.Sex.unique()


# In[46]:


df['Sex'] = df.Sex.map({'male': 1, 'female': 0}).astype(int)


# In[47]:


df.head()


# # Final check and final thoughts 

# \Looks like we finished, but we are not.  
# Now we should split data set back and prepare X and y for future modeling.

# In[48]:


# get a raw data sets q-ty of rows:
te, tr = raw_test.shape[0], raw_train.shape[0]


# In[49]:


# splitting back:
train = df[:tr]
test = df[tr:]


# In[50]:


X = train.values
y = target
print('Train set for modeling shape is:', X.shape, 'Target shape is:', y.shape, sep='\n')


# Because I'll not show modeling part in this notebook, I'll prepare the data frame for submission now:

# In[51]:


result = pd.DataFrame()
result['PassengerId'] = raw_test['PassengerId'].astype('int')
result.shape


# In[52]:


# After modeling you'll add column...
#result['Survived'] = y_pred.astype('int')

# ... and save file...
# result.to_csv("Titanic_prediction.csv", index=False)

# ... and do not forget to check before submission
# print(result.shape, result.head(), sep='\n')


# That's it!  
# 
# Remember, I am new here, so any feedback is very, very welcome!  
# And english language is not my native - sorry me, if I use it in crazy way sometimes.
# 
# By the way, if you're an experienced Data Scientist and you don't have enough time for cleaning data for all your projects, I may help (my Upwork profile here - https://www.upwork.com/o/profiles/users/_~014dcb1a76dd7f7892/).
# 
# Thank you for your time!
# 
# wbr,  
# Lana

# P.S.  
# I didn't show any visualization here, because of the topic.  
# It's coming soon as a separate notebook.   
# Here, on Kaggle.

# In[ ]:




