#!/usr/bin/env python
# coding: utf-8

# # Titanic Competition by Kaggle
# predict who's gonna survived :) 

# In[4]:


import numpy as np 
import pandas as pd

import os
# print(os.listdir("../input"))


# ## Load Data 

# In[5]:


train_original = pd.read_csv('../input/train.csv')
test_original = pd.read_csv('../input/test.csv')


# #### look at the Data

# In[6]:


train_original.head()


# In[7]:


test_original.head()


# ### Work and play with a Copy of the Data
# I'll work on a copy of the original data so nothing will get lost.  
# also, I'll do the pre-processing on the training and the testing data.

# In[8]:



datasets = [pd.DataFrame(train_original,copy=True),pd.DataFrame(test_original,copy=True)]

for df in datasets :
    print(df.info())
    print()


# # Pclass
# A proxy for socio-economic status  
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# no missing values | no categorical values
# 
# so lets check if it has impact on the result

# In[9]:


# Impact
df = datasets[0]
print(df[['Survived','Pclass']].groupby(['Pclass']).mean())


# good. keeping it!
# 
# # Family Size
# SibSp - number of siblings/spouse  
# Parch - number of parents/children
# 
# i'll sum it up to get the hole family size

# In[10]:


# CREATE
for df in datasets :
    df['FamilySize'] = df.SibSp+df.Parch+1
    df.FamilySize = df.FamilySize.astype('int64')

# IMPACT
df = datasets[0]
print(df[['Survived','FamilySize']].groupby(['FamilySize']).mean())


# good. keeping it!
# 
# ### Alone
# acording to family size lets check a new feature - Alone or not  
# if it has an impact - i'll keep it

# In[11]:


# CREATE
for df in datasets :
    df['Alone'] = 0
    df.loc[df.FamilySize==1,'Alone'] = 1
    df.Alone = df.Alone.astype('int64')

# IMPACT
df = datasets[0]
print(df[['Survived','Alone']].groupby(['Alone']).mean())


# good. keeping it!
# 
# # Sex
# from catagory to binary  
# ~ male to 1  
# ~ female to 0

# In[12]:


# CREATE
for df in datasets :
    df['IsMale'] = df.Sex.map(lambda s : 1 if s=='male' else 0)
    df.IsMale = df.IsMale.astype('int64')

# IMPACT
df = datasets[0]
print(df[['Survived','IsMale']].groupby(['IsMale']).mean())


# good. keeping it!
# 
# # Embark
# embarked Port of Embarkation  
# C = Cherbourg | Q = Queenstown | S = Southampton
# 
# ~ fill out missing value  
# ~ numberize it  
# ~ check impact  

# In[13]:


for df in datasets :
    # Fill Missing Values
    df.Embarked.fillna('S',inplace=True)
    
    # CREATE
    df['EmbarkedNumber'] = 0;

    # numberize it 
    dic = {'S':0,'C':1,'Q':2}
    df.EmbarkedNumber = df.Embarked.replace(dic,inplace=False)
    df.EmbarkedNumber = df.EmbarkedNumber.astype('int64')

# IMPACT
df = datasets[0]
print(df[['Survived','EmbarkedNumber']].groupby(['EmbarkedNumber']).mean())


# good. keeping it!
# 
# # Fare
# price of the ticket
# 
# ~ missing values    
# ~ devide for ranges  
# ~ check for impact  
# 

# In[14]:


# TRAIN
df = datasets[0]

# Missing Values - just in case
fare_median = df.Fare.median()
df.Fare.fillna(fare_median,inplace=True) 

# devide for Fare Range
df['FareRange'],bins = pd.qcut(df.Fare,4,labels=[0,1,2,3],retbins=True)
df.FareRange = df.FareRange.astype('int64')
# else :
#     df['FareRange'] = pd.cut(df.Fare,labels=[0,1,2,3],bins=bins)
# print(df['FareRange'].value_counts())

# IMPACT
print(df[['Survived','FareRange']].groupby(['FareRange']).mean())
###################

# TEST
df = datasets[1]

# Missing Values - just in case
df.Fare.fillna(fare_median,inplace=True) 

# devide for Fare Range
df['FareRange'] = pd.cut(df.Fare,labels=[0,1,2,3],bins=bins)
df.FareRange = df.FareRange.astype('int64')


# # Age
# the age of the person
# 
# ~ missing values    
# ~ devide for ranges  
# ~ check for impact  

# In[15]:


# TRAIN
df = datasets[0]

# Missing Values
age_median = int(df.Age.median())
age_std = int(df.Age.std())
r = lambda : np.random.randint(age_median - age_std,age_median + age_std)
df.Age.fillna(r(), inplace=True)
df.Age = df.Age.astype('int64')

# devide for Fare Range
df['AgeRange'],bins = pd.qcut(df.Age,5,labels=[0,1,2,3,4],retbins=True)
df.AgeRange = df.AgeRange.astype('int64')

# IMPACT
print(df[['Survived','AgeRange']].groupby(['AgeRange']).agg(['mean','sum']))
#########

# TEST
df = datasets[1]

# Missing Values
df.Age.fillna(r(), inplace=True)
df.Age = df.Age.astype('int64')

# devide for Fare Range
df['AgeRange'] = pd.cut(df.Age,labels=[0,1,2,3,4],bins=bins)
df.AgeRange = df.AgeRange.astype('int64')


# # Name to Title
# extract the Title from the name

# In[16]:



# TRAIN
df = datasets[0]

# CREATE
def get_title(name):
    import re
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

df['Title'] = df.Name.map(lambda n : get_title(n))

# IMPACT
# print(df[['Title','Survived']].groupby(['Title']).agg(['mean','sum']))

# FROM CATAGORY TO NUMBERS
titles = df[['Title','Survived']].groupby(['Title']).agg(['mean','sum']).reset_index()  
titlesRare = titles.Title[titles.Survived['sum']<titles.Survived['sum'].mean()].tolist()
titlesNotRare = titles.Title[titles.Survived['sum']>=titles.Survived['sum'].mean()].tolist()
di = {'Master':1, 'Miss':2, 'Mr':3, 'Mrs':4}
df['TitleNumber'] = df['Title'].replace(to_replace=titlesRare, value=0).replace(di)
df.TitleNumber = df.TitleNumber.astype('int64')

# IMPACT
df = datasets[0]
print(df[['TitleNumber','Survived']].groupby(['TitleNumber']).agg(['mean','sum']))
#################

df = datasets[1]
df['Title'] = df.Name.map(lambda n : get_title(n))
df['TitleNumber'] = df['Title'].replace(to_replace=titlesRare, value=0).replace(di).replace('Dona',0)
df.TitleNumber = df.TitleNumber.astype('int64')




# In[17]:


for df in datasets:
    print(df.head())


# # Clean Data
# 
# ~ drop cols  
# ~ drop catagory
# 

# In[18]:


datasets_clean = []

for df in datasets:
    # Drop SibSp & Parch cuz FamilySize & Alone
    # Drop Sex cuz IsMale
    # Drop Embarked cuz EmbarkedNumber
    # Drop Fare cuz FareRange
    # Drop Age cuz AgeRange
    # Drop Name & Title cuz TitleNumber
    cols_to_drop = ['SibSp','Parch','Sex','Fare','Embarked','Fare','Age','Name','Title']

    # Drop Column i did not have the power to extract valuable features
    cols_to_drop.append('Ticket')
    cols_to_drop.append('Cabin')

    datasets_clean.append(df.drop(columns=cols_to_drop,inplace=False,errors='ignore'))

for df in datasets_clean:
    print(df.info())
    print()


# # Training-Fitting-Predict

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier
# checked RandomForestClassifier, AdaBoostClassifier
# checked LogisticRegressionCV
# checked SVC

df_clean = datasets_clean[0]

X_original = df_clean.loc[:, df_clean.columns != 'Survived']
y_original = df_clean.Survived

X_train, X_test, y_train, y_test = train_test_split(X_original,y_original,train_size=0.9,test_size=0.1,random_state=0)

model = GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pre = model.predict(X_test)
target_names = ['0-not Survived', '1-Survived']
print(classification_report(y_test, y_pre, target_names=target_names))


# In[23]:


df_clean = datasets_clean[1]

y_pre = model.predict(df_clean)

test_original['Survived'] = y_pre



# In[33]:


# sub = pd.read_csv('../input/gender_submission.csv')
# sub.head()

my_submission = pd.DataFrame({'PassengerId': test_original.PassengerId, 'Survived': test_original.Survived})
my_submission.to_csv('submission.csv', index=False)

