#!/usr/bin/env python
# coding: utf-8

# **Input Access And  Initializations**
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fancyimpute import MICE#Imputation
import missingno as msno #Missing Value Viualization
import matplotlib.pyplot as plt #Graphs
import seaborn as sns #Graphs


# **Accessing Input Files**
# *Using Pandas package,we read the input files and view a few head observations.*

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()


# **Data Type Check**
# *Dtypes used to check data type of each column in dataframe.*
# **Results**
# *Name,Sex,Ticket,Fare need a type change and rest of columns are in correct type*

# In[ ]:


print(train_df.dtypes)
print("===================================================================================")
print(test_df.dtypes)


# **Viewing type change columns**
# *"Name","Sex","Ticket","Cabin","Embarked"*

# In[ ]:


obj_to_cat = ["Name","Sex","Ticket","Cabin","Embarked"]
train_df[obj_to_cat].head()


# **Checking NA columns in DataFrame**
# *Age,Cabin,Embarked need NA treatment*

# In[ ]:


train_df.isnull().any()


# **Having Quick Look at Summary**
# *Summary dataframe holds statistical data with count percentage.*
# **Average** 
# *Average of meaningful columns are Age and Fare[29.6,32.2]*

# In[ ]:


train_df.drop_duplicates()
summary_df = train_df.describe().transpose()
summary_df['count%']=(summary_df['count']/summary_df['count'].max())*100
summary_df


# **NA Randomness checking using Missingno Package**
# Cabin has high NA compared to other columns.
# Out of 12 columns 10 are frequently free of NA.

# In[ ]:


msno.matrix(train_df.sample(891))


# **Converting Object type to Category**
# *We are considering few columns that doesnt need imputation are converted into category.*

# In[ ]:


#train_df.drop(['PassengerID'],axis=1,inplace=True)
train_df['Pclass'] = train_df['Pclass'].astype('category')
train_df['Sex'] = train_df['Sex'].astype('category')
train_df['Ticket'] = train_df['Ticket'].astype('category')

train_df.dtypes


# **Extracting Feature from NAME column**
# *Name can be converted to new category using Title name.*

# In[ ]:


train_df["Name"] = train_df["Name"].str.extract("([A-Za-z]+)\.",expand=False).astype('category')


# **Imputation of Age**
# *Using the name feature we can extract mean and impute in missing values of respective title.*
# *Example: Dr - Mean imputation based on title : 42*

# In[ ]:


passenger = ["Name","Sex","Age","SibSp","Parch"]
train_df[passenger].groupby('Name').describe().transpose()


# In[ ]:


train_df[['Name','Age']].isnull().any()
train_df.groupby('Name')['Age'].mean()
def age_na_fill(x):
    x['Age'] = train_df.groupby('Name')['Age'].mean()[x['Name']]
    return x['Age']    
train_df.loc[train_df['Age'].isnull(),["Age"]] = train_df.loc[train_df['Age'].isnull(),["Age","Name"]].apply(age_na_fill,axis=1)
train_df['Age'].isnull().any()


# **Imputation of Cabin and embarked**
# *Cabin has high level of NA,hence we convert NA into a new category.Emarked only two NA values and its imputed using MODE(Most repeated category)*

# In[ ]:


#pd.set_option('display.max_rows', None)
ship = ["Ticket","Fare","Cabin","Embarked","Pclass","PassengerId"]
train_df[ship].groupby("Ticket").count()


# In[ ]:


train_df["Ticket_cat"]=train_df['Ticket'].str.extract('([a-zA-z])',expand=False)
train_df["Ticket_cat"] = train_df["Ticket_cat"].fillna("N")#numerical
train_df["Ticket_cat"] = train_df["Ticket_cat"].astype('category')


# In[ ]:


train_df['Cabin_cat'] = train_df['Cabin'].str.extract('([a-zA-z])',expand=True)
train_df['Cabin_cat'] = train_df['Cabin_cat'].fillna('U')#unknown
train_df['Cabin_cat'] = train_df['Cabin_cat'].astype('category')


# In[ ]:


mode_val = train_df['Embarked'].mode().to_string(index =False)
train_df['Embarked']=train_df['Embarked'].fillna(value = "S")
train_df['Embarked'] = pd.Categorical(train_df['Embarked'],categories=["C","Q","S"])
train_df.drop(['Cabin'],inplace=True,axis=1)


# In[ ]:


train_df.dtypes
train_df.isnull().any()#After imputation DataFrame is free from NA


# **Features Engineering**
# * *Family Size = Adding Sisiter/Brother and Parents/Childrens*
# * *Alone = Whether got a family or not*

# In[ ]:


train_df.head()
train_df[passenger]
train_df['family_size']=train_df['Parch'] + train_df['SibSp']


# In[ ]:


train_df['Alone'] = 0
train_df.loc[train_df['family_size'] > 0,'Alone'] = 0 #has family
train_df.loc[train_df['family_size'] == 0,'Alone'] = 1
train_df.head()


# **Dropping unwanted Columns**
# * Ticket - Got too many category Hence Dropped.
# * PassengerId - Completely uniqueID Hence Dropped.

# In[ ]:


train_df.drop(['Ticket','PassengerId'],axis=1,inplace=True)


# **Final DataFrame After Pre-Processing And Imputation.**

# In[ ]:


train_df.head()


# ***Data Visualization***

# * **Correlation Plot**
# Helps to find correlated columns in dataframe

# In[ ]:


plt.figure(figsize=(10,10))
plt.title('Pearson Correlation of Features', size=15)
sns.heatmap(train_df.corr(),cmap=plt.cm.RdBu,annot=True)
plt.show()


# * Survived is positively correlated to Fare and Negatively correlated to Family_size
# * Alone and Fare are negatively correlated.

#  * **BarPlot** 
#  *Analysis of Categorical Value*

# In[ ]:


sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Embarked")
plt.show()


# 1.  Cherbourg = Many male passenger didnt survive the disaster.
# 2.  Queenstown = Many Female passenger didnt survive the disaster
# 3.  Southampton = Many Female passenger suvived the disaster

# In[ ]:


plt.figure(figsize=(20,20))
sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Cabin_cat")
plt.show()


# * Cabin A,B,D female got survived than men in the respective Cabins 

# In[ ]:


plt.figure(figsize=(20,20))
sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Ticket_cat")
plt.show()


# * Ticket type W got significant amount of female survival rate compared to other Types.

# In[ ]:


plt.figure(figsize=(20,20))
sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Name")
plt.show()


# In[ ]:


plt.figure(figsize=(20,20))
sns.factorplot('Survived','Age',hue='Sex',data =train_df,kind="bar",col="Pclass")
plt.show()


# * First Class passenger survived well when compared to other classes

# In[ ]:


plt.figure(figsize=(20,20))
sns.factorplot('Survived','Age',hue='Alone',data =train_df,kind="bar",col="Pclass")
plt.show()


# * Lone Travellers survived the disaster well when compared to family
