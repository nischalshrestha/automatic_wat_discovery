#!/usr/bin/env python
# coding: utf-8

# In[3]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[4]:


train_df = pd.read_csv('../input/train.csv', index_col="PassengerId")
test_df = pd.read_csv('../input/test.csv', index_col="PassengerId")


# In[5]:


#Lets create the Column Survived in the Test data set
test_df['Survived'] = -777


# In[6]:


#Lets combain two Test data set and Train data set all together 
dataset = pd.concat((train_df, test_df),axis=0)


# **Working with missing values:**

# Embarked:

# In[7]:


dataset.info()


# In[8]:


dataset[dataset.Embarked.isnull()]


# In[9]:


# how many people embarked at different points
dataset.Embarked.value_counts()


# In[10]:


dataset.groupby(['Pclass', 'Embarked']).Fare.median()


# In[11]:


# replace missing values with 'C'
dataset.Embarked.fillna('C', inplace=True)


# Fare:

# In[12]:


dataset[dataset.Fare.isnull()]
dataset.groupby(['Pclass', 'Embarked']).Fare.median()
dataset.Fare.fillna(8.05, inplace=True)


# Age:

# In[13]:


dataset[dataset.Age.isnull()]


# In[14]:


# Function to extract the title from the name 
def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title


# In[15]:


# lets use map function to assosiate each Name with Title
dataset.Name.map(GetTitle)


# In[16]:


dataset.Name.map(lambda x : GetTitle(x)).unique()


# In[17]:


# Function to extract the title from the name 
def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]


# In[18]:


# create Title feature
dataset['Title'] =  dataset.Name.map(lambda x : GetTitle(x))

# Box plot of Age with title
dataset[dataset.Age.notnull()].boxplot('Age','Title');

# replace missing values
title_age_median = dataset.groupby('Title').Age.transform('median')
dataset.Age.fillna(title_age_median , inplace=True)


# Working with Outliers

# In[19]:


# histogram for fare 
dataset.Fare.plot(kind='hist',title='histogram for fare', bins=20)
# binning. We are splitting some different ranges of values into 4 bins
pd.qcut(dataset.Fare,4)


# In[20]:


pd.qcut(dataset.Fare, 4, labels=['very_low','low','high','very_high']) #discretization


# In[21]:


# create fare feature bin
dataset['Fare_Bin']=pd.qcut(dataset.Fare, 4, labels=['very_low','low','high','very_high'])


# Feature Engineering:

# In[22]:


#Age State based on Age
dataset['AgeState']=np.where(dataset['Age'] >= 18, 'Adult', 'Child')
# AgeState counts
dataset['AgeState'].value_counts()


# In[23]:


# cross tab
pd.crosstab(dataset[dataset.Survived != -777].Survived, dataset[dataset.Survived !=-777].AgeState)


# In[24]:


#family size
# Family: Adding parents with siblings
dataset['FamilySize'] = dataset.Parch + dataset.SibSp + 1 # 1 for self


# In[25]:


# lets create crosstable to see a family size impact on survival rate
pd.crosstab(dataset[dataset.Survived != -777].Survived, dataset[dataset.Survived != -777].FamilySize)


# In[26]:


#Feature IsMother
# a lady aged more than 18 which has parch>0 and is married (not miss). 1 and 0 in the end of where function means
# if condition is true -> assign 1 else assign 0
dataset['IsMother']=np.where(((dataset.Sex=='female') & (dataset.Parch>0) & (dataset.Age>18) & (dataset.Title !='Miss')),1,0)


# In[27]:


# crosstab with IsMother
pd.crosstab(dataset[dataset.Survived !=-777].Survived, dataset[dataset.Survived !=-777].IsMother)


# Deck:

# In[28]:


dataset.loc[dataset.Cabin =='T','Cabin'] = np.NaN


# In[29]:


# extract first character of Cabin string to the deck
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
dataset['Deck'] = dataset['Cabin'].map(lambda x : get_deck(x))


# In[30]:


#check counts
dataset.Deck.value_counts()
# use crosstab to look into survived feature cabin wise
pd.crosstab(dataset[dataset.Survived !=-777].Survived,dataset[dataset.Survived != -777].Deck)


# Categorical Feature Encoding

# In[31]:


# sex
dataset['IsMale'] = np.where(dataset.Sex == 'male', 1, 0)


# In[32]:


# columns Deck, Pclass, Title, AgeState
dataset = pd.get_dummies(dataset,columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])


# Drop some usless columns

# In[33]:


dataset.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1,inplace=True)
columns = [column for column in dataset.columns if column != 'Survived']
columns = ['Survived'] + columns
dataset = dataset[columns]


# In[34]:


dataset.info()


# In[35]:


X_train = dataset.loc[dataset['Survived'] != -777, 'Age':'AgeState_Child'].values
y_train = dataset.loc[dataset['Survived'] != -777, 'Survived'].values


# In[36]:


X_test = dataset.loc[dataset['Survived'] == -777, 'Age':'AgeState_Child']


# In[37]:


# Fitting logistic regression in to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[40]:


test_df.head()


# In[66]:


data_to_submit = pd.DataFrame({
    'PassengerId':X_test.index,
    'Survived':y_pred
})


# In[71]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)

