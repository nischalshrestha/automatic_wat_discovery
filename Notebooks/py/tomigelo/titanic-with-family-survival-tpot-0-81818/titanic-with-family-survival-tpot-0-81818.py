#!/usr/bin/env python
# coding: utf-8

# # Titanic Prediction Competition
# # Intro
# This notebook demonstrates a few approaches to the Titanic prediction competition including using a gender-only model (no machine learning at all!) as well as some machine learning algorithms using random forest classifiers. The last model was powered by the AutoML tool [TPOT](https://epistasislab.github.io/tpot/) which resulted in a **public score of 0.81818 (top 4%).**
# 
# Here is the official competition description from [Kaggle](https://www.kaggle.com/c/titanic):
# 
# > ### Competition Description
# > The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# > 
# > One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# > 
# > In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# # Data Loading

# In[1]:


# import libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[2]:


# list data files that are connected to the kernel
import os
os.listdir('../input/')


# In[81]:


# read the train.csv file into a datframe
df_train = pd.read_csv('../input/train.csv')
print('Shape: ', df_train.shape)
df_train.head()


# In[82]:


# read the test.csv file into a datframe
df_test = pd.read_csv('../input/test.csv')
print('Shape: ', df_test.shape)
df_test.head()


# In[91]:


# create df_full by merging both train and test data
df_full = df_train.append(df_test, sort=False)
print('Shape: ', df_full.shape)


# # Exploratory Data Analysis

# In[6]:


df_train.info()


# In[7]:


df_test.info()


# In[99]:


# import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# create function EDA_helper to do which is doing 3 things: binning, encoding of the feature and calculating the impact on the target feature
def EDA_helper(feature, bin_number=5, train_size=891):
    '''function creates a new column as 'old feature name_bin' and bins the values (only if the number of unique values is more than 10). After that it calculates the sum, count and mean of the feature values'''    
    # calculate number of unique values for the feature
    unique_features = len(list(df_full[feature].unique()))
    # if there are more than 10 unique values
    if unique_features > 10:
        print('Number of unique features is %d, starting to bin...' % unique_features)
        # create a new column for the bins
        df_full[feature + '_bin'] = pd.qcut(df_full[feature], bin_number)
        # assign the bins to the train and test dataframe
        df_train[feature + '_bin'] = df_full[feature + '_bin'][:train_size]
        df_test[feature + '_bin'] = df_full[feature + '_bin'][train_size:]
        # define LabelEncoder instance 
        label = LabelEncoder()
        # fit and transform the data
        df_full[feature + '_bin_code'] = label.fit_transform(df_full[feature + '_bin'].astype(str))
        # assign the encoded bins to the train and test dataframe
        df_train[feature + '_bin_code'] = df_full[feature + '_bin_code'][:train_size]
        df_test[feature + '_bin_code'] = df_full[feature + '_bin_code'][train_size:]
        print('Binning successful, calculating impact...')
        # calculate the statistics
        impact = df_full[[feature + '_bin', 'Survived']].groupby([feature + '_bin']).agg(['sum','count','mean']).rename(columns={'sum':'Yes','count':'Total','mean':'In %'})
    else:
        print('Number of unique features is %d, binning not needed. Calculating impact...' % unique_features)
        # define LabelEncoder instance 
        label = LabelEncoder()
        # fit and transform the data
        df_full[feature + '_code'] = label.fit_transform(df_full[feature])
        # assign the encoded bins to the train and test dataframe
        df_train[feature + '_code'] = df_full[feature + '_code'][:train_size]
        df_test[feature + '_code'] = df_full[feature + '_code'][train_size:]
        # calculate the statistics for not binned features
        impact = df_full[[feature, 'Survived']].groupby([feature]).agg(['sum','count','mean']).rename(columns={'sum':'Yes','count':'Total','mean':'In %'})
    return impact


# ### PassengerId

# In[105]:


# using the function on the 'PassengerId' column
EDA_helper('PassengerId')


# ### Survived

# In[20]:


# unique value counts in 'Survived' column
df_train['Survived'].value_counts()


# ### Pclass

# In[21]:


EDA_helper('Pclass')


# ### Name

# In[12]:


# extract the title from the 'Name' column
for name in df_full['Name']:
    df_full['Title'] = df_full['Name'].str.extract('([A-Za-z]+)\.', expand=False)

# check how the different titles are distributed by gender
print(pd.crosstab(df_full['Title'], df_full['Sex']))


# In[22]:


# categorize titles
for title in df_full['Title']:
    df_full['Title'] = df_full['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
                                               )
    df_full['Title'] = df_full['Title'].replace('Mlle', 'Miss')
    df_full['Title'] = df_full['Title'].replace('Ms', 'Miss')
    df_full['Title'] = df_full['Title'].replace('Mme', 'Mrs')
    
# use the EDA_helper function
EDA_helper('Title')


# ### Sex

# In[106]:


EDA_helper('Sex')


# ### Age

# In[57]:


# fill the missing age info with median from the full dataset
for age in df_full['Age']:
    df_full['Age'].fillna(df_full['Age'].median(), inplace=True)

# using the EDA_helper function and setting number of bins to 4
EDA_helper('Age', 4)


# ### SibSp & Parch

# In[113]:


# combine both columns as 'Family size'
df_full['Family Size'] = df_full['SibSp'] + df_full['Parch']

EDA_helper('Family Size')


# ### Ticket

# In[100]:


# import library
import re

# remove non-digits from the ticket and change to numeric type
for ticket in df_full['Ticket']:
    df_full['Ticket'] = df_full['Ticket'].apply(lambda x: x if x.isdigit() else re.sub('\D','', x))

# changing the type to numeric
df_full['Ticket'] = df_full['Ticket'].apply(pd.to_numeric)
    
EDA_helper('Ticket')


# ### Fare

# In[124]:


# fill the missing fare info with median fare from the full dataset
for fare in df_full['Fare']:
    df_full['Fare'].fillna(df_full['Fare'].median(), inplace=True)

EDA_helper('Fare')


# ### Cabin

# In[107]:


# fill the missing info with string 'N' and extract the first letter as new column 'N' for the full dataset
for cabin in df_full['Cabin']:
    df_full['Cabin'].fillna('N', inplace=True)
    df_full['Deck'] = df_full['Cabin'].apply(lambda x: 'N' if pd.isnull(x) else x[0])

EDA_helper('Deck')


# ### Embarked

# In[129]:


# fill the missing info with the most common value
for cabin in df_full['Embarked']:
    df_full['Embarked'].fillna('S', inplace=True)

EDA_helper('Embarked')


# # Feature Engineering and Selection
# 
# ### New Feature: Family Survival
# 
# This is a feature from [S.Xu's](https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever) and [Konstantin's kernel](https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83/). 

# In[109]:


# extract the last name from the 'Name' column (using the full_data)
for name in df_full['Name']:
    df_full['Last Name'] = df_full['Name'].str.extract('([A-Za-z]+)\,', expand=False)

DEFAULT_SURVIVAL_VALUE = 0.5
df_full['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in df_full[['Survived','Name', 'Last Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      df_full.loc[df_full['Family_Survival']!=0.5].shape[0])


# In[110]:


for _, grp_df in df_full.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df_full.loc[df_full['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(df_full[df_full['Family_Survival']!=0.5].shape[0]))

train_size = len(df_train)

# Family_Survival in df_train and df_test:
df_train['Family_Survival'] = df_full['Family_Survival'][:train_size]
df_test['Family_Survival'] = df_full['Family_Survival'][train_size:]


# ### Selecting Most Important Features

# In[1]:


# define a list of columns to work with going further
columns_to_keep = ['Sex_code', 'Pclass', 'Fare_bin_code', 'Age_bin_code', 'Family Size_code', 'Family_Survival']

# create new datafames with the desired columns
train = df_train[columns_to_keep]
test = df_test[columns_to_keep]

# save the target column for later use
train_labels = df_train['Survived']

print('Train data shape: ', train.shape)
print('Test data shape: ', test.shape)


# In[127]:


train.head()


# # Modeling

# ### Gender Model
# 
# This simple model predicts that all female passengers survive while males don't.

# In[44]:


# create simple predicition based on gender (women live, men die)
gender_pred = df_test['Sex'].apply(lambda x: '1' if x=='female' else '0')
gender_pred.value_counts()


# When submitted, this gender-only model will get a **score of 0.76555.**

# ### Random Forest Model

# In[75]:


# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# define the scaler instance
scaler = StandardScaler()

# fit on train data, transform both train and test data
train = scaler.fit_transform(train)
test = scaler.transform(test)
        
# define the classifier instance
clf = RandomForestClassifier(n_estimators=100, random_state = 42)

# fit the classifier on the train data and previously saved train labels
clf.fit(train, train_labels)

# predict on test data
rf_results = clf.predict(test)


# In[ ]:


# make a submission dataframe
submit = df_test.loc[:, ['PassengerId']]
submit.loc[:, 'Survived'] = rf_results

# save the submission dataframe
submit.to_csv('RF_submission.csv', index = False)


# When submitted, this random forest model will get a **score of 0.79425**

# ### TPOT Model
# 
# Last but not least, I will let the AutoML tool [TPOT](https://epistasislab.github.io/tpot/) run for 2 hours to automatically find a machine learning pipeline using genetic programming.
# 
# Interestingly, TPOT also selected a random forest model which** scored 0.81818.**

# In[3]:


# import TPOT
from tpot import TPOTClassifier

# create instance
pipeline_optimizer = TPOTClassifier(max_time_mins=120, n_jobs = -1, random_state=42, verbosity=2, cv=5)

# fit TPOT on the train data
# commented out after the run
#pipeline_optimizer.fit(train, train_labels)

# export optimized code
# commented out after the run
#pipeline_optimizer.export('tpot_titanic_pipeline.py')

# import libraries
from sklearn.pipeline import make_pipeline

# create the pipeline from TPOT
# original pipeline inluded a Binarizer and RBFSampler which scored only 0.78947 
exported_pipeline = make_pipeline(
    RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.45, min_samples_leaf=14, min_samples_split=13, n_estimators=100)
)

# fit the pipeline on the train data
exported_pipeline.fit(train, train_labels)

# predict on the test data
results = exported_pipeline.predict(test)


# In[ ]:


# make a submission dataframe
submit = df_test.loc[:, ['PassengerId']]
submit.loc[:, 'Survived'] = results

# save the submission dataframe
submit.to_csv('TPOT_submission.csv', index = False)

