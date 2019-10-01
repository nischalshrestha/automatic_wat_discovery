#!/usr/bin/env python
# coding: utf-8

# ## Imports
# Import here all libraries that will be needed for loading, inspecting, and cleaning the data: pandas, numpy, matplotlib.pyplot, and sklearn. Show version of sklearn.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import sklearn
print (sklearn.__version__)
# suppress warnings
import warnings  
warnings.filterwarnings('ignore')


# ## Load and preview data 
# Load the Titanic train & test csv files into DataFrames `train_df` and `test_df`. 
# We're also creating a DataFrame `df` that contains both train and test data by concatenating both datasets.

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")
# use both datasets for inspection by saving them in a new dataframe df
df = pd.concat([train_df,test_df], axis=0, ignore_index=False) 
print('Size of training set: {:d}'.format(len(train_df)))
print('Size of test set: {:d}'.format(len(test_df)))


# Take a look at some data by showing the first 10 lines of `df`. We can already see that a few Cabin values are missing.

# In[ ]:


df.head(10)


# Take a look at all data by showing some statistics. 

# In[ ]:


train_df.describe()


# ## Clean up the data
# Look for null fields in the data. The `Cabin` data seems to be pretty incomplete. Since I can't think of a way to use such information anyway, I'm going to drop that column. 
# 
# There are two missing records for `Embarked` in the training set and one missing record for `Fare` in the test set.

# In[ ]:


# print summary of missing fields in the training set
print("Training set\n")
print(train_df.isnull().sum(axis=0))
print("\n")
print("Train and test set\n")
print(df.isnull().sum(axis=0))
# drop Cabin 
train_df = train_df.drop(['Cabin'], axis=1)
df = df.drop(['Cabin'], axis=1)


# Take a look at data again -- notice there are many missing values in the `Age` column.
# Show percent of missing values per column.

# In[ ]:


print('Missing values per column in %')
print(((1 - train_df.count()/len(train_df.index))*100).apply(lambda x: '{:.1f}%'.format(x)))


# In[ ]:


# look at the row containing awkward minimum value (.42) for Age
train_df.loc[train_df['Age'].idxmin(axis=1)]


# Replace missing `Age` fields with mean age (computed by class). Same with missing `Fare`.

# In[ ]:


# look at a few records with missing Age
train_df[train_df.Age.isnull()].head()


# In[ ]:


train_df['Age'] = train_df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
df['Age'] = df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
# check -- now there should be no rows with null Age fields
train_df[train_df.Age.isnull()].head()


# In[ ]:


train_df['Fare'] = train_df.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))
df['Fare'] = df.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))
# check -- now there should be no rows with null Age fields
train_df[train_df.Fare.isnull()].head()


# For `Embarked` we look at the occurrences of each value and replace null values with the most frequent one, which is "S".

# In[ ]:


# just two records with missing `Embarked` data
print('Number of records with missing Embarked data: {}'.format(len(train_df[train_df.Embarked.isnull()])))

# check occurrences of different types of embarkment
train_df['Embarked'].value_counts(dropna=False)


# In[ ]:


# fill the two missing values with the most frequent value, which is "S".
train_df["Embarked"] = train_df["Embarked"].fillna("S")
df['Embarked'].value_counts(dropna=False)


# Are there any more `NaN`s in the training set?

# In[ ]:


# any more nans?
len(train_df[pd.isnull(train_df).any(axis=1)])


# In[ ]:


train_df[train_df['Survived']==0]


# ## Enrich data
# By looking at non survivors I spotted some foreign looking names. So we're going to look at names. Names ending with "ff" sound Russian. There are also other names that sound foreign, namely those ending in "sson" (Swedish?) "ic" and "ff" (Russian?), and some other endings: , "i", "o", "u", "ski", "a". I'm going to use these name endings to populate a new column `Foreign`.
# 
# I'm also going to add a `Name_len` column with the name length and a column `Name_end` with the last character of the family name (not sure how these columns might contribute to the model).

# In[ ]:


train_df[train_df['Survived']==0]


# In[ ]:


# add `FirstName` and `LastName` columns that will be needed later
train_df['LastName'],train_df['FirstName'] = train_df['Name'].str.split(',', 1).str
df['LastName'],df['FirstName'] = df['Name'].str.split(',', 1).str


# In[ ]:


# foreign names
train_df['Foreign'] = False
train_df['Foreign'] = train_df['LastName'].str.endswith(("ic", "sson", "ff", "i", "o", "u", "ski", "a"))
df['Foreign'] = False
df['Foreign'] = df['LastName'].str.endswith(("ic", "sson", "ff", "i", "o", "u", "ski", "a"))
train_df[train_df['Foreign']].head(10)


# In[ ]:


# are names ending in "ff" Russian?
# how many of them survived?
train_df[['FirstName', 'LastName', 'Survived']][train_df['LastName'].str.endswith("ff")].head()


# In[ ]:


# none survived
train_df[train_df['LastName'].str.endswith("ff") & train_df['Survived']>0]


# In[ ]:


# also for names ending in "ic" there are very few (1/20) survivors
train_df[train_df['LastName'].str.endswith("ic")]['Survived'].value_counts()


# Add columns `Name_len` and `Name_end` (last character of the last name).

# In[ ]:


train_df['Name_len'] = train_df['Name'].apply(lambda x: len(x)).astype(int)
train_df['Name_end'] = train_df['LastName'].str[-1:]
df['Name_len'] = df['Name'].apply(lambda x: len(x)).astype(int)
df['Name_end'] = df['LastName'].str[-1:]


# In[ ]:


# ADD FEATURES
print(df.columns)
dummy_features=['Age', 'Embarked', 'Fare', 'Name', 'Parch', 'Pclass', 'Sex', 'SibSp',
       'Survived', 'Ticket', 'LastName', 'FirstName', 
       'Name_len', 'Foreign']

df_dummies = df[dummy_features]
df = pd.get_dummies(df[dummy_features])
train_features = df.iloc[:891,:]
train_labels = train_features.pop('Survived').astype(int)
test_features = df.iloc[891:,:].drop('Survived',axis=1)


# In[ ]:


print(df.columns)


# ## Build a predictive model
# 
# This time we're going to use `scikit-learn`, evaluate a few models on our dataset and choose the best for our submission.
# 

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


# We're going to try out a few models and rate them using cross-evaluation.

# In[ ]:


models=[KNeighborsClassifier(), LogisticRegression(), GaussianNB(), SVC(), DecisionTreeClassifier(),
        RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier()]
names=['KNN', 'LR', 'NB', 'SVM', 'Tree', 'RF', 'GB', 'Ada']
for name,model in zip(names, models):
    score = cross_val_score(model, train_features, train_labels, cv=5)
    print('{} :: {} , {}'.format(name, score.mean(), score))


# Logistic regression and Random Forest seem to produce the best models, so we will use these methods again to build models. But this time we are going to first scale the features using a [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

# In[ ]:


models=[LogisticRegression(),RandomForestClassifier()]
names=['LR','RF']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features_scaled = scaler.fit(train_features).transform(train_features)
test_features_scaled = scaler.fit(test_features).transform(test_features)
for name,model in zip(names,models):
    score = cross_val_score(model,train_features_scaled,train_labels,cv=5)
    print('{} :: {} , {}'.format(name,score.mean(),score))


# In[ ]:


# Initialize the model class
model = RandomForestClassifier()

# Train the algorithm using all the training data
model.fit(train_features,train_labels)

# Make predictions using the test set.
predictions = model.predict(test_features)

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": predictions
    })

# uncomment to save submission file
# submission.to_csv("./output/RandomForestClassifierNormName.csv", index=False)


# With this kernel I got my currently best score of 80%

# In[ ]:




