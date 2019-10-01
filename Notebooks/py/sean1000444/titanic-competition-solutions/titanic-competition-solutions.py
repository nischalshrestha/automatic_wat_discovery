#!/usr/bin/env python
# coding: utf-8

# # Titanic Competition Solutions

# ## Import necessary libraries

# In[ ]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


# ## Read in data to dataframes

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
gender_sub = pd.read_csv('../input/gender_submission.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Identify problem
# * Description: Predict the survival or death of members of the Titanic given data about them as inputs
# * Type of Problem: Categorical

# ## Describe data
# 
# * Numerical
#   * Continuous: Fare
#   * Discrete: PassengerId, Age, SibSp, Parch
# * Categorical: Survived, Sex, Embarked
# * Ordinal: Pclass
# * Mixed: Cabin, Ticket
# * String: Name

# In[ ]:


train_df.head()


# ## Analyze data with pivot tables (Sex and Pclass)
# 
# Let's start off with Sex and Pclass just from intuition that they may have an impact on survival. We can use pivot tables for these as they contain less than 5 categories each.
# 
# **Obervations**
# * Sex and Pclass have significant coorelations with survival rate.

# In[ ]:


train_df[['Sex', 'Survived']].groupby(['Sex']).mean()


# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# ## Analyze data by visualization (Age)
# 
# The data for Age is continuous thus makes creating a pivot table currently not very weildly due to the length it would be. To fix this, we are first going to viusalize the distribution of who survived based on age and then split age up into bands so that we can see the survival probability distribution of certain age brackets.
# 
# **Observations**
# 
# It is clear that the Age plays a significant part in who survives as seen in that over half of those from infant age until 16 survived, while less than 10% of those over 64 survived.

# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby('AgeBand').mean().sort_values(by='AgeBand', ascending=True)


# After this analysis, we will not need the AgeBand column anymore so we will drop it.

# In[ ]:


del train_df['AgeBand']


# ## Feature Engineering and Creation
# 
# We have now discovered that Age, Sex, and Pclass all have strong coorelations to survival ratings, so we decide to include them in our model. 
# 
# However, in order for models to be trained on this data, the data must be in encoded form according to how it should be weighted, ie. Pclass should not be ordinal as 3rd class doesn't have mean that this characteristic is worth three of a first-class characteristic-it should just be an identifier.
# 
# 1. Let's make an array of train_df and test_df as we will need these changes on both in order for our model testing as well as submitted prediction to take them into account. 
# 2. We are going to create dummy columns then one-hot encode these features as described below:
#   * Age
#   * Sex
#   * Pclass

# In[ ]:


combine = [train_df,test_df]


# In[ ]:


for dataset in combine:
    dataset["Age1"] = 0
    dataset["Age2"] = 0
    dataset["Age3"] = 0
    dataset["Age4"] = 0
    dataset["Age5"] = 0

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age1'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age2'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age3'] = 1
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age4'] = 1
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 80), 'Age5'] = 1


# In[ ]:


for dataset in combine:
    dataset['Male'] = 0
    dataset['Female'] = 0
    
for dataset in combine:
    dataset.loc[dataset['Sex'] == 'male', 'Male'] = 1
    dataset.loc[dataset['Sex'] == 'female', 'Female'] = 1


# In[ ]:


for dataset in combine:
    dataset['Pclass1'] = 0
    dataset['Pclass2'] = 0
    dataset['Pclass3'] = 0
    
for dataset in combine:
    dataset.loc[dataset['Pclass'] == 1, 'Pclass1'] = 1
    dataset.loc[dataset['Pclass'] == 2, 'Pclass2'] = 1
    dataset.loc[dataset['Pclass'] == 3, 'Pclass3'] = 1


# ## Cleaning Up Data
# 
# Now that we have one-hot encoded Age, Sex, and Pclass, we can drop the Age, Sex, and Pclass columns from the tables that we are working with.
# 
# We also are going to drop the columns that we are not working on currently to create a base model centered on Age, Sex, and Pclass that we can build upon.
# However, we are going to save the Passenger Ids of test_df so that we can use them in the submission process.
# 
# **Dropping:**
# * Passenger_Id
# * Age
# * Pclass
# * Sex
# * Name
# * SibSp
# * Parch
# * Ticket
# * Fare
# * Cabin
# * Embarked
# 
# **Saving:**
# * test_df's Passenger_Id as Test_Pass_Id

# In[ ]:


Test_Pass_Id = test_df['PassengerId']


# In[ ]:


train_df = train_df.drop(['PassengerId', 'Age', 'Pclass', 'Sex', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.drop(['PassengerId', 'Age', 'Pclass', 'Sex', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)


# ## Model Fitting and Evaluation
# 
# Our data is all in order now and we are going to run through the different models that we have at our disposal, training each one on the data and looking at its accuracy.
# 
# Models to test:
# * Logistic Regression
# * Support Vector Machines
# * Random Forest Classifier
# 
# Steps:
# 1. Split train_df into train_X and train_Y which will be all data other than Survived and Survived respectively.
# 2. Split the data into train_X, train_Y, test_Y, and test_Y.
# 3. Fit each model to train_X and train_Y, then score with test_X and test_Y.
# 4. Then use cross_val_score for each model to use cross validation with each model to ensure that the fit is not skewed by which parts of train_df are used as X and Y respectively.
# 5. Compare these two scores for each, but ultimately use the cross validation score to rank models.

# ### Logistic Regression

# In[ ]:


train_X = train_df.drop(['Survived'], axis=1)
train_Y = train_df['Survived']

# With Cross Validation
lreg = LogisticRegression()
acc_logreg_cross_val = cross_val_score(lreg, train_X,train_Y, cv=5).mean()

# With Single Split
lreg = LogisticRegression()
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
lreg.fit(train_X, train_Y)
acc_logreg_single_split = lreg.score(test_X, test_Y)

print('Single Split:', acc_logreg_single_split, 'Cross Validation:', acc_logreg_cross_val)


# ### Support Vector Machines

# In[ ]:


train_X = train_df.drop(['Survived'], axis=1)
train_Y = train_df['Survived']

# With Cross Validation
svc = SVC(C = 0.1, gamma=0.1)
acc_svc_cross_val = cross_val_score(svc, train_X,train_Y, cv=5).mean()

# With Single Split
svc = SVC(C = 0.1, gamma=0.1)
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
svc.fit(train_X, train_Y)
acc_svc_single_split = svc.score(test_X, test_Y)

print('Single Split:', acc_svc_single_split, 'Cross Validation:', acc_svc_cross_val)


# ### Random Forest Classifier

# In[ ]:


train_X = train_df.drop(['Survived'], axis=1)
train_Y = train_df['Survived']

# With Cross Validation
rand_forest = RandomForestClassifier()
acc_rf_cross_val = cross_val_score(rand_forest, train_X,train_Y, cv=5).mean()

# With Single Split
rand_forest = RandomForestClassifier()
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
rand_forest.fit(train_X, train_Y)
acc_rf_single_split = rand_forest.score(test_X, test_Y)

print('Single Split:', acc_rf_single_split, 'Cross Validation:', acc_rf_cross_val)


# ## Model Evaluation and Submission

# In comparing the cross validation accuracies of our three models, we decide that Random Tree has the highest confidence score and is thus the model we would like to predict the data which we are going to submit with.

# In[ ]:


compare_models = pd.DataFrame({'Model': ['Logistic Regression', 'SVC', 'Random Tree'], 'Score' : [acc_logreg_cross_val, acc_svc_cross_val, acc_rf_cross_val]})
compare_models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({'PassengerId' : Test_Pass_Id, 'Survived' : rand_forest.predict(test_df)})
submission.to_csv('titanic_csv_submission.csv', index=False)

