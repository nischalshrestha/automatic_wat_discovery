#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn, sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

sns.set_style('whitegrid')

##from subprocess import check_output
##print(check_output(["ls", "../input"]).decode("utf8"))

#### import the data
test   = pd.read_csv('../input/test.csv')
train    = pd.read_csv('../input/train.csv')

# Any results you write to the current directory are saved as output.


# In[ ]:


#


# In[ ]:


#train.head()
#train[train['Survived'] == 1]['Name']


# In[ ]:


##train['Ticket_only_number'] = np.where(train['Ticket'].str.isdigit(), 1, 0)
train['Ticket_group'] = np.where(train['Ticket'].str.isdigit(), train['Ticket'].astype(str).str[0], train['Ticket'].str[:1])
train['Ticket_length'] = train['Ticket'].apply(lambda x: len(x))
test['Ticket_group'] = np.where(test['Ticket'].str.isdigit(), test['Ticket'].astype(str).str[0], test['Ticket'].str[:1])
test['Ticket_length'] = test['Ticket'].apply(lambda x: len(x))

train["NameLength"] = train["Name"].apply(lambda x: len(x))
test["NameLength"] = test["Name"].apply(lambda x: len(x))

fig, (axis1) = plt.subplots(1,1,figsize=(10,5))

ticket_group_mean = train[["Ticket_group", "Survived"]].groupby(['Ticket_group'],as_index=False).mean().sort('Survived')
sns.barplot(x='Ticket_group', y="Survived", data=ticket_group_mean, palette="Set3", ax=axis1)


# In[ ]:


##train['Ticket_number_details'] = train['Ticket'].apply(lambda x: len(x))
###train['Ticket_group'] = np.where(train['Ticket'].str.isdigit(), 'only number', train['Ticket'].str[:1])
##
##ticket_group_det_mean['number_only'] = np.where(train['Ticket'].str.isdigit(), 1, 0)
##train_sample = ticket_group_det_mean[ticket_group_det_mean['number_only'] == 1]
##
##fig, (axis1, axis2) = plt.subplots(2,1,figsize=(10,10))
##
##ticket_group_det_mean = train_sample[["Ticket_number_details", "Survived"]].groupby(['Ticket_number_details'],as_index=False).mean()
##sns.barplot(x='Ticket_number_details', y="Survived", data=train, palette="Set3", ax=axis1)
##sns.countplot(x='Ticket_number_details', hue = 'Survived', data=train, palette="husl", ax=axis2)


# In[ ]:


########## this counts the number of spaces in the Name column
import re

at = re.compile(r" ", re.I)
def count_spaces(string):
    count = 0
    for i in at.finditer(string):
        count += 1
    return count

train["spaces_in_name"] = train["Name"].map(count_spaces)
test["spaces_in_name"] = test["Name"].map(count_spaces)


# In[ ]:


# This function returns the title from a name.
def title(name):
# Search for a title using a regular expression. Titles are made of capital and lowercase letters ending with a period.
    find_title = re.search(' ([A-Za-z]+)\.', name)
# Extract and return the title If it exists. 
    if find_title:
        return find_title.group(1)
    return ""

train["Title"] = train["Name"].apply(title)
test["Title"] = test["Name"].apply(title)


# In[ ]:


#train.head(5)

#### here I want a univariate p-value analysis between the target and the variable 

train['Cabin_first_ltr'] = np.where(train['Cabin'].isnull(), 'Null', 'Not Null')
##train['Parch_grouped'] = np.where(train['Parch'] > 0, '1', '0')
train['FamilySize'] = train['SibSp'] + train['Parch']
train['withfamily'] = np.where(train['FamilySize'] > 0, 1, 0)
train['Female'] = np.where(train['Sex'] == 'female', 1, 0)

train['miss'] = np.where(train['Name'].str.contains("Miss. "), 1, 0)
train['mrs'] = np.where(train['Name'].str.contains("Mrs. "), 1, 0)



#df[df['date'].astype(str).str.contains('07311954')]
#train['Cabin_number'] = np.where(train['Cabin'].isnull(), 'Null', 'Not Null') #train['Cabin'].str[1:])

#train['Cabin_first_ltr'] = train['Cabin'].str[:1]
#train['Cabin_first_ltr'][train['Cabin'].isnull()] = 'Null'

#df['Age_Group'][df['Age'] > 40] = '>40'

fig, (axis1, axis2, axis3, axis4) = plt.subplots(4,1,figsize=(5,15))
sns.countplot(x='spaces_in_name', hue = 'Survived', data=train, palette="husl", ax=axis1)
sns.countplot(x='mrs', hue = 'Survived', data=train, palette="husl", ax=axis2)
#sns.countplot(x='dr', hue = 'Survived', data=train, palette="husl", ax=axis3)

#### Look at the % survived 
mrs_mean = train[["mrs", "Survived"]].groupby(['mrs'],as_index=False).mean()
sns.barplot(x='mrs', y="Survived", data=mrs_mean, palette="Set3", ax=axis3)

miss_mean = train[["Ticket_group", "Survived"]].groupby(['Ticket_group'],as_index=False).mean()
sns.barplot(x='Ticket_group', y="Survived", data=miss_mean, palette="Set3", ax=axis4)


# In[ ]:


## to run a random forest we need to make sure the dataset doens't contain any missing values.
### does it contain missing values
if train.isnull().values.any() == True:
    print("there are some missing values")
else: 
    print("there are no missing values")


# In[ ]:


## Make adjustments to the test dataset to match the train dataset

test['Cabin_first_ltr'] = np.where(test['Cabin'].isnull(), 'Null', 'Not Null')
test['FamilySize'] = test['SibSp'] + test['Parch']
test['withfamily'] = np.where(test['FamilySize'] > 0, 1, 0)
test['Female'] = np.where(test['Sex'] == 'female', 1, 0)

test['miss'] = np.where(test['Name'].str.contains("Miss. "), 1, 0)
test['mrs'] = np.where(test['Name'].str.contains("Mrs. "), 1, 0)


# In[ ]:


fig, (axis1) = plt.subplots(1,1, figsize = (5, 5))
FamilySize_mean = train[["withfamily", "Survived"]].groupby(["withfamily"], as_index = False).mean()
sns.barplot(x = 'withfamily', y = 'Survived', data = FamilySize_mean, palette = "Set3", ax = axis1)


# In[ ]:


##### this removes the missing values
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    


### this will transfer the categorical variables to floats for the algo
def do_treatment(df):
    for col in df:
        if df[col].dtype == np.dtype('O'):
            df[col] = df[col].apply(lambda x : hash(str(x)))

    


# In[ ]:



train_imputed = DataFrameImputer().fit_transform(train)
test_imputed = DataFrameImputer().fit_transform(test)

do_treatment(train_imputed)
do_treatment(test_imputed)


# In[ ]:


##train_imputed.head()
#### this tells us which format each of the variables are in 
##train_imputed.columns.to_series().groupby(train_imputed.dtypes).groups


# In[ ]:


##train_independent_vars.head()


# In[ ]:


## Make adjustments to the test dataset to match the train dataset

test['Cabin_first_ltr'] = np.where(test['Cabin'].isnull(), 'Null', 'Not Null')
test['Parch_grouped'] = np.where(test['Parch'] > 0, '1', '0')
test['withfamily'] = np.where(test['FamilySize'] > 0, 1, 0)

test['miss'] = np.where(test['Name'].str.contains("Miss. "), 1, 0)
test['mrs'] = np.where(test['Name'].str.contains("Mrs. "), 1, 0)


# In[ ]:


train_imputed['withfamily'].unique()


# In[ ]:


##### is there any major collinarity between the variables? 

import statsmodels.api as sm

train_cols = ['Title', 'NameLength', 'Pclass', 'Female', 'Age', 'Ticket_group', 'Cabin_first_ltr']

vals_removed_bc_p_value_too_small = ['Ticket_length', 'withfamily', 'Fare', 'Embarked', 'spaces_in_name']

####### 
logit = sm.Logit(train_imputed['Survived'].astype(float), train_imputed[train_cols].astype(float))
result = logit.fit()


# In[ ]:


result.summary()


# In[ ]:


######## Creating the random forest model 
# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 200, max_features = 'sqrt',
                             max_depth = None, verbose = 1, n_jobs = -1)

# Fit the training data to the Survived labels and create the decision trees
#train_independent_vars = train_imputed.drop(['Survived'], axis = 1)
train_independent_vars = train_imputed[['Ticket_length', 'Title', 'NameLength', 'Pclass', 'Female', 'Age', 'withfamily', 'Ticket_group', 'Fare', 'Embarked', 'Cabin_first_ltr', 'spaces_in_name']]
train_independent_vars = train_independent_vars

train_dependent_vars = train_imputed['Survived']

forest = forest.fit(train_independent_vars, train_dependent_vars)

# Take the same decision trees and run it on the test data
output = forest.predict(train_imputed[['Ticket_length', 'Title', 'NameLength', 'Pclass', 'Female', 'Age', 'withfamily', 'Ticket_group', 'Fare', 'Embarked', 'Cabin_first_ltr', 'spaces_in_name']])

### combine the passengerid with the prediction
output_df = pd.DataFrame(test_imputed.PassengerId).join(pd.DataFrame(output))
output_df.columns = ['PassengerId', 'Survived']
#### create the final output dataframe
final_output = DataFrame(columns=['PassengerId', 'Survived'])
final_output = final_output.append(output_df[['PassengerId', 'Survived']])
#### convert to csv
final_output.to_csv('output.csv', index = False, header = ['PassengerId', 'Survived'])


# In[ ]:


#
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in indices:
    print(train_independent_vars.columns[f], importances[f])


# In[ ]:


#


# In[ ]:


#


# In[ ]:


#


# In[ ]:


#


# In[ ]:


#


# In[ ]:




