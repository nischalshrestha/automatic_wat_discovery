#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use('ggplot')

def preprocess_dataset(dataset):
    # Remove PassengerId, Name, Ticket and Cabin. 
    dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # Convert Sex and Embarked to numerical value:
    replace = {'Sex': {'male': 0, 'female': 1}, 'Embarked': {'C': 0, 'Q': 1, 'S': 2}}
    dataset = dataset.replace(replace)
    # Fix Embarked (there are NaN):
    dataset.Embarked = dataset.Embarked.fillna(2)
    return dataset


# ## Dataset
# 
# Load the datasets:

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
dataset = pd.concat([test_df, train_df])


# Take a look in the dataset:

# In[ ]:


print('COLUMNS', dataset.columns.values)
print('')
dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.describe(include=['O'])


# In[ ]:


for column in ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'Ticket']:
    print(column, 'unique values:', dataset[column].unique())


#  - **passengerid**: integer
#  - **survived**: 0 or 1
#  - **pclass** (ticket class): 1, 2 or 3 1st = Upper 2nd = Middle 3rd = Lower
#  - **name** (passeger name): string
#  - **sex** (passenger sex): 'male' or 'female'
#  - **age** (passenger age): NaN and float
#  - **sibsp** (number of siblings/spouses aboard the titanic): 1 0 3 4 2 5 8
#  - **parch** (number of parents/children aboard the titanic): 0 1 2 5 3 4 6
#  - **ticket**: string, alphanumeric
#  - **fare**: float
#  - **cabin** (cabin number): NaN, string, alphanumeric
#  - **embarked** (port of embarkation): NaN, C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Types:
# 
# - **categorical**: survived, pclass, sex, embarked
# - **continous**: age, fare
# - **discrete**: sibsp, parch
# - **alphanumeric**: ticket, cabin, name
# - **Null, NaN**: cabin, embarked, age

# **PREPROCESSING TODO**
# 
# - [X] Remove PassangerId as it doesn't mean anything for us
# - [X] Remove Ticket
# - [X] Remove Cabin
# - [X] Remove Name
# - [X] Convert Sex to numerical value
# - [X] Convert Embarked to numerical value
# - [ ] Fix Age (there are NaN)
# - [X] Fix Embarked (there are NaN)
# - [ ] Convert Age to some ranges
# - [ ] Convert Fare to some ranges
# - [ ] Maybe we can try to use Cabin and Ticket later. I don't know if they mean something.

# In[ ]:


# Remove PassengerId, Name, Ticket and Cabin. 
dataset = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# Convert Sex and Embarked to numerical value:
replace = {'Sex': {'male': 0, 'female': 1}, 'Embarked': {'C': 0, 'Q': 1, 'S': 2}}
dataset = dataset.replace(replace)


# In[ ]:


print(dataset.Embarked.isnull().sum())
dataset.groupby('Embarked').Embarked.count()


# In[ ]:


dataset.Embarked = dataset.Embarked.fillna(2)


# In[ ]:


dataset.Embarked.isnull().sum()


# In[ ]:


from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataset, alpha=0.2, figsize=(30, 30), diagonal='hist')


# Fix Age (there are NaN). Select some Age ranges:

# In[ ]:


print(dataset.Age.isnull().sum())
print(dataset.Age.min(), dataset.Age.max())


# In[ ]:


dataset.Age.hist(bins=8)


# In[ ]:


print('AGE MEAN:', dataset.Age.mean(), 'AGE STD: ', dataset.Age.std())


# In[ ]:


temp = dataset.copy(deep=True)
temp.Age = temp.Age.fillna(-1)
temp[temp.Age == -1][['Age', 'Pclass']].groupby('Pclass').count()


# In[ ]:


dataset[dataset.Pclass == 3].Age.hist(bins=10)


# In[ ]:


dataset[dataset.Pclass == 2].Age.hist(bins=10)


# In[ ]:


dataset[dataset.Pclass == 1].Age.hist(bins=10)


#  Ok, so we identified that there are more missing data in the 3 class and identified the distribution in the 3 classes.

# In[ ]:





# In[ ]:


train_df = preprocess_dataset(train_df)
test_df = preprocess_dataset(test_df)
test_df.head()


# Train using:
# 
# * [ ] Linear Regression
# * [ ] Logistic Regression
# * [ ] SVM
# * [ ] Random Forrest
# * [ ] Neural Networks
