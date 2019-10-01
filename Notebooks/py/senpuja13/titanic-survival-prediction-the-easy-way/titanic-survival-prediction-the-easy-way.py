#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **LOAD THE DATA FIRST**

# In[ ]:


train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
train_set.head(5)


# Let's observe the training and test dataset first.

# In[ ]:


train_set.shape


# In[ ]:


train_set.info()


# In[ ]:


test_set.head(5)


# In[ ]:


test_set.shape


# In[ ]:


test_set.info()


# **Observations:**
# 1. Number of rows in the train dataset = 891 
# 2. Number of columns in the train dataset = 12 
# 3. Some data missing in some columns like Age, Cabin, and Embarked
# 4. Number of rows in the test dataset = 418 
# 5. Number of columns in the test dataset = 11
# 6. The missing column in the test dataset is the Survived column whose values of course we need to find out. 
# 7. Some values are missing in other columns like Age, Fare, Cabin and Embarked

# ** Feature Engineering¶**
# Feature vectors are used to represent numeric or symbolic characteristics (called features, features are basically the measurable properties).
# 
# Feature engineering is the process of using domain knowledge of the data to create feature vectors that make machine learning algorithms work.
# 
# What are the things that we need to do here?
# 
# Guess the missing values first
# Then map everything into possible numeric values
# 
# Let's see how we can do it.

# In[ ]:


train_set.head(5)


# Let's combine both the datasets as we are gonna perform all the operations on both the sets.

# In[ ]:


train_test_dataset = [train_set, test_set]


# The only value that is useful in the Name of a person in this case is the title of that person. So we will extract the title and map it accordingly

# In[ ]:


for dataset in train_test_dataset:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train_set['Title'].value_counts()


# In[ ]:


test_set['Title'].value_counts()


# **Map the title**
# Mr : 0 Miss : 1 Mrs: 2 Master: 3 Others: 4

# In[ ]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4,
                 "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }
for dataset in train_test_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


train_set.head(5)


# In[ ]:


test_set.head(5)


# we can drop the name from both the datasets as it's not needed now

# In[ ]:


train_set.drop('Name', axis = 1, inplace = True)
test_set.drop('Name', axis = 1, inplace = True)


# In[ ]:


train_set.head(5)


# In[ ]:


test_set.head(5)


# **Map the sex**

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_dataset:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


train_set.head(5)


# In[ ]:


test_set.head(5)


# **Do something about the age**¶
# Fill up the missing values and then map them.
# Let's fill up the missing values with the median of passengers' ages by grouping according to their title.

# In[ ]:


train_set['Age'].fillna(train_set.groupby("Title")['Age'].transform("median"), inplace = True)
test_set['Age'].fillna(test_set.groupby("Title")['Age'].transform("median"), inplace = True)


# In[ ]:


train_set.head(20)


# **Map the Age**¶
# kids:0 teenagers: 1 Adults: 2 Middle-Aged: 3 Old: 4

# In[ ]:


for dataset in train_test_dataset:
    dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 20), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 35), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 50), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4


# In[ ]:


train_set.head(5)


# **Map the embarked after filling up the missing values**

# In[ ]:


Pclass1 = train_set[train_set['Pclass']==1]['Embarked'].value_counts()
Pclass1


# In[ ]:


Pclass2 = train_set[train_set['Pclass']==2]['Embarked'].value_counts()
Pclass2


# In[ ]:


Pclass3 = train_set[train_set['Pclass']==3]['Embarked'].value_counts()
Pclass3


# In[ ]:


for dataset in train_test_dataset:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_dataset:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train_set.head(5)


# **Family**
# Calculate the family size as the possibility of a person to survive the tragedy is more if he has a family on board.

# In[ ]:


train_set["Family"] = train_set["SibSp"] + train_set["Parch"] + 1
test_set["Family"] = test_set["SibSp"] + test_set["Parch"] + 1


# In[ ]:


train_set.drop('Parch', axis = 1, inplace = True)
test_set.drop('Parch', axis = 1, inplace = True)
train_set.drop('SibSp', axis = 1, inplace = True)
test_set.drop('SibSp', axis = 1, inplace = True)


# In[ ]:


train_set.head(10)


# In[ ]:


train_set['Family'].value_counts()


# In[ ]:


test_set['Family'].value_counts()


# the minimum family size is 1 and the maximum family size is 11

# In[ ]:


family_mapping = {1: 0, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1, 7: 1.2, 8: 1.4, 9: 1.6, 10: 1.8, 11: 2}
for dataset in train_test_dataset:
    dataset['Family'] = dataset['Family'].map(family_mapping)


# In[ ]:


train_set.head(5)


# **Map the fare**

# In[ ]:


train_set["Fare"].fillna(train_set.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_set["Fare"].fillna(test_set.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


for dataset in train_test_dataset:
    dataset.loc[ dataset['Fare'] <= 15, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


train_set.head(5)


# **Do something about the cabin**

# In[ ]:


train_set['Cabin'].value_counts()


# In[ ]:


Pclass1 = train_set[train_set['Pclass']==1]['Cabin'].value_counts()
Pclass1


# In[ ]:


Pclass2 = train_set[train_set['Pclass']==2]['Cabin'].value_counts()
Pclass2


# In[ ]:


Pclass3 = train_set[train_set['Pclass']==3]['Cabin'].value_counts()
Pclass3


# In[ ]:


for dataset in train_test_dataset:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# what we can notice from this is that First class cabins mostly start with A, B , C and D Second class cabins are D, E, F and third class cabins are E, F and G Now we will map the Cabin and then will fill up the blanks with the median

# In[ ]:


cabin_mapping = {"A": 0, "B": 0.2, "C": 0.4, "D": 0.6, "E": 0.8, "F": 1, "G": 1.2, "T": 1.4}
for dataset in train_test_dataset:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train_set


# In[ ]:


train_set["Cabin"].fillna(train_set.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_set["Cabin"].fillna(test_set.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# Ticket isn't that much needed here, so we'll just drop that column

# In[ ]:


train_set.drop('Ticket', axis = 1, inplace = True)
test_set.drop('Ticket', axis = 1, inplace = True)
train_set.drop('PassengerId', axis = 1, inplace = True)


# In[ ]:


train_data = train_set.drop('Survived', axis=1)
target = train_set['Survived']

train_data.shape, target.shape


# In[ ]:


train_data


# In[ ]:


train_data.info()


# **Data Modelling**
# Here comes the data modelling part. We will be using SVC classifier for this prediction.

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


clfr = SVC()
clfr.fit(train_data, target)

test_data = test_set.drop("PassengerId", axis=1).copy()
prediction = clfr.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()

