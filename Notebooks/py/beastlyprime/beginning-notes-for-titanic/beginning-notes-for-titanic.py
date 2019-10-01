#!/usr/bin/env python
# coding: utf-8

# # Data processing
# 
# ## Display the data
# 
# * Use `.describe()`
# 
#     In this way we can get some statistical characters of data. When the count of one column is less than the whole, it may have missing values ( See column "Age" ).
# 
# * Use `.unique()`
# 
#     See how many different values are in the data.
# 
# * Use `.value_counts()`
# 
#    See how many different values are in the data, and their counts.
# 
# ## Initialize a new column
# 
# * Use 0
# * Use "NaN"

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train["family_size"] = 0
print("Unique:", train["family_size"].unique())
print("Value Counts:", train["family_size"].value_counts())
train["family_size"] = float("NaN")
test["family_size"] = float("NaN")
print(train.describe())
print(test.describe())


# ## Select data
# 
# * Use a second `[]`-operator to serve a boolean test

# In[ ]:


# Convert the male and female groups to integer form
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
print(train["Sex"].value_counts())


# ## Cleaning
# **Null** 
# 
# * Use pandas.DataFrame method: .fillna()
# 
# 

# In[ ]:


train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test.Fare = test.Fare.fillna(test["Fare"].median())


# ## Save data
# 
# * Use [pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
# 
#     pd.DataFrame( data, index, columns, dtype, copy )

# In[ ]:


ID = np.arange(0,10)
age = np.arange(10,20)
people = pd.DataFrame(age, ID, columns = ["Age"])
people.to_csv("people.csv", index_label = ["ID"])


# # Models
# 
# ## Decision Tree

# In[ ]:


# Convert the Embarked classes to integer form
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
test.loc[train["Embarked"] == "S", "Embarked"] = 0
test.loc[train["Embarked"] == "C", "Embarked"] = 1
test.loc[train["Embarked"] == "Q", "Embarked"] = 2

# Add some new features
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1

# Construct features and the target
features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
features_test = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values
target = train["Survived"]

# Train on a tree
decision_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
decision_tree = decision_tree.fit(features, target)
print(decision_tree.score(features, target))

# Make prediciton
prediction_dt = decision_tree.predict(features_test)
PassengerId =np.array(test["PassengerId"]).astype(int)
solution_dt = pd.DataFrame(prediction_dt, PassengerId, columns = ["Survived"])
solution_dt.to_csv("solution_dt.csv", index_label = ["PassengerId"])

