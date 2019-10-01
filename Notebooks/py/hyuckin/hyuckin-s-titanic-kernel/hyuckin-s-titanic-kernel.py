#!/usr/bin/env python
# coding: utf-8

# **Hello! This is my first Kaggle Kernel and my first foray into random forests.**

# This is my first iteration, which is a simple version of the decision tree adapted from the datacamp tutorial.
# 
# This iteration is a simple decision tree that uses Sex, Age, Passenger Class, and Fare as variables.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import tree

# Copy the training and testing data to 'train' and 'test' respectively
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},)
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},)

# Impute missing variables with the median value
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Convert categorical data to discrete integers
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

# Create arrays to store target and feature values
target = train["Survived"].values
features1 = train[["Sex", "Age", "Pclass", "Fare"]].values

# Create a tree using target and features
decTree1 = tree.DecisionTreeClassifier()
decTree1 = decTree1.fit(features1, target)
print("Solution 1 score:")
print(decTree1.score(features1, target))
print("Feature importances: Sex, Age, Pclass, Fare")
print(decTree1.feature_importances_)

# Extract the test feature data
test_features1 = test[["Sex", "Age", "Pclass", "Fare"]].values

# Make the prediction using the data tree
prediction1 = decTree1.predict(test_features1)

# Copy the prediction into a DataFrame
PassengerId = np.array(test["PassengerId"]).astype(int)
solution1 = pd.DataFrame(prediction1, PassengerId, columns = ["Survived"])

# Export the solution to a csv file
solution1.to_csv('solution1.csv', index_label= ["PassengerId"])


# This submission scored a 0.72727 on the public leaderboard.

# I will now take the number of siblings and spouses aboard (SibSp), the number of parents and children aboard (Parch), port of embarkation (Embarked), and Cabin into account. For Cabin, I will assign a 1 if the person had a cabin and a 0 if not.

# In[ ]:


# Impute missing variables with mode 
train["Embarked"] = train["Embarked"].fillna("S")

# Convert categorical data to discrete integers
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

train.loc[pd.notnull(train["Cabin"]), "Cabin"] = 1
train.loc[pd.isnull(train["Cabin"]), "Cabin"] = 0
test.loc[pd.notnull(test["Cabin"]), "Cabin"] = 1
test.loc[pd.isnull(test["Cabin"]), "Cabin"] = 0

# Create array to store feature values
features2 = train[["Sex", "Age", "Pclass", "Fare", "Embarked", "Cabin", "SibSp", "Parch"]].values

# Create a tree using target and features
decTree2 = tree.DecisionTreeClassifier()
decTree2 = decTree2.fit(features2, target)
print("Solution 2 score:")
print(decTree2.score(features2, target))
print("Feature importances: Sex, Age, Pclass, Fare, Embarked, Cabin, SibSp, Parch")
print(decTree2.feature_importances_)

# Extract the test feature data
test_features2 = test[["Sex", "Age", "Pclass", "Fare", "Embarked", "Cabin", "SibSp", "Parch"]].values

# Make the prediction using the data tree
prediction2 = decTree2.predict(test_features2)

# Copy the prediction into a DataFrame
PassengerId = np.array(test["PassengerId"]).astype(int)
solution2 = pd.DataFrame(prediction2, PassengerId, columns = ["Survived"])

# Export the solution to a csv file
solution2.to_csv('solution2.csv', index_label= ["PassengerId"])


# This submission scored a 0.67464 on the public leaderboard, lower than my previous attempt. This suggests that the decision tree overfitted the data, which makes sense as we introduced more features. To reduce the number of features, I will remove "Embarked," which had the lowest feature importance value of  0.013. It also intuitively makes sense that the place of embarkation would not be a major contributive factor to survival. I will also combine "SibSp" and "Parch" into one variable, "famSize". 

# In[ ]:


# Add famSize column to train and test, where famSize = SibSp + Parch
train = train.assign(famSize = train["SibSp"] + train["Parch"])
test = test.assign(famSize = test["SibSp"] + test["Parch"])

# Create array to store feature values
features3 = train[["Sex", "Age", "Pclass", "Fare", "Cabin", "famSize"]].values

# Create a tree using target and features
decTree3 = tree.DecisionTreeClassifier()
decTree3 = decTree3.fit(features3, target)
print("Solution 3 score:")
print(decTree3.score(features3, target))
print("Feature importances: Sex, Age, Pclass, Fare, Cabin, famSize")
print(decTree3.feature_importances_)

# Extract the test feature data
test_features3 = test[["Sex", "Age", "Pclass", "Fare", "Cabin", "famSize"]].values

# Make the prediction using the data tree
prediction3 = decTree3.predict(test_features3)

# Copy the prediction into a DataFrame
PassengerId = np.array(test["PassengerId"]).astype(int)
solution3 = pd.DataFrame(prediction3, PassengerId, columns = ["Survived"])

# Export the solution to a csv file
solution3.to_csv('solution3.csv', index_label= ["PassengerId"])


# This submission scored a 0.59809 on the public leaderboard, which suggests that manual feature selection is not productive. I will try again with different sets of features:

# Sex, Age, Fare, famSize: 0.64593

# Sex, Age, Fare: 0.68900

# 

# In[ ]:




