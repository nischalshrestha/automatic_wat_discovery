#!/usr/bin/env python
# coding: utf-8

# **Titanic dataset** is more or less like the "Hello World!" of Data Science and Machine Learning. This is a beginner friendly tutorial for anyone who wants to explore Titanic Dataset. 

# In[ ]:


# Let's first do the necessary imports.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


# Load the train and test datasets to create two dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **Time to understand the structure of your data**
# 
# .describe() summarizes the columns/features of the DataFrame, including the count of observations, mean, max and so on. 
# 
# .shape gives the dimensions of the dataframe

# In[ ]:


train.describe() 


# In[ ]:


train.shape


# **Data Dictionary** (to understand more please visit: https://www.kaggle.com/c/titanic/data)
# 
#       Variable         Definition                          Key
# 
#       survival         Survival                            0 = No, 1 = Yes
#       pclass           Ticket class                        1 = 1st, 2 = 2nd, 3 = 3rd
#       sex              Sex    
#       Age              Age in years    
#       sibsp            # of siblings / spouses aboard the Titanic    
#       parch            # of parents / children aboard the Titanic    
#       ticket           Ticket number    
#       fare             Passenger fare    
#       cabin            Cabin number    
#       embarked         Port of Embarkation                 C = Cherbourg, Q = Queenstown, S=  Southampton
# 
# 

# Before we dive deeper in, let's see if we can answer some basic questions from data directly. 
# 
# * **How many people in your training set survived the disaster with the Titanic? **
# * **Number of males that survived vs number of males who did not survive**
# * **Number of femailes that survived vs number of females who did not survive**
# * **Does Age play a role?**

# **How many people in your training set survived the disaster with the Titanic? **

# In[ ]:


# absolute numbers
print(train["Survived"].value_counts())

# percentages
print(train["Survived"].value_counts(normalize = True))


# You will see that 549 individuals died which is about 61.61 % and 342 survived which is about 38.38 %.  

# **Number of males that survived vs number of males who did not survive**
# 
# **Number of femailes that survived vs number of females who did not survive**

# In[ ]:


# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))


# It looks like it makes sense to predict that all females will survive, and all men will die. (A novice prediction!)

# **Does age play a role?**
# 
# Another variable that could influence survival is age; since it's probable that children were saved first. You can test this by creating a new column with a categorical variable Child. Child will take the value 1 in cases where age is less than 18, and a value of 0 in cases where age is greater than or equal to 18.

# In[ ]:


# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column

train["Child"][train["Age"] < 18] = 1
train["Child"][train["Age"] >= 18] = 0

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


# As you can see from the survival proportions, age does certainly seem to play a role.

# **Simple Decision Tree**
# 
# We will need the following to build a decision tree
# * target: A one-dimensional numpy array containing the target/response from the train data. (Survival in this case)
# * features: A multidimensional numpy array containing the features/predictors from the train data. (ex. Sex, Age)
# 

# In[ ]:


# Print the train data to see the available features
print(train.columns.values)


# Let us choose our target and feature variables. 
# * Target: Survived Column
# * Feature: Passenger, Class, Sex, Age, Fare
# * Build a decision tree tree_one to predict survival using features and target.
# * Look at the importance of features in your tree and compute the score 

# Before we proceed further with Decision Tree we need to do some cleanup. Let's first check for missing values in any of the columns that we will be using in our feature array. 

# In[ ]:


# Import 'tree' from scikit-learn library
from sklearn import tree


# Let's check for missing values. 

# In[ ]:


train.isnull().sum()


# For our feature array we are using Age column and it has 177 missing values. Let's first impute values on this column with the median of all present values.

# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].median())


# In[ ]:


# Let us take another look at the values in dataset again.
train.head()


# Sex and Embarked are categorical values but they are in non-numeric format. We need to convert them to numeric format. Emarked also has some missing values and we need to impute them with the majority value that is 'S' before converting to numeric format. 

# In[ ]:


# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2


# In[ ]:


# Let's take a look at train.head() again. 
train.head()


# It looks good, so let's proceed with creating target and features array.

# In[ ]:


# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


# Based on our tree, we can see that "Fare" contributed more than other features. That is an analysis we are driving out from this decision tree. 
# 
# Let's now take a look at our test dataset and see if there are any null values. 

# In[ ]:


# Looking at test.info() we see that Fare has one null value. 
test.isnull().sum()


# Age has 86 null values and Fare has 1 null value. Let's do imputation similar to what we did for train set. 

# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1


# Let's extract features from test set and make our prediction based on my_tree_one model.

# In[ ]:


# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# In[ ]:


get_ipython().system(u'ls')


# You can upload my_solution_one.csv to kaggle. 

# I have written this code based on the following free tutorial on datacamp.  https://www.datacamp.com/community/open-courses/kaggle-python-tutorial-on-machine-learning I hightly recommend you take this beginner friendly course from datacamp. It further goes onto introduce Random Forest which gives a much better prediction score.

# In[ ]:




