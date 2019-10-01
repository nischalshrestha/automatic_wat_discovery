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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# In[ ]:


test.head()


# In[ ]:


train.head()


# # **2. Exploring Data**

# In[ ]:


print(test.describe()) # test set
print(test.shape)


# In[ ]:


print(train.describe())
print(train.shape)


# ## **2.1. Survival Rate by Gender**

# In[ ]:


# Passengers that survived vs passengers that passed away
print(train['Survived'].value_counts())

# As proportions
print(train['Survived'].value_counts(normalize=True))

# Males that survived vs males that passed away
print(train['Survived'][train['Sex'] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train['Survived'][train['Sex'] == 'female'].value_counts())

# Normalized male survival
print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize=True))

# Normalized female survival
print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize=True))


# ## **2.2. Survival by Age**

# In[ ]:


# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train['Child'][train['Age'] < 18] = 1
train['Child'][train['Age'] >= 18] = 0

# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# # Print normalized Survival Rates for passengers 18 or older
print(train['Survived'][train['Child'] == 0].value_counts(normalize = True))


# # **3. Naive Prediction: Females survive vs. Males die**

# **Comment:** The pass away is majority. By the majority rules, the naive prediction is always saying "pass away". When we take a look on Gender, females have more than 50% of survive (74.20%) when males have less than 50% (18.89%). 
# It motivates us to always predict "survived" for females, and "die" for males (as in the sample result).
# 
# **The idea is find the subset with high survival rate. **

# In[ ]:


# Create a copy of test: test_one
test_one = test

# Initialize a Survived column to 0
test_one['Survived'] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one['Survived'][test_one['Sex'] == 'female'] = 1

test_one['Survived'].head()


# # **4. Decision Tree Preduction **

# Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat. The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations in that node determine how to predict for new observations that end up in that terminal node.

# In[ ]:


# Import the Numpy library
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree


# ### **4.1. Cleaning and Formatting Data**
# * Missing values should be imputed (by median or majority)
# * Categorical Variables should be numeric

# In[ ]:


print(train['Sex'].describe())
print(train['Embarked'].describe())
print(train['Embarked'].unique())


# In[ ]:


train.columns


# In[ ]:


train['Age'].unique()


# In[ ]:


# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train['Embarked'].fillna("S")
train['Age'] = train['Age'].fillna(train['Age'].median())

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

print(train['Sex'].head(5))
print(train['Embarked'].head(5))


# ### **4.2. First Decision Tree**
# 1. Create `my_tree_one` as `DecisiontreeClassifier` object
# 2. `fit` method on `DecisiontreeClassifier` taking `np.array` as input
# 3. Request attribute `feature_importances_` to see the weigh by importances for each included features
# 4. Compute **mean accuracy** by calling the method score

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


# ** Comment:** The `Fare` is the most important feature

# ### **4.3. Prediction**

# In[ ]:


# on the test set, also manipulate nan values
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

# categorical var into numeric
test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1


# In[ ]:


# Extract the features from the test set: Pclass, Sex, Age, and Fare (to numpy.array)
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("solution_1.csv", index_label = ["PassengerId"])


# ### **4.4. Control Overfitting**
# 
# The default decision tree set `max_depth` and `min_samples_split` as none, which is not good. Because more complex model means more overfitting. Thus, the above model performs worse than the naive model in gender

# In[ ]:


# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))


# In[ ]:


# Convert the Embarked classes to integer form
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


# In[ ]:


# Extract the features from the test set: Pclass, Sex, Age, and Fare (to numpy.array)
test_features_two = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Make your prediction using the test set
my_prediction_two = my_tree_two.predict(test_features_two)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_two = pd.DataFrame(my_prediction_two, PassengerId, columns=["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)


# ### **4.5. Feature-enginerring**
# Creatively engineering your own features by combining the different existing variables

# In[ ]:


train_two = train.copy()
train_two["family_size"] = train_two['SibSp'] + train_two['Parch'] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))


# # **5. Random Forecast**
# The Random Forest technique handles the overfitting problem you faced with decision trees. It grows multiple (very deep) classification trees using the training set. At the time of prediction, each tree is used to come up with a prediction and every outcome is counted as a vote. For example, if you have trained 3 trees with 2 saying a passenger in the test set will survive and 1 says he will not, the passenger will be classified as a survivor. This approach of overtraining trees, but having the majority's vote count as the actual classification decision, avoids overfitting.

# In[ ]:


# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))


# In[ ]:


''' (7) Compare and Interpret'''

#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))


# 

# 
