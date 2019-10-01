#!/usr/bin/env python
# coding: utf-8

# # From Zero to Random Forest Classifier (Explained)

# ## 1. Importing data

# We start by getting the data with pandas. We also import some other libraries for later use.

# In[ ]:


import pandas as pd
import numpy as np # For linear algebra
import matplotlib.pyplot as plt # For visualization
from sklearn import tree # For creating trees

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Fun fact: We can instantly create a classifier with a 0.76555 public score with the 2 lines of code below.
# It is analogous to gender_submission.csv:
# test["Survived"] = 0
# test["Survived"][test["Sex"] == "female"] = 1


# ## 2. Exploratory Data Analysis (EDA)

# We can now inspect the first 5 rows of the training data.

# In[ ]:


train.head()


# We can go ahead and drop the "Ticket" feature. If the ticket numbers encode anything at all, it would be likely that they encode things like cabin number, ticket class, embarkment port, and these features are already present.

# In[ ]:


train.drop("Ticket", 1, inplace = True)
test.drop("Ticket", 1, inplace = True)


# The `.shape` attribute represents (rows, columns).

# In[ ]:


train.shape


# `.describe()` allows us to easily explore the DataFrame.

# In[ ]:


train.describe()


# Using `value_counts()`, we can inspect the proportion of people who survived and died:

# In[ ]:


train["Survived"].value_counts()


# In[ ]:


train["Survived"][train["Sex"] == "female"].value_counts(normalize = True) # normalize = True returns percentages instead of raw counts


# ## 3. Handling missing data

# Before proceeding, we need to solve issues caused by null values in the data.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# We can start with the 2 `NaN` cases in the "Embarked" column of the training data.

# In[ ]:


fig, ax = plt.subplots()
ax.hist(train["Embarked"])
ax.set_title("Embarked Classes")
plt.show()


# Since "S" is the most common case, we can assign them the value "S".

# In[ ]:


train["Embarked"].fillna("S", inplace = True)


# In order to impute the missing "Age" values, it would be nice to extract the titles from the "Name" field and then correlate titles such as "Master" or "Miss" with a younger age etc... For an implementation of that, check out the kernel [Titanic Dataset Preprocessing](https://www.kaggle.com/wencesvillegas/titanic-dataset-preprocessing) by Wences Villegas.
# 
# However, for the purposes of this kernel, we will replace the empty entries with the median age.

# In[ ]:


median_age_train = train["Age"].dropna().median()
median_age_test = test["Age"].dropna().median()
print(median_age_train, median_age_test)


# As seen above, the median age for the training and test data is off by 1. Since we have to choose one, and the training data contains more records, we will choose `28.0`.

# In[ ]:


train["Age"].fillna(median_age_train, inplace = True)
test["Age"].fillna(median_age_train, inplace = True)


# The test set has one missing "Fare" entry.

# In[ ]:


test["Fare"].fillna(test["Fare"].mean(), inplace = True) # It would have been better of course to use the mean for his/her passenger class.


# Finally, individual cabin codes are not likely to have any predictive power in our problem. It would be better to split the cabin codes into categories based on the starting letter. These letters might encode cabin class which might have predictive power on the survival odds due to it being influenced by social class. 
# 
# Since we don't know the ordering of the cabin categories, if there is any at all, this feature will be a nominal feature. Passengers without a class will have entries of "Unknown".

# In[ ]:


def getCabinCat(cabin_code):
    if pd.isnull(cabin_code):
        cat = "Unknown"
    else:
        cat = cabin_code[0]
    return cat

cabin_cats_train = np.array([getCabinCat(cc) for cc in train["Cabin"].values])
cabin_cats_test = np.array([getCabinCat(cc) for cc in test["Cabin"].values])

# We can now add this as a new "CabinCat" feature in the DataFrames, and remove the "Cabin" feature.
train = train.assign(CabinCat = cabin_cats_train)
train = train.drop("Cabin", axis = 1)
test = test.assign(CabinCat = cabin_cats_test)
test = test.drop("Cabin", axis = 1)

# We can now investigate the distribution of passengers over the cabin categories in the training set:
print("Number of passengers:\n{}".format(train["CabinCat"].groupby(train["CabinCat"]).size()))


# We can now verify that there are no null values remaining.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ## 4. More EDA

# In order to get an idea of how much age played a role in survival, we can create a feature called "Child" as follows.

# In[ ]:


train["Child"] = np.where(train["Age"] < 18, 1, 0)
# NB all passengers with missing ages were imputed an age which was > 18, which is not ideal since some of them could have been children.


# In[ ]:


print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))


# In[ ]:


print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))


# ## 5. Encoding categorical values

# Categorical data has to be encoded before it can be used in statistical models. We can use sklearn's `LabelEncoder` to do that.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
categorical_classes_list = ["Sex", "Embarked"]
#encode features that are cateorical classes
for column in categorical_classes_list:
    le = LabelEncoder()
    le.fit(train[column])
    train[column] = le.transform(train[column])
    test[column] = le.transform(test[column])


# ## 6. First decision tree

# In order to build a decision tree, we need the following:
# * `features`: A multidimensional numpy array containing the features/predictors from the train data.
# * `target`:  A one-dimensional numpy array containing the target/response from the training data (survival in our case)

# In[ ]:


# Create the features and target numpy arrays: features_one, target
features_one = train[["Pclass", "Sex", "Age", "Fare", "Embarked"]].values
target = train["Survived"].values


# We can now create a `tree` object from the `DecisionTreeClassifier` scikit-learn class. We can then fit it to  `features_one` and `target`.

# In[ ]:


my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)


# We can now check the importance of each feature in the decision tree. They will be listed in the order we had them in `features_one`:

# In[ ]:


print(my_tree_one.feature_importances_)


# A quick metric we can get is the mean accuracy using the `.score()` function. Note that this is the accuracy when trying to classify the training **not** the test data:

# In[ ]:


print(my_tree_one.score(features_one, target))


# We can also visualize the decision tree.

# In[ ]:


import graphviz # Open source graph visualization software
dot_data = tree.export_graphviz(my_tree_one, out_file = None) # Instead of outputing a .out file with the description of the model, we store it in a variable.
graph = graphviz.Source(dot_data) 
graph


# We can now use the model to make a prediction on the test set.

# In[ ]:


my_prediction = my_tree_one.predict(test[["Pclass", "Sex", "Age", "Fare", "Embarked"]].values)


# We create a `DataFrame` with two columns: "PassengerId" and "Survived". We then write it to a `.csv` file.

# In[ ]:


PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


# ## 7. Better decision tree

# We don't need to test this prediction in order to realize it will not perform well. The decision tree which we visualized was quite complex and clearly [overfitted](https://www.wikiwand.com/en/Overfitting) the training data.
# 
# This is because the default argument for `max_depth` and `min_samples_split` is `None`.
# 
# The `max_depth` parameter determines when the splitting up of the decision tree stops.
# The `min_samples_split` parameter monitors the amount of observations in a bucket. If a certain threshold is not reached (e.g. minimum 10 passengers) no further splitting can be done.

# In[ ]:


# Create a new array of features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1) # For a good explanation of what the random_state does, see here: https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))


# Note the score of this model on the training set is lower than the previous one's. However, it will be much higher on the test set.

# ## 8. More Feature Engineering

# Before we proceed to make an even better model, it would be a good idea to apply more feature engineering. 
# Some ideas are:
# 1. Creating a "Title" feature based on titles extracted from "Name"
# 2. Creating a feature for age bracket as having so many individual ages in the model might lead to overfitting
# 3. Creating a fare bracket feature, for the same reason as before
# 4. Creating a "Race" feature using `library(wru)`, the who r u library, to guess a person's race from their surname. There might have been racial bias which is worth investigating
# 5. Creating a variable for the number of people on a single ticket to give a count for group size
# 6. Creating a Family size feature which is determined by the variables "SibSp" and "Parch", which indicate the number of family members a certain passenger is traveling with

# For our purposes, we will only create one new predictive attribute at this point: the last one in the list. A valid assumption is that larger families need more time to get together on a sinking ship, and hence have a lower probability of surviving. 
# 
# We will add a "FamilySize" feature, which is the sum of "SibSp" and "Parch" plus one (the person her/himself), to the test and training set.

# In[ ]:


# Create train_two with the newly defined feature
train_two = train.copy()
train_two["FamilySize"] = train["SibSp"] + train["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "FamilySize"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree, and remember lower is probably better
print(my_tree_three.score(features_three, target))


# ## 9. Random Forests

# Is there a way to achieve a better model instead of modifying `max_depth` and `min_samples_split` and hoping to get lucky?
# 
# Yes, **random forests**. 

# Random forests operate by constructing a multitude of decision trees at training time and outputting the class that is the mean prediction (regression) or mode of the classes (classification, our case) of the individual trees.
# 
# When we instantiate sklearn's `RandomForestClassifier` class, `n_estimators` needs to be set. This argument allows us to set the number of trees we wish to plant and average over.

# In[ ]:


# Import the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a features variable with the features we want
features_forest = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

print(my_forest.score(features_forest, target))


# In[ ]:


# Compute predictions on our test set features
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)

# Create DataFrame and .csv file
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_solution_random_forest.csv", index_label = ["PassengerId"])


# In[ ]:


# The files can be found here:
import os
print(os.listdir("../working"))

