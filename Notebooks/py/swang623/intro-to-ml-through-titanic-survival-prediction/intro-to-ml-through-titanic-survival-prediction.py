#!/usr/bin/env python
# coding: utf-8

# __Reference__
# 
# This notebook referenced the following Kaggle Kernels:
# -  [Nadin Tamer, Titanic Survival Predictions (Beginner)](https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)
# -  [Omar El Gabry, A Journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# -  [Anisotropic, Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# - [Sina, Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier)

# ## Introduction to Machine Learning through Titanic Project
# 
# ###  Critical Steps
# 1. Importing & Exploring Necessary Libraries
# 2. Read in Data
# 3. Feature Exploration
# 4. Data Manipulation
# 5. Running Machine Learning Algorithms
# 6. Creating Submission File to Kaggle

# # 1. Import Libraries

# In[ ]:


# Data Analysis Libraries
import numpy as np
import pandas as pd

# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


# # 2. Read in Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# # 3. Feature Exploration
# 
# In this step, we will get a basic sense of the data and visualize the features to figure out which ones are relevant for the analysis.

# In[ ]:


# A basic look at the training data
train.sample(5)


# In[ ]:


# Summary of the training data
train.describe(include = "all")


# In[ ]:


# Get a clearer understanding of data types and missing values
train.info()
print('***************************************************')
test.info()


# ## 3.1 Pclass

# In[ ]:


# Explore if survival rate depends on passenger class
sns.barplot(x = "Pclass", y = "Survived", data = train)
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean()


# There seems to be a significant difference in survival rate for passengers in different classes. This feature should go into the model.

# ## 3.2 Sex

# In[ ]:


# Explore if survival rate depends on passenger gender
sns.barplot(x = "Sex", y = "Survived", data = train)
train[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean()


# Sex should definitely go into the model as well.

# ## 3.3 Age 

# In[ ]:


# Age is a continuous variable with 20% of the data missing. 
# We will first look at the distribution
sns.distplot(train["Age"].dropna(), bins = 70, kde = False)


# - Age is not normally distributed so we cannot simply generate random numbers following a normal distribution to fill in the missing numbers. 
# - Instead of treating age as a continuous variable, it might be better to categorize age intervals since one year difference in age would probably not determine if the person survive.
# - In the next section, we will come up ways to fill in the missing value and categorize age.

# ## 3.3 SibSp

# In[ ]:


# Explore if survival rate depends on the number of siblings/spouses abroad the Titanic
sns.barplot(x = "SibSp", y = "Survived", data = train)
sibsp = pd.DataFrame()
sibsp["Survived Mean"] = train[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean()["Survived"]
sibsp["Count"] = train[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).count()["Survived"]
sibsp["STD"] = train[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).std()["Survived"]
print(sibsp)
train[(train["SibSp"] == 5)|(train["SibSp"] == 8)]


# - In the next step, we will group "SibSp" into [0, 1, 2 or more]
# - It is surprising that none of the members in the two families with 5 and 8 SibSp survived. Looking at the available "Age" data points, it seems that most of them are kids. It would be a good idea to fill in the rest ages as "teenagers" or "kids". However, there are only 7 records that needs to be filled in in this way so in this analysis, we will not treat them differently. 

# ## 3.4 Parch

# In[ ]:


# Explore if survival rate depends on the number of parents/children abroad the Titanic
sns.barplot(x = "Parch", y = "Survived", data = train)
sibsp["Survived Mean"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean()["Survived"]
sibsp["Count"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index = False).count()["Survived"]
sibsp["STD"] = train[["Parch", "Survived"]].groupby(["Parch"], as_index = False).std()["Survived"]
print(sibsp)


# - In the next step, we will group "Parch" into [0, 1, 2 or more]

# ## 3.5 Fare

# In[ ]:


# See the distribution of Fare
#sns.distplot(train["Fare"][train["Pclass"]==1].dropna(), bins = 10, kde = False)
print(train[["Fare", "Survived"]].dropna().groupby(["Survived"]).count())
fare_hist = sns.FacetGrid(train, col="Survived")
fare_hist = fare_hist.map(plt.hist, "Fare")

train[["Fare", "Survived"]].dropna().groupby(["Survived"]).median()


# - The outputs above indicate that the distribution of fare for the group who survived and the group who did not is different. So we will include fare in the model.
# - We will also categorize fare.

# ## 3.6 Cabin

# In[ ]:


# There are many missing values in this colomn
(train["Survived"][train["Cabin"].isnull()].count())/(train["Cabin"].count())


# - Since there are much more missing values than available values, we will leave this variable out from the model.

# ## 3.7 Embarked

# In[ ]:


# Explore if survival rate depends on the port passenger embarked
sns.barplot(x = "Embarked", y = "Survived", data = train)
train[["Survived", "Embarked"]].groupby(["Embarked"]).mean()


# - We will keep this variable in our model.

# ## Insights from Feature Exploration & Next Steps
# - Some variables may not have valuable information and can be dropped from the dataset.
# - Missing values in both the training dataset and testing dataset should be addressed.
# - Continuous variables should be categorized.

# # 4. Data Manipulation

# ## 4.1 Dropping Unnecessary Variables
# 
# From the outputs above, we get a basic sense of the variables and it is intuitive that "PassengerId", "Name"and "Ticket" are not likely to be valuable for the analysis. Therefore, we will drop these variables from both the training and testing dataset. 
# 
# From the summary statistics, we also realize that the column "cabin" has too many missing values to draw information from. We will also exclude this column from the datasets.

# In[ ]:


PassengerId = test['PassengerId']
train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)
test = test.drop(["PassengerId","Name", "Ticket", "Cabin"], axis = 1)


# In[ ]:


#train.info()
#print('***************************************************')
#test.info()


# ## 4.2 Dealing with Missing Values
# 
# There are two variabels with missing values in the training dataset: "Age" and "Embarked"

#  ## *4.2.1 Embarked -- categorical data* 
#  
#  ### It's common to replace missing values of a categorical variable with mode. We will find the count for each unique values and replace is with the most appeared value.

# In[ ]:


print(train["Embarked"].unique())
print(train.groupby(["Embarked"])["Survived"].count().reset_index())
train["Embarked"] = train["Embarked"].fillna("S")
train.groupby(["Embarked"])["Survived"].count().reset_index()


# ## *4.2.2 Age*
# 
# ### About 20% of the "Age" column is missing. As inspied by *A Journey through Titanic*, we will replace missing values with random numbers between (mean - std) and (mean + std)

# In[ ]:


# Calculate mean and standard deviation of "Age" column
train_mean = train["Age"].mean()
train_std = train["Age"].std()
test_mean = test["Age"].mean()
test_std = test["Age"].std()

# Count missing values
count_na_train = train["Age"].isnull().sum()
count_na_test = test["Age"].isnull().sum()

# generate random numbers
np.random.seed(66)
train_rand = np.random.randint(train_mean - train_std, train_mean + train_std, size = count_na_train)
test_rand = np.random.randint(test_mean - test_std, test_mean + test_std, size = count_na_test)

# Fill missing values with random numbers
train["Age"][np.isnan(train["Age"])] = train_rand
test["Age"][np.isnan(test["Age"])] = test_rand

# Convert into int
train["Age"] = train["Age"].astype(int)
test["Age"] = test["Age"].astype(int)


# ## *4.2.3 Fare*
# 
# ### There is one missing value for fare in the test data. We will simply replace it with the median

# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].median())


# In[ ]:


# Confirm that all missing values are taken care of
#train.info()
#print('***************************************************')
#test.info()


# ## 4.3 Categorize Numeric Values

# ## 4.3.1 Age

# In[ ]:


# Map Age to categorical groups
train.loc[train["Age"] <= 16, "age_c"] = "1"
train.loc[(train["Age"] <= 32)&(train["Age"] > 16), "age_c"] = "2"
train.loc[(train["Age"] > 32)&(train["Age"] <= 48), "age_c"] = "3"
train.loc[(train["Age"] > 48)&(train["Age"] <= 64), "age_c"] = "4"
train.loc[(train["Age"] > 64), "age_c"] = "5"

test.loc[test["Age"] <= 16, "age_c"] = "1"
test.loc[(test["Age"] <= 32)&(test["Age"] > 16), "age_c"] = "2"
test.loc[(test["Age"] > 32)&(test["Age"] <= 48), "age_c"] = "3"
test.loc[(test["Age"] > 48)&(test["Age"] <= 64), "age_c"] = "4"
test.loc[(test["Age"] > 64), "age_c"] = "5"

train[["age_c","Survived"]].groupby(["age_c"]).mean()


# In[ ]:


# set up two new dataframes for the final model
m_train = train
m_test = test

#m_train.info()
#print('***************************************************')
#m_test.info()


# In[ ]:


# Generate Dummy Variable for Age
# Dropped the first one to avoid multicollinearity
#age_dummy_train = pd.get_dummies(train["age_c"], drop_first = True)
#age_dummy_test = pd.get_dummies(test["age_c"], drop_first = True)

# Concatenate Age dummy with the original training dataset
#m_train = pd.concat([m_train, age_dummy_train], axis = 1)
#m_test = pd.concat([m_test, age_dummy_test], axis = 1)

# Drop original Age and age_c
#m_train = m_train.drop(["age_c", "Age"], axis = 1)
#m_test = m_test.drop(["age_c", "Age"], axis = 1)

#m_train.sample(5)


# ## 4.3.2 SibSp
# 
# Categorize into 0,1,2 or more

# In[ ]:


# Map SibSp into categories
train.loc[train["SibSp"] == 0, "sib_c"] = "0"
train.loc[train["SibSp"] == 1, "sib_c"] = "1"
train.loc[train["SibSp"] >1 , "sib_c"] = "2"

test.loc[test["SibSp"] == 0, "sib_c"] = "0"
test.loc[test["SibSp"] == 1, "sib_c"] = "1"
test.loc[test["SibSp"] >1 , "sib_c"] = "2"

# Generate Dummy Variable
#sib_dummy_train = pd.get_dummies(train["sib_c"], drop_first = True)
#sib_dummy_test = pd.get_dummies(test["sib_c"], drop_first = True)


# Append sib_dummy to m-train
#m_train = pd.concat([m_train, sib_dummy_train], axis = 1)
#m_train = m_train.drop(["SibSp"], axis = 1)

#m_test = pd.concat([m_test, sib_dummy_test], axis = 1)
#m_test = m_test.drop(["SibSp"], axis = 1)

#m_train.sample(5)


# ## 4.3.3 Parch
# 
# Categorize into 0,1, 2 and more

# In[ ]:


# Map Parch into categories
train.loc[train["Parch"] == 0, "pc_c"] = "0"
train.loc[train["Parch"] == 1, "pc_c"] = "1"
train.loc[train["Parch"] >1 , "pc_c"] = "2"

test.loc[test["Parch"] == 0, "pc_c"] = "0"
test.loc[test["Parch"] == 1, "pc_c"] = "1"
test.loc[test["Parch"] >1 , "pc_c"] = "2"

# Generate Dummy Variable
#pc_dummy_train = pd.get_dummies(train["pc_c"], drop_first = True)
#pc_dummy_test = pd.get_dummies(test["pc_c"], drop_first = True)


# Append sib_dummy to m-train/m-test
#m_train = pd.concat([m_train, pc_dummy_train], axis = 1)
#m_train = m_train.drop(["Parch"], axis = 1)

#m_test = pd.concat([m_test, pc_dummy_test], axis = 1)
#m_test = m_test.drop(["Parch"], axis = 1)

#m_train.sample(5)


# ## 4.3.4 Fare

# In[ ]:


# Map fare values into categories
train["fare_c"] = pd.qcut(train["Fare"], 4, labels = ["1", "2", "3","4"])
test["fare_c"] = pd.qcut(test["Fare"], 4, labels = ["1", "2", "3","4"])

# Generate dummy variables for both train and test
#fare_dummy_train = pd.get_dummies(train["fare_c"], drop_first = True)
#fare_dummy_test = pd.get_dummies(test["fare_c"], drop_first = True)

# Append dummy variables to the original data frames
#m_train = pd.concat([m_train, fare_dummy_train], axis = 1)
#m_train = m_train.drop(["Fare"], axis = 1)

#m_test = pd.concat([m_test, fare_dummy_test], axis = 1)
#m_test = m_test.drop(["Fare"], axis = 1)

#m_train.sample(5)


# ## 4.4 Assign numerical values to categorical variables

# ## 4.4.1 Sex

# In[ ]:


train.loc[train["Sex"] == "male", "sex_c"] = "0"
train.loc[train["Sex"] == "female", "sex_c"] = "1"

test.loc[test["Sex"] == "male", "sex_c"] = "0"
test.loc[test["Sex"] == "female", "sex_c"] = "1"

# Generate dummy variables for both train and test
#sex_dummy_train = pd.get_dummies(train["Sex"], drop_first = True)
#sex_dummy_test = pd.get_dummies(test["Sex"], drop_first = True)

# Append dummy variables to the original data frames
#m_train = pd.concat([m_train, sex_dummy_train], axis = 1)
#m_train = m_train.drop(["Sex"], axis = 1)

#m_test = pd.concat([m_test, sex_dummy_test], axis = 1)
#m_test = m_test.drop(["Sex"], axis = 1)

#m_train.sample(5)


# ## 4.4.2 Embarked

# In[ ]:


train.loc[train["Embarked"] == "S", "emk_c"] = "0"
train.loc[train["Embarked"] == "Q", "emk_c"] = "1"
train.loc[train["Embarked"] == "C", "emk_c"] = "2"

test.loc[test["Embarked"] == "S", "emk_c"] = "0"
test.loc[test["Embarked"] == "Q", "emk_c"] = "1"
test.loc[test["Embarked"] == "C", "emk_c"] = "2"

# Generate dummy variables for both train and test
#emk_dummy_train = pd.get_dummies(train["Embarked"], drop_first = True)
#emk_dummy_test = pd.get_dummies(test["Embarked"], drop_first = True)

# Append dummy variables to the original data frames
#m_train = pd.concat([m_train, emk_dummy_train], axis = 1)
#m_train = m_train.drop(["Embarked"], axis = 1)

#m_test = pd.concat([m_test, emk_dummy_test], axis = 1)
#m_test = m_test.drop(["Embarked"], axis = 1)

#m_train.sample(5)

train.sample(5)


# ## 4.4.3 Pclass

# In[ ]:


# Map Parch into categories
#train.loc[train["Pclass"] == 1, "class_c"] = "class1"
#train.loc[train["Pclass"] == 2, "class_c"] = "class2"
#train.loc[train["Pclass"] == 3, "class_c"] = "class3"

#test.loc[train["Pclass"] == 1, "class_c"] = "class1"
#test.loc[train["Pclass"] == 2, "class_c"] = "class2"
#test.loc[train["Pclass"] == 3, "class_c"] = "class3"

# Generate dummy variables for both train and test
#class_dummy_train = pd.get_dummies(train["class_c"], drop_first = True)
#class_dummy_test = pd.get_dummies(test["class_c"], drop_first = True)

# Append dummy variables to the original data frames
#m_train = pd.concat([m_train, class_dummy_train], axis = 1)
#m_train = m_train.drop(["Pclass"], axis = 1)

#m_test = pd.concat([m_test, class_dummy_test], axis = 1)
#m_test = m_test.drop(["Pclass"], axis = 1)

#m_train.sample(5)


# In[ ]:


# Drop non-numeric categorical variables
train = train.drop(["Embarked","Sex", "Age", "SibSp", "Parch", "Fare"], axis = 1)
test = test.drop(["Embarked","Sex", "Age", "SibSp", "Parch", "Fare"], axis = 1)


# In[ ]:


train.sample(5)


# In[ ]:


test.sample(5)


# In[ ]:


# Check dataset status before modelling
train.info()
print('***************************************************')
test.info()


# # 5. Running Machine Learning Algorithms
# 
# The datasets are finally ready for modeling!!!
# 
# We will explore the following models:
# - Gaussian Naive Bayes
# - Logistics Regression
# - Support Vector Machine
# - Decision Tree Classifier
# - Random Forest Classifier
# - K-Nearest Neighbors
# 
# * Note that all parameters are set as default as of 1/10/2018; To be adjusted

# In[ ]:


# As inspired by Nadin, we will use 80% of the data for training,
# and the rest 20% to test the accuracy of the model

predictors = train.drop(["Survived"], axis = 1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)


# In[ ]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
# y_pred = gaussian.predict(x_val)
acc_gaussian = gaussian.score(x_val, y_val)
acc_gaussian


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
#y_pred = logreg.predict(x_val)
acc_logreg = logreg.score(x_val, y_val)
acc_logreg


# In[ ]:


# Support Vector Machine
svc = SVC()
svc.fit(x_train, y_train)
#y_pred = logreg.predict(x_val)
acc_svc = svc.score(x_val, y_val)
acc_svc


# In[ ]:


# Decision Tree Classifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
#y_pred = decisiontree.predict(x_val)
acc_decisiontree = decisiontree.score(x_val, y_val)
acc_decisiontree


# In[ ]:


# Random Forest Classifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
#y_pred = randomforest.predict(x_val)
acc_randomforest = randomforest.score(x_val, y_val)
acc_randomforest


# In[ ]:


# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
#y_pred = knn.predict(x_val)
acc_knn = knn.score(x_val, y_val)
acc_knn


# Based on the outputs above, Random Forest seems to work the best.

# # 6. Create a Submission File
# 

# In[ ]:


# Generate Predictions
prediction = randomforest.predict(test)

submission_titanic = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': prediction })
#print(submission_titanic)
submission_titanic.to_csv("submission_titanic.csv", index = False)


# In[ ]:




