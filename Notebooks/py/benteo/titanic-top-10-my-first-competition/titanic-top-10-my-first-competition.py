#!/usr/bin/env python
# coding: utf-8

# # Titanic data. Exploring classifier trees and forest with sklearn and xgboost.
# 
# In this notebook I take a look at the data from the kaggle Titanic training competition and try to predict if one passenger will survive based on various informations. I will use first a simple classifier trees from sklearn, a random forest and try to tune it. Finally I do the same operation with the xgboost library.
# 
# ## Import libraries and data
# 
# First let's import all the libraries that we will need in this notebook.

# In[ ]:


get_ipython().magic(u'matplotlib inline')
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


# Then, let's load the training and test datas in two separate dataframes

# In[ ]:


# Load the training data in a dataframe
train = pd.read_csv("../input/train.csv")

# Load the test data in a dataframe
test = pd.read_csv("../input/test.csv")


# ## Description of variables 
# 
# The following description of variables comes from the competition input
# 
# | Variable | Definition	| Key              |
# | -------- | ---------- | ---------------- |
# | survival | Survival   | 0 = No, 1 = Yes  |
# | pclass   | Ticket class |	1 = 1st, 2 = 2nd, 3 = 3rd |
# | sex	   | Sex
# | Age	   | Age in years | Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# | sibsp	   | # of siblings / spouses aboard the Titanic	
# | parch	   | # of parents / children aboard the Titanic	
# | ticket   | Ticket number	
# | fare	   | Passenger fare	
# | cabin	   | Cabin number	
# | embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton
# 
# We can take a look at the data to see what we have to deal with.

# In[ ]:


# Display some data
train.head()


# Now we can take a first look at what we loaded by displaying the column names and their type.

# In[ ]:


# Show the columns of the dataframe and their original type
for column in train.columns:
    print(column, train[column].dtype)


# Some columns are treated as integer but should be categories. It is easy to convert them.

# In[ ]:


# Convert some columns into categorical data
train["Survived"] = train["Survived"].astype('category')
train["Pclass"] = train["Pclass"].astype('category')
test["Pclass"] = test["Pclass"].astype('category')
train.describe()


# ## Analyze data
# 
# Before doing any operations and predictions, we can take a look at the distribution of some variables to get an idea on what could depend the survival rate.
# 
# First, we can see that only one third of the passengers survived. This could be a first dummy model.

# In[ ]:


train["Survived"].value_counts().plot.pie(figsize=(4, 4),
                                          autopct='%.2f',
                                          title="Percentage of survivors",
                                          fontsize = 10)


# One third of the passengers are female. Is that correlated with the survival rate ?

# In[ ]:


train["Sex"].value_counts().plot.pie(figsize=(4, 4),
                                     autopct='%.2f',
                                     title="Percentage of Male and Female passengers",
                                     fontsize = 10)


# Well it appears that it is the case. You have a better chance to survive if you are female.

# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=train);


# Also, about half of the passengers are on the third class.

# In[ ]:


train["Pclass"].value_counts().plot.pie(figsize=(4, 4),
                                        autopct='%.2f',
                                        title="Percentage of passengers per Class",
                                        fontsize = 10)


# And if the class distribution in the survived who survived is approximately equal, most of the third class passengers died.

# In[ ]:


sns.countplot(x="Survived", hue="Pclass", data=train);


# From this first look at the data, it is easy to say that you have a better chance of survival if you are female and from the first class, which is not surprising.
# 
# It is possible to further analyse the data with plots and correlations. But the goal is to build a model, not from personal insights, but from machine learning.
# 
# ## Data treatment / Feature engineering
# 
# In order to feed data to a machine learning model, some preparations must be made first.
# I wrote one subroutine by feature I engineered. Look at the subroutine documentations for informations.

# In[ ]:


def process_family(data):
    """ Aggregate the family size"""
    print("Processing family")
    data["familysize"] = data["SibSp"] + data["Parch"]
    print("    Done")
    return data

def process_ticket(data):
    """ Get further informations from the ticket number.
    Some passengers have the same ticket number. 
    We assume that it means they belong to the same group of people.
    """
    print("Processing ticket")
    data["ticketgroupsize"] = data.groupby("Ticket")["Ticket"].transform("count") - 1
    print("    Done")
    return data

def find_nan(data, feature, error=False):
    """ Look for missing values in a specific feature,
    count them and display the number. Raise an error 
    if asked. """
    if data[feature].isnull().values.any():
        print("    Missing values: ",
              data[feature].isnull().sum(),
              "over",
              len(data[feature].index),
              "(",
              data[feature].isnull().sum()/len(data[feature].index),
              "%)")
        if error:
            raise ValueError("NaN")
    return data[feature].isnull().values.any()

def process_names(data):
    """ Get further informations from the name title and put
    it into a new categorical data named type."""
    print("Processing Names")
    find_nan(data, "Name")
    # It is chosen to regroup titles by relevant categories.
    # Modifying that should have a big impact on the model results.
    name_dict = {"Capt":       "officer",
                 "Col":        "officer",
                 "Major":      "officer",
                 "Dr":         "officer",
                 "Rev":        "officer",
                 "Jonkheer":   "snob",
                 "Don":        "snob",
                 "Sir" :       "snob",
                 "the Countess":"snob",
                 "Dona":       "snob",
                 "Lady" :      "snob",
                 "Mme":        "married",
                 "Ms":         "married",
                 "Mrs" :       "married",
                 "Miss" :      "single",
                 "Mlle":       "single",
                 "Mr" :        "man",
                 "Master" :    "boy"
                }
    data['prefix'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    data['type'] = data['prefix'].map(name_dict)

    # Dummy encoding
    # Create one column for each value of the categorical data and assign a
    # one or zero value.
    # This is needed for sklearn that can only deal with numbers.
    column_dummies = pd.get_dummies(data['type'],prefix='type')
    data = pd.concat([data,column_dummies],axis=1)
    print("    Done")
    
    return data

def process_sex(data):
    """ Dummy encoding for the sex. 
    Map one to male and zero to female """
    print("Processing Sex")
    find_nan(data, "Sex", error=True)
    data['Sex'] = data['Sex'].map({'male':1,'female':0})
    print("    Done")
    
    return data

def process_age(data):
    """ Deal with missing age values. 
    A lot of data is missing in the age column.
    This is filled by a categorized median age.
    """
    print("Processing Age")
    find_nan(data, "Age")
    
    medianage = data.groupby(['Sex', 'Pclass', 'type'])["Age"].median()
    
    def fillna_age(row, medianage):
        age = medianage.loc[row["Sex"], row['Pclass'], row["type"]]
        return age

    data["Age"] = data.apply(lambda row : fillna_age(row, medianage) if np.isnan(row['Age']) else row['Age'], axis=1)
    find_nan(data, "Age")
    print("    Done")
    
    return data

def process_fare(data):
    """ Deal with missing fare values."""
    print("Processing Fare")
    find_nan(data, "Fare")
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    print("    Done")
    
    return data

def process_embarked(data):
    """ Deal with missing embarked data and create
    the dummy encoding. """
    print("Processing Embarked")
    find_nan(data, "Embarked")
    # Find the most common occurence of the categorical data
    most_common = data['Embarked'].value_counts().index[0]
    # Replace NaN values with the most common occurence
    data['Embarked'].fillna(most_common, inplace=True)
    
    # dummy encoding
    # Create one column for each value of the categorical data
    column_dummies = pd.get_dummies(data['Embarked'],prefix='Embarked')
    data = pd.concat([data,column_dummies],axis=1)
    # Drop the now irrelevant column
    data.drop('Embarked',axis=1,inplace=True)
    print("    Done")
    
    return data

def process_cabin(data):
    """ Get the deck information from the cabin feature
    and deal with missing values. """
    print("Processing Cabin")
    find_nan(data, "Cabin")
    # Replace NaN values with U for unknown
    data['Cabin'].fillna("U", inplace=True)
    # Extract the deck information
    data['deck'] = data["Cabin"].map(lambda row: row[0])
    # dummy encoding
    # Create one column for each value of the categorical data
    column_dummies = pd.get_dummies(data['deck'],prefix='deck')
    data = pd.concat([data,column_dummies],axis=1)
    print("    Done")
    
    return data

def process_all(data):
    """ Process all the dataset features and return the dataset """
    data = process_family(data)
    data = process_ticket(data)
    data = process_sex(data)
    data = process_names(data)
    data = process_age(data)
    data = process_fare(data)
    data = process_embarked(data)
    data = process_cabin(data)
    return data

def write_results(data, model):
    """ Write results in the csv format for competition submission """
    with open("titanic.csv","w") as outfile:
        outfile.write("PassengerId,Survived\n")
        for passenger in data.index:
            line = str(data.at[passenger, "PassengerId"]) + "," + str(int(data.at[passenger, model])) + "\n"
            outfile.write(line)


# All this feature preprocessing will have a big impact on the model precision.
# Let's apply it to the train and test dataframes.
# 
# Combine the train and test dataset in one dataset for feature engineering.
# Then resplit them in two separate datasets.

# In[ ]:


# Load
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Merge & process
merged = train.append(test)
merged = process_all(merged)
# Split
train = pd.DataFrame(merged.head(len(train)))
test = pd.DataFrame(merged.iloc[len(train):])


# In[ ]:


train.columns


# ## First stupid models and evaluation
# 
# Before doing any advanced machine learning models, we can build stupid model to then know if complex models do better than that. The score is evaluated with the root mean squared error.
# 
# The first stupid model is a weighted random from the observation we made. Only one third of passengers survive.

# In[ ]:


train["eval_random"] = [np.random.choice([0,1], p=[0.62, 0.38]) for passenger in train.index]

rmse_random = np.sqrt(mean_squared_error(train["Survived"], train["eval_random"]))
print(rmse_random)


# The second stupid model is rich woman. If you are rich, and a woman, you survive.

# In[ ]:


def richwoman(passenger, data):
    """ If you are a female from the first class, you survive. """
    if data.at[passenger, "Sex"] == "female" and data.at[passenger, "Pclass"] == 1:
        return 1
    else:
        return 0
train["eval_richwoman"] = [richwoman(passenger, train) for passenger in train.index]

rmse_richwoman = np.sqrt(mean_squared_error(train["Survived"], train["eval_richwoman"]))
print(rmse_richwoman)


# So we see that building a model from feature informations does a better job than an overall probability. Now the goal is to build a more complex model to get a better score.
# 
# # Sklearn
# ## Single tree
# 
# Building a single classifier decision tree with sklearn is easy. You have to provide the list of features that you want the decisions to be made on. Here we give the list of all the features that we have. The multiplication of features comes from the dummy encoding preprocessing that is need for sklearn.

# In[ ]:


features_names = ["Fare",
                  "SibSp",
                  "Parch",
                  "familysize",
                  "ticketgroupsize",
                  "Pclass",
                  "Sex", 
                  "Age",
                  "Embarked_C", 
                  "Embarked_Q",
                  "Embarked_S",
                  "type_boy",
                  "type_officer",
                  "type_married",
                  "type_single",
                  "type_snob",
                  "type_man",
                  "deck_A",
                 "deck_B",
                 "deck_C",
                 "deck_D",
                 "deck_E",
                 "deck_F",
                 "deck_G",
                 "deck_U"]

# We select here all the features above.
features = train[features_names] 
# The target is to predict the survived category
target = train["Survived"]
# The tree is a decision tree classifier.
my_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
# The tree is fitted to the train data.
my_tree = my_tree.fit(features, target)

# Sklearn has a built in score that we can look at and from that score tune the tree parameters.
# Here we look at the score on the train data so beware of overfitting.
print("Score of tree on train data: ", my_tree.score(features, target))

# Use the tree to evaluate its answer on the train data.
train["eval_tree"] = my_tree.predict(features)
# Look at the RMSE score on the train data.
rmse_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_tree"]))
print("RMSE:", rmse_tree)


# With this simple tree we have a way better model than the rich woman model (look at the RMSE score). Now we can look at the feature importance. That is to say, what are the most important features when deciding if one passenger will survive ?

# In[ ]:


features_imp = pd.DataFrame()
features_imp['feature'] = features_names
features_imp['importance'] = my_tree.feature_importances_
features_imp.sort_values(by=['importance'], ascending=True, inplace=True)
features_imp.set_index('feature', inplace=True)
features_imp.plot(kind='barh', figsize=(20, 20))


# So the most important features are if you are a man or not, how much you paid for your ticket and your family size. It is coherent with the first observations that we made.
# 
# We can now use this tree to predict the survival outcome of the passengers in the test data, write the results in a csv and then submit it to kaggle.

# In[ ]:


test["eval_tree"] = my_tree.predict(test[features_names].values)

write_results(test, "eval_tree")


# ## Random forest
# 
# Instead of getting a decision from one tree, it is possible to get the answer from a panel of trees, that is to say a forest. Each tree in the forest is different and is built around a sample of features. It is always better to get an answer from a diverse jury and this way overfitting is limited.

# In[ ]:


# The forest will have 50 trees 
# and the max number of features by trees is the square root of the total features number
my_forest = RandomForestClassifier(n_estimators=50, max_features='sqrt')
my_forest = my_forest.fit(features, target)


# Let's look if using a forest has an impact on the feature importance compared to a single tree.

# In[ ]:


features_imp = pd.DataFrame()
features_imp['feature'] = features_names
features_imp['importance'] = my_forest.feature_importances_
features_imp.sort_values(by=['importance'], ascending=True, inplace=True)
features_imp.set_index('feature', inplace=True)
features_imp.plot(kind='barh', figsize=(20, 20))


# The feature importance is not exactly the same but it is close. The main difference is that the Sex is now in fourth position instead of being insignificant with a single tree.

# In[ ]:


print("Score of forest on train data: ", my_forest.score(features, target))

train["eval_forest"] = my_forest.predict(features)

rmse_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_forest"]))
print("RMSE:", rmse_tree)


# This random forest model is way better than the single tree model when comparing the rmse score on train data.
# We can also use it on the test data and write the csv for kaggle submission.

# In[ ]:


test["eval_forest"] = my_forest.predict(test[features_names].values)

write_results(test, "eval_forest")


# **The kaggle score is 0.73684**

# ## Tuned Random Forest
# 
# Single trees and forest have a number of tuning parameters that have a big impact on model performance and overfitting.
# (see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
# 
# One way to pick the best set of parameters is to brute force test them all with a parameter grid.

# In[ ]:


def tune_forest(features, targets):
    """ Find the best parameters for the random forest """
    #parameter_grid = {
    #             'max_depth' : [5, 6, 7],
    #             'n_estimators': [20],
    #             'max_features': ['sqrt', 'auto', 'log2'],
    #             'min_samples_split': [2, 5, 10, 15],
    #             'min_samples_leaf': [3, 10],
    #             'bootstrap': [True, False],
    #             }
    parameter_grid = None
    parameter_grid = {
                 'max_depth' : [8, 10, 12],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier(n_jobs=2)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=3,
                               n_jobs=2,
                               verbose=1)

    grid_search.fit(features, targets)
    model = grid_search.best_estimator_
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best estimator: {}'.format(grid_search.best_estimator_))
    
    return model, parameters


# In[ ]:


model, parameters = tune_forest(features, target)
train["eval_tuned_forest"] = model.predict(train[features_names].values)
rmse_tuned_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_tuned_forest"]))
print("RMSE:", rmse_tuned_tree)


# In[ ]:


test["eval_tuned_forest"] = model.predict(test[features_names].values)

write_results(test, "eval_tuned_forest")


# ** The kaggle score is 0.80861. ** Even if the rmse score of this tuned random forest is worse than the previous random forest, the kaggle score is better. This is due to overfitting effects.

# # XGboost
# 
# XGboost is a popular machine learning library when using decision trees.
# Work in progress

# In[ ]:





# In[ ]:


model = xgb.XGBClassifier()
model.fit(features, target)

print("Score of tree on train data: ", model.score(features, target))

train["eval_xgb_tree"] = model.predict(features)

rmse_xgb_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_xgb_tree"]))
print("RMSE:", rmse_xgb_tree, "1-RMSE:", 1.0-rmse_xgb_tree)


# In[ ]:


features_imp = pd.DataFrame()
features_imp['feature'] = features_names
features_imp['importance'] = model.feature_importances_
features_imp.sort_values(by=['importance'], ascending=True, inplace=True)
features_imp.set_index('feature', inplace=True)
features_imp.plot(kind='barh', figsize=(20, 20))


# In[ ]:


test["eval_xgb_tree"] = model.predict(test[features_names])

write_results(test, "eval_xgb_tree")


# In[ ]:


def tune_xgb_tree(features, targets):
    parameter_grid = {
                 'max_depth' : [7, 8, 9],
                 'max_delta_step': [1],
                 'n_estimators': [20, 40, 60, 80],
                 'colsample_bylevel': [0.8, 0.9, 1.0],
                 'colsample_bytree': [0.6, 0.8, 1.0],
                 'subsample': [0.3, 0.4, 0.5, 0.6],
                 }
    xgb_model = xgb.XGBClassifier()
    print(xgb_model.get_params().keys())

    grid_search = GridSearchCV(xgb_model,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=3,
                               n_jobs=2,
                               verbose=1)

    grid_search.fit(features, targets)
    model = grid_search.best_estimator_
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best estimator: {}'.format(grid_search.best_estimator_))
    
    return model, parameters


# In[ ]:


model, parameters = tune_xgb_tree(features, target)


# In[ ]:


train["eval_tuned_xgb_tree"] = model.predict(train[features_names])
rmse_tuned_tree = np.sqrt(mean_squared_error(train["Survived"], train["eval_tuned_xgb_tree"]))
print("RMSE:", rmse_tuned_tree, "1-RMSE:", 1.0-rmse_tuned_tree)


# In[ ]:


parameters


# In[ ]:


features_imp = pd.DataFrame()
features_imp['feature'] = features_names
features_imp['importance'] = model.feature_importances_
features_imp.sort_values(by=['importance'], ascending=True, inplace=True)
features_imp.set_index('feature', inplace=True)
features_imp.plot(kind='barh', figsize=(20, 20))


# In[ ]:


test["eval_tuned_xgb_tree"] = model.predict(test[features_names])

write_results(test, "eval_tuned_xgb_tree")


# # Best score
# 0.79904

# In[ ]:




