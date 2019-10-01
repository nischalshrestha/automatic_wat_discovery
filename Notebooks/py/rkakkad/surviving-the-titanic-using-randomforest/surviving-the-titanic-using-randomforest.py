#!/usr/bin/env python
# coding: utf-8

# # The Problem
# The sinking of the RMS Titanic after colliding with an icerberg was one of the most tragic incidents in modern history. It killed 1502 of its 2224 passengers and crew. 
# 
# In this problem we have been provided data about each of the passengers and whether they survived or not. Our problem is to analyze the data and build a model to predict which type of passengers will survive the incident. The accuracy of the model will be measured based on the accuracy of the predicted result of passengers relative to the truth. 
# 
# This is a  classification problem in which we need to build a model to classify a passenger as likely surviver or not. We will be building a randomforrest model. We will broadly following the following steps
# 1. Set Up: Import the libraries & Load the datasets
# 2. Data Exploration: Analyze the fields in the dataset and build hypothesis around how they can be used
# 3. Feature Engineering: Setting up our features in line with the data exploration results
# 4. Set the hyperparameters
# 5. Build the model and the prediction
# Let's get started! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import collections
import os
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Set up the data

# In[ ]:


# Importing the dataset
dataset_train = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')


# # Data Exploration
# We will understand the overall structure of our dataset and the dive in to each of the key parameters. At a broad level we try to answer key questions about each parameter like
# 
# - Is this likely to be an important predictor?
# - What is the coverage of this data point (ie how many values are populated). How should we populate blank values?
# - What is the quality of the field - is it a clean field or a dirty field & if dirty - what cleaning can be performed?
# - Can any parameters be derived from this field that may be good predictors.
# - How is our dataset distributed across the fiedl. 
# 
# As a process we would like to explore all our data fields first before manipulating our data. Let's get started.

# In[ ]:


# Get info on training dataset
dataset_train.info()


# Training Data Set: We have 891 observations in total with 12 columns. Our target variable is "Survived". Of these the Age parameter has only 714 entries, Cabin has only 204 &  Embarked has 889. We will need to explore filling the missing values in these parameters. 

# In[ ]:


# Get info on testing dataset
dataset_test.info()


# Testing Data Set: The testing data set has 418 entries, i.e. testing data as a % of the total sample is 418/(418 + 891) = 32%. This is on the higher side, as we usually use about 20-25% of a sample for testing, however in this case we take this a given and do not change the datasets. The testing data set has gaps in Age, Cabin and Fare fields. Let's move on...

# In[ ]:


# checking the column names & sample data for dataset_train
dataset_train.head(10)


# In[ ]:


# Passenger ID 
dataset_train.PassengerId.describe()


# This is an index that increments by 1 for every passenger. Since we do not know how this field has been derived, we cannot formulate hypotheses for how it may impact Survival. However, if this field is related to some real world phenomenon such as if it is provided in order of making a reservation then it would mean that the lower indexes are for the earliest booked passengers - which may be correlated with passengers being more planned in their lives v/s the passengers with later indexes which were booked late or last minute. Since we are not sure if this will impact us - I tried leaving it in the base model - but it did not show as a important factor - hence decided to drop it. 

# In[ ]:


# Data Exploration - Survived
dataset_train.Survived.describe()


# In[ ]:


dataset_train["Survived"].value_counts()


# In[ ]:


dataset_train["Survived"].value_counts(normalize = True)


# Survived is our target variable. It is a binary field with 0 - indicating did not survive and 1 - indicating survive. We can see from our data that ~62% people died & 38% survived. These are good benchmarks to keep in mind going forward. 

# In[ ]:


# Data Exploration - Pclass
dataset_train["Pclass"].value_counts(normalize = True)


# Passenger class shows that 24% passengers are in first class, 21% second class and 55% in 3rd class. Lets see how survival is correlated with these. 

# In[ ]:


dataset_train["Survived"].groupby(dataset_train["Pclass"]).mean()


# This is interesting. We can see that Passengers in 1st class have a 62% survival rate compared to 38% benchmark. Similarly 3rd class passengers have only a 24% survival rate. This is a good variable to keep in our model.

# In[ ]:


# Data exploration - Name
dataset_train["Name"].describe()


# The field Name has 891 unique values. 

# In[ ]:


dataset_train["Name"].head(20)


# Name is a dirty field (as is most often the case) and is unusable as-is, but we can glean potentially useful information from the titles embedded in the name. We see that every name  starts with surname followed by comma and the title which ends in a dot. So we can split the name to get a new column called title - which could impact survival rates.
# 
# Upon analyzing the data we also find that there are some titles with lots of occurences like Mr., Mrs. etc and there are lots of titles with single occurences. During the feature engineering - we will club the singleton titles in to 1 group.

# In[ ]:


# Data exploration - Sex
dataset_train["Survived"].groupby(dataset_train["Sex"]).mean()


# We see a higher survival rate among women, we can keep this field as is.

# In[ ]:


# Data exploration Age
dataset_train["Survived"].groupby(dataset_train["Age"].isnull()).count()


# About 177 passengers do not have an age. Let's see if this impacts survival rate

# In[ ]:


# Check if null age impacts survival rate
dataset_train["Survived"].groupby(dataset_train["Age"].isnull()).mean()


# Passengers with null age have only 29% survival rate. We can impute the age by taking the mean of age for the group of passengers having the same class and title as the passenger with a null age. This would give us a reasonable approximation. Note as is standard practice - we take the averages of values from the training data set for the testing dataset too.

# In[ ]:


# Data exploration - SibSp
dataset_train["SibSp"].value_counts()


# In[ ]:


dataset_train["Survived"].groupby(dataset_train["SibSp"]).mean()


# Having siblings or spouse on board could improve ones chances of survival. We see that people with greater than 2 siblings or spouses have a significantly lower rate of survival.
# Let's move on

# In[ ]:


# Data exploration - Parch
dataset_train["Parch"].value_counts()


# In[ ]:


dataset_train["Survived"].groupby(dataset_train["Parch"]).mean()


# Data gets really thin in the Parch > 2 region...so we should not read too much in to variations in that range. For now since it stands to reason that having parents or children on board will impact ones survival rate ... we will keep it in the mix. 
# 
# In initial versions of my model I had kept the SibSp and Parch fields as separate predictors. However, they were both very weak hence I combined the two in to a single predictor called family size.

# In[ ]:


# Data exploration - Ticket
dataset_train["Ticket"].value_counts()


# While the ticket field is dirty as provided - an interesting observation is that there are group tickets and individual tickets...ie some tickets have as many as 7 passengers. This would indiciate a group traveling together which may impact survival rates. We can append the group ticke flag and see if it has predictive value. Further, ticket letters and numbers could be indicative of the location on the ship - however we have not used those in this model.

# In[ ]:


# Data exploration - Fare
dataset_train.head()


# Fare should be a factor in survival as it will likely determine the location the seat of the passenger, the place they were when the iceberg struck, etc...This should also be highly correlated with the passenger class and may be even correlated with the port of embarkment. Let's take a look.

# In[ ]:


# Fare pentiles
pd.qcut(dataset_train["Fare"], 5).value_counts(sort = False)


# In[ ]:


# Checking fare correlation with passenger class
pd.crosstab(pd.qcut(dataset_train["Fare"], 5), columns = dataset_train["Pclass"] )


# In[ ]:


# Checking fare correlation with port of embarkment
pd.crosstab(pd.qcut(dataset_train["Fare"], 4), columns = dataset_train["Embarked"]).apply(lambda r: r/r.sum(), axis=1)


# While there is a correlation between ticket price and port of embarkment, across the board most passengers have boarded from port S. Let's see the correlation of ticket fare with survival rate

# In[ ]:


# Correlation of ticket fare with survival rate
dataset_train["Survived"].groupby(pd.qcut(dataset_train["Fare"], 5)).mean()


# We see a strong correlation between fare and survival rate so we will keep this predictor. Let's move to the next variable.

# In[ ]:


# Data exploration - Cabin
dataset_train["Cabin"].describe()


# In[ ]:


dataset_train["Cabin"].head()


# In[ ]:


# Frequency count of Cabin values
dataset_train["Cabin"].value_counts()


# In[ ]:


# Checking if null cabin impacts survival
dataset_train["Survived"].groupby(dataset_train["Cabin"].isnull()).mean()


# So we see that a passenger with a null cabin has a far lower chance of survival. Also, similar to ticket, some cabins have many passengers which could impact the rate of survival. So let's add the 3 group categories - Group Cabin, single cabin or "Not present" in the feature engineering stage.

# In[ ]:


# Data exploration - Embarked
dataset_train["Embarked"].describe()


# Embarked is the port from where the passenger has embarked. It has 2 missing values. While S is the most common port we should check other values ot see if there's a better way to update the port of embarkment. Let's check the fare of these tickets.

# In[ ]:


# Checking the ticket of passengers with empty embarkment
print(dataset_train.loc[dataset_train["Embarked"].isnull()])


# These passengers are traveling together on a ticket worth $80. They are also first class passengers - lets check teh correlation between class and port. 

# In[ ]:


# checking correlation between class and port
pd.crosstab(dataset_train["Pclass"], columns = dataset_train["Embarked"]).apply(lambda r: r/r.sum(), axis=1)


# Passengers of Class 1 mostly came from port S so we will add that to the embarked port for these passengers. Ok lets get started with feature Engineering.

# # Feature Engineering - Function Definition
# We will create a series of functions to modify our data as discussed in the data exploration stage. This will allow us to modify our training and test datasets simultaneously and in the same mannner. It is risky to separate the manipulation of testing and training datasets as there is chance of errors creeping in.

# In[ ]:


# Deleting the passengerID
del dataset_train["PassengerId"]
del dataset_test["PassengerId"]


# In[ ]:


# Function for extracting titles and removing the Name Column
def titles(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i["Title"] = i["Name"].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i["Name"]
    return dataset_train, dataset_test


# In[ ]:


# Function for removing the low incidence titles and bucketing them in to others
def titleGroups(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i.loc[i["Title"] == "Col.",["Title"]] = "Other" 
        i.loc[i["Title"] == "Major.",["Title"]] = "Other" 
        i.loc[i["Title"] == "Mlle.",["Title"]] = "Other" 
        i.loc[i["Title"] == "Ms.",["Title"]] = "Miss." 
        i.loc[i["Title"] == "Sir.",["Title"]] = "Mr." 
        i.loc[i["Title"] == "Capt.",["Title"]] = "Other" 
        i.loc[i["Title"] == "Lady.",["Title"]] = "Mrs." 
        i.loc[i["Title"] == "Don.",["Title"]] = "Other" 
        i.loc[i["Title"] == "the",["Title"]] = "Other" 
        i.loc[i["Title"] == "Mme.",["Title"]] = "Other" 
        i.loc[i["Title"] == "Jonkheer.",["Title"]] = "Other" 
    return dataset_train, dataset_test


# In[ ]:


# Function to fill missing age values in the dataset
def fillAges(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        data = dataset_train.groupby(['Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return dataset_train, dataset_test


# In[ ]:


# Function to convert siblings and parch to family size
def familySize(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i["FamilySize"] = np.where((i["SibSp"]+i["Parch"]) == 0 , "Single", np.where((i["SibSp"]+i["Parch"]) <= 3,"Small", "Big"))
        del i["SibSp"]
        del i["Parch"]
    return dataset_train, dataset_test


# In[ ]:


# Function to append ticketCounts to dataset & delete ticket
def ticketCounts(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i["TicketCount"] = i.groupby(["Ticket"])["Title"].transform("count")
        del i["Ticket"]
    return dataset_train, dataset_test


# In[ ]:


# Fill the na Fares with mean of fares from the set.
dataset_train['Fare'].fillna(dataset_train['Fare'].mean(), inplace = True)
dataset_test['Fare'].fillna(dataset_train['Fare'].mean(), inplace = True)


# In[ ]:


# Function to add Cabin count flag
def cabinCount(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i["CabinCount"] = i.groupby(["Cabin"])["Title"].transform("count")
        del i["Cabin"]
    return dataset_train, dataset_test


# In[ ]:


# Function to convert cabinCount to cabinType Flag
def cabinCountFlag(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i["CabinType"] = "Missing"
        i.loc[i["CabinCount"] == 1,["CabinType"]] = "Single" 
        i.loc[i["CabinCount"] == 2,["CabinType"]] = "Double" 
        i.loc[i["CabinCount"] >= 3,["CabinType"]] = "ThreePlus" 
        del i["CabinCount"]
    return dataset_train, dataset_test 


# In[ ]:


# Function to fill the missing values of Embarked
def fillEmbarked(dataset_train, dataset_test):
    for i in [dataset_train, dataset_test]:
        i["Embarked"] = i["Embarked"].fillna("S")
    return dataset_train, dataset_test


# In[ ]:


# Encoding our categorical variables as dummy variables to ensure scikit learn works
def dummies(dataset_train, dataset_test, columns = ["Pclass", "Sex", "Embarked","Title","TicketCount","CabinType","FamilySize"]):
    for column in columns:
        dataset_train[column] = dataset_train[column].apply(lambda x: str(x))
        dataset_test[column] = dataset_test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in dataset_train[column].unique() if i in dataset_test[column].unique()]
        dataset_train = pd.concat((dataset_train, pd.get_dummies(dataset_train[column], prefix = column)[good_cols]), axis = 1)
        dataset_test = pd.concat((dataset_test, pd.get_dummies(dataset_test[column], prefix = column)[good_cols]), axis = 1)
        del dataset_train[column]
        del dataset_test[column]
    return dataset_train, dataset_test


# # Feature engineering - Running the functions
# Now that our functions are set up - let's run our datasets through them and get the final datasets.

# In[ ]:


dataset_train, dataset_test = titles(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = titleGroups(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = fillAges(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = familySize(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = ticketCounts(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = cabinCount(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = cabinCountFlag(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = fillEmbarked(dataset_train, dataset_test)


# In[ ]:


dataset_train, dataset_test = dummies(dataset_train, dataset_test,columns = ["Pclass", "Sex", "Embarked","Title","TicketCount","CabinType", "FamilySize"])


# # Checking the final datasets

# In[ ]:


dataset_train.head()


# In[ ]:


dataset_train.describe()


# In[ ]:


dataset_test.head()


# In[ ]:


dataset_test.describe()


# # Building the model
# We will start building our random forrest model by setting up our optimal hyperparameters.

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_features='auto', 
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)


# In[ ]:


# Creating the Grid Search Parameter list
parameters = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1, 5, 10],
             "min_samples_split" : [12, 16, 20, 24],
             "n_estimators": [100, 400, 700]}


# In[ ]:


# Setting up the gridSearch to find the optimal parameters
gridSearch = GridSearchCV(estimator=classifier,
                  param_grid=parameters,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)


# In[ ]:


# Getting the optimal grid search parameters
gridSearch = gridSearch.fit(dataset_train.iloc[:, 1:], dataset_train.iloc[:, 0])


# In[ ]:


# Printing the out of bag score and the best parameters values
print(gridSearch.best_score_)
print(gridSearch.best_params_)


# In[ ]:


# building the random forrest classifier
classifier = RandomForestClassifier(criterion='entropy', 
                             n_estimators=100,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
classifier.fit(dataset_train.iloc[:, 1:], dataset_train.iloc[:, 0])
print("%.5f" % classifier.oob_score_)


# In[ ]:


# Creating the list of important features
pd.concat((pd.DataFrame(dataset_train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(classifier.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# From the factor importance we can see that a number of the factors we built have become important predictors, most notable Title Mr. is the 3rd most important factor, CabinType_Missing & FamilySize both turned out important as well!

# In[ ]:


# Making the predictions on the test set
predictions = classifier.predict(dataset_test)


# In[ ]:


# Making the predictions file for submission
predictions = pd.DataFrame(predictions, columns=['Survived'])
passengerIds = pd.read_csv('../input/test.csv')
predictions = pd.concat((passengerIds.iloc[:, 0], predictions), axis = 1)


# In[ ]:


# To save our results to a csv locally
predictions.to_csv('predictions.csv', index = False)

