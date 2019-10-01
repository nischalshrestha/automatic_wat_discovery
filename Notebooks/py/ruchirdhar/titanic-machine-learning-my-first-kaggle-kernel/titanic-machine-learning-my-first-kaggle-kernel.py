#!/usr/bin/env python
# coding: utf-8

# 

# **Objective:**
# Objective of this exercise is to use simple Machine Learning approach in Python and Pandas to predict survival rate of the test data provided as part of the exercise.
# Steps performed in-order to predict survival of the passengers are as follows:
# * Load Train and Test dataset.
# * Analyse dataset to identify missing values
# * Perform data wrangling taking simplistic assumption - in doing so goal is not to overly complicated exercise from a beginner perspecitve.
# * Convert qualitative variables into quantifiable variables by creating dummy variables
# * Drop un-necessary columns from dataframes once data wrangling is complete
# * Use Logistic regression to predict survival rate
# * Thats all folks easy peasy...isnt it ?
# 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# Lets all describe method to see how Training set data looks like. Note that this function is returning details of only those columns which are numeric i.e. quantitative
# 
# **Oops** there are missing age values. We know that certain features played a role in survival of passengers and one of the key feature was age - obviously closely related to your social status and gender.
# 
# We shall need to fix this problem..Anyways lets check Test dataset as well

# In[ ]:


train.describe()


# **Ummm....** some missing age in Test dataset as well. We'll fix this as well...

# In[ ]:


test.describe()


# Can surname help me to comeup with some way to fix Age problem ? Not a bad idea....Lets create a new column in dataframe with surname of each passenger across train and test dataset. Afterall we dont have any missing names

# In[ ]:


train["Title"] = train["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip().upper())
test["Title"] = test["Name"].apply(lambda x: x.split(',')[1].split('.')[0].strip().upper())


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# Quick and dirty Analysis which will form the basis of assumption on how to Create Age Group bucket for passengers as I dont want to come up with Age value for this dataset
# 
# The objective of below analysis is to find out that in the given population of the train dataset which has Age data available, can we draw some simplistic assumption on the age group. I plan to create following age group:
# 
# * Kids : Age < 16
# * Young Adults: Age >=16 and Age <  30
# * Adults: Age >=30 and Age <55
# * Old: Age > = 50
# I plan to predict the age group of missing passenders in the train data using title, for which I need to perform an analysis of Title v/s Age group based on known dataset. It  will help me to predict Age Group just using Title column.

# In[ ]:


train.Title.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MASTER' and Age >= 10")


# In[ ]:


train[train.Age.notnull()].query("Title =='MASTER' and Age >= 10").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MISS' and Age <= 16").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MISS' and Age > 16 and Age < 30").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MISS' and Age >= 30 and Age < 55").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MISS' and Age > 55").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MR' and Age < 16").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MR' and Age >= 16 and Age < 30").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MR' and Age >= 30 and Age < 55").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MRS' and Age < 16").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MRS' and Age >= 16 and Age < 30").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MRS' and Age >= 30 and Age < 55").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query("Title =='MRS' and Age >=55").Survived.value_counts()


# In[ ]:


train[train.Age.notnull()].query( "Age >=55").Survived.value_counts()


# In[ ]:


train[train.Age.isnull()].Title.value_counts()


# **Summary of analysis **
# * All of the passengers with Title "Master" were less than age of 13.
# * Survival rate of passengers having Title "Mr" and Age group 16-29 and Age group 30-55 was almost same...well surival rate of Mr's age group 16-29 was less however at high level they were least likely to survive.
# * Survival rate of passengers having Title "Miss" and Age group 16-29 was lesser than those of with the same title but Age group 30-55.
# * Survival rate of passengers having Title "Mrs" were almost the same across age group 16-29 and 30-55
# * I have ignored other surnames due to it being small in number and therefore not significantly providing enough information to draw any conclusion.
# 
# Basis Above analysis, considering worst case and reducing chances of misclassifying any non surviver as surviver I am creating following AgeGroup Function
# 
# *  Function will take arguments Title and Age :
#     If Age is missing:
#         * If Title is "MASTER" : Assign AgeGroup "KID"
#         * If Title is "MISS": assign AgeGroup "YoungAdult"
#         * If Title is "MR" : assign AgeGroup "YoungAdult"
#         * If Title is "MRS" assign AgeGroup "Adult' 
#         * for All Missing Age assign AgeGroup "Adult" 
# * If Age is not missing, then assign AgeGroup using following Criteria of Age:
#     * Kid if Age < 16
#     * YoungAdult if 16 =< Age < 30
#     * Adult if 30=< Age < 55
#     * Old if Age >=55    
# 

# In[ ]:


def AgeGroup(title, age):
    if pd.isnull(age):
        if (title == "MASTER"):
            return "KID"
        elif (title == "MR" or title =="MISS"):
            return "YOUNGADULT"
        else:
            return "ADULT"
    elif (age < 16):
        return "KID"
    elif (age >= 16 and age < 30 ):
        return "YOUNGADULT"
    elif (age>=30 and age <=55):
        return "ADULT"
    else:
        return "OLD"


# In[ ]:


train["AgeGroup"] = np.vectorize(AgeGroup)(train["Title"],train["Age"])
test["AgeGroup"] = np.vectorize(AgeGroup)(test["Title"],test["Age"])


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# So far so good...but there is an issue, how will the model use Sex, Title and AgeGroup for prediction these are qualitatie vaues in-order to make these usable for the model prediction lets create dummy columns for these which will be used by the model for prediction.. how to do that ? pretty simple..Lets see

# In[ ]:


test=pd.concat([test,pd.get_dummies(test.Sex)],axis=1)
test=pd.concat([test,pd.get_dummies(test.AgeGroup)],axis=1)
train=pd.concat([train,pd.get_dummies(train.Sex)],axis=1)
train=pd.concat([train,pd.get_dummies(train.AgeGroup)],axis=1)


# In[ ]:


test.head(5)


# In[ ]:


train.head(5)


# I think by now I have all required information to pass it to Model for prediction..but before I do that..Lets clean all this up and get rid of un-necessary columns..also create a vector of survived passengers from our train dataset.

# In[ ]:


trainingSetSurvivedPassengerVector =train.Survived


# In[ ]:


train.columns


# In[ ]:


train.drop(['Survived','Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup'], axis=1, inplace=True)


# In[ ]:


train.head(5)


# In[ ]:


test.drop(['Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'AgeGroup'], axis=1, inplace=True)


# In[ ]:


test.head(5)


# Now we are good to call the Model for prediction, for this exercise I shall call logistic classfier. Lets do it.....

# In[ ]:


regression = LogisticRegression()
regression.fit(train, trainingSetSurvivedPassengerVector)
testSetSurvivedPassengerVector = regression.predict(test)
modelAccuracy = round(regression.score(train, trainingSetSurvivedPassengerVector) * 100, 2)
modelAccuracy


# In[ ]:


submitResult = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": testSetSurvivedPassengerVector})


# In[ ]:


submitResult.to_csv('submitResults.csv', index=False)


# In[ ]:




