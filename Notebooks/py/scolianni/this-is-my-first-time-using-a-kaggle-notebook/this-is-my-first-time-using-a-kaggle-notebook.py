#!/usr/bin/env python
# coding: utf-8

# ## This is my first time using a Kaggle notebook...
# ...And I am not really sure what I am doing.  My main goal here is to write a really sketchy submission for the Titanic competition.
# 
# ### My goals are as follows
# 
# 1.  Write a "really bad" first submission to the competition that works.  Gotta jump in the pool some time.
# 2.  Improve on #1 using what I already know to create a simple model that uses sklearn.
# 3.  Make sure I'm adding in all this cool dank Pandas (a library I'm still learning) knowledge.  It definitely saves time.
# 4.  Look at some other kernels / the forum and make minor improvements.
# 5.  When I feel comfortable navigating the Kaggle environment, take a look at another competition.

# ## Read in the data

# In[ ]:


# The first order of business, as always, is to read in the mother fucking data

import pandas as pd
dfTrain = pd.read_csv('../input/train.csv')
dfTest = pd.read_csv('../input/test.csv')


# ## Look at the format of the data
# 
# We need to know the expected data types for each column to be able to effectively clean the data of NaN's.  
# 
# We can see the following:
# 
# 1.  PassengerID - int
# 2.  Survived - int
# 3.  Pclass - int
# 4.  Name - string
# 5.  Sex - string
# 6.  Age - float
# 7.  SibSp - int
# 8.  Parch - int
# 9.  Ticket - string
# 10.  Fare - float
# 11.  Cabin - String
# 12.  Embarked - String

# In[ ]:


dfTrain.head()


# ## Clean the NaN's from the data

# In[ ]:


# Assign default values for each data type
defaultInt = -1
defaultString = 'NaN'
defaultFloat = -1.0

# Create lists by data tpe
intFeatures = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch']
stringFeatures = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
floatFeatures = ['Age', 'Fare']

# Clean the NaN's
for feature in list(dfTrain):
    if feature in intFeatures:
        dfTrain[feature] = dfTrain[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        dfTrain[feature] = dfTrain[feature].fillna(defaultString)
    elif feature in floatFeatures:
        dfTrain[feature] = dfTrain[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)
    
for feature in list(dfTest):
    if feature in intFeatures:
        dfTest[feature] = dfTest[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        dfTest[feature] = dfTest[feature].fillna(defaultString)
    elif feature in floatFeatures:
        dfTest[feature] = dfTest[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)


# ## Encode data
# 
# Encode all features except PassengerId, as this needs to be untouched for the Kaggle grading script to run properly.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
dfCombined = pd.concat([dfTrain, dfTest])
for feature in list(dfCombined):
    
    le = LabelEncoder()
    le.fit(dfCombined[feature])
    
    if feature in dfTrain:
        if feature != 'PassengerId':
            dfTrain[feature] = le.transform(dfTrain[feature])
    if feature in dfTest:
        if feature != 'PassengerId':
            dfTest[feature] = le.transform(dfTest[feature])


# ## Split into a training and test set
# 
# Sklearn can take a dataframe as input to its functions??  Cool!

# In[ ]:


from sklearn.model_selection import train_test_split

X = dfTrain.drop(['Survived'], axis=1)
y = dfTrain['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=num_test, random_state=23)


# ## Train ML classifier
# 
# For this purpose, we will use Random Forests.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)


# ## Calculate accuracy

# In[ ]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print(accuracy)


# ## Form predictions on test set

# In[ ]:


# Generate predictions
clf = RandomForestClassifier()
clf.fit(X, y)
dfTestPredictions = clf.predict(dfTest)

# Write predictions to csv file
results = pd.DataFrame({'PassengerId': dfTest['PassengerId'], 'Survived': dfTestPredictions})
results.to_csv('results.csv', index=False)
results.head()


# ## Possible Improvements
# 
# 1.  Instead of default values for NaN's, clean missing values with column averages.
# 2.  Remove columns that offer little or no information.  Eg. PassengerId.
# 3.  Add features related to family size.
# 4.  Assess the accuracy by implementing k-fold cross validation.
# 5.  Implement Gird Search to tune hyper parameters
# 6.  Learn how to make an evaluation for how valuable each feature is.  Remove unimportant features.

# # Things I Learned
# 
# 1.  In Python 3 specifically, for LabelEncoder to work, encodings of NaN's must be to a data type homogeneous with the rest of the data in a column. 
# - This is not a problem in Python 2.7
# 2.  You can pass a Pandas data frame into an Sklearn classifier.  That's pretty fucking cool, and saves a lot of time.
