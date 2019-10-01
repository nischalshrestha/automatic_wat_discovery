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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Exploratory Analysis**
# 
# Let's take a look at the dataset.  Here are notes on the dataset as provided by Kaggle:
# 
# Variable: Definition Key
# survival: Survival	0 = No, 1 = Yes
# 
# pclass:	Ticket class:	1 = 1st, 2 = 2nd, 3 = 3rd
# sex: Sex	
# Age: Age in years	
# sibsp:	# of siblings / spouses aboard the Titanic	
# parch:	# of parents / children aboard the Titanic	
# ticket:	Ticket number	
# fare: Passenger fare	
# cabin: Cabin number	
# embarked: Port of Embarkation:	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[ ]:


# Importing the datasets
training_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")
training_set.head(20)


# In[ ]:


test_set.head(20)


# I am attempting to predict the survival of each passenger based on the independent variables in the dataset of which there are 11.  Some of these 11 have no relationship or correllation to the dependent variable "Survived".  At first glance these apear to be: PassengerId, Name, Ticket, Fare, and Embarked.  This leaves us with Pclass, Sex, Age, SibSp, Parch, and Cabin as usable IV's.   

# In[ ]:


# Display the number of missing values for each feature(column)
training_set.isnull().sum()


# In[ ]:


test_set.isnull().sum()


# I am not using Embarked so 2 missing values does not matter.  Therefore, Age and Cabin are the only variables to be considered.  For age I will fill the missing values with the mean age.  I do not know the format for the cabin identifier and cannot fill in the missing values, so I will leave the variable out of the model.
# 1. First I will split the dataset into the matrix of features and the dependent variable vector.
# 2. Secondly, I will take care of the missing data in Age with the mean of the column

# **Split the data into the Training Set and the Test Set**

# In[ ]:


# Create Matrix of Features(Training set)
X_train = training_set.iloc[:, [2,4,5,6,7]].values # Pclass, Sex, Age, SibSp, Parch

# Create Dependent Variable Vector(Training set)
y_train = training_set.iloc[:, 1].values # Survived

# Create Matrix of Features(Test set)
X_test = test_set.iloc[:, [1,3,4,5,6]].values # Pclass, Sex, Age, SibSp, Parch

# No y_test as I'm not given the "survived" feature in the test set as this is a competition

# Take care of missing values in Age for both Training Set and Test Set
from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer(strategy = 'mean')
imputer_test = SimpleImputer(strategy = 'mean')

# Training Set
imputer_train = imputer_train.fit(X_train[:, 2:3]) # Fit Age column to imputer
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3]) # Convert NaN's to mean of whole column

# Test Set
imputer_test = imputer_test.fit(X_test[:, 2:3]) # Fit Age column to imputer
X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3]) # Convert NaN's to mean of whole column

# Couldn't easily find a built in numpy way to compute this so I just made my own function to count how many NaN's are in the imputed X_train ndarray
def numMissing(X_train):
    num_nans = 0
    for y in X_train:
        if y[2] == np.nan:
            count = count + 1
    return num_nans

print("Training Set: Number of missing values in age: {}".format(numMissing(X_train)))
print("Test Set: Number of missing values in age: {}".format(numMissing(X_test)))


# In[ ]:


X_train


# In[ ]:


X_test


# Age has now been dealt with

# **Encoding Categorical Data**
# 
# Missing data has been dealt with, now the categorical data must be encoded.  The Matrix of Features X_train contains 5 features: Pclass, Sex, Age, SibSp, and Parch.  
# * Pclass is already in a numerical format, but one is not necessarily better than the other for survival, so it must be One Hot Encoded
# * Sex is either Male or Female, so it must be Label Encoded first and then One Hot Encoded
# * Age is already in a numerical format, and fine the way it is
# * SibSp and Parch are in a numerical format, but one or the other value  is not necessarily better for survival, so it must be One Hot Encoded.
# 
# y_train does not need any adjusting as it is only if they survived or not which is either 0 or 1.
# 
# The same encoding will be done for X_test as well

# In[ ]:


# Encoding categorical data for X_train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encode Sex feature in X_train and X_test
labelencoder_train_sex = LabelEncoder()
X_train[:, 1] = labelencoder_train_sex.fit_transform(X_train[:, 1])
labelencoder_test_sex = LabelEncoder()
X_test[:, 1] = labelencoder_test_sex.fit_transform(X_test[:, 1])

# OneHotEncode X_train
onehotencoder_train = OneHotEncoder(n_values = 'auto', categories = 'auto')
X_train = onehotencoder_train.fit_transform(X_train[:, [0,1,3,4]]).toarray()

# OneHotEncode X_test
onehotencoder_test = OneHotEncoder(n_values = 'auto', categories = 'auto')
X_test = onehotencoder_test.fit_transform(X_test[:, [0,1,3,4]]).toarray()

# Print shape
print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))


# **Feature Scaling**

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


X_train


# In[ ]:


X_test


# **Fitting Logistic Regression to the Training Set**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# **Predict Test Set**

# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test_set["PassengerId"],
    "Survived": y_pred
})


# In[ ]:


submission.shape


# In[ ]:


submission


# In[ ]:


submission.to_csv('survive_or_not.csv')
print(os.listdir("../working"))


# In[ ]:




