#!/usr/bin/env python
# coding: utf-8

# # The titanic data set
# 
# Chris Kulp
# 
# July 26, 2018
# 
# This is my first attempt at a Kaggle competition. I chose to use a Random Forest Classifier because another research  project that I am working on uses one as well. I wanted to get some experience using Random Forests before applying it to my research projet.
# 
# This notebook imports the data, targets the features I wanted to use, trains a RandomForest classifier, and makes predictions on the test data using the trained model.
# 
# This is the first kernel that I have uploaded to Kaggle.

# In[ ]:


#load the necessary libraries

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# ## Import and inspect data

# In[ ]:


df_train = pd.read_csv('../input/train.csv',delimiter=",")
df_train.head()


# In[ ]:


df_test = pd.read_csv('../input/test.csv',delimiter=",")
df_test.head()


# ## Data Cleaning and Feature Engineering
# I wanted to avoid columns with NaN because I didn't want to deal with replacing those values with interpolations, averages, etc... on my first attempt. Hence, I eliminated any columns with a NaN. In addition, I removed Fare on this attempt thinking that it is likely correlated with Pclass (i.e. higher class tickets cost more).
# 
# I also replaced the categorical data, Embarked and Sex so that they are computable. 
# 

# In[ ]:


def clean_data(df):
    df_elim_cols=df.drop(['Age','Cabin','Ticket','Fare','Embarked'],axis=1) #eliminate columns I don't want
    df_replace_sex = df_elim_cols.replace({'Sex': {'female': 1, 'male': 2}}) #replace Sex data
    return df_replace_sex


# In[ ]:


clean_train = clean_data(df_train)
clean_test = clean_data(df_test)
clean_train.head()


# In[ ]:


clean_test.head()


# Inspired by other kernels, I decided to do some feature engineering. I created a new feature that is based on the title of the passenger.

# In[ ]:


def engineer_title_column(df):
    '''This function will look at the name of each individual, determine the title in the name and assign 
    that title a number. It will then create and append a Title column to the data frame. Note that I am sure
    that there are better ways of doing this!'''
    
    title_list = [] #dummy list of title numbers
    
    for item in df['Name']: #go through each person and identify their title and assign it a number
        if "Mr." in item:
            title_list.append(1)
        elif "Mrs." in item:
            title_list.append(2)
        elif "Miss." in item:
            title_list.append(3)
        elif "Master." in item:
            title_list.append(4)
        elif "Rev." in item:
            title_list.append(5)
        else:
            title_list.append(6) #a "catch-all" for any other title
        
    titles_to_column = pd.Series(title_list) #create a pandas series
    df['Title'] = titles_to_column.values #append the above series to the data frame
    name_dropped = df.drop(['Name'],axis=1) #remove the name column
    return name_dropped


# In[ ]:


training_df=engineer_title_column(clean_train)
test_df = engineer_title_column(clean_test)
training_df.head()


# In[ ]:


test_df.head()


# ## Train the model
# 
# First I convert the data frames to NumPy arrays so that I can use the data to train the random forest classifier. 

# In[ ]:


X_train = training_df.iloc[:,2:].values #.values turns the dataframe into a numpy array (Gets rid of index)
y_train = training_df.iloc[:,1].values

X_test = test_df.iloc[:,1:].values #.values turns the dataframe into a numpy array (Gets rid of index)


# Train the classifier.

# In[ ]:


estimator = RandomForestClassifier(n_estimators=10)
rnd_clf = estimator
rnd_clf.fit(X_train,y_train)


# Check accuracy of model on the training data. I also compute the confusion matrix. 

# In[ ]:


y_pred = rnd_clf.predict(X_train)
print(accuracy_score(y_train,y_pred))


# In[ ]:


confusion_matrix(y_train, y_pred)


# Additional feature engineering and hyperparameter tuning would likely improve the above scores, but since this is a first attempt, we'll keep moving forward. 

# ## Make predictions on test data

# In[ ]:


test_pred = rnd_clf.predict(X_test)


# ## Prepare test results for submission
# 
# I do this by adding a column of test results to the cleaned up and engineered test data frame and removing all of the feature columns. There are probably better ways of doing this...

# In[ ]:


test_df['Survived'] = test_pred
output_df = test_df.drop(['Pclass','Sex','SibSp','Parch','Title'],axis=1) 
output_df.head()


# In[ ]:


#finally, export results to a submission file
#output_df.to_csv('submission_file.csv',index=False)


# While there is clearly room for improvement here, I found this exercise to be very helpful in learning how to put to practice my studies in machine learning. I hope you find this kernel useful in some way. Thank you for reading it.

# In[ ]:




