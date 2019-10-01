#!/usr/bin/env python
# coding: utf-8

# Introduction
# ------------
# 
# This Kaggle Notebook is meant to serve as an introduction to those on Kaggle for the first time. We walk through reading in data, manipulating the data, running multiple ML algorithms, and writing to a file.
# We look at a combination of different methods of analyzing the Titanic Survivor data using Machine Learning.
# 
# Data Cleaning techniques:
# 
#  - Simple (eliminate inconvenient features)
# 
#  - Complex (create new features)
# 
# Machine Learning algorithms:
# 
#  - Random Forest Classifier
# 
#  - Gradient Boosting Classifier
# 
# We apply our algorithms to both data cleaning techniques, leading to four different possible outputs.

# ##Imports##
# 
# 
# First, we import the tools we will need:
# 
#  - CSV: writing comma separated value files at the end (output)
#  - Numpy: data manipulation
#  - Pandas: data storage & manipulation
#  - SKLearn: Machine Learning framework
#  - Subprocess: see what files we will be working with

# In[ ]:


import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ##Simple Data Cleaning##
# Our goal with this function is to turn gender into numeric variables and eliminate all other features that aren't already simple.
# 
# First, we use Pandas to read in the CSV file. The data turns into a DataFrame.
# 
# Next, we create a new feature called 'Gender'. We assign 'Gender' to always be 3 as a placeholder. Now, we go ahead and correctly fill in Gender. We use Panda's built-in mapping function to turn "male" or "female" into 1 or 0, respectively.
# 
# Finally, we drop everything that isn't simple, remove anything that's null, and return our DataFrame.

# In[ ]:


def clean_data_simple(file_url):
    df = pd.read_csv(file_url,header=0)
    
    df['Gender'] = 3
    df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(int)
    
    df.loc[(df.Age.isnull()),'Age'] = 0
    
    df = df.drop(['Name','Sex','Ticket','Fare','Cabin','Embarked'],axis=1)
    
    return df


# ##Complex Data Cleaning##
# 
# Our goal with this function is to create a more robust set of features. We still turn gender into numeric variables but also create new features, such as the total party size, the place of embarkment, and the adjusted fare.
# 
# First, we use Pandas to read in the CSV file. The data turns into a DataFrame.
# 
# Next, we create a new feature called 'Gender'. We assign 'Gender' to always be 3 as a placeholder. Now, we go ahead and correctly fill in Gender. We use Panda's built-in mapping function to turn "male" or "female" into 1 or 0, respectively.
# 
# We follow the same procedure as 'Gender' to create 'EmbarkCode'. Remember, with the frameworks we are using, all features should be a numeric digit. We convert the location of embarkment to either 0, 1, or 2.
# 
# Instead of using separate features ('SibSp': Siblings+Spouses & 'Parch': Parents+Children), we combine these two features into 'TotalParty' to find the size of the total group.
# 
# Fare is currently a large number that tends to range between 0 and 40. To not give this feature too much weight, we divide all fares by 10 (FareAdjusted). This idea may or may not work... tuning is the core of Machine Learning, so give it (and other ideas!) a try.
# 
# Finally, we drop everything that isn't simple, remove anything that's null, and return our DataFrame.

# In[ ]:


def clean_data_complex(file_url):
   df = pd.read_csv(file_url,header=0)

   df = df.drop(['Name','Ticket','Cabin'],axis=1)
   
   df['Gender'] = 3
   df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(int)
   df = df.drop(['Sex'],axis=1)
   
   df['EmbarkCode'] = 3
   df['EmbarkCode'] = df['Embarked'].map({'S':0,'C':1,'Q':2})
   df = df.drop(['Embarked'],axis=1)
   
   df['TotalParty'] = df['SibSp'] + df['Parch']
   df = df.drop(['SibSp','Parch'],axis=1)
   
   df['FareAdjusted'] = df['Fare'] / 10.0
   df = df.drop(['Fare'],axis=1)
   
   df.loc[(df.Age.isnull()),'Age'] = 0
   df.loc[(df.EmbarkCode.isnull()),'EmbarkCode'] = 3
   df.loc[(df.FareAdjusted.isnull()),'FareAdjusted'] = 1
   
   return df


# ##Clean Training Data##
# 
# Now let's run our cleaning functions on out training input data. We will receive two DataFrames, one for the simple cleaning and one for the complex algorithm.
# 
# To verify everything is working, we'll print the head of the DataFrames.

# In[ ]:


train_dataframe = clean_data_simple('../input/train.csv')
print(train_dataframe.head())

train_dataframe_complex = clean_data_complex('../input/train.csv')
print(train_dataframe_complex.head())


# ##Clean Test Data##
# 
# Now let's run our cleaning functions on out test input data. We will receive two DataFrames, one for the simple cleaning and one for the complex algorithm.
# 
# To verify everything is working, we'll print the head of the DataFrames.
# Keep in mind, this data does not have the `Survived` feature, as it is test data.

# In[ ]:


test_dataframe = clean_data_simple('../input/test.csv')
print(test_dataframe.head())

test_dataframe_complex = clean_data_complex('../input/test.csv')
print(test_dataframe_complex.head())


# ##DataFrame -> Numpy##
# 
# Unfortunately, SciKit Learn doesn't play nice with DataFrames. We need to convert our DataFrames to Numpy arrays. Luckily, it's a quick one-liner.

# In[ ]:


train_data = train_dataframe.values
test_data = test_dataframe.values

train_data_complex = train_dataframe_complex.values
test_data_complex = test_dataframe_complex.values


# ##Random Forest Classifier##
# 
# Time to get (machine) learning! We repeat these steps twice... once with the simple data cleaning, once with the complex. We save our outputs as separate variables.
# 
# When we fit our data, we need to be careful what data we are selecting.
# 
#     train_data[0::,2::]
# 
# This code takes all rows and columns starting at column 3 onward (remember, 0 is the first row/column). We capture all the data we need but avoid PassengerId and Survived-status.
# 
# 
#     train_data[0::,1]
# 
# This code takes all rows and only the Survived column.

# In[ ]:


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,2::],train_data[0::,1])
forest_output = forest.predict(test_data[0::,1::])

forest_complex = RandomForestClassifier(n_estimators = 100)
forest_complex = forest_complex.fit(train_data_complex[0::,2::],train_data_complex[0::,1])
forest_complex_output = forest_complex.predict(test_data_complex[0::,1::])


# ##Gradient Boosting Classifier##
# 
# Time to get (machine) learning! We repeat these steps twice... once with the simple data cleaning, once with the complex. We save our outputs as separate variables.
# When we fit our data, we need to be careful what data we are selecting.
# 
#     train_data[0::,2::]
# 
# This code takes all rows and columns starting at column 3 onward (remember, 0 is the first row/column). We capture all the data we need but avoid PassengerId and Survived-status.
# 
#     train_data[0::,1]
# 
# This code takes all rows and only the Survived column.
# 
# *We have commented out the code below because it takes too long to learn. We'll discuss alternatives in future updates.*

# In[ ]:


#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(train_data[0::,2::], train_data[0::,1])
#gradient_output = clf.predict(test_data[0::,1::])  

#clf_complex = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(train_data_complex[0::,2::], train_data_complex[0::,1])
#gradient_complex_output = clf.predict(test_data_complex[0::,1::])


# ##Data Output##
# 
# Here, we save our outputs to separate files. Later, we may update this notebook to discuss comparing data outputs and seeing if our intuition can help point out which algorithm combination worked best.

# In[ ]:


output = forest_output
predictions_file = open("forest_output.csv", "w")
open_file_object = csv.writer(predictions_file)
ids = test_dataframe['PassengerId'].values
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Saved "forest_output" to file.')

output = forest_complex_output
predictions_file = open("forest_complex_output.csv", "w")
open_file_object = csv.writer(predictions_file)
ids = test_dataframe['PassengerId'].values
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Saved "forest_complex_output" to file.')

