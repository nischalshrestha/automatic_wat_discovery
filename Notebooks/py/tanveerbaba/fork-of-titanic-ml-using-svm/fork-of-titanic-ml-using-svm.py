#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let's Perform the Exploratory Data Analysis on a Titanic Training Data Set

# In[ ]:


#Our Train data set
train.head()


# In[ ]:


#Summary of a Dataframe
train.info()


# In[ ]:


#Generates Descriptive Statistics
train.describe()


# In[ ]:


#To show missing data I will use Heatmap
plt.figure(figsize = (12,8))
sns.heatmap(train.isnull(), yticklabels = False, cmap = 'viridis', cbar = False)


# As we can see Age column and Cabin have missing data. To Filling or Cleaning the data we can use different kind of methods 

# In[ ]:


#Let's first check how people survived in a training set 
sns.countplot(train['Survived'], palette = 'RdBu_r')
#plt.legend(train['Sex'])


# As we can almost 330 people are survivors

# In[ ]:


#Now we can another property to visualize Survivors along with their class 
sns.countplot(train['Survived'], palette = 'RdBu_r', hue = train['Pclass'])


# Visualization shows that passengers who didn't survive belongs to Class 3(Poor people we can say) and Class 1(Rich people) survived more than others

# In[ ]:


#Now we can another property to visualize how many were Males and Females
sns.countplot(train['Survived'], palette = 'RdBu_r', hue = train['Parch'])


# Above visualization shows that most males died than females

# In[ ]:


#Now we can visualize Age of Passengers
plt.figure(figsize =(12,6))
sns.set_style('whitegrid')
sns.distplot(train['Age'].dropna(), kde = False, bins = 30)


# As we can see most Ages are between 19 to 32

# In[ ]:


#Here we show Passengers count of Siblings and Spouses
plt.figure(figsize = (12,6))
sns.countplot(train['SibSp'], palette = 'viridis')


# Visualization shows most passengers don't have Siblings and Spouses in a Ship.

# In[ ]:


#Let's Visualize the Embarked Feature
plt.figure(figsize = (12,6))
sns.countplot(train['Embarked'])


# 
# 
# 
# # Before Jumping to the Data Analysis on a Titanic Data Test, let's perform data Cleaning on a Titanic Training Data Set.

# During Exploratory Data Analysis we find out that 'Age' feature and 'Cabin' feature were missing data.
# Lot's of ways we can handle these kind of features. In this Data Cleaning process, we will perform simple techniques

# In[ ]:


#Let's visualize to find out the Average Age's of passengers based on their 'Pclass'
sns.boxplot(x= 'Pclass', y = 'Age', data = train)


# As we can see, Average Age of Class 1 passengers is 37.

# Average Age of Class 2 passengers is 29.

# Average Age of Class 3 passengers is 24.

# In[ ]:


#Now let's define a function to fill the 'Age' column's missing fields by the average age of particular classes
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


#call a function to fill the missing 'Age' data
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)


# In[ ]:


#Let's visualize the training set again by HeatMap
plt.figure(figsize = (12,6))
sns.heatmap(train.isnull(), cmap = 'viridis', yticklabels = False, cbar = False)


# As we can see 'Age' column is filled with the Average ages of each particular passenger class.

# In[ ]:


#It's better for us to drop the 'Cabin' column because lot of data is missing in the 'cabin' column
train.drop('Cabin', axis = 1, inplace = True)


# In[ ]:


#Let's fill in the 'Embarked' Feature
train['Embarked'].fillna(value = 'S', inplace = True)


# In[ ]:


train.head()


# Now let's look out those features we don't need in our model:
# 1. PassengerID
# 2. Name
# 3. Ticket
# 4. Embarked

# In[ ]:


#Now let's drop these features in our train data set
train.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = True)


# Before we use this data set to our model we need to create dummy variables for categorial features to make it easy for the Model.

# We can look into our data set, there is only one categorical feature 'Sex' and create Dummy variables for 'Embarked'' for better performance. Let's create dummy for it.

# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first = True)


# In[ ]:


#Now let's concatenate the sex Dataframe with train Dataframe
train = pd.concat([train,sex], axis = 1)


# In[ ]:


#Rename the 'male' column and drop the 'Sex' column
train.drop('Sex',axis = 1,inplace = True)


# In[ ]:


Embarked = pd.get_dummies(train['Embarked'])


# In[ ]:


#Now let's concatenate the Embarked Dataframe with train Dataframe
train = pd.concat([train,Embarked], axis = 1)


# In[ ]:


#Drop the 'Embarked' column
train.drop('Embarked',axis = 1,inplace = True)


# In[ ]:


#Our cleaned Train Data set:
train.head()


# # Let's concentrate on the Testing Data set and clean it up

# In[ ]:


#Our Test data set
test.head()


# In[ ]:


#Summary of a Dataframe 
test.info()


# In[ ]:


#Generates Descriptive Statistics
test.describe()


# In[ ]:


#visualize the missing data by heatmap
plt.figure(figsize = (12,8))
sns.heatmap(test.isnull(), cbar = False, yticklabels = False, cmap = 'viridis')


# We need to perform same operations on test data set for cleaning features(where data is missing)
# 
# Let's fill in the missing data of 'Age' column with average age with respect to passenger class

# In[ ]:


#Let's visualize to find out the Average Age's of passengers based on their 'Pclass'
sns.boxplot(x = 'Pclass', y = 'Age', data = test)


# As we can see, Average Age of Class 1 passengers is 42.
# 
# Average Age of Class 2 passengers is 27.
# 
# Average Age of Class 3 passengers is 24.

# In[ ]:


#Now let's define a function to fill the 'Age' column's missing fields by the average age of particular classes
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 27
        else:
            return 24
    else:
        return Age


# In[ ]:


#call a function to fill the missing 'Age' data
test['Age'] = test[['Age','Pclass']].apply(impute_age, axis = 1)


# In[ ]:


#visualize the missing data of test data by heatmap
plt.figure(figsize = (12,8))
sns.heatmap(test.isnull(), cbar = False, yticklabels = False, cmap = 'viridis')


# As we can see 'Age' column is filled with the Average ages of each particular passenger class.

# In[ ]:


#Let's first drop the features we don't need to test for predictions 
PassengerId = pd.DataFrame(test['PassengerId'])
test.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)


# In[ ]:


#Let's visualize to find out the Average Fare's of passengers based on their 'Pclass'
plt.figure(figsize = (10,5))
sns.boxplot(x = 'Pclass', y = 'Fare', data = test)
plt.ylim(0,150)


# In[ ]:


#Let's Visualize 'Embarked' feature of Test data set
sns.countplot(test['Embarked'])


# In[ ]:


#Let's fill in the 'Embarked' Feature
test['Embarked'].fillna(value = 'S', inplace = True)


# In[ ]:


#Now feature 'Fare' has some missing data. Let's try to handle it
def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Fare):
        if Pclass == 1:
            return 60
        elif Pclass == 2:
            return 16
        else:
            return 8
    else:
        return Fare


# In[ ]:


#call a function to fill the missing 'Fare' data
test['Fare'] = test[['Fare','Pclass']].apply(impute_fare, axis = 1)


# Now, our last task is to convert categorical features into dummy variables

# In[ ]:


sex = pd.get_dummies(test['Sex'], drop_first = True)


# In[ ]:


#Now let's concatenate the sex Dataframe with test Dataframe
test = pd.concat([test,sex], axis = 1)


# In[ ]:


#Rename the 'male' column and drop the 'Sex' column
test.drop('Sex',axis = 1,inplace = True)


# In[ ]:


Embarked = pd.get_dummies(test['Embarked'])


# In[ ]:


#Now let's concatenate the Embarked Dataframe with test Dataframe
test = pd.concat([test,Embarked], axis = 1)


# In[ ]:


#Drop the 'Embarked' column
test.drop('Embarked',axis = 1,inplace = True)


# In[ ]:


#Our cleaned Test Data set:
test.head()


# # Now, it's time to train the model to predict whether the passenger Survived or not.

# In[ ]:


X_train = train.drop('Survived', axis = 1)
y_train = train['Survived']


# In[ ]:


X_test = test


# In[ ]:


#Import the model 
from sklearn.svm import SVC


# In[ ]:


#Object creation
model = SVC()


# In[ ]:


#Let's fit the model with the training data set
model.fit(X_train,y_train)


# In[ ]:


#Calculate the predictions
predictions = model.predict(X_test)


# In[ ]:


from sklearn.grid_search import GridSearchCV


# In[ ]:


param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose = 3)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


grid_predictions = pd.DataFrame(grid_predictions)


# In[ ]:


titanic = pd.concat([PassengerId,grid_predictions], axis = 1)
titanic.columns = ['PassengerId','Survived']


# In[ ]:


titanic.to_csv('Improved_predictions.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




