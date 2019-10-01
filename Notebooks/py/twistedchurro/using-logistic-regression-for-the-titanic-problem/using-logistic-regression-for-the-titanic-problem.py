#!/usr/bin/env python
# coding: utf-8

# # Titanic Prediction using Python
# ### A huge thank you to Jose Portilla and his Udemy course for teaching me https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4

# ## Imports and reading in files

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/train.csv")


# In[ ]:


#Check that the file was read in properly and explore the columns
df.head()


# ## Data exploration

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')


# From the heat map, we can see that a lot of the 'Cabin' row information is missing. However, while the age column is also missing some data, we can use imputation to fill in some of the data later. Additionally, the 'Embarked' column has so few rows missing, that we can just delete those.

# In[ ]:


sns.set_style('darkgrid')


# In[ ]:


sns.countplot(x='Survived', data=df, hue='Pclass')


# We can see here that those who did not survive were predominantly from the 3rd Passenger Class (Pclass).

# In[ ]:


sns.countplot(x='SibSp', data=df, hue='Survived')


# In[ ]:


df['Fare'].hist(bins=40)


# Here, we impute the age of those we do not have information on. We use a boxplot to estimate the median age of each class, and impute that into the age for the rows with missing age.

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass', y='Age', data=df)


# In[ ]:


def inpute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else: return 24
    else: return Age


# In[ ]:


df['Age']=df[['Age','Pclass']].apply(inpute_age, axis=1)


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')


# You can see now the data is cleaner, but we still need to clean the 'Cabin' and 'Embarked' columns. For now, we will simply drop the 'Cabin' column and drop the rows where 'Embarked' is missing.

# In[ ]:


df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(),cbar=False, yticklabels=False, cmap='viridis')


# The data is now clean of null values, but we still need to take care of objects that a machine learning algorithm can't handle, namely strings.

# In[ ]:


df.info()


# We can see that 'Name', 'Sex', 'Ticket', and 'Embarked' are all objects. In this case, they indeed are all strings. We will use Pandas built in getDummies() funciton to convert those to numbers.

# In[ ]:


#We make a new 'Male columns because getDummies will drop one the the dummy variables
#to ensure linear independence.
df['Male'] = pd.get_dummies(df['Sex'], drop_first=True)


# In[ ]:


#The embarked column indicates where the passenger boarded the Titanic.
#It has three values ['S','C','Q']
embarked = pd.get_dummies(df['Embarked'], drop_first=True)
df = pd.concat([df, embarked], axis=1)


# In[ ]:


#These columns do not provide us any information for the following reasons:
#PassengerID: we consider 'PassengerID' a randomly assigned ID thus not correlated with surviability
#Name: we are not performing any feature extraction from the name, so we must drop tihs non-numerical column
#Sex: the 'Male' column already captures all information about the sex of the passenger
#Ticket: we are not performing any feature extraction, so we must drop this non-numerical column
#Embarked: we have extracted the dummy values, so those two numerical dummy values encapsulate all the embarked info

df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)


# In[ ]:


#Take a look at our new dataframe
df.head()


# In[ ]:


df.info()


# ##Build and train the model

# In[ ]:


#Seperate the feature columns from the target column
X = df.drop('Survived', axis=1)
y = df['Survived']


# In[ ]:


#Split the data into two. I don't think this is necessary since there are two files.
#I will keep this here for now
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X, y)


# In[ ]:


#Read in the test data
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


#Clean the test data the same way we did the training data
test_df['Age']=test_df[['Age','Pclass']].apply(inpute_age, axis=1)
test_df.drop('Cabin', axis=1, inplace=True)
test_df.dropna(inplace=True)
test_df['Male'] = pd.get_dummies(test_df['Sex'], drop_first=True)
embarked = pd.get_dummies(test_df['Embarked'], drop_first=True)
test_df = pd.concat([test_df, embarked], axis=1)
pass_ids = test_df['PassengerId']
test_df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)


# In[ ]:


test_df.tail()


# In[ ]:


predictions = logmodel.predict(test_df)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": pass_ids,
        "Survived": predictions
    })
submission.to_csv('titanic.csv', index=False)

