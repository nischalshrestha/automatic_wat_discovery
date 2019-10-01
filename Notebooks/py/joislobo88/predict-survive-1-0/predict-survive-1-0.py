#!/usr/bin/env python
# coding: utf-8

# My second notebook. Any feedback is greatly appreciated

# **Load Data :**
#    First import the training and test data from csv to data frame

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB # Naive Bayes classifier
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

dfTrain = pd.read_csv("../input/train.csv") # importing training set
dfTest = pd.read_csv("../input/test.csv") # importing test set


# **Find Missing values in the data : **

# In[ ]:


dfTrain.shape


# Check whether the training set has any null values for the attributes by calculating the sum of null values

# In[ ]:


print(dfTrain.isnull().sum())


# Based on the data Age,Cabin,Embarked has missing values in the training set and in the test set age,cabin have missing values

# In[ ]:


print(dfTest.isnull().sum())


# **Handle the missing values : **   
#  Fill the missing values with the mean value  in training and test set.

# In[ ]:


dfTrain.fillna(dfTrain.mean(), inplace=True)
dfTest.fillna(dfTest.mean(), inplace=True)


# **Pictorial representation of total number of passengers who survived and who did not survive **

# In[ ]:


ax = dfTrain.Survived.value_counts().plot(kind='bar',color="rg")
ax.set_ylabel("Number of passengers")
ax.set_xlabel("Survived [No-0 , Yes -1]")
rects = ax.patches

# Add labels

for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, 1 + height, '%d' % int(height), ha='center', va='bottom')


# Encoding the input into categorical numeric type

# In[ ]:


dfTrain['Sex'] = pd.factorize(dfTrain['Sex'])[0]
dfTest['Sex'] = pd.factorize(dfTest['Sex'])[0]


# **Select the attributes of interest for prediction  : **    
# i)Pclass,
# ii)Age,
# iii)Sex,
# iv)Fare 
# Assuming they are independent  

# In[ ]:


trainAttributeData = pd.DataFrame.as_matrix(dfTrain[['Pclass','Age','Sex','Fare']])
testAttributeData =  pd.DataFrame.as_matrix(dfTest[['Pclass','Age','Sex','Fare']])
trainPredictAttribute =  pd.DataFrame.as_matrix(dfTrain[['Survived']]).ravel()


# **Train the model using Naive Bayes Classifier**

# In[ ]:


trainPredictAttribute.shape
classifier = GaussianNB()
classifier.fit(trainAttributeData,trainPredictAttribute)
GaussianNB(priors=None)


# **Predict the values for Survived on the test data set **

# In[ ]:


predictValues = pd.DataFrame(classifier.predict(testAttributeData),columns=['Survived'])


# **Create results dataframe with the PassengerId and the Survived column and export to excel**

# In[ ]:


passengerIdValues = pd.DataFrame()
passengerIdValues['PassengerId'] = dfTest['PassengerId']


# In[ ]:


finalResult = passengerIdValues.join(predictValues)


# In[ ]:


finalResult.to_csv("Survivor_prediction.csv", index = False)


# In[ ]:


finalResult.head()

