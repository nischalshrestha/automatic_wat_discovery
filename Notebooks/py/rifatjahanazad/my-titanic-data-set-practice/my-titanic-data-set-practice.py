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
print(os.listdir('../input'))
### default given helpful info ###
# Any results you write to the current directory are saved as output.


# # Other Necessary Imports

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
get_ipython().magic(u'matplotlib inline')


# # Loading Data

# In[ ]:


baseDirectory = '../input/'
trainFile = baseDirectory + 'train.csv'
testFile = baseDirectory + 'test.csv'
gender_submissionFile = baseDirectory + 'gender_submission.csv'

trainData = pd.read_csv(filepath_or_buffer=trainFile)
testData = pd.read_csv(filepath_or_buffer=testFile)
gender_submissionData = pd.read_csv(filepath_or_buffer=gender_submissionFile)


# # Data Exploring

# In[ ]:


# to display ass rows
pd.options.display.max_rows = 999
print (trainData)
print(trainData.head())
print(testData.head())
print(gender_submissionData.head())


# ## Train Data

# In[ ]:


print(trainData.info())
print(trainData.describe())


# In[ ]:


sns.pairplot(trainData)


# ## Test Data

# In[ ]:


print(testData.info())
print(testData.describe())


# ## Total Entries

# In[ ]:


print('Total entry (test + trian) = {0}'.format(str(len(trainData) + len(testData))))


# ## Missing Value Checking

# ### Train data

# In[ ]:


# exploring isnull method of pandas
print(trainData.isnull().head())


# In[ ]:


sns.heatmap(trainData.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Test data

# In[ ]:


sns.heatmap(testData.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Checking Survival Rate

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=trainData,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=trainData,palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=trainData,palette='rainbow')


# In[ ]:


sns.distplot(trainData['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


sns.countplot(x='SibSp',data=trainData)


# In[ ]:


trainData['Fare'].hist(color='green',bins=40,figsize=(8,4))


# # Data Cleaning

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=trainData,palette='winter')


# In[ ]:


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


def cleanData(data):
    data['Age'] = data[['Age','Pclass']].apply(impute_age,axis=1)
    data.drop('Cabin',axis=1,inplace=True)
    data.fillna(0,inplace=True)
    # data.dropna(inplace=True)


# In[ ]:


cleanData(trainData)


# In[ ]:


sns.heatmap(trainData.isnull(),yticklabels=False,cbar=False,cmap='viridis')
trainData.info()
trainData.describe()
print(trainData.head())


# # Converting Values

# In[ ]:


def convertingValues(data):
    sex = pd.get_dummies(data['Sex'],drop_first=True)
    embark = pd.get_dummies(data['Embarked'],drop_first=True)
    data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    data = pd.concat([data,sex,embark],axis=1)


# In[ ]:


convertingValues(trainData)


# In[ ]:


y = trainData['Survived']
x = trainData.drop('Survived',axis=1)
logmodel = LogisticRegression()
logmodel.fit(x,y)


# # Preparing Test Data

# In[ ]:


cleanData(testData)
convertingValues(testData)


# In[ ]:


sns.heatmap(testData.isnull(),yticklabels=False,cbar=False,cmap='viridis')
testData.info()


# # Predict

# In[ ]:


predictions = logmodel.predict(testData)


# # Evaluation

# In[ ]:


gender_submissionData.info()


# In[ ]:


print(classification_report(gender_submissionData['Survived'],predictions))
print('accuracy - {0}'.format(logmodel.score(testData,gender_submissionData['Survived'] )))


# In[ ]:


confusion_matrix(gender_submissionData['Survived'], predictions)


# In[ ]:


gender_submissionData['Survived'] = pd.DataFrame(predictions)


# In[ ]:


gender_submissionData.to_csv("result.csv", index=False)


# In[ ]:


gender_submissionData.to_string(index=False)


# # References
# 01. [pyhton notebook from Pierian Data](https://github.com/lytemar/Python-for-Data-Science-and-Machine-Learning-Bootcamp---Udemy/blob/master/Machine%20Learning%20Sections/Logistic-Regression/Logistic%20Regression%20with%20Python.ipynb)

# In[ ]:





# In[ ]:





# In[ ]:




