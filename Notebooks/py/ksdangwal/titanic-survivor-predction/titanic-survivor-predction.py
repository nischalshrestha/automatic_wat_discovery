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


# In[ ]:


# read form data file

trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
genderData = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


testData.head()


# In[ ]:


# top 5 rows of train data
trainData.head()


# In[ ]:


trainData.shape


# In[ ]:


# top 5 rows of genderData data
genderData.head()


# In[ ]:


# some groups of people were more likely to survive than others, such as women, children, and the upper-class.

# Survival Based on Gender
testData['isSurvived'] = np.where(testData['Sex'] == 'female', '1', '0')

# Survival Based on Class
#testData.loc[testData.Pclass == 1, 'isSurvived'] = '1' 

# top 5 rows of testData with new column isSurvived
testData.head()


# In[ ]:


# bottom 5 rows of trainData with new column isSurvived
testData.tail()        


# In[ ]:


# get total number of survived male and female in our testData, so far...
testData.groupby(['Sex']).count()['isSurvived'].reset_index()


# In[ ]:


trainEmbarkedPercentage = trainData.groupby(['Embarked']).count()['Survived'].reset_index()

# get total survived percentage based on embarked
totalTrainData = trainData['PassengerId'].count()
trainEmbarkedPercentage['embarkedPercentage'] = (trainEmbarkedPercentage['Survived'] / totalTrainData)*100
trainEmbarkedPercentage


# In[ ]:


testEmbarkedPercentage = testData.groupby(['Embarked']).count()['isSurvived'].reset_index()

# get total survived percentage based on embarked parameter
totalTestData = testData['PassengerId'].count()
testEmbarkedPercentage['embarkedPercentage'] = (testEmbarkedPercentage['isSurvived'] / totalTestData)*100
testEmbarkedPercentage


# In[ ]:


genderData.head()


# In[ ]:


genderData['Survived'] = testData['isSurvived']


# In[ ]:


genderData.to_csv('passenger_survived_02.csv', index=False)

