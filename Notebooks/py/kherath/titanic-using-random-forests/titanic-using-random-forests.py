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


train_file_path = '../input/train.csv'


data = pd.read_csv(train_file_path)


# In[ ]:


data


# ### Deal with Null Values 

# #### 1. Age column Null Values 

# In[ ]:


data.Age.fillna(data.Age.mean(),inplace=True)


# #### 2. Cabin Column
# 
# I'm going to drop this coulmn

# In[ ]:


data.drop(['Cabin'],axis=1,inplace=True)


# #### 3. Embarked Column
# I'm going to drop this too

# In[ ]:


data.drop(['Embarked'],axis=1,inplace=True)


# In[ ]:


data.head(20)


# In[ ]:





# In[ ]:





# ## Now building the model 

# In[ ]:


#1. define the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

titanic_model = RandomForestRegressor(n_estimators = 100, oob_score = True,random_state = 42)

#2. Instantiate the model 

predictors = ['Age', 'Pclass','Fare','SibSp']

X = data[predictors]

y = data['Survived']

#3. Fit the model

titanic_model.fit(X,y)


# In[ ]:





# ### Let's look at the validity of the model using the oob_score and C-Stat

# In[ ]:


titanic_model.oob_score_


# In[ ]:


## C-stat
y_oob = titanic_model.oob_prediction_
print('C-stat: ',roc_auc_score(y,y_oob))


# In[ ]:





# In[ ]:





# ### TIme to make predictions

# #### Loading the test data and clean up the data

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.isnull().sum()


# In[ ]:





# In[ ]:


test.Age.fillna(test.Age.mean(),inplace=True)


# In[ ]:


test.Fare.fillna(test.Fare.mean(),inplace=True)


# In[ ]:





# ### NOw change all the needed data types to float

# In[ ]:


test.dtypes


# In[ ]:





# In[ ]:


test['Pclass'] = test.Pclass.astype(float)


# In[ ]:


test['SibSp'] = test.SibSp.astype(float)


# In[ ]:





# In[ ]:





# ### Load the test model 

# In[ ]:


predictors = ['Age', 'Pclass','Fare','SibSp']


test_X = test[predictors]


# In[ ]:





# ## Finally make predictions. 

# In[ ]:


survival = titanic_model.predict(test_X)

print(survival)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':survival})

#Visualize the first 5 rows
submission.head()


# In[ ]:





# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




