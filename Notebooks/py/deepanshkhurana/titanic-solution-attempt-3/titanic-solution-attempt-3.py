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


np.random.seed(0)
#Creating variables for train_data and test_data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

#Describing train_data to see if there are any missing values.
train_data.describe()


# # Cleaning The Data

# In[ ]:


#As we can see there are some missing values in 'Age', let's inspect the data
train_data['Age'].head(20)

#As suspected 'Age' has NaN values. To fix them we can substitute the ages with the mean age which is 29.69, roughly 30.
train_data['Age'] = train_data['Age'].fillna(30.0)
test_data['Age'] = test_data['Age'].fillna(30.0)

#There are also NaN values in test_data's 'Fare' column. We'll just replace it with the mean fare as well.
test_data['Fare'] = test_data['Fare'].fillna(30.0)

#For our model to work, we will also need to change the age to numeric values.
#We'll use a simple convention where female=1, male=0
train_data['Sex'] = train_data['Sex'].replace('male',0)
train_data['Sex'] = train_data['Sex'].replace('female',1)
test_data['Sex'] = test_data['Sex'].replace('male',0)
test_data['Sex'] = test_data['Sex'].replace('female',1)
        
#Let's describe the data now
test_data.describe()
#train_data.describe()


# In[ ]:


#Looks like we've fixed the 'Age' column and we're back to 891 values. The mean is also not as changed as it could have been.

#Let's print all columns to find the prediction target
train_data.columns


# In[ ]:


#It's clear we're looking at the 'Survived' column when we're looking to make predictions.

y = train_data.Survived


# # Creating X
# 
# Let's start defining our X now. We can see that most columns would affect the survival rate of the passengers.
# 
# Let's say the following columns affect the survival rate. Namely,
# 
# * Pclass
# * Sex
# * Age
# * SibSp
# * Parch
# * Fare
# 
# The reason I've chosen to ignore ticket, cabin, and embarked is because none of them are of obvious significance. The fare is included because more affluent people could have access to better cabins and facilities. In general, affluent people would have larger chances because of their position in the hierarchy. This isn't true but an assumption on my part.
# 

# In[ ]:


#Let's define X

features = ['Pclass','Sex','Age','SibSp','Parch','Fare']

X = train_data[features]


# # Reviewing The Data

# In[ ]:


#Let's review X before we proceed

print(X.describe())
print("\n")
print(X.head())


# # Splitting The Data

# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size = 0.15)


# # Get The Maximum Number of Leaf Nodes

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

leaf_nodes = [5, 25, 50, 100, 250, 500, 1000, 2000]

for node in leaf_nodes:
    print(node, get_mae(node,train_X,val_X,train_y,val_y))


# # Building The Model Using Random Forests

# In[ ]:


titanic_model = RandomForestRegressor(random_state=1, max_leaf_nodes = 50)

titanic_model.fit(train_X,train_y)
prediction = titanic_model.predict(val_X)
val_mae = mean_absolute_error(val_y,prediction)

print(val_mae)


# # Making The Predictions

# In[ ]:


final_X = test_data[features]
final_predictions = titanic_model.predict(final_X)
final_predictions = np.round(final_predictions)
final_predictions = final_predictions.astype(int)
print(final_predictions[1:10])


# # Creating the submission

# In[ ]:


#creating the object
submission = pd.DataFrame({
    'PassengerId':test_data['PassengerId'],
    'Survived':final_predictions
})
print(submission.head())
submission.to_csv('titanic_submission.csv', index = False)

