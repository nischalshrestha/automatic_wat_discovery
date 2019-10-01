#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# Load data

training_set=pd.read_csv('../input/train.csv')
test_set=pd.read_csv('../input/test.csv')


# In[ ]:


# Data preprocessing step

print(training_set.info())


# Based on the output above, I shall use a DecisionTreeRegressor to try out using columns with no missing data.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

y=training_set.Survived
X_features=list(training_set.columns)
print(X_features)

#the column Sex will be converted into numerical binary

# First check on number of unique values (male, female) to ensure there's no typo etc
print('There are %s unique values' %training_set.Sex.nunique())

# Creating a new column to map the data
training_set['Gender'] = training_set['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#remove target variable, Survived
X_features.remove('Survived')

#remove features with missing values
X_features.remove('Cabin')
X_features.remove('Embarked')
X_features.remove('Age')

#for this very basic DecisionTreeRegressor, we shall remove non numerical columns
X_features.remove('Name')
X_features.remove('Ticket')
X_features.remove('Sex')
print(X_features)
X=training_set[X_features]
print(X.info())


# In[ ]:





# In[ ]:


# Running the first basic model

first_model=DecisionTreeRegressor(random_state=42)
_=first_model.fit(X, y)

# We shall do a simple test on the first five results of the training set.
# As it is data that the model has seen, the predicted results are identical to the real results.
print(first_model.predict(X.head()))
print(y.head())


# In[ ]:


#Now, to test this basic model on new data. First, check out the test set
print(test_set.head())

# Creating a new column to map the Sex column
# doing the same for the test set which was earlier done for the training set
test_set['Gender'] = test_set['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Extract out the relevant features that were used for the basic model
X_test=test_set[X_features]
print(X_test.info())


# In[ ]:


# Notice that there is a missing data in the Fare column. We will have to fill in the null data first.

X_test=X_test.fillna(method='ffill')
print(X_test.info())


# In[ ]:


y_pred=first_model.predict(X_test)
y_final=pd.DataFrame(y_pred)
submission=pd.concat([X_test['PassengerId'],y_final],axis=1)
submission.columns = ['PassengerId', 'Survived']
print(submission.shape)
# Read to a CSV file
submission.to_csv('submission.csv',index=False)
print('done')

