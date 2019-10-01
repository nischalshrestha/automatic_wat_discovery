#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Regressor modeling¶
# 
# Based on Kaggle Tutorial: https://www.kaggle.com/dansbecker/your-first-machine-learning-model

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


#check the first rows of train data set (to check if the import was Ok)
train.tail()


# In[ ]:


#check the first rows of test data set (to check if the import was Ok)
test.tail()


# Reclassify all the literal data (except Name; NaN - is not literal, it is Null-value) to numeric to use it in regression models:

# In[ ]:


# Train data set
train.loc[train['Sex'] == 'male', 'Sex'] = 1
train.loc[train['Sex'] == 'female', 'Sex'] = 0
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 0
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 0

# Test data set
test.loc[test['Sex'] == 'male', 'Sex'] = 1
test.loc[test['Sex'] == 'female', 'Sex'] = 0
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 0
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 0


# Fill the missing values with our assumptions

# In[ ]:


# Train data set
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(1)
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

# Test data set
test['Age'] = test['Age'].fillna(train['Age'].median())
test['Embarked'] = test['Embarked'].fillna(1)
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# ***

# # Modeling
# 
# ### Decision Tree Regressor modeling
# 
# Based on Kaggle
# https://www.kaggle.com/dansbecker/your-first-machine-learning-model

# In[ ]:


predictors = ["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# ***

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
dtree = DecisionTreeRegressor(random_state=1)

# Fit model
dtree.fit(train[predictors], train['Survived'])
predictions = dtree.predict(test[predictors])

# convert results to binary
n = 0
for i in predictions:
    if(i > 0.5):
        predictions[n] = 1
    else:
        predictions[n] = 0
    n += 1

# change type of the predictions array to integer
predictions = predictions.astype(int)
predictions.dtype


# In[ ]:


#Confusion matrix
#from sklearn.metrics import confusion_matrix
#confusion_matrix(predictions, train['Survived'])


# In[ ]:


# Accuracy
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(train['Survived'], predictions)
#accuracy


# ***

# # Submission

# We need to put our csv-file with the results into Kaggle /input directory like so:
# 
# (LB score: 0.69377)

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("dtree_submit2.csv", index=False)


# ## Kaggle API to submit
# 
# API: https://github.com/Kaggle/kaggle-api
# 
# Check submissions:
# 
# `kaggle competitions submissions`
# 
# Submit to competition:
# 
# `kaggle competitions submit titanic -f Git\kaggle-titanic\src\dtree_submit2.csv -m "decision tree"`
