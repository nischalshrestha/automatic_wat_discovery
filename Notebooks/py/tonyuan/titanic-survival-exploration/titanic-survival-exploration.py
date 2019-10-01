#!/usr/bin/env python
# coding: utf-8

# # Solving titanic survival exploration without using Sklearn

# *This is my first try in kaggle and I don't want to use sklearn or any other packages to help me solve this problem. In this notebook, I just use numpy and pandas to do this work.*

# In[4]:


import numpy as np
import pandas as pd
from IPython.display import display
get_ipython().magic(u'matplotlib inline')

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
display(train.head())


# In[6]:


# Store the 'Survived' feature in a new variable and remove it from the dataset 
train_outcomes = train['Survived']
train_features = train.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(train_features.head())


# In[14]:


def predictions(train_features):
    predictions = []
    for _, passenger in train_features.iterrows():
        if passenger['Sex'] == 'female':   
            if passenger['Pclass'] in [1, 2]:
                predictions.append(1)
            else:
                if passenger['Embarked'] in ['C', 'Q']:
                    predictions.append(1)
                else:
                    predictions.append(0)
        elif passenger['Age'] <= 10:      
            if passenger['SibSp'] >= 3:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"
## Test the 'accuracy_score' function
# predictions = pd.Series(np.ones(5, dtype = int))
# print accuracy_score(outcomes[:5], predictions)

# The result of training set
predictions_train = predictions(train_features)
print(accuracy_score(train_outcomes, predictions_train))
# The result of testing set
predictions_test = predictions(test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_test
    })
submission.to_csv('titanic.csv', index=False)

