#!/usr/bin/env python
# coding: utf-8

# **Project: Titanic Survival Exploration**
# -----------------------------------------
# 
# In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths of most of its passengers and crew. In this introductory project, we will explore a subset of the RMS Titanic passenger manifest to determine which features best predict whether someone survived or did not survive. To complete this project, you will need to implement several conditional predictions and answer the questions below. Your project submission will be evaluated based on the completion of the code and your responses to the questions.

# In[ ]:


import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

test_data = pd.read_csv('../input/test.csv')
train_data = pd.read_csv('../input/train.csv')
combined_data = pd.concat([train_data.drop('Survived', axis=1), test_data])


# In[ ]:


outcomes = train_data['Survived']
train = train_data.drop('Survived', axis=1)
train.head()


# In[ ]:


def accuracy_score(truth, pred):
    if len(truth) == len(pred):
        return 'Predictions have an accuracy of {:.2f}%'.format((truth == pred).mean()*100)
    else:
        return 'Number of predictions does not match the number of outcomes'


# In[ ]:


def predictions_0(data):
    
    predictions = []
    
    for _, passenger in data.iterrows():
        predictions.append(0)
    return pd.Series(predictions)

predictions = predictions_0(train)
accuracy_score(outcomes, predictions)


# In[ ]:


def predictions_1(data):
    
    predictions = []
    
    for _, passenger in data.iterrows():
        
        predictions.append( 0 if passenger['Sex'] == 'male' else 1)
        
    return pd.Series(predictions)

predictions = predictions_1(train)
accuracy_score(outcomes, predictions)


# In[ ]:


def accuracy_score(truth, pred):
    if len(truth) == len(pred):
        return 'Predictions have an accuracy of {:.2f}%'.format((truth == pred).mean()*100)
    else:
        return 'Number of predictions does not match the number of outcomes'


# In[ ]:


def predictions_0(data):
    
    predictions = []
    
    for _, passenger in data.iterrows():
        predictions.append(0)
    return pd.Series(predictions)

predictions = predictions_0(train)
accuracy_score(outcomes, predictions)


# In[ ]:


def predictions_1(data):
    
    predictions = []
    
    for _, passenger in data.iterrows():
        
        predictions.append(0 if passenger['Sex'] == 'male' else 1)
        
    return pd.Series(predictions)

predictions = predictions_1(train)
accuracy_score(outcomes, predictions)


# In[ ]:


def predictions_2(data):
    
    predictions = []
    
    for _, passenger in data.iterrows():
        
        predictions.append(1 if passenger['Age'] < 10 or passenger['Sex'] == 'female' else 0)
        
    return pd.Series(predictions)

predictions = predictions_2(train)
accuracy_score(outcomes, predictions)


# In[ ]:


def predictions_3(data):
    
    predictions = []
    
    for _, passenger in data.iterrows():
        
        if (passenger['Sex'] == 'male'):
            if (passenger['Pclass'] == 2):
                if (passenger['Age'] > 15):
                    predictions.append(0)
                else:
                    predictions.append(1)
            else:
                predictions.append(0)
        else:
            if passenger['SibSp'] > 2:
                predictions.append(0)
            else:
                predictions.append(1)
        
    return pd.Series(predictions)

predictions = predictions_3(train)
accuracy_score(outcomes, predictions)


# In[ ]:


predictions = predictions_3(test_data)
predictions.head()


# In[ ]:


submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)

