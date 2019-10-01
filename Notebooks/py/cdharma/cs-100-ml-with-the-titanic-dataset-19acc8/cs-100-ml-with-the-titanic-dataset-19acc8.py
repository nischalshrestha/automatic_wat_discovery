#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# # Loading Data

# In[ ]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


test_data


# In[ ]:


training_data['Sex'].value_counts()


# In[ ]:


training_data['Pclass'].value_counts()


# Male, over 8 under 18, 

# In[ ]:


training_data[(training_data['Age'] <= 18) & (training_data['Age'] >= 8) & (training_data['Sex'] == 'male')].sort_values(by=['Fare']).drop(columns=['Embarked', 'PassengerId', 'Name'])


# In[ ]:


training_data[(training_data['Sex'] == 'male')]


#  # Data Analysis

# In[ ]:


training_data[(training_data['Parch'] >= 1) & (training_data['Age'] <= 18) & (training_data['Sex'] == 'female')]


# In[ ]:


training_data[(training_data['Sex'] == 'female')
             & (training_data['Parch'] >= 1)
             & (training_data['Age'] <= 70)
             & (training_data['Age'] >= 50)].sort_values(by='Survived').drop(columns=['Ticket', 'Fare', 'Cabin'])


# In[ ]:


training_data[training_data['Pclass'] == 3].sort_values(by='Age')


# # Predictions

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female':
        if row['Pclass'] <= 2:
            predictions.append(1)
        elif row['Age'] <= 5:
            predictions.append(1)
        elif row['SibSp'] <= 2:
            predictions.append(1)
        elif row['Age'] <= 40 and row['Age'] >= 30:
            predictions.append(1)
        elif row['Age'] >= 50 and row['Age'] <= 70:
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        if row['Pclass'] == 1:
            if row['Age'] <= 15:
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            predictions.append(0)


# In[ ]:


print(len(predictions))


# In[ ]:


num_correct = 0
for idx, row in training_data.iterrows():
    if row['Sex'] == 'female':
        if row['Pclass'] <= 2:
            survived = 1
        elif row['Age'] <= 5:
            survived = 1
        elif row['SibSp'] <= 2:
            survived = 1
        elif row['Age'] <= 40 and row['Age'] >= 30:
            survived = 1
        elif row['Age'] >= 50 and row['Age'] <= 70:
            survived = 1
        else:
            survived = 0
    else:
        if row['Sex'] == 'male':
            if row['Pclass'] == 1:
                if row['Age'] <= 17:
                    survived = 1
            else:
                survived = 0
    if row['Survived'] == survived:
        num_correct += 1
print('Accuracy =', num_correct / len(training_data))


# In[ ]:


len(predictions)


# In[ ]:


test_data['Survived'] = predictions


# # Generating a Submission

# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# In[ ]:




