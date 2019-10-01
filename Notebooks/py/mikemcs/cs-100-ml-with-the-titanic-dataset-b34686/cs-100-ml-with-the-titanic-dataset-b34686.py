#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# # Loading Data

# In[3]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# # Exploring the Data

# In[ ]:


training_data.info()


# In[ ]:


test_data.info()


# In[ ]:


training_data[:10]


# In[ ]:


training_data.describe() # automatically summarize numeric columns


# In[ ]:


training_data.describe(include=['O']) # summarize columns containing strings


# ## By-Column Queries

# In[ ]:


training_data['Sex']


# In[ ]:


training_data['Embarked'].value_counts()


# In[ ]:


training_data[training_data['Age'] <= 14]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Parch'] > 3)]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
              & (training_data['Age'] <= 6.5) & (training_data['Embarked'] == 'S')]['Survived'].value_counts()


# In[ ]:


training_data[['Sex', 'Survived']]


# In[ ]:


training_data[['Sex', 'Survived']].groupby(['Sex']).mean()


# In[ ]:


training_data[['Age', 'Survived']].groupby(['Age']).mean()


# # Working with Rows

# In[ ]:


for idx, row in test_data.iterrows():
    print(row['Name'])


# # Making Some Predictions

# In[4]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female' and row['Fare'] > 25 and row['Pclass'] == 3:
        predictions.append(0) # survived
    elif row['Sex'] == 'female' and row['Parch'] > 3:
        predictions.append(0) # survived
    elif row['Sex'] == 'female':
        predictions.append(1) # survived
    elif row['Sex'] == 'male' and row['Pclass'] < 3 and row['Fare'] < 10:
        predictions.append(0) # survived
    elif row['Age'] <= 6.5 and row['Pclass'] <= 2.5:
        predictions.append(1) # survived
    elif row['Age'] <= 6.5 and row['Embarked'] == 'S':
        predictions.append(1) # survived
    elif row['Age'] <= 15 and row['Embarked'] == 'S' and row['Fare'] <= 23:
        predictions.append(1) # survived
    elif row['Age'] <= 15 and row['SibSp'] <= 2.5:
        predictions.append(1) # survived
    else:
        predictions.append(0) # perished


# In[ ]:


len(predictions)


# In[ ]:


test_data['Survived'] = predictions


# In[ ]:


test_data[:10]


# # Generating a Submission

# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)

