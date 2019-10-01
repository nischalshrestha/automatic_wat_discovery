#!/usr/bin/env python
# coding: utf-8

# # Kasper K and Abinahd J

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


training_data['Sex'].value_counts()


# In[ ]:


training_data[training_data['Age'] <= 15]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
              & (training_data['Age'] <=12)]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Age'] > 60)
              & (training_data['Pclass'] == 1)
              #& (training_data['Embarked'] == 'S')
              #& (training_data['Parch'] == 0 )
              & (training_data['Sex'] == 'male')]['Survived'].value_counts()


# In[ ]:


training_data[['Sex', 'Survived']]


# In[ ]:


training_data[['Sex', 'Survived']].groupby(['Sex']).mean()


# In[ ]:


training_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# In[ ]:


training_data[['Embarked', 'Survived']].groupby(['Embarked']).mean()


# # Working with Rows

# In[ ]:


for idx, row in test_data.iterrows():
    print(row['Name'])


# # Making Some Predictions

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female': 
        if row['Pclass'] == 1:
            predictions.append(1)
        elif row['Pclass'] == 2:
            predictions.append(1)
        elif row['Pclass'] == 3:
            if row['Age'] <= 7:
                predictions.append(1)
            elif row['Age'] > 7:
                    if row['Embarked'] == 'C':
                        predictions.append(1)
                    else:
                        predictions.append(0)
            else:
                predictions.append(0)
        else:
            predictions.append(0)
    elif row['Sex'] == 'male':
        if row['Pclass'] == 1:
            if row['Age'] <=17:
                predictions.append(1)
            else:
                predictions.append(0)
        elif row['Pclass'] == 2:
            if row['Age'] <= 15:
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            predictions.append(0)
            
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


# In[ ]:




