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


training_data[training_data['Sex'] == 'female']['Survived'].value_counts()


# In[ ]:


training_data[training_data['Sex'] == 'male']['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 1)]['Survived'].value_counts()


# In[ ]:


training_data[[ 'Survived','Sex']]


# In[ ]:


training_data[['Pclass','Sex','Age', 'Fare','Survived']].groupby(['Pclass','Sex','Age']).mean()


# A# Working with Rows

# In[ ]:


for idx, row in test_data.iterrows():
    print(row['Name'])


# # Making Some Predictions

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female':
            if row['Age'] <= 70 and row['Age'] > 16:
                if row['Pclass'] == '1' or '2':
                    predictions.append(1) # survived
                elif row['Pclass'] == '3':
                    predictions.append(0)
                    
            elif row['Age'] <= 16:
                predictions.append(1)
            else:
                predictions.append(0)
   # elif row['Sex'] == 'male':
    #    if row['Fare'] > 100:
       #     predictions.append(1)
      #  elif row['Age'] < 15:
     #       predictions.append(1)
      #  else:
       #     predictions.append(0)
    else:
        predictions.append(0) # perished


# In[ ]:





# In[ ]:


len(predictions)


# In[ ]:


test_data['Survived'] = predictions


# # Generating a Submission

# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




