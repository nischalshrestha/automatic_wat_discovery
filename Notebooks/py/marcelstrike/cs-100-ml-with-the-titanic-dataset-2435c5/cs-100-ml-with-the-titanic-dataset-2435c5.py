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


training_data[(training_data['Sex'] == 'male') 
              & (training_data['Pclass'] == 1)]['Survived'].value_counts()


# In[ ]:


training_data[['Sex', 'Survived']]


# In[ ]:


training_data[['Sex', 'Survived']].groupby(['Sex']).mean()


# In[ ]:


training_data[['Pclass', 'Survived', 'Sex']].groupby('Pclass').mean()


# In[ ]:


training_data[['Age', 'Survived']].groupby('Age').mean()


# ## Visualization

# In[ ]:


training_data.hist(figsize=(10,10))
plt.show()


# In[ ]:


training_data.plot(kind='density', subplots=True, sharex=False, figsize=(10,10))
plt.show()


# In[ ]:


training_data.corr()


# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(training_data, figsize=(10,10))
plt.show()


# # Working with Rows

# In[ ]:


for idx, row in test_data.iterrows():
    print(row['Name'], row['Pclass'])


# # Making Some Predictions

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female':
        predictions.append(1) # survived
    elif row['Sex'] == 'male' and row['PassengerId'] == 5 or 6 or 7 or 8 or 13 or 14 or 17:
        predictions.append(0)
    elif row['Age'] == 3 or 4 or 5 or 2 or 1 or 0.42:
        predictions.append(1)
    elif row['Age'] >= 46 and row['Survived'] == 0:
         predictions.append(0) # perished
    elif row['Sex'] == 'male' and row['PassengerId'] == 
        predictions.append(0)
    elif row['Sex'] == 'female' and row['PassengerId'] == 2 or 3 or 9 or 10 or 11 or 12 or 16 or  or 20 :
        predictions.append(1)
    elif row
    else:
        predictions.append(0)


# In[ ]:


len(predictions)


# In[ ]:


test_data['Survived'] = predictions


# In[ ]:


test_data[:10]


# In[ ]:


test_data


# # Generating a Submission

# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




