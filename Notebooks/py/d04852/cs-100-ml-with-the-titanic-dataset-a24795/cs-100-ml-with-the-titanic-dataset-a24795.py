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


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3)]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)].describe()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)]


# In[ ]:


from pandas.plotting import scatter_matrix
training_data[(training_data['Sex'] == 'female')  
              & (training_data['Pclass'] == 3)].corr()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)]


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
        if row['Sex'] == 'female' and row['SibSp'] < 2:
            predictions.append(1)
        else:
            if row['Age'] < 10:
                predictions.append(1)
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




