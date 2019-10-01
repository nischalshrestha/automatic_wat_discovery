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


training_data[training_data['Sex'] == 'male']['Survived'].value_counts()


# # 

# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Age'] <= 30) & (training_data['Age'] >= 13) & (training_data['SibSp'] <= 0)]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'male')
              & (training_data['Age'] <= 36) & (training_data['Age'] >= 25) & (training_data['SibSp'] <= 0) & (training_data['Pclass'] == 1)]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)]


# In[ ]:


from pandas.plotting import scatter_matrix
training_data[(training_data['Sex'] == 'female')  
              & (training_data['Pclass'] == 2)].corr()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
              & (training_data['Pclass'] == 1) & (training_data['Survived'] == 0)]


# In[ ]:


training_data[['Sex', 'Survived']]


# In[ ]:


training_data[['Sex', 'Survived']].groupby(['Sex']).mean()


# In[ ]:


training_data[['Pclass', 'Survived', 'Sex']].groupby('Survived').mean()


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


import random


# In[ ]:


random.randint(0,100)


# In[ ]:


47/347


# 

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female':
        if row['Pclass'] == 1:
            predictions.append(1)
        elif row['Fare'] >= 300:
            predictions.append(1)
        elif row['Pclass'] == 2:
            predictions.append(1)
        elif row['Pclass'] == 3:
            if row['SibSp'] >= 1 and row['Parch'] >= 2:
                predictions.append(0)
            else:
                predictions.append(1)
        elif row['Age'] <= 1:
            predictions.append(1)
        else:
            predictions.append(0)
    elif row['Sex'] == 'male':
        if row['Fare'] >= 300:
            predictions.append(1)
        elif row['Age'] <= 16 and row['SibSp'] < 2:
            predictions.append(1)
        elif row['Age'] < 17 and row['Pclass'] <= 2:
            predictions.append(1)
        elif row['Age'] <= 1:
            predictions.append(1)
        else:
            if random.randint(1,100) == 1:
                predictions.append(1)
            else:
                predictions.append(0)
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

