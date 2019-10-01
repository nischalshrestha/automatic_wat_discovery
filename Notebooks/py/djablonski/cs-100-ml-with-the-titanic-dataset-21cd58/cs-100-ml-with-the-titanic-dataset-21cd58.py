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


training_data[(training_data['Sex'] == 'male') & (training_data['Embarked'] == 'Q')]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'male') & (training_data['Pclass'] == 1) & (training_data['Age'] <40)]['Survived'].value_counts()


# 5# training_data[(training_data['Sex'] == 'female') 
#               & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)].describe()

# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)]


# In[ ]:


from pandas.plotting import scatter_matrix
training_data[(training_data['Sex'] == 'male')].corr()


# In[ ]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 3) & (training_data['Survived'] == 0)]


# In[ ]:


training_data[['Sex', 'Survived']]


# In[ ]:


training_data[['Pclass', 'Survived', 'Fare', 'Sex']].groupby(['Pclass']).mean()


# In[ ]:


training_data[['Pclass', 'Survived', 'Sex']].groupby('Pclass').mean()


# In[ ]:


training_data[['Sex','Age', 'Survived']].groupby('Age').mean()


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
        if row['Pclass'] < 3:
            predictions.append(1) # survived
        elif row['Fare'] < 25:
            predictions.append(1)
        else:
            predictions.append(0)
    else:
        if row['Age'] < 10:
            predictions.append(1)
        else:
            predictions.append(0) # perished


# In[ ]:


num_correct = 0
for idx, row in training_data.iterrows():
    if row['Sex'] == 'female':
        if row['Age'] < 15:
            survived = 1
        if row['Pclass'] < 3:
            survived = 1 # survived
        elif row['Fare'] < 25:
            survived = 1
        else:
            survived = 0
    else:
        if row['Age'] < 10:
            survived = 1
        elif row['Age'] == 1 and row['Age'] < 40 :
            survived = 1 # perished
        else:
            survived = 0
    if row['Survived'] == survived:
        num_correct += 1
print('Accuracy =', num_correct / len(training_data))


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




