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


training_data['Age'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'male')
              & (training_data['Age']<= 15) & (training_data['SibSp'] <= 2)
             ]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Pclass'] <= 2) &
             (training_data['Age'] <= 16) 
             ]['Survived'].value_counts()


# In[ ]:


training_data[
             (training_data['Fare'] >= 275)
             ]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Fare'] >= 300)
             ]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Pclass'] == 2) & (training_data['Sex'] == 'female')
              & (training_data['SibSp'] + training_data['Parch'] < 2)
             ]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Pclass'] == 2) & (training_data['Sex'] == 'female')
             ]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['SibSp'] >= 3) 
           ]['Survived'].value_counts()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
                & (training_data['Survived'] == 1)].describe()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
                & (training_data['Survived'] == 0)].describe()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
                & (training_data['Survived'] == 1)
              & (training_data['Pclass']==3) 
                ].describe()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 
              & (training_data['Survived'] == 1)
              & (training_data['Pclass']==3) 
                ].describe()


# In[ ]:


training_data[(training_data['Sex'] == 'male') 

              & (training_data['Age']>=42) 
                ].describe()


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


training_data[['Embarked', 'Survived']].groupby('Embarked').mean()


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
    if row['Age'] <= 16 and row['Pclass']<=2:
        predictions.append(1)
    elif row['Age'] <=1:
        predictions.append(1)
    elif row['Fare'] >=300:
        predictions.append(1)
    elif row['Sex'] == 'female' and row['Pclass']<= 2:
        predictions.append(1)
    elif row['Sex'] == 'female' and (row['SibSp'] + row['Parch'] > 3) and row['Pclass']== 3:
        predictions.append(0)
    elif row['Sex'] == 'female' and row['Pclass']== 3:
        predictions.append(1)
    elif row['Sex'] == 'male' and row['Age']<= 15 and row['SibSp'] <= 2:
        predictions.append(1)
    else:
        predictions.append(0) # perished


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




