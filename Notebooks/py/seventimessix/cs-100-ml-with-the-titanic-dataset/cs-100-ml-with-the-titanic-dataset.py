#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# # Loading Data

# In[2]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# # Exploring the Data

# In[3]:


training_data.info()


# In[4]:


test_data.info()


# In[5]:


training_data[:10]


# In[6]:


training_data.describe() # automatically summarize numeric columns


# In[7]:


training_data.describe(include=['O']) # summarize columns containing strings


# ## By-Column Queries

# In[8]:


training_data['Name']


# In[9]:


training_data['Sex'].value_counts()


# In[10]:


training_data[training_data['Sex'] == 'female']['Survived'].value_counts()


# In[11]:


training_data[(training_data['Sex'] == 'female') 
              & (training_data['Pclass'] == 1)]['Survived'].value_counts()


# In[12]:


training_data[['Survived','Sex' , 'Embarked', 'Fare', 'Pclass', 'Age', 'Parch', 'SibSp']]


# In[13]:


training_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# # Working with Rows

# In[ ]:


ct = 0
ct1 = 0

for idx, row in training_data.iterrows():
    if(row['Survived'] == 1 and row['Sex'] == 'female' and row['Pclass'] < 4 and row['Cabin'] == None and row['Age'] < 40):
        
        ct += 1
       
    elif(row['Survived'] == 0 and row['Sex'] == 'female' and row['Pclass'] < 4 and row['Cabin'] == None and row['Age'] < 40):
        
        ct1+=1
        
print(ct)
print(ct1)


# In[ ]:





# In[ ]:





# # Making Some Predictions

# In[15]:


predictions = []
   
for idx, row in test_data.iterrows():
    if row['Sex'] == 'female':
        if row['Sex'] == 'female' and row['Pclass'] == 2 and row['Parch'] > 1:
            predictions.append(1)
        elif(row['Pclass'] == 3 and row['SibSp'] > 2):
            predictions.append(0)
        elif(row['Pclass'] == 3 and row['Embarked'] == 'S'):
            predictions.append(0)
        elif row['Pclass'] == 3 and row['Parch'] > 1:
            predictions.append(0)
        else:
            predictions.append(1)
    elif row['Sex'] == 'male' and row['Age'] < 16 and row['Pclass'] != 3:
        predictions.append(1)
    elif row['Sex'] == 'male' and row['Parch'] > 0 and row['Age'] < 5:
        predictions.append(1)
    else:
        predictions.append(0) # perished


# In[9]:


len(predictions)


# In[19]:


len(correctArr)


# In[20]:


correctArr = []
correct = 0
for idx, row in training_data.iterrows():
    if(row['Survived'] == 1):
        correctArr.append(1)
    else:
        correctArr.append(0)
for i in range(0, len(predictions)):
    if(predictions[i] == correctArr[i]):
        correct += 1
        
print(100 * correct / len(predictions))        


# In[ ]:


test_data['Survived'] = predictions


# # Generating a Submission

# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




