#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100


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


training_data[:20]


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
              & ((training_data['Parch'] <= 1) | (training_data['SibSp'] <= 1))
              & (training_data['Pclass'] == 3 )  
              & ((training_data['Fare'] <= 20 ) | (training_data['Fare'] >= 100 ))
              & (training_data['Age'] > 16)
              ]['Survived'].value_counts()


# In[ ]:





# In[ ]:


training_data[['Sex', 'Survived']]


# In[ ]:


training_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# # Working with Rows

# In[ ]:


for idx, row in test_data.iterrows():
    print(row['Name'])


# # Making Some Predictions

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():
    if row['Sex'] == 'male' and (row['Pclass'] == 1 or row['Pclass'] == 2) and row['Age'] < 16:
        predictions.append(1)
    elif row['Sex'] == 'male':
        predictions.append(0)
    elif row['Sex'] == 'female' and row['Pclass'] == 3 and row['Parch'] > 1 and row['SibSp'] > 1:
        predictions.append(0)
    elif row['Sex'] == 'female' and row['Pclass'] == 3 and (row['Parch'] <= 1 or row['SibSp'] <= 1) and row['Fare'] > 20 and row['Fare'] < 100:
        predictions.append(0)
    elif row['Sex'] == 'female':
        predictions.append(1)
    


# In[ ]:


len(predictions)


# In[ ]:


test_data['Survived'] = predictions


# In[ ]:


test_data[:50]


# # Generating a Submission

# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




