#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv('../input/test.csv')


# In[ ]:


# combine the two datasets for now to make filling the age easier
combined_data = pd.concat([titanic_train.drop('Survived',axis=1),titanic_test],axis=0,sort=False).reset_index()
combined_data.tail()


# In[ ]:


# identify the number of null values in each column
null_cols = combined_data.apply(lambda x: sum(x.isnull()))
null_cols


# In[ ]:


null_ages = combined_data[combined_data['Age'].isnull()]
len(null_ages)


# In[ ]:


# try to find out the title to better predict the ages
combined_data['Title'] = combined_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0])


# In[ ]:


null_ages = combined_data[combined_data['Age'].isnull()]
ages_dict = dict(null_ages['Title'].value_counts())
ages_dict


# In[ ]:


# find out the mean age for each group
grouped_df = combined_data.groupby('Title')
# replace the missing age values with mean age for each group
f = lambda x: x.fillna(x.mean())
transformed_ages = grouped_df['Age'].transform(f)
new_combined_data = pd.concat([combined_data,transformed_ages.rename('transformed_ages')],axis=1)
new_combined_data = new_combined_data.reset_index()


# In[ ]:


new_combined_data.head()
new_combined_data[new_combined_data['Name'] == 'Kelly, Mr. James']


# In[ ]:


train_features = new_combined_data.loc[0:890].drop(['Age','Cabin','Ticket','index','PassengerId','Name'],axis=1)
test_features = new_combined_data.loc[891:].drop(['Age','Cabin','Ticket','index','PassengerId','Name'],axis=1)


# In[ ]:


# replace the missing embarked value with S , since it is the highest
train_features['Embarked'] = train_features['Embarked'].fillna('S',axis=0)


# In[ ]:


# lets drop the NA values for now
train_features.apply(lambda x: sum(x.isnull()))


# In[ ]:


x = train_features
x = pd.get_dummies(x,columns=['Sex','Embarked','Pclass','Title'],drop_first=True)
y = titanic_train['Survived']


# In[ ]:


lg = LogisticRegression()
model = lg.fit(x,y)


# In[ ]:


model.score(x,y)


# In[ ]:


null_cols = test_features.apply(lambda x: sum(x.isnull()))
null_cols


# In[ ]:


# replace Nas with the mean values
test_features['Fare'] = test_features['Fare'].fillna(np.mean(test_features['Fare']),axis=0)


# In[ ]:


x_test = pd.get_dummies(test_features,columns=['Sex','Embarked','Pclass','Title'],drop_first=True)


# In[ ]:


missing_columns = set(x.columns) - set(x_test.columns)
# Add a missing column in test set with default value equal to 0
for c in missing_columns:
    x_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
x_test = x_test[x.columns]


# In[ ]:


x_test.head()


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


pass_ids = pd.Series(np.arange(892,1310))
predicts = pd.Series(predictions)


# In[ ]:


predicted_data = pd.concat([pass_ids,predicts],axis=1)
predicted_data.columns = ['PassengerId','Survived']


# In[ ]:


predicted_data.to_csv('titanic_predictions.csv',index=False)


# In[ ]:





# In[ ]:




