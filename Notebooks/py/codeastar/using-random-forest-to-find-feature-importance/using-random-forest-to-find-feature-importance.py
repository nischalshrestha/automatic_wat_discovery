#!/usr/bin/env python
# coding: utf-8

# It is a short demostration for using Random Forest to get Feature Importance from Titanic Survivors data set. This technique is handy for feature selection process.
#     
# If you want to look for the complete Titantic Survivors code reference instead, feel free to visit http://www.codeastar.com/data-wrangling then.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#get the training data set
df_train = pd.read_csv("../input/train.csv")


# Just take a look at what the training data looks like:

# In[ ]:


df_train.head(5)


# Well, categorical variables are not good for machine, don't worry, I have a plan!

# In[ ]:


df_train_all_num = (df_train.apply(lambda x: pd.factorize(x)[0]))


# Now all the fields are factorized as numeric values

# In[ ]:


df_train_all_num.head(5)


# In[ ]:


#load our model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[ ]:


#prepare training X, Y data set
train_y = df_train_all_num['Survived']
#drop unused fields
train_x = df_train_all_num.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)      


# In[ ]:


#thit is how we get the feature importance with simple steps:
model.fit(train_x, train_y)
# display the relative importance of each attribute
importances = model.feature_importances_
#Sort it
print ("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(train_x)), reverse=True)
print (sorted_feature_importance)


# Then the  model tells us "**Sex**", "**Age**" and the "**Fare**" are the three most importance features within the Titanic Survivors data set.  
