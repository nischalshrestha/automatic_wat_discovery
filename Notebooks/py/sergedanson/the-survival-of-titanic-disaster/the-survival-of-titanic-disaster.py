#!/usr/bin/env python
# coding: utf-8

# 
# 

# <p style="text-align: center; font-size:18pt; color:#2980b9; font-weight:500;">Predicting the Survival of Titanic disaster with Python</p>
# 
# <p style="text-align: center; font-size:12pt; color:#2980b9; font-weight:500;">Let's roll the Leaderboard together...</p>
# 
# 
# ![](http://www.globalnerdy.com/wordpress/wp-content/uploads/2015/08/if-the-titanic-sank-today1.jpg)

# In[2]:


import pandas as pd
import numpy as np


# In[6]:


#Let's get dataset that we are going to train our model
train=pd.read_csv("../input/train.csv")
#view the 5 top most records in our dataset
train.head()


# In[12]:


#create a list that holds the columns that we are going to use for now

feature_cols=['Pclass','Parch','Sex','Age']

# Select the columns from our dataset using pandas to create the "PREDICTORS"
x=train.loc[:,feature_cols]

x['Sex'].replace(['female','male'],[0,1],inplace=True)


#total number of NaN of Ages
x['Age'].isnull().sum()

#Deal with NaN
#x.dropna()   => can delete data with NaN which is not a good practice => it reduces quality of our model by reducing our sample
#filling NaN with mean
mean_value=x['Age'].mean()
x['Age']=x['Age'].fillna(mean_value)

x['Age'].isnull().sum()



# In[13]:


#create the prediction target variable "y"
y=train.Survived


# In[14]:


#Classification model
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(x,y)


# In[16]:


#get the test dataset
test=pd.read_csv("../input/test.csv")
#test dataset doesn't have the Survive column since it's exactly what we are testing
test.head()


# In[17]:


x_test=test.loc[:,feature_cols]

x_test['Sex'].replace(['female','male'],[0,1],inplace=True)

#total number of NaN of Ages
x_test['Age'].isnull().sum()

#Deal with NaN
#x.dropna()   => can delete data with NaN which is not a good practice => it reduces quality of our model by reducing our sample
#filling NaN with mean
mean_value=x_test['Age'].mean()
x_test['Age']=x['Age'].fillna(mean_value)

x_test['Age'].isnull().sum()


# In[20]:


test_pred_class= log_reg.predict(x_test)
#test_pred_class
#create a submission file 

pd.DataFrame({"PassengerId":test.PassengerId,"Survived":test_pred_class}).set_index("PassengerId").to_csv("sub.csv")

