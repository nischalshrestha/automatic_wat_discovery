#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This is my first kernel!**
# **As a memorial to starting my kaggler's life, I want to leave a first step. **
# 
# To make simple, I only do minimum processes to submit to Titanic competition. 
# 
# The most important thing is to submit my predictions without giving up, so I'm not aim to get highly score. 
# 
# Let' get start!

# # Import libraries
# I use scikit-learn's LogisticRegression, because I think this is the simplest classification algorithm. 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression


# # Import datasets
# Use pandas. It's my first pandas.

# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# # Check datasets
# Pandas DataFrame can use head method to show first 5 rows.
# 
# I use this method with print method. Mmm... It's not beautiful.

# In[3]:


print(train.head())


# Without print. OK, Beautiful.

# In[4]:


train.head()


# And test data shows. Beautiful.

# In[5]:


test.head()


# # Preprocessing data - drop columns
# Both train and test data have unnecesary columns to ｍachine learning. So let's drop these columns.

# In[6]:


drop_train = train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
drop_test = test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)


# Check drop train data. Ok.

# In[7]:


drop_train.head()


# And test data is Ok, too.

# In[8]:


drop_test.head()


# # Preprocessing data - make X_train, Y_train
# To use LogisticRegression, I need X_train data which is training vector, features and Y_train which is target vector. 

# In[9]:


X_train = drop_train.drop(['Survived'], axis=1)


# In[10]:


X_train.head()


# In[11]:


Y_train = drop_train.drop(['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'],axis=1)


# In[12]:


Y_train.head()


# # Preprocessing data - OneHot encoding
# I tred fit above data, but I couldn't. It seems that I need to change categorical features to OneHot.

# In[13]:


oneHot_X_train = pd.get_dummies(X_train)


# In[14]:


oneHot_X_train.head()


# Not bad. Although Pclass looks like numrical feature but  it is categorical feature. So I use columuns' parameters.
# 
# OK, beautilful.

# In[15]:


oneHot_X_train = pd.get_dummies(X_train, columns=['Pclass','Sex','Embarked'])


# In[16]:


oneHot_X_train.head()


# And test data OneHot, too.

# In[17]:


oneHot_test = pd.get_dummies(drop_test, columns=['Pclass','Sex','Embarked'])


# In[18]:


oneHot_test.head()


# # Preprocessing data - Impute
# I tred fit above data again, but I couldn't too. It seems that I need to complete missing values.
# 
# Both oneHot_X_train and oneHot_test contain missing values.

# In[19]:


print(oneHot_X_train.isnull().sum())
print(oneHot_test.isnull().sum())


# To conplete missing values, I use scikit-learn's Imputer.
# Import library and fit it.

# In[20]:


from sklearn.preprocessing import Imputer
my_imputer = Imputer()


# In[21]:


imputer_X_train = my_imputer.fit_transform(oneHot_X_train)
imputer_test = my_imputer.fit_transform(oneHot_test)                              


# It seems that when I fit Imputer data type have changed from pd.dataframe to np.ndarray. 
# 
# So I can't use pd's head method, instead I use np's code.

# In[22]:


print(type(imputer_X_train))
print(imputer_X_train.shape)
print(imputer_X_train[0:5,])
print(type(imputer_test))
print(imputer_test.shape)
print(imputer_test[0:5,])


# # Fitting
# Preprocessing has been finished, so let's fit!
# 
# Is this Succeeded? There is some warning message. 
# 
# It seems that I should pass Y_train as the shape (n_samples, ). Actualy, I could get score, so it's not must. But, I fixed it just in case.

# In[29]:


lr = LogisticRegression().fit(imputer_X_train, Y_train)


# In[30]:


print(lr.score(imputer_X_train,Y_train))


# Before fixed Y_train shape

# In[31]:


print(Y_train.shape)


# Let's fix it by ravel.

# In[32]:


reshaped_Y_train = Y_train.values.ravel()
print(reshaped_Y_train.shape)


# Fit again. Ok, This time no warning. And it is same score with above one.

# In[33]:


lr = LogisticRegression().fit(imputer_X_train, reshaped_Y_train)


# In[34]:


print(lr.score(imputer_X_train, reshaped_Y_train))


# # Predict
# Ok, it's time to predict.
# 0, 0, 0, 0, 1... Looks like not bad.

# In[35]:


test_pred = lr.predict(imputer_test)


# In[36]:


print(test_pred.shape)
print(test_pred[0:5,])


# # Submit
# To submit, I have to make file as submission data form. I used pd.DataFrame.

# In[37]:


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})


# In[39]:


submission.head()


# In[40]:


submission.to_csv('submission.csv', index=False)


# # Conclusion
# I'm grad to have submited my prediction.
# 
# My score is 0.75598. No bad No good, isn't it?
# 
# I will carry on studying ML. And from next time, I would like to use other ML algorithm such as  decision tree, neural network or grid search, pipeline and so on...
