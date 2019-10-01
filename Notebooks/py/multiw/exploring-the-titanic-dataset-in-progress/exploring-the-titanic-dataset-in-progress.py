#!/usr/bin/env python
# coding: utf-8

# # An Introduction to Machine Learning: Exploring the Titanic Dataset
# In this kernel, I will be exploring the Titanic dataset using machine learning techniques.
# 
# ## Introduction to the Titanic Dataset
# The dataset we will be working on can be found [here](https://www.kaggle.com/c/titanic). This is a dataset from a beginner's competition on Kaggle.
# 
# This dataset holds attributes about passengers on the Titanic (such as age, sex, fare price, etc.) along with whether they survived. In this kernel, we will build a model to predict whether a given passenger survived based on their attributes. For example, given a passenger with who is 18 years old, male, and bought a $50 ticket, our model should predict whether he survived or not. 
# 
# ## Initial Assessment of the Dataset
# ### Importing the dataset
# To import our data, we use a python package called ```pandas```.  Note that we have two datasets: training data (```train_df```) and testing data (```test_df```). In supervised learning, we usually build our models from the training data and analyze our model's accuracy using the testing data. Both datasets contain a list of passengers, except they are mutually exclusive.

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import csv data as DataFrame objects.
train_df = pd.read_csv('../input/train.csv');
test_df = pd.read_csv('../input/test.csv');


# In Python, variable types are not explicitly declared in our code. Thus, here is the use-case documentation for [```DataFrame```](https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe).
# 
# ### First look at the data
# Here's a preview of our training datatset. The information about passengers will help us build our model to predict a passengers' survival.

# In[ ]:


# Display the first 5 samples (passengers) of our dataset.
train_df.head()


# Here are some explanations about each passenger attribute. Note some columns cateorical data rather than numerical, such as survival status, ticket class, and port of embarkation. This may affect the way we create our model using these categorical attributes.
# 
# | Variable | Definition | Key |
# | ---------- | ------------ | ----- |
# | survival | Survival | 0 = No, 1 = Yes
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd
# | sex | Sex | |
# | Age | Age in years | |
# | sibsp | # of siblings / spouses aboard the Titanic | |
# | parch | # of parents / children aboard the Titanic | |
# | ticket | Ticket number | |
# | fare | Passenger fare | |
# | cabin | Cabin number | |
# | embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
# 
# ## Building Our First Model
# 

# ## Something to Discuss Later
# ### Preprocessing our Data
# Before we can build a model, let's process our data to make it usable. As we can see when sampling our data, we have some ```NaN``` values in our cabin and age columns.

# In[ ]:


# Display the last 5 samples of our dataset.
train_df.tail()


# We can get a better summary of our dataset using the following.

# In[ ]:


train_df.info()


# ## Next Steps
# Tutorial
# * [Here](https://www.kaggle.com/startupsci/titanic-data-science-solutions)'s the tutorial I am referencing to use python packages and machine learning practices.
# 
# Learning the environment
# * Check out what kaggle/python docker image is. To quote the starter code, "This Python 3 environment comes with many helpful analytics libraries installed. It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python".

# In[ ]:




