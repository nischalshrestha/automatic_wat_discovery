#!/usr/bin/env python
# coding: utf-8

# # Logistic-Regression-in-Python
# Applied Logistic Regression (Supervised learning model) on Titanic: Machine Learning from Disaster Data-set from Kaggle
# 
# ## Overview 
# The following jupyter notebook contains steps I used for drawing out predictions on the survived column. Since there is a target column/ variable present, this problem gets classified as a supervised learning problem.
# Steps:
# 1. Importing necessary libraries. (numpy, pandas, matplotlib, seaborn).
# 2. Reading in data.
# 2. Data Exploration and Visualization.
# 3. Feature Engineering. 
# 4. Construction of a predictive model( Any machine learning model in Surpervised Learning)
# 5. Fitting featurized data into the predictive model.
# 6. Drawing out predictions with test data.
# 
# ## Dependencies:
# * conda install numpy
# * conda install pandas
# * conda install scikit-learn
# * conda install matplotlib

# 
# ### Importing Libraries

# In[ ]:


# Standard libraries
import pandas as pd
import numpy as np


# In[ ]:


# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# #### Reading Data

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# ### Data exploration and visualization

# In[ ]:


train.info()


# In[ ]:


train.describe()


# #### Checking for null values

# In[ ]:


train.isnull()


# Plotting the null values on a heatmap for better visualization

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
# every yellow line means missing data 


# 
# * Age and Cabin columns have missing data.
# *  Age column has relatively less missing data when compared to Cabin

# In[ ]:


# First of all lets plot who survived and who didn't
sns.countplot(x = 'Survived', data = train)
# So about 580 didn't survive 
# And 320 survived


# * *Dataset has a greater number rows with survived label as 0*
# * *About 580+ classify as 0*
# * *And 320 as 1*

# In[ ]:


# Lets look at survived with a hue of gender
sns.countplot(x = 'Survived', data = train, hue = 'Sex', palette='RdBu_r')


# In[ ]:


# Lets look at survived with a hue of Pasenger class
sns.countplot(x = 'Survived', data = train, hue = 'Pclass')


# In[ ]:


# Lets get an idea about the age of people in the data set
sns.distplot(train['Age'].dropna(), kde= False, bins = 30)


# In[ ]:


sns.countplot(x = 'SibSp', data = train)
# By looking at this plot, most people on board neither had  siblings / spouses aboard


# In[ ]:


# Another column which we haven't explored yet is the fare column
train['Fare']


# In[ ]:


train['Fare'].hist( bins = 40, figsize = (10,4))
# most of the distribution is between 0 and 100 


# In[ ]:


plt.figure(figsize = (10,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
# The figure shows that the Passengers in class 1 have older people 
# And younger people in lower Pclass


# In[ ]:


# Filling in null age values
def substitution(columns):
    Age = columns[0]
    Pclass = columns[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 36         # approx mean value from blue box
        elif Pclass == 2:
            return 29        # approx mean value from orange box
        else:
            return 23         # approx mean value from green box  
    else:
        return Age           # is not null


# In[ ]:


train['Age'] = train[['Age', 'Pclass']].apply(substitution, axis = 1)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
# no more missing values in Age 


# In[ ]:


sns.heatmap(train.corr(), annot= True)
# Checking for correlation between columns


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
# there are so many missing columns in cabin
# that it seems right to drop it


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
# Final check for null values


# In[ ]:


train.head()


# In[ ]:


pd.get_dummies(train['Sex'])
# we need to convert the sex column
# otherwise the machine learning alogorithm won't be able process the data


# In[ ]:


pd.get_dummies(train['Sex'], drop_first= True)
# now you can not feed both these columns as male and female are opposite
# and it will mess up the machine learning algorthim


# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first= True)


# In[ ]:


sex.head()


# In[ ]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)
# same process with Embarked column


# In[ ]:


embark.head()


# In[ ]:


# Since Pclass is also a categorical column
pclass = pd.get_dummies(train['Pclass'],drop_first=True)


# In[ ]:


train = pd.concat([train, sex, embark, pclass], axis = 1)


# In[ ]:


train.head()
# now, we don't need sex, embarked, plcass column because we have encoded them.


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket', 'Pclass'],axis=1,inplace=True)
# dropping columns which we are not going to use


# In[ ]:


train.head()
# looks perfect for our machine learning algorithm
# all data is numeric


# In[ ]:


train.head()


# In[ ]:


# Features
X = train.drop('Survived', axis = 1)

# Target variable
y = train['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression
# Supervised learning 


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X,y)


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test.columns


# In[ ]:


test.info()


# In[ ]:


# Preparing test data according to the model
sex = pd.get_dummies(test['Sex'], drop_first= True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
pclass = pd.get_dummies(test['Pclass'],drop_first=True)

test = pd.concat([test, sex, embark, pclass], axis = 1)

test.drop(['Sex','Embarked','Name','Ticket', 'Pclass', 'Cabin'],axis=1,inplace=True)


# In[ ]:


test.columns


# In[ ]:


# Checking for null values in test
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# In[ ]:


# Dropping null values of age
test.dropna(inplace=True)


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# In[ ]:


test.info()


# In[ ]:


predictions = logmodel.predict(test)


# In[ ]:


predictions


# In[ ]:


predictions = pd.DataFrame(predictions)
predictions.to_csv('gender_submission.csv')


# In[ ]:




