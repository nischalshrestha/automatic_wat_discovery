#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import pandas as pd
import seaborn as sns

# Importing the training dataset
train = pd.read_csv("../input/train.csv")

# Viewing the number of rows and columns in the training dataset
train_shape = train.shape
print(train_shape)


# **Topic: Machine Learning From Disaster**
# 
# **Challenge:** To Completethe analysis of what sort of people were likely to survive. 
# 
# Apply machine learining and its tools to predict which passengers survived the tragedy.
# Skills Required: Python, R | Binary Classification
# 
# **Basic Information: **
# The data has been split into two groups:
# 
# training set (train.csv)
# test set (test.csv)
# 
# **Training Set**: 
# It is used here to build your machine learning models. 
# Ground truth is provided for each passenger.
# Model is based on “features” like passengers’ gender and class.
# 
# **Test set**
# To check the performance of model on unseen data.
# Ground truth is not provided for any passenger.
# Model is used to  model to predict whether or not they survived the sinking of the Titanic.
# 
# Step 1: We have dataset of different information about passengers on the board - Titanic. This information will be used to predict whether those people survived or not.

# Step 2: To Get the Information of all the data stored in the Table

# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


# We can use DataFrame.pivot_table() to easily do this
# Importing the library for plotting
import matplotlib.pyplot as plt
# Calling the pivot_table() function for Sex
sex_pivot = train.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Female', 'Male'
sizes = [70, 20]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Male')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[ ]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
train = train.drop(['PassengerId','Name','Ticket'], axis=1)


# In[ ]:


# Embarked

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
train["Embarked"] = train["Embarked"].fillna("S")

# plot
sns.catplot('Embarked','Survived', data=train,height=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=train, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(train['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

#embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
#embark_dummies_test.drop(['S'], axis=1, inplace=True)

train = train.join(embark_dummies_titanic)
#test_df    = test_df.join(embark_dummies_test)

train.drop(['Embarked'], axis=1,inplace=True)
#test_df.drop(['Embarked'], axis=1,inplace=True) 


# In[ ]:


train.describe()


# **Function for Correlation:** 
# 
# **Correlation** is a measure of how strongly one variable depends on another. Consider a hypothetical dataset containing information about professionals in the software industry. ... Correlation can be an important tool for feature engineering in building machine learning models.

# In[ ]:


def plot_correlation_map( df ):
    corr = train.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )


# In[ ]:


plot_correlation_map( train )


# **Exploratory Data Analysis**
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.
# 
# **Missing Data**
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:





# In[ ]:




