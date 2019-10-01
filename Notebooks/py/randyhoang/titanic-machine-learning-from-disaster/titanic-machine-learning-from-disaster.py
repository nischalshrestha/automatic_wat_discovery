#!/usr/bin/env python
# coding: utf-8

# <img src="https://media.giphy.com/media/9ABgKHIu3acWA/giphy.gif"/>

# # Titantic: Machine Learning from Disaster  
#  

# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  Titanic was a passenger liner that sank in the North Atlantic Ocean on 15 April 1912 after colliding with an iceberg during her first voyage from Southampton, UK to New York City, US. The sinking of Titanic caused the deaths of 1,502 people in one of the deadliest peacetime maritime disasters in history. Titanic was the largest ship at the time and was called 'the unsinkable'. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# After leaving Southampton on 10 April 1912, Titanic called at Cherbourg in France and Queenstown (now Cobh) in Ireland before heading westwards towards New York. On 14 April 1912, four days into the crossing and about 375 miles (600 km) south of Newfoundland, she hit an iceberg at 11:40 pm (ship's time; GMT−3). The glancing collision caused Titanic's hull plates to buckle inwards in a number of locations on her starboard side and opened five of her sixteen watertight compartments to the sea. Over the next two and a half hours, the ship gradually filled with water and sank. Passengers and some crew members were evacuated in lifeboats, many of which were launched only partly filled. A disproportionate number of men -- over 90% of those in Second Class -- were left aboard due to a "women and children first" protocol followed by the officers loading the lifeboats. Just before 2:20 am Titanic broke up and sank bow-first with over a thousand people still on board. Those in the water died within minutes from hypothermia caused by immersion in the freezing ocean. The 710 survivors were taken aboard from the lifeboats by RMS Carpathia a few hours later.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

# **“Fifteen-hundred people went into the sea, when Titanic sank from under us. There were twenty boats floating nearby… and only one came back. One. Six were saved from the water, myself included. Six… out of fifteen-hundred. Afterward, the seven-hundred people in the boats had nothing to do but wait… wait to die… wait to live… wait for an absolution… that would never come.” —Rose**
# ***
# 
# <img src="https://media.giphy.com/media/YE9A1qSEn0gV2/giphy.gif"/>
# 
# 
# 

# ## About This Dataset
# ***
# Source: https://www.kaggle.com/c/titanic
# 
# **Overview: The data has been split into two groups:**
# 
# training set (train.csv) test set (test.csv) The training set should be used to build machine learning models. For the training set, the outcome is included (also known as the “ground truth”) for each passenger. My model will be based on “features” like passengers’ gender and class.
# 
# The test set should be used to see how well my model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. For each passenger in the test set, use the model to predict whether or not they survived the sinking of the Titanic.
# 
# gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
# 
# Data Dictionary Variable	Definition	Key survival	Survival	0 = No, 1 = Yes pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd sex	Sex Age Age in years sibsp	# of siblings / spouses aboard the Titanic parch	# of parents / children aboard the Titanic ticket	Ticket number fare	Passenger fare cabin	Cabin number embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton Variable Notes pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way... Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way... Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.
# 

# ## Problem
# ***
# *The Titanic is about to sink and we want to predict who survives and who dies*

# ## Objective 
# ***
# 
# *We would like to see the likelyhood of a passenger surviving this fatal crash, many factors come into play in predicting whether or not a passenger survives.  The goal is to find out what those factors are and how much of an impact they make in this situation.*

# ## OSEMN Pipeline
# ****
# 
# *I’ll be following a typical data science pipeline, which is call “OSEMN” (pronounced awesome).*
# 
# 1. **O**btaining the data is the first approach in solving the problem.
# 
# 2. **S**crubbing or cleaning the data is the next step. This includes data imputation of missing or invalid data and fixing column names.
# 
# 3. **E**xploring the data will follow right after and allow further insight of what our dataset contains. Looking for any outliers or weird data. Understanding the relationship each explanatory variable has with the response variable resides here and we can do this with a correlation matrix. 
# 
# 4. **M**odeling the data will give us our predictive power on whether a passenger will survive or not. 
# 
# 5. I**N**terpreting the data is last. With all the results and analysis of the data, what conclusion is made? What factors contributed most to survival? What relationship of variables were found? 

# # Part 1: Obtain the Data  
# ***

# In[ ]:


# Import libraries for data manipulation and data visulization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# Read the csv file and store the data into a dataframe "titanic_df"
titanic_df = pd.read_csv('../input/train.csv', index_col=None)


# # Part 2: Scrubbing the Data
# ***

# Usually cleaning and scrubbing the data would be tedious and can take many steps to prepare the data,  but thankfully, this particular data set is fairly clean and only has a few flaws.  I have to make sure that the dataset is not missing any values. 
# 
# Before scrubbing the data, I am going to do a exploratory data analysis (EDA), which is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.
# 
# 

# ## Part 2a: Exploratory Data Analysis
# Let's get a first overview of the train and test dataset
# How many rows and columns are there?
# What are the names of the features (columns)?
# Which features are numerical, which are categorical?
# How many values are missing?
# The **shape** and **info** methods answer these questions
# **head** displays some rows of the dataset
# **describe** gives a summary of the statistics (only for numerical columns)

# In[ ]:


titanic_df.shape


# In[ ]:


titanic_df.info()


# In[ ]:


# Check to see if there is any missing values in our dataset
titanic_df.isnull().any()


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df.describe()


# ## Part 2a:  The Scrubbing
# Since it looks like the column "Cabin" is missing many values, so we are going to drop the nulls and I am going to call it "deck"
# I am going to ignore other columns with missing values, because it should not have much impact on the accuracy of my analysis.

# In[ ]:


deck = titanic_df['Cabin'].dropna()


# Taking a look at the cabin column, you can see that the values contain a Letter followed by numbers.  Since we only need the letter to determine their cabin location.

# In[ ]:


deck.head


# In[ ]:


levels = []
for level in deck:
    levels.append(deck[0])
    
# plotting the new data
cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']
sns.factorplot('Cabin', data=cabin_df,palette='winter_d', kind="count", order =['A','B','C','D','E','F','G','T'])


# # Part 3: Exploratory Analysis - Exploring the Data
# *** 
#  <img  src="https://s-media-cache-ak0.pinimg.com/originals/32/ef/23/32ef2383a36df04a065b909ee0ac8688.gif"/>

# ## Part 3a: Demographic of the people onboard the Titanic
# ***

# In[ ]:


# Count of sex onboard
sns.catplot('Sex', data=titanic_df,kind='count')


# In[ ]:


# Count of each sex on which "Class" 
sns.catplot('Pclass', data=titanic_df,kind='count',hue='Sex')


# In[ ]:


# Making a function for whether or not the passenger is a "child" or not
# Under the age of 16 = Child, over the age of 16 = "Sex"
def male_female_child(passenger):
    age, sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[ ]:


# Creating a row named 'person' that will display if the passenger is a child, if not child, then display sex
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[ ]:


# Let's take a look at the dataset to see if our column worked out 
titanic_df[0:10]


# In[ ]:


# List out the number of male, female, and children on each class
sns.catplot('Pclass',data=titanic_df,kind='count',hue='person')


# In[ ]:


# Histogram of the distribution of ages
titanic_df['Age'].hist(bins=70)


# In[ ]:


# Average age of passengers onboard
titanic_df['Age'].mean()


# In[ ]:


# The number of female/male/child
titanic_df['person'].value_counts()


# In[ ]:


# FacetGrid allows me to make multiple plots
# aspect changes is necessary to change the aspect ratio so the graph fits nicely
fig = sns.FacetGrid(titanic_df, hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

# Set a variable to equal the highest age in the data
oldest = titanic_df['Age'].max()

# Set a limit from yongest to oldest
fig.set(xlim=(0,oldest))

# Add a legend to the graph
fig.add_legend()


# In[ ]:


# Let's do the same plot, but now adding the children
# FacetGrid allows me to make multiple plots
# aspect changes is necessary to change the aspect ratio so the graph fits nicely
fig = sns.FacetGrid(titanic_df, hue='person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)

# Set a variable to equal the highest age in the data
oldest = titanic_df['Age'].max()

# Set a limit from yongest to oldest
fig.set(xlim=(0,oldest))

# Add a legend to the graph
fig.add_legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




