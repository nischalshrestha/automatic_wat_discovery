#!/usr/bin/env python
# coding: utf-8

# # The Titanic problem from a noob perspective
# 

# Hello all! First off I want to say that I am a total noob in this and I am sure I will be making lots of silly mistakes while working with this Machine Learning project (it's my first one and also the first time I use Jupyter Notebook). So please, I would really appreciate any suggestions or improvements you may have about my code as I am sure you all are way more experienced than me. Thanks! :)

# I want to state that I am familiarized with the Python language but not at all with the main data processing libraries such as *Pandas* and *Numpy* and the data visualization ones like *Matplotlib* and *Seaborn*. I will be updating my code while I learn more about these.
# 
# *Note:* This will not be an in depth review, as pointed before, my skills are very very limited so there may even be some mistakes or unnecessary steps. For a way more detailed explanation, please look at this very well written <a href="https://www.kaggle.com/startupsci/titanic-data-science-solutions"> review</a>. Some ideas implemented in this notebook come from there.

# So this notebook is going to be divided in the following sections:
# 
# - Description of the problem
# - Knowing more about the data (still in process)
# - Transforming the data (still in progress)
# - Predicting new input (not yet started)

# ## Description of the problem

# In this project I am going to try to approach the Titanic problem although I am sure I will need help from other people's reviews to finish it. 
# 
# The very first thing I consider one should do is to carefully review what the problem is all about. The most important things we should note from here I think are:
# 
# - The Titanic sank after colliding with an iceberg, killing **1502 out of 2224** passengers and crew. So we have 67% of people dead.
# - Some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# - **In this challenge [...] we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.**
# 
# In the *Data* section we find how we should submit the results as well as all the data specification. Among the data from the **Data Dictionary** section, we find two very interesting ones: 
# 
# - **Sibsp:** # of siblings/spouse
# - **Parch:** # of parents/children
# 
# So, after knowing what to do, let's see what kind of data we are going to deal with:

# We first have to import all the necessary libraries (it should be updated every time a new resource is needed). The following ones are for data treatment and visualization:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[ ]:


# We read from the csv files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# and view how the first lines are composed
train.head()


# ______________________________________________________________________________________________________________________________

# ## Knowing more about the data

# In order to know more about the kind of data that we are dealing with, let's see for each column their correspondent values and their count. 
# 
# *Note:* As I have discovered, doing this also helps you to see if there is any incongruity within the training data.

# In[ ]:


# Get the total number of passengers we have in our dataset
n_passengers = train.shape[0]

# Now we discard some of the columns that would render unique when counting their values
train_to_print = train.drop(["PassengerId", "Name", "Age", "Ticket", "Cabin", "Fare"], axis=1)


for item in train_to_print:
    print(train_to_print[item].value_counts())
    print("{0}: Missing data from {1} passengers".format(item, n_passengers - train[item].count()))
    print("\n")
    


# Let's see how many information we have left to fill:

# In[ ]:


train.count()


# Now, let's make several assumptions in regards to the survival:
# - The survival rate for women was larger than the one for men
# - The survival rate for children between 0 - 10 was one of the greatest amongst different ranges of ages
# - The survival rate for passengers in higher class was greater
# 
# We can check these by looking at the stats:

# In[ ]:


train[["Survived", "Sex"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Sex")


# In[ ]:


children = train.loc[(train.Age > 0) & (train.Age <= 10)]
mean_child = children[["Survived", "Age"]].groupby(["Age"], as_index=False).mean()["Survived"].mean()
print("The average child between 0 and 10 had a {:.2f}% chance of survival".format(mean_child*100))

others = train.loc[(train.Age > 10)]
mean_others = others[["Survived", "Age"]].groupby(["Age"], as_index=False).mean()["Survived"].mean()
print("The mean chances of survival for the rest were: {:.2f}%".format(mean_others*100))


# In[ ]:


train[["Survived", "Pclass"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Pclass")


# Also, just for fun and training with the *matplotlib* library, here are the charts for each of the three assumptions we made:

# ### Survival plots based on gender

# In[ ]:


# Create the array with the x locations for the groups
x_locations = np.arange(1)  
# Set the bar's width
bar_width = 0.2      

# Set each pair of values we'll use
survived = [0, 1]
gender = ["female", "male"]
colors = ['c', 'b']
spacing = [x_locations, x_locations + 0.3] 

# Declare the figure of our plot and its size
fig, axes = plt.subplots(1, 2, figsize=(8,3.8))

# Create both plots 
for ax, surv in zip(axes, survived):
    for sex, sp, c in zip(gender, spacing, colors):
        gender_surv_df = train.loc[(train.Sex == sex) & (train.Survived == surv)]
        count = gender_surv_df.PassengerId.count()

        ax.bar(sp, count, bar_width, color=c)
        ax.set_title('Survived=' + str(surv))
        ax.set_xticks([0, 0.3])
        ax.set_xticklabels(('Female', 'Male'))



# **In progress...**

# ------

# ## Transforming the data

# Now that we know more about our data, one of the things we can do is to "purge" or delete that information we may think is not very useful. 
# 
# In my unexperienced case, I consider that the columns **PassengerId**, **Fare** and **Cabin** do not give too much information about if the passenger itself will survive or not (which at the end of the day is what we need to predict). So we may want to drop them:

# In[ ]:


# First fill the array with the columns we want to drop
to_drop = ["PassengerId", "Fare", "Cabin"]

# and we drop them. The parameter axis=1 means that these values
# are contained in the columns.
new_train = train.drop(to_drop, axis=1)

new_train.head(3)


# **In progress..**
