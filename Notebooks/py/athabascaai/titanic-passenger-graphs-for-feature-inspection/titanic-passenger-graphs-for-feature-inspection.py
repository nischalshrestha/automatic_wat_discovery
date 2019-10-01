#!/usr/bin/env python
# coding: utf-8

# Looking into the data from the Titanic Disaster data set.
# ---------------------------------------------------------

# In[ ]:


# Author: Jamie de Domenico, March 22, 2017
# Graphing of the Titanic Data
# This is a exploritory look into the data provided from the 
# test data of the Titanic training data

get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
from subprocess import check_output

pd.options.display.max_rows = 100


# **Read in the training data file**  
# Simply read in the data files for the training.
# The training file is a known entity with most of the fields filled in.
# I say this with trepidation 

# In[ ]:


data = pd.read_csv('../input/train.csv')


# **Display the head of the train.csv file, default is 5 rows.**   
# This will provide us with a look at the column headings and possible features of the data set so that we can predict with a level of confidence if a passenger survived or not.  The first 5 rows also provides us with a glimpse of the data type(s) and this allows us to figure out how to convert this into a use-able type for a machine learning algorithm to use.

# In[ ]:


data.head()


# **This will provide a description of the data.**   
# This provides a description of the integer based data so that we can see the 
# distribution from a statistical point of view.  This leaves out the missing values so the description may be skewed.  This is the reason we need to fill the missing data in since we have a picture of the scenario just not a complete picture.  

# In[ ]:


data.describe()


# 
# **Fill in the missing age values with the a median age value.**   
# Here we are going to fill in the missing ages with a median value.
# There are many ways to do this, this is just one of the most common methods.
# If we find that age along with any other features has a high relevancy for predicting the data then we will need to find a better way to fill in the missing values so that they is a better approximation to what a real value would be in the case of the missing data.

# In[ ]:


data['Age'].fillna(data['Age'].median(), inplace=True)

data.describe()


# **Survival by Sex.**  
# It is easy to see that the men had the highest mortality rate.
# It is interesting to see that none of the training data had a missing value for sex.
# If there was a missing value for this we could easily fill in a correct value from a combination of the title and perhaps the first name along with some other attributes.
# It is interesting to note that the survival by sex indicates that **Males**  Had the highest mortality rate from the training data.  On the same note looking at the survived bar it looks like the dead bar but upside down, where Females had a much higher survival rate.
# So what does this tell us?
# Well first and for most is that **Males** had a low chance of survival. This is our first indication of data importance.

# In[ ]:


survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5), color=['pink','blue'], title='Survival by the Sex')


# **Survival by Age.**  
# Earlier we filled in the missing data for age with a mean value so that we could provide a first time graph of survival by age.  Seems that anyone between the age of 25 to 35 had the highest mortality.
# This tells use that age is another important feature that we need to use when predicting the survival rate of a passenger on the Titanic.

# In[ ]:


figure = plt.figure(figsize=(10,5))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'], 
         bins = 30, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.title('Survival by Age')


# **Ticket Price:**  
# Ticket price really relates into class, someone who bought a low priced ticket probably had a cabin in the lower part of the boat and was considered part of the working class. Looking at the graph you can see that the lower the ticket price the higher the mortality or lower the survival.

# In[ ]:



data['Fare'].fillna(data['Fare'].median(), inplace=True)
figure = plt.figure(figsize=(10,5))
plt.hist(
            [data[data['Survived']==1]['Fare'],
            data[data['Survived']==0]['Fare']], 
            stacked=True, color = ['g','r'],
            bins = 30,
            label = ['Survived','Dead']
        )
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.title('Survival by Ticket Price')


# **Scatter diagram with age and as the characteristics for survival**
# There is some relevance here where we can observe that higher fares tend to have higher survival along with an age that is not between 25 and 30.  see the peak around the 27 year old age with fares coming close to 100.00.  So there is some relationship between age and fare price.

# In[ ]:


plt.figure(figsize=(10,5))
plt.title('Scatter Diagram of Survival By Age & Fare')
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


# **Fare by Class**
# This bar chart gives us a view of the average cost of each of the 3 class .
# Pclass 1 is 80.00 plus
# Pclass 2 is 20.00
# Pclass 3 is approximately 12.00
# 
# This tells us that if we have the cost of a ticket per passenger then we can predict with a high level of confidence the class Pclass they are in or the level the cabin is on.  

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(
                                           title = 'Average Fare by Class',
                                           kind='bar',
                                           figsize=(10,5), 
                                           ax = ax,
                                           x = 'Passanger Class',
                                           color=['green', 'yellow', 'blue']
                                           )


# **Survival by the Class** 
# 
#  1.    **Q** - *First Class is Green*   
#  2.   **C** - *Second Class is Red*
#  3.   **S** - *Third Class is Black*
# 

# In[ ]:


survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(
        kind='bar',stacked=True, 
        figsize=(10,5), 
        title = 'Survival By Class',
        color=['blue', 'yellow', 'green']
       )


# **Conclusion**  
# So we have the data graphed and we can see that there are a number of strong indicators for survival.
# 
#  1. Sex
#  2. Pclass
#  3. Age
#  4. Ticket price ( but this relates directly to the Pclass)
