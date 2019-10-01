#!/usr/bin/env python
# coding: utf-8

# **To Predict Titanic Survivors**
# 
# The wreck  the Titanic was one of the worst shipwrecks in history, and is certainly the most well-known.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 
# One of the reasons that the shipwreck lead to such loss of life is that were not enough lifeboats for the passengers and crew.  Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, like women, children, and the upper-class.
# In this analysis I will try to complete the analysis of what sorts of people were likely to survive.  
# 
# VARIABLE DESCRIPTIONS:
# 
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
#                 
# survival        Survival
#                 (0 = No; 1 = Yes)
#                 
# name            Name
# 
# sex             
# 
# age             Age
# 
# sibsp           Number of Siblings/Spouses Aboard
# 
# parch           Number of Parents/Children Aboard
# 
# ticket          Ticket Number
# 
# fare            Passenger Fare
# 
# cabin           Cabin
# 
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)
#                 
# home.dest       Home/Destination
# 
# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 

# **1. Importing Libraries and Packages**
# 
# We will use these packages to help us manipulate the data and visualize the features/labels as well as measure how well our model performed. Numpy and Pandas are helpful for manipulating the dataframe and its columns and cells. We will use matplotlib along with Seaborn to visualize our data.

# In[ ]:


import numpy as np 
import pandas as pd 

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings("ignore")


# **2. Loading and Viewing Data Set**
# 
# Before we begin, we should take a look at our data table to see the values that we'll be working with. We can use the head and describe function to look at some sample data and statistics. We can also look at its keys and column names.

# In[ ]:


training=pd.read_csv("../input/train.csv")


# In[ ]:


training.head()


# In[ ]:


training.describe()


# In[ ]:


print(training.keys())


# **3. Dealing with NaN Values (Imputation)**
# 
# There are NaN values in our data set in the age column. Furthermore, the Cabin column has too many missing values and isn't useful to be used in predicting survival. We can just drop the column as well as the NaN values which will get in the way of our analysis.We also need to fill in the NaN values with replacement values in order for the model to have a complete prediction for every row in the data set. This process is known as imputation and we will show how to replace the missing data.

# In[ ]:


def null_table(training):
    print('Trainig Dataframa')
    print(pd.isnull(training).sum())
   

null_table(training)


# In[ ]:


training.drop(labels=['Cabin','Ticket'],inplace=True,axis=1)


null_table(training)


# In[ ]:


training['Age'].fillna(training['Age'].median(),inplace=True)

training['Embarked'].fillna('S',inplace=True)


null_table(training)


# **4. Plotting and Visualizing Data**
# 
# It is very important to understand and visualize any data . By visualizing, we can see the trends and general associations of variables like Sex and Age with survival rate. We can make several different graphs for each feature we want to work with to see the entropy and information gain of the feature.

# In[ ]:


sns.barplot(x='Sex',y='Survived', data=training)
plt.title('Distribution of survival based on Gender')
plt.show()

total_survived_females = training[training.Sex == "female"]["Survived"].sum()
total_survived_males = training[training.Sex == "male"]["Survived"].sum()

print("Total people survived is: " + str((total_survived_females + total_survived_males)))
print("Proportion of Females who survived:") 
print(total_survived_females/(total_survived_females + total_survived_males))
print("Proportion of Males who survived:")
print(total_survived_males/(total_survived_females + total_survived_males))


# Note that the numbers printed above are the proportion of male/female survivors of all the surviviors ONLY. The graph shows the propotion of male/females out of ALL the passengers including those that didn't survive.

# Gender appears to be a very good feature to use to predict survival, as shown by the large difference in propotion survived. Let's take a look at how class plays a role in survival as well.

# **Class**

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=training)
plt.ylabel("Survival Rate")
plt.title("Distribution of Survival Based on Class")
plt.show()

total_survived_one = training[training.Pclass == 1]["Survived"].sum()
total_survived_two = training[training.Pclass == 2]["Survived"].sum()
total_survived_three = training[training.Pclass == 3]["Survived"].sum()
total_survived_class = total_survived_one + total_survived_two + total_survived_three

print("Total people survived is: " + str(total_survived_class))
print("Proportion of Class 1 Passengers who survived:") 
print(total_survived_one/total_survived_class)
print("Proportion of Class 2 Passengers who survived:")
print(total_survived_two/total_survived_class)
print("Proportion of Class 3 Passengers who survived:")
print(total_survived_three/total_survived_class)


# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue="Sex", data=training)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# In[ ]:


sns.barplot(x="Sex", y="Survived", hue="Pclass", data=training)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")


# It appears that class also plays a role in survival, as shown by the bar graph. People in Pclass 1 were more likely to survive than people in the other 2 Pclasses.

# **Age**

# In[ ]:


survived_ages = training[training.Survived == 1]["Age"]
not_survived_ages = training[training.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1, 2, 2)
sns.distplot(not_survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Didn't Survive")
plt.subplots_adjust(right=1.7)
plt.show()


# In[ ]:


sns.stripplot(x="Survived", y="Age", data=training, jitter=True)


# It appears as though passengers in the younger range of ages were more likely to survive than those in the older range of ages, as seen by the clustering in the strip plot, as well as the survival distributions of the histogram.
