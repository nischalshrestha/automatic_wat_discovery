#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of Titanic Data Set 
# 

# <h4><span style="Color:Black"> 1. ) Import libraries </strong></span></h4>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# <h4><span style="Color:Black"> 2. ) Read in  Titanic Data Set using <strong>read_csv</strong></span></h4>

# In[ ]:


titanic_train = pd.read_csv('../input/train.csv')


# <h4><span style="Color:Black"> 2. ) View Titanic Data Set and checked data types for columns  </span></h4>

# In[ ]:


titanic_train.head(20)


# In[ ]:


titanic_train.dtypes


# <h4><span style="Color:Black"> 3.) Rearrange Columns    </span></h4>

# In[ ]:


titanic_train.columns


# In[ ]:


new_order = ['Survived', 'Pclass','Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Embarked','Name', 'PassengerId', 'Ticket', 'Cabin']


# In[ ]:


titanic_train = titanic_train[new_order]


# In[ ]:


titanic_train.head()


# <h4><span style="Color:Black"> 4. ) Checking the dimensionality of the DataFrame and used the isnull( ) method to locate any null values located within the DataFrame.    </span></h4>
# 

# In[ ]:


titanic_train.shape


# In[ ]:


titanic_train.isnull().sum()


# <h4><span style="Color:Black"> 5. )Univariate Analysis (Graphical\Non-Graphical) : </span></h4> Checked the distinct observations within DataFrame, generated descriptive and graphical statistical analysis using the .value_counts( ) and .describe( ) methods. Also, used the seaborn library to graph desired variables. 

# In[ ]:


titanic_train.Survived.value_counts(normalize=True)*100


# In[ ]:


titanic_train.Pclass.value_counts(normalize=True)*100


# In[ ]:


titanic_train.Sex.value_counts(normalize=True)*100


# In[ ]:


titanic_train.Parch.value_counts(normalize=True)*100


# In[ ]:


titanic_train.Embarked.value_counts(normalize=True)*100


# In[ ]:


titanic_train.describe(include=['object'])


# In[ ]:


titanic_train.describe()


# <h4><span style="Color:Black"> 4. ) Perform univariate analysis </span></h4>

# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(16, 10), sharex=False, sharey=False)

sns.countplot(x='Pclass', data=titanic_train, ax=axes[0,0])
sns.countplot(x='Survived', data=titanic_train, ax=axes[0,1])
sns.countplot(x='Sex', data=titanic_train, ax=axes[0,2])
sns.countplot(x='SibSp', data=titanic_train, ax=axes[0,3])
sns.countplot(x='Embarked', data=titanic_train, ax=axes[1,0])
sns.countplot(x='Parch', data=titanic_train, ax=axes[1,1])
sns.distplot(titanic_train['Fare'], ax= axes[1,2])
sns.distplot(titanic_train['Age'].dropna(), ax = axes [1,3]);


# <h4><span style="Color:Black"> 5. )Bivariate Analysis (Graphical\Non-Graphical) : </span></h4> Checked the distinct observations within DataFrame, generated descriptive and graphical statistical analysis using the .value_counts( ) and .describe( ) methods. Also, used the seaborn library to graph desired variables. 

# <h4><span style="Color:Black">Below, I used the heat map in the seaborn library to see both the positive and negative correlations between the variables associated with this DataFrame.</span></h4>

# In[ ]:


titanic_corr = titanic_train.corr()


# In[ ]:


sns.heatmap(titanic_corr,annot=True) 


# In[ ]:


fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10), sharex=False, sharey=False)

sns.stripplot(x="Sex", y="Age", data=titanic_train , jitter=True ,ax=axes2[0,0])
sns.barplot(x="Pclass", y="Survived", data=titanic_train, ax=axes2[0,1])
sns.barplot(x="Embarked", y="Survived", data=titanic_train, ax = axes2 [0,2])
sns.barplot(x="Pclass", y="Age", data=titanic_train, ax = axes2[1,1])
sns.barplot(x='Embarked', y='Fare', data=titanic_train, ax = axes2[1,2])
sns.barplot(x='Sex', y='Survived', data=titanic_train, ax = axes2[1,0]);


# # Conclusion
# 
# <h4><span style="Color:Black">The observations from univariate analysis were the following: </span></h4>
#    * 68% of the passengers did not survive.
#    * Majority of the passengers on the Titanic were considered lower class.  
#    * 78% of the passengers were from Southhampton, 18% of the passengers were from Cherbourg, and 8%  were from         Queenstown
#    * Average fare was 32.20 pounds which would be approximately 3,500 dollars if the ticket was bought today. 
#    
# <h4><span style="Color:Black">The observations from bivariate analysis were the following: </span></h4>
#    * There is a correlation between age and Pclass. The mean age in the upper class is higher and had a better chance on surviving.   
#    * From the univariate plots, there was more people on the Titanic from Southampton. But, from the graph bivariate plots the mean survival was higher for individuals from Cherbourg. 
#    * The mean fare was the highest for passengers from Cherbourg. 
# 
# 
# 
#    

# In[ ]:




