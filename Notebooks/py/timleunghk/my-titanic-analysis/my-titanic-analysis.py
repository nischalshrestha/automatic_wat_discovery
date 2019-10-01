#!/usr/bin/env python
# coding: utf-8

# ## My Titanic Analysis

# # 1 Business Understanding
# 
# ## 1.1 Objective
# 
# Predict survival on the Titanic
# 
# ## 1.2 Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On 15/4/2012, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew,722 passengers and crew were survived. Survial rate was therefore 32% (722/2224). This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, I would like the analysis of what sorts of people were likely to survive by using Deep Learning mechanism.
# 
# 

# ## 1.3 Prepare Panda & Load Data
# 
# According to Kaggle, there are two datasets 
# 
# Location of dataset
# train csv: https://storage.googleapis.com/bwdb/acceleratehk/10%20-%20kaggle%20class/train.csv
# test 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


df_train = pd.read_csv("../input/train.csv")

# Configure visualisations
get_ipython().magic(u'matplotlib inline')
sns.set_style('white')
#pylab.rcParams['figure.figsize']=8,6


# In[ ]:


def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()


# In[ ]:


df_train.info()


# In[ ]:


df_train


# In[ ]:


df_train['Age'].unique()


# We found NaN in Age Field, In order to keep all records and prevent from affecting Age Result,
# we use mean instead of replace 0 in fillna function
# 
# 

# Prepare a new dataframe, the difference mean of Age replace NaN in Age field.

# In[ ]:


df_train['Embarked'].unique()


# In[ ]:


age_mean=df_train['Age'].mean()

titanic = pd.DataFrame()

titanic['Survived'] = df_train['Survived']
titanic['Name'] = df_train['Name']
titanic['Age'] = df_train['Age'].fillna(age_mean)
titanic['Pclass'] = df_train['Pclass']
titanic['Sex']=df_train['Sex']
titanic['SibSp']=df_train['SibSp']
titanic['Parch']=df_train['Parch']
titanic['Fare']=df_train['Fare']
titanic['Cabin']=df_train['Cabin']
titanic['Embarked']=df_train['Embarked'].fillna('NA')



# ## View Description of new dataframe.. Titanic

# In[ ]:


titanic.describe()


# ## View all values of these  fields

# In[ ]:


titanic['Age'].unique()


# In[ ]:


titanic['Fare'].unique()


# In[ ]:


titanic['Pclass'].unique()


# In[ ]:


titanic['Name'].unique()


# In[ ]:


titanic['SibSp'].unique()


# In[ ]:


titanic['Parch'].unique()


# In[ ]:


titanic['Fare'].unique()


# In[ ]:


titanic['Age'].mean()


# In[ ]:


titanic.corr()  ## Find relationship


# ## 1.4 Finding Relationship between varibles .... by Heatmap
# 
# Heatmap uses Seaborn

# In[ ]:


#plot_correlation_map(titanic)
sns.heatmap(
        titanic.corr(),
        cbar_kws={ 'shrink' : .9 },
        annot = True, 
        annot_kws={'fontsize': 12}
)
sns.set_style('white')



# According to above heat map, we found some important points to focus
# 
# 1. Fare VS Survied = 0.26, Fare expansive >> Survied  
# (-1<=r<=1, -1<=r<0 , inversely proportional, 0 < r<=1, Directly proportional, 0: No relationship)
# 
# 2. Class VS Survied = -0.34, Class Lower (by number) >> Survied
# 3. Age VS Survied = -0.7, Age smaller(means younger) >> Survied
# 4. SibSp VS Survied = -0.035, least relationship but inversely proportional
# 5. Parch VS Survied 0.0082, Fare expansive >> Survied  
# 

# ### 1.6 Distribution Models
# 
# Prepare Model for learning according to Age, Class, SibSp, Fare VS Survied
# 
# 

# In[ ]:


plot_distribution(titanic,var='Pclass',target='Survived')


# In[ ]:


plot_distribution(titanic,var='Age',target='Survived',row='Sex')


# In[ ]:


plot_distribution(titanic,var='SibSp',target='Survived')


# In[ ]:


plot_distribution(titanic,var='Parch',target='Survived')


# In[ ]:


cols = ['Pclass','Age']##,'SibSp']

traindf_info = titanic.loc[:,cols]
traindf_info.shape

traindf_survived=titanic['Survived']

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(traindf_info,traindf_survived)


# In[ ]:


df_test_load= pd.read_csv("../input/test.csv")


df_test_tobepredicted = df_test_load.loc[:,cols]

#Start build model

#1. Fill NaN by filling mean in order to prevent from affecting the calculation and building model.
age_mean = df_test_tobepredicted['Age'].mean()

titanic_tobepredicted = pd.DataFrame()

titanic_tobepredicted['Pclass'] = df_test_tobepredicted['Pclass']
titanic_tobepredicted['Age'] = df_test_tobepredicted['Age'].fillna(age_mean)

titanic_tobepredicted.describe()



# In[ ]:


#Start predict who in df_test is survived 

my_titanic_predict = logreg.predict(titanic_tobepredicted)
my_titanic_predict


# In[ ]:


df_result=pd.DataFrame({'PassengerId':df_test_load['PassengerId'],'Survied':my_titanic_predict})
df_result.to_csv('titanic_result.csv')


# In[ ]:





# In[ ]:





# In[ ]:




