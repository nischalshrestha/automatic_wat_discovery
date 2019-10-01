#!/usr/bin/env python
# coding: utf-8

# # Titanic Challenge 
# 
# This code will be a trip around the principals algorithms for classification problems. We will see 3 phases, Exploration Data Analysis, Predictive Models and ML Advance. I hope that like my first code in Kaggle.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Data visualization
import seaborn as sns #Data Visualization
import os
import warnings
import plotly.plotly as py
import plotly.tools as tls

warnings.filterwarnings("ignore")


# In[ ]:


print(os.listdir("../input"))# See, how much files we have?


# In[ ]:


data = pd.read_csv('../input/train.csv',sep=',')
data.head(10)


# In[ ]:





# ## Analyzing and Undestanding features and labels

# In[ ]:


data.describe()


# ### Kind of classification
# 
# Titanic challenge give us a binary classification problem but our job is identify if objetive variable presents a balanced or unbalanced problem, because the response allow to take into account the optimal strategic for resolve the predictive task.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(17,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# In[ ]:


classes = data['Survived'].value_counts()
diference = classes[0]-classes[1]
print ("the difference between classes are {} rows".format(diference))


# ### Features
# Passenger Class 

# In[ ]:


pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='summer_r')


# Sex

# In[ ]:


data.groupby(['Sex']).count()


# In[ ]:


data.groupby(['Sex','Survived'])['Survived'].count()


# In[ ]:


pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


sns.barplot('Pclass','Survived',hue='Sex',data=data)
plt.show()


# ## Data cleaning

# In[ ]:


nulls=data.isnull().sum()
print (nulls)
nulls.plot.bar()


# In[ ]:




