#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from pandas import Series, DataFrame 
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[3]:


t_df = pd.read_csv('../input/train.csv')
t_df


# In[4]:


t_df.head()


# In[5]:


t_df.info()


# In[6]:


t_df['Age'].hist(bins=70)


# In[7]:


sns.factorplot('Sex', data=t_df, kind="count")


# In[8]:


t_df.groupby('Sex')[['Survived']].mean()


# In[9]:


sns.factorplot('Pclass',data=t_df,hue='Sex', kind='count')


# In[10]:


#t_df.groupby(['Sex', 'Pclass'])['Survived'].aggregate('mean').unstack()
t_df.pivot_table('Survived', index= 'Sex', columns= 'Pclass')


# In[11]:


Age = pd.cut(t_df['Age'], [0,18,80])
t_df.pivot_table('Survived', index= [Age, 'Sex'], columns = 'Pclass')


# In[12]:


t_df.pivot_table(index='Sex', columns='Pclass',
                aggfunc={'Survived':sum, 'Fare':'mean'})


# In[22]:


def male_female_child(passenger):
    Age,Sex = passenger
    if Age < 16:
        return 'child'
    else:
        return Sex


# In[23]:


t_df['person'] = t_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[24]:


t_df[0:10]


# In[ ]:


sns.factorplot('Pclass',data=t_df,hue='person', kind='count')


# In[25]:


t_df['person'].value_counts()


# In[26]:


t_df.pivot_table('Survived', index='person', columns='Pclass', margins= True, margins_name="%survival")


# In[27]:


deck = t_df['Cabin'].dropna()
deck.head()


# In[28]:


levels = []

for level in deck:
    levels.append(level[0])  #prendi la prima lettera  

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,palette='winter_d', kind='count')


# In[29]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.factorplot('Cabin',data=cabin_df,palette='summer', kind='count')


# In[30]:


sns.factorplot('Embarked',data=t_df,hue='Pclass', kind='count')


# In[31]:


t_df.pivot_table('Survived', index= 'Embarked', columns= 'Pclass')


# In[32]:


t_df['Alone'] =  t_df.Parch + t_df.SibSp
t_df['Alone']

t_df['Alone'].loc[t_df['Alone'] >0] = 'With Family'
t_df['Alone'].loc[t_df['Alone'] == 0] = 'Alone'

sns.factorplot('Alone',data=t_df,palette='Blues', kind='count')


# In[33]:


t_df["Survivor"] = t_df.Survived.map({0: "no", 1: "yes"})

sns.factorplot('Survivor',data=t_df,palette='Set1', kind='count')


# In[34]:


t_df.pivot_table('Survived', index='Alone', columns='Pclass', margins= True, margins_name="%survival")


# In[37]:


t_df.pivot_table('Survived', index='Alone', columns='Embarked', margins= True, margins_name="%survival")


# In[ ]:




