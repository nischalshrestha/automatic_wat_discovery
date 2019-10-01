#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import Series, DataFrame


# In[ ]:


titanic_df = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df.info()

1) Who were the passangers on the titanic? (Ages, Gender, CLass...)
2) What deck were the passengers on and how does that relate to their class?
3) Where did the passengers come from?
4) Who was alone and who was with family?
5) What factors helf someone survive the sinking?
# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


sns.countplot('Pclass',data=titanic_df,hue='Sex')


# In[ ]:


def male_female_child(passenger):
    age, sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[ ]:


titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis=1)


# In[ ]:


titanic_df[0:10]


# In[ ]:


sns.countplot('Pclass',data=titanic_df,hue='person')


# In[ ]:


titanic_df['Age'].hist(bins=70)


# In[ ]:


mean_age = titanic_df['Age'].mean()
mean_age


# In[ ]:


titanic_df['person'].value_counts()


# In[ ]:


fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))
fig.add_legend()


# In[ ]:


titanic_df.head()


# In[ ]:


deck = titanic_df['Cabin'].dropna()


# In[ ]:


deck.head()


# In[ ]:


levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']



# In[ ]:


cabin_df = cabin_df[cabin_df != 'T']
sns.countplot(x='Cabin',data=cabin_df, palette='summer', order=['A', 'B', 'C', 'D', 'E', 'F'])


# In[ ]:


titanic_df.head(10)


# In[ ]:


sns.countplot('Embarked', data=titanic_df, palette='muted', hue='Pclass', order=['C', 'Q', 'S'])


# In[ ]:


# Who was alone? Who was with family?
titanic_df.head(10)


# In[ ]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[ ]:


titanic_df['Alone']


# In[ ]:


titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[ ]:


titanic_df.head()


# In[ ]:


sns.countplot('Alone', data=titanic_df, palette='Blues')


# In[ ]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})

sns.countplot('Survivor', data=titanic_df, palette='Set1')


# In[ ]:


sns.factorplot(x='Pclass', y='Survived', data=titanic_df)


# In[ ]:


sns.factorplot(x='Pclass', y='Survived', data=titanic_df, hue='person')


# In[ ]:


sns.lmplot('Age', 'Survived', data=titanic_df)


# In[ ]:


sns.lmplot('Age', 'Survived', data=titanic_df, hue='Pclass', palette='winter')


# In[ ]:


generations = [10,20,40,60,80]

sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins = generations)


# In[ ]:


sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins = generations)

1) Did the deck have an effect on the passengers survival rates? 
2) Did having a family member increase the odds of surviving?
# In[ ]:


sns.lmplot('Age', 'Survived', hue='Embarked', data=titanic_df, palette='winter', x_bins = generations)


# In[ ]:


sns.lmplot('Age', 'Survived', hue='Alone', data=titanic_df, palette='winter', x_bins = generations)


# In[ ]:





# In[ ]:





# In[ ]:




