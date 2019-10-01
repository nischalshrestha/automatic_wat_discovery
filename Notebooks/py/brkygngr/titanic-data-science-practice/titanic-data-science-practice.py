#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# In[ ]:


#Read data
data = pd.read_csv('../input/train.csv')


# Get column names in data

# In[ ]:


data.columns


# In[ ]:


data.columns.size


# In[ ]:


data.shape


# Turn all column names to lower case

# In[ ]:


data.columns = [column.lower() for column in data.columns]
data.columns


# In[ ]:


#I could not understand what some columns name represents so I changed them.
data = data.rename(
    columns = {
        'passengerId': 'id',
        'pclass': 'ticketClass',
        'sibsp': 'sisBroSpo',
        'parch': 'parentChild'
    }
)


# In[ ]:


#More understandable.
data.columns


# Get type's in data

# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# Numeric values inside the data.

# In[ ]:


data.describe()


# In[ ]:


#From output of describe method above it looks like;
#More people were travelling with ticketClass 3 then other groups
#Most people have no sisBroSpo(sibsb) aboard
#Most people have no parentChild(parch) aboard
#sisBroSpo and parentChild could be connected?
data.plot()
plt.show()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.boxplot(column = ['age', 'sisBroSpo', 'parentChild'])
plt.show()


# In[ ]:


data.plot(kind = 'scatter', x = 'sisBroSpo', y = 'parentChild')
plt.show()


# 491 people had  ticket class 3, 184 people had 2 and 216 people had 1

# In[ ]:


#Ticket class number's
print(data.ticketClass.unique())
print(data.ticketClass.value_counts())


# In[ ]:


pd.melt(
    frame = data, 
    id_vars = 'age', 
    value_vars = ['sisBroSpo', 'parentChild']
).head(10)


# In[ ]:


pd.melt(
    frame = data, 
    id_vars = 'age',
    value_vars = ['sisBroSpo', 'parentChild']
).tail(10)


# In[ ]:


filtered = data[(data.age >= 1) & (data.survived == 1)]
filtered.loc[filtered.age == filtered.age.max()]


# ^ Oldest surviver who's age is above 1

# In[ ]:


filtered = data[(data.age >= 1) & (data.survived == 0)]
filtered.loc[filtered.age == filtered.age.max()]


# ^ Oldest person to die who's age is above 1

# In[ ]:


filtered = data[(data.age <= 1) & (data.survived == 1)]
filtered.loc[filtered.age == filtered.age.min()]


# ^ Youngest person who survived

# In[ ]:


filtered = data[(data.age <= 1) & (data.survived == 0)]
filtered.loc[filtered.age == filtered.age.min()]


# ^ Youngest person who did not survived.

# In[ ]:


#data.loc[:,['sisBroSpo', 'parentChild']].plot()
data.loc[:,['sisBroSpo', 'parentChild']].plot(subplots = True)
plt.legend(loc = 'upper right')
plt.show()


# People above age 50 features

# In[ ]:


data[data.age > 50].plot()
plt.show()


# In[ ]:


data[data.age > 50].plot(subplots = True)
plt.show()


# In[ ]:


data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'parentChild',
    color = 'yellow'
)
plt.show()


# In[ ]:


data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'sisBroSpo',
    color = 'yellow'
)
plt.show()


# People below age 50 features

# In[ ]:


data[data.age < 50].plot()
plt.show()


# In[ ]:


data[data.age < 50].plot(subplots = True)
plt.show()


# In[ ]:


data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'parentChild',
    color = 'yellow'
)
plt.show()


# In[ ]:


data[data.age > 50].plot(
    kind = 'hist', 
    x = 'age', 
    y = 'sisBroSpo',
    color = 'yellow'
)
plt.show()


# In[ ]:


data.sort_values('age').age.unique()


# In[ ]:


keep = ['name', 'age', 'sisBroSpo', 'parentChild', 'survived']
ageAboveFifty = data[data.age > 50].filter(items = keep).head()
ageBelowFifty = data[data.age < 50].filter(items = keep).head()


# In[ ]:


ageAboveFifty = ageAboveFifty.sort_values('age')
ageAboveFifty


# In[ ]:


ageBelowFifty = ageBelowFifty.sort_values('age')
ageBelowFifty


# In[ ]:


ageAboveFifty.where(ageAboveFifty.values != ageBelowFifty.values)


# In[ ]:


data.describe()


# In[ ]:


#Find percentage of people surviving if they are above 28
ageAbove28 = data[
    (data.age >= 28) & (data.survived == 1)
].age.count() / data[(data.survived == 1)].age.count()


# In[ ]:


#Find percentage of people surviving if they are below 28
ageBelow28 = data[
    (data.age < 28) & (data.survived == 1)
].age.count() / data[(data.survived == 1)].age.count()


# In[ ]:


ticketOne = data[
    (data.ticketClass == 1) & (data.survived == 1)
].ticketClass.count() / data[(data.survived == 1)].ticketClass.count()


# In[ ]:


ticketTwo = data[
    (data.ticketClass == 2) & (data.survived == 1)
].ticketClass.count() / data[(data.survived == 1)].ticketClass.count()


# In[ ]:


ticketThree = data[
    (data.ticketClass == 3) & (data.survived == 1)
].ticketClass.count() / data[(data.survived == 1)].ticketClass.count()


# In[ ]:


fareAbove = data[
    (data.fare >= 14) & (data.survived == 1)
].fare.count() / data[(data.survived == 1)].fare.count()


# In[ ]:


fareBelow = data[
    (data.fare < 14) & (data.survived == 1)
].fare.count() / data[(data.survived == 1)].fare.count()


# In[ ]:


pSurvivor = data[(data.survived == 1)].passengerid.count() / data.passengerid.count()
pSurvivor


# In[ ]:




