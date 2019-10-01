#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[ ]:


get_ipython().magic(u'matplotlib inline')

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt


# 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# 

# In[ ]:


# Get an overview of the data

print(train.info())
print(test.info())


# 

# In[ ]:


# Now we will have a look at individual features to see if everything's fine

print(train.PassengerId.describe())
print(train.Name.head(20))


# 

# In[ ]:


# View the Passenger class
print(train.Pclass.value_counts())
plt.hist(train.Pclass,bins=3)
plt.show()


# 

# In[ ]:


# Check the Sex column
print(train.Sex.value_counts())


# 

# In[ ]:


# Next check the Age column
print(train.Age.dropna().describe())
plt.hist(train.Age.dropna(),bins=50,histtype='bar')
plt.show()


# 

# In[ ]:


# Next see the number of siblings/spouse on-board
print(train.SibSp.describe())
plt.hist(train.SibSp,bins=8)
plt.show()


# 

# In[ ]:


# Have a look at number of parent/children on-board
print(train.Parch.describe())
plt.hist(train.Parch,bins=6)
plt.show()


# 

# In[ ]:


# Have a look at the ticket number and Cabin

print(train.Ticket.head())
print(train.Cabin.head())

print(train.Ticket.describe())
print(train.Cabin.describe())


# 

# In[ ]:


print(train.Fare.describe())

fig = plt.figure()

ax1 = fig.add_subplot(131)
ax1.boxplot(train.Fare[train.Fare<100])

ax2 = fig.add_subplot(132)
ax2.boxplot(train.Fare[train.Fare<300])

ax3 = fig.add_subplot(133)
ax3.boxplot(train.Fare)

plt.show()


# 

# In[ ]:


print(train.Embarked.value_counts())


# 
