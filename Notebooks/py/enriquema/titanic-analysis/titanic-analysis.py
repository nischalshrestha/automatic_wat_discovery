#!/usr/bin/env python
# coding: utf-8

# In[53]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[54]:


train = pd.read_csv('../input/train.csv')


# In[55]:


print(train)


# # Categorical and Numerical variables analysis
# It's important analyze all the variables to reduce dimensionality of dataset, so we will analyze all the variables an how much representative are to the target (Survived).
# Columns to analyze:

# In[56]:


list(train)


# First we gonna study the variables over the target(Survived), starting for "PassengerId"

# In[57]:


import matplotlib.pyplot as plt
plt.scatter(train['PassengerId'], train['Survived'])
plt.xlabel('PassengerId')
plt.ylabel('Survived')
plt.show()


# We will drop this variable because there are not corelation between "PassengerId" and "Survivla" variables.

# In[58]:


train = train.drop(['PassengerId'], axis=1)


# The next variable are Pclass, that means the Ticket class and can have 1,2 or 3 as a value, that means:
# 	1 = 1st
#     2 = 2nd
#     3 = 3rd

# In[59]:


plt.scatter(train['Pclass'], train['Survived'])
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()


# Seeing the plot we cannot abstract anything about the graphic, but the variable is categorical so we will let the variable like is for now.

# Now we can see the relation between "Name" and "Survived"

# In[60]:


plt.scatter(train['Name'], train['Survived'])
plt.xlabel('Name')
plt.ylabel('Survived')
plt.show()


# Like with "PassengerId" we can drop this variable because is not representative to the "Survived" variable

# In[61]:


train = train.drop(['Name'], axis=1)


# Study of the "Sex" variable

# In[62]:


plt.scatter(train['Sex'], train['Survived'])
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.show()


# Like with "Pclass" we cannot see anything representative, but for the values we can catalog this variable like categorical.

# Study of "Age" variable

# In[63]:


plt.scatter(train['Age'], train['Survived'])
plt.xlabel('Ticket')
plt.ylabel('Survived')
plt.show()


# Looking the plot we can see that the aren't a direct relation between this variable over "Survived", but this variable represent numerical values so we can do it in the future cluster to categorize this posible values, in first way we don't do it any modification to understand how many precision we can be.

# Study of the SibSp variable

# In[64]:


plt.scatter(train['SibSp'], train['Survived'])
plt.xlabel('SibSp')
plt.ylabel('Survived')
plt.show()


# With this scatter plot we can see the distribution and we can predice that all that have more than 5 brother will die, so we can transform this variable to one like "have less than 5 brother" that represent 0 if have it or 1 if don't

# In[65]:


train['HaveLessThanFiveBrothers'] = ''
train['HaveLessThanFiveBrothers'][train['SibSp'] < 5] = 0
train['HaveLessThanFiveBrothers'][train['SibSp'] >= 5] = 1
train = train.drop(['SibSp'], axis=1)


# Study of "Parch"

# In[66]:


plt.scatter(train['Parch'], train['Survived'])
plt.xlabel('Parch')
plt.ylabel('Survived')
plt.show()


# In first way this variable dont say anything clear, so like with "Age" we will keep this variable like is to understand how important is to the final model.

# Study "Ticket" variable
# We can supose ticket, that it is the ticket number, cannot apport anything to our model, so we must verify if tis import or not.

# In[67]:


plt.scatter(train['Ticket'], train['Survived'])
plt.xlabel('Ticket')
plt.ylabel('Survived')
plt.show()


# As we can see, the plot dont show any clear distribution, so we can drop this variable because the variable are not representative to the model.

# In[68]:


train = train.drop(['Ticket'], axis=1)


# Study "Fare" variable

# In[69]:


plt.scatter(train['Fare'], train['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()


# This variable represent the cost of the fare, we can define cluster or range of fare but in first way we will not do it anything as we do it with "Age".

# Study "Cabin" variable
# First we will show the posible values of Cabin, because we cannot do it a graph.

# In[70]:


train['Cabin'].unique()


# As we can see, there are a lot of kind of posible values so we will transform this to get the word that represent the cabin. For this we will create a new variable call "Sector" and will replace the number to anything to get the posible values and with the purpose of represent the distributtion between this variable over the "Survival" variable, we will convert the new variable to "string".

# In[71]:


train['Sector'] = train['Cabin'].str.replace('[0-9]+', '').fillna('Z').astype(str)


# In[72]:


train['Sector'].unique()


# As we can see the posible values are less than before the transformation, so now represent it!

# In[73]:


plt.scatter(train['Sector'], train['Survived'])
plt.xlabel('Sector')
plt.ylabel('Survived')
plt.show()


# We can see that the distribution dont apport us a lot of information, just that some groups survive like "F G" or "D D", I think that this variable have a lot of dispersion and is not so much descriptive because in so many cause can have values like "B B" or "B B B" or "C C"... and this is not valid because one person cannot be hosted in more than 1 cabin.
# 
# This is a supose, so like i work the variable and can be identify for many cause that depending of the sector can survive or not, we will keep the transformation of "Cabin" variable and drop the "Cabin" variable because the dispersion that have.

# In[74]:


train = train.drop(['Cabin'], axis=1)


# Study "Embarked" variable

# In[75]:


plt.scatter(train['Embarked'], train['Survived'])
plt.xlabel('Embarked')
plt.ylabel('Survived')
plt.show()


# As we can see all that don't embarked or there are not registry about his embarked are alive, so we can transform this variable to know who is embarked or not. 
# For that we gonna transform the variable "Embarked" to the variable "isEmbarked" that will have 1 if the person is embarked or 0 if not.
# 
# Finally we will delete the "Embarked" variable.

# In[84]:


train['isEmbarked'] = train['Embarked'].str.replace('C|Q|S', '1').fillna(0)

# Remove Embarked variable
train = train.drop(['Embarked'], axis=1)


# In[85]:


plt.scatter(train['isEmbarked'], train['Survived'])
plt.xlabel('isEmbarked')
plt.ylabel('Survived')
plt.show()


# # First Classification
# Finally we have our dataset as you can see with all the transformations nada clean variables, so now we must do it the clasificator to know how good are our transformation to predict who will survival or not.

# In[88]:


list(train)


# It's important prepare all our data and do it the transformation in TestDataset like we do it in train, so let's start!

# In[89]:


#Charge test
test = pd.read_csv('../input/train.csv')


# In[ ]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[87]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# 
