#!/usr/bin/env python
# coding: utf-8

# **Hello!**
# 
# This is just a simple example, using **Random Forest** in the **Titanic dataset**.
# Hope you enjoy, it's only a small try to make a kernel, hehe.
# 
# And, "**May the Force be with You"!**

# In[ ]:


#First, the "import part"
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import os
print(os.listdir("../input"))


# In[ ]:


#Reading the data
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# In[ ]:


#Here I'm removing things that I think that wouldn't be useful
# (of course this is just a supposition, in fact, they are useful if right used)
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace = True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace = True)

#Get_dummyes convert categorical variable into dummy/indicator variables
data_train = pd.get_dummies(train)
data_test = pd.get_dummies(test)

#"Filnna" fills empty values, I'm replacing they for the mean of the values.
# (but, again, there are better ways to do this... this is just an example)
data_train['Age'].fillna(data_train['Age'].mean(), inplace = True)
data_test['Age'].fillna(data_test['Age'].mean(), inplace = True)
data_test['Fare'].fillna(data_test['Fare'].mean(), inplace = True)


# In[ ]:


# Separating the data into variables to analyze (x) and the result (y) that we expect
x = data_train.drop('Survived', axis=1)
y = data_train['Survived']


# In[ ]:


# Parametrizing the method and testing using cross-validation
# You can change the parameters to get better results; ^^
classifier_rf = RandomForestClassifier(
                criterion='gini',
                max_depth=50,
                n_estimators=100,
                n_jobs=-1)
    
scores_rf = cross_val_score(classifier_rf, x, y, scoring='accuracy', cv=5)
print(scores_rf.mean())


# In[ ]:


# Training the model with the data
classifier_rf.fit(x, y)


# In[ ]:


# Creating a submission
submission = pd.DataFrame()
submission['PassengerId'] = data_test['PassengerId']
submission['Survived'] = classifier_rf.predict(data_test) #Here is were the predictions were made! ^^

submission.to_csv('submission.csv', index=False)


# Now just go there and post your submission! ^^
# 
# Of course this was just a simple example, there are other, much better ways to do this.
# 
# **Thanks for reading and have a great day!**
# 
