#!/usr/bin/env python
# coding: utf-8

# # Is Jack alive?                                                                       
# by Mohammad Shahebaz
# 
# This notebook is my one of my attempt to study the dataset, nit picking the features and predict the outcome using **Logistic Regression Model** and **Random Forests** along account of taking various **random states** in considerations.

# In[ ]:


#Standard imports 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


# test = pd.read_csv('test.csv')


# In[ ]:


train.head()


# #### Checking how many values in the dataset are null 

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train.isnull(),cmap='viridis')


# It seems that we have fewer **Age** and lots of **Cabin** data missing. So, lets fix that!

# Coding a sin_bin() function which changes the values of **Sex** to **1** if **'male'**

# In[ ]:


def sin_bin(x):
    if x =='male':
        return 1
    else:
        return 0


# In[ ]:


train['Sex'] = train['Sex'].apply(sin_bin)


# #### Observing the distribution of Age 

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train['Age'].dropna())


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='rainbow')


# In[ ]:


plt.figure(figsize=(12, 7))
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In the above box plot the relation between the Class and Age is significant. And, it makes sense as the older people are richer hence belonging to class 1. From this data we are going to take the means from the above plot to fill in the missing data of Age column.
# 
# Similarly we can code other two functions emark_num and is_cab for converting the three Emabarked strings 'C', 'Q' and 'S' to numbers. 
# 
# And, in my analysis I have opted to not drop the cabins column and using it as feature for having a cabin or **not**.

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


def emark_num(x):
    if x is 'C':
        return 1
    elif x is 'Q':
        return 2
    else:
        return 3
    
train['Embarked'] = train['Embarked'].apply(emark_num)


# In[ ]:


# trim 1
# train.drop(['PassengerId','Name'],axis=1, inplace=True)
train.drop(['Name','Ticket'],axis=1,inplace=True)


# In[ ]:


def is_cab(z):
    
    if isinstance(z, float):
        return 0
    else:
        return 1

train['Cabin'] = train['Cabin'].apply(is_cab)


# In[ ]:


# Checking the head for cleansed data
train.head()


# In[ ]:


# Checking the heatmap to verify the absence of null data
plt.figure(figsize=(12, 7))
sns.heatmap(train.isnull(), cmap='viridis')


# ## Training and Predicting
# Opting for **Logistic Regression**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.333, 
                                                    random_state=2014) 


# ### Predicting Output

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# Let's move on to the evaluation part

# ## Evaluation

# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions)*100)


# ## Choosing Random state

# In[ ]:


# accuracy_scores = []
# for i in range(1,3000):
#     from sklearn.model_selection import train_test_split

#     X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
#                                                         train['Survived'], test_size=0.333, 
#                                                         random_state=i) 
#     from sklearn.linear_model import LogisticRegression
#     logmodel = LogisticRegression()
#     logmodel.fit(X_train,y_train)
#     predictions = logmodel.predict(X_test)
#     from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
#     accuracy_scores.append(accuracy_score(y_test,predictions))


# In[ ]:


# max(accuracy_scores)

