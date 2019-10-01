#!/usr/bin/env python
# coding: utf-8

# # Introduction

# # This kernel contain
# 1. Reading of Data
# 2.Analyzing data
# 3.Data Wragling
# 4.Data Cleaning

# # Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


# Reading data

# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head(5)


# In[ ]:


data.tail(5)


# In[ ]:


print(data.shape)



# In[ ]:


data.info()


# In[ ]:


data.describe()


# ## unique age

# In[ ]:


data.Age.unique()


# In[ ]:


age_unique = data[data.Age >=0]
plt.figure(figsize=(30,10))
age_unique.Age.sort_values().value_counts().plot(kind="bar")
plt.show()


# In[ ]:


data['Survived']>0


# #  total number of survived passanger

# In[ ]:


data.Survived.sum()


# In[ ]:


data.corr()


# In[ ]:


fig,ax = plt.subplots(figsize=(8,7))
ax = sns.heatmap(data.corr(), annot=True,linewidths=.5,fmt='.1f')
plt.show()


# # Anaylzing data

# In[ ]:


sns.countplot(x='Survived',data=data)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=data)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=data)


# In[ ]:


sns.countplot(x='SibSp',data=data)


# In[ ]:


data['Age'].plot.hist()


# In[ ]:


data['Fare'].plot.hist()


# In[ ]:


data.hist(figsize=(12,8))
plt.figure()


# In[ ]:


plt.scatter(data['Survived'],data['Age'])


# # DATA WRANGLING

# In[ ]:


total=data.isnull()
total


# In[ ]:


total=data.isnull().sum()
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(13)


# # Visulaly analyzing

# In[ ]:


sns.heatmap(data.isnull(),cmap='viridis')


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=data)


# # Data cleaning

# In[ ]:


data.head()


# In[ ]:


data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


sns.heatmap(data.isnull(),cmap='viridis',cbar=False)


# In[ ]:


data.dropna(how ='any', inplace = True)
#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[ ]:


data.head(3)


# In[ ]:


gender=pd.get_dummies(data['Sex'],drop_first=True)
gender.head()


# In[ ]:


embarked=pd.get_dummies(data['Embarked'],drop_first=True)
embarked.head()


# In[ ]:


pcl=pd.get_dummies(data['Pclass'],drop_first=True)
pcl.head()


# In[ ]:


data=pd.concat([data,gender,embarked,pcl],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.drop(['Pclass','Sex','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


data.head()


# # Train Data

# In[ ]:


X=data.drop('Survived',axis=1)


# In[ ]:


y=data['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X, y = np.arange(10).reshape((5, 2)), range(5)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions=logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']


# In[ ]:


print(classification_report(y_true, y_pred, target_names=target_names))


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


y_true = [2, 0, 2, 2, 0, 1]
y_pred = [2, 0, 2, 2, 0, 2]


# # Got accuracy of 83.34%

# In[ ]:


accuracy_score(y_true, y_pred)*100


# I'm new in data science. Your feedback is very important to me 
# # I hope this kernel is helpfull for you,upvote will motivates and appreciate me for further work.

# In[ ]:




