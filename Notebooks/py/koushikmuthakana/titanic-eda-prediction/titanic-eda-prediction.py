#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernal mainly worked on representing the analysis of titanic dataset of several parameters using the logistic regression. later I will explore this in other ML techniques, stating with the logistic regression helps to understand and analyse simple. I'm open for all reviews and suggestions on this  kernal. Please review and suggest me for better work.
#            The following kernal contains basic steps for logistic regression
#              1. Import python libraries & Data
#              2. Access Data Quality and  Missing values
#              3. Exploratory Data Analysis
#              4. Logistic Regression

# # 1. Import python libraries & Data
#     
# 

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
sns.set(style='whitegrid',color_codes=True)
import os
#print(os.listdir("../input"))



# In[ ]:


#Reading train CSV Data into DataFrame
train=pd.read_csv('../input/train.csv')

#Reading test CSV Data into DataFrame
test=pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


print('The number of observations in train dataset is {0}'.format(len(train)))


# In[ ]:


test.head()


# In[ ]:


print('The number of observations in test dataset is {0}'.format(len(test)))


# # 2. Data Quality & Missing Value Assessment

# In[ ]:


train.info()
# we have total 891 entries of 12 columns.
# We have null values in Age, Cabin, Embarked attributes 


# In[ ]:


#Checking null values
print(train.isnull().sum())

# % of missing values
print("% of missing values of 'Age' column: {0}".format(round((train.Age.isnull().sum()/len(train))*100)))
print("% of missing values of 'Cabin' column: {0}".format(round((train.Cabin.isnull().sum()/len(train))*100)))
print("% of missing values of 'Embarked' column: {0}".format((train.Embarked.isnull().sum()/len(train))*100))


# ###### Note: Cabin column has more than 70% of missing values so using this variable in prediction may not be useful
# 
# 
# 

# In[ ]:


train_data=train.drop('Cabin',axis=1)
train_data.head()


# ## Visualizing the missing values columns

# In[ ]:


# Missing value columns: 'Age'
train_data['Age'].hist(bins=15,density=True,alpha=0.7)
train_data['Age'].plot(kind='density',color='green')
plt.xlabel('Age')



# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(x='Age',data=train_data)


# In[ ]:


# Missing value column: 'Embarked'
sns.countplot(x='Embarked',data=train_data)


# ##### Filling the missing values in 'Age' and "Embarked" columns

# In[ ]:


# Impute the Age column with median value
print("The median of 'Age' :",train_data.Age.median())
train_data['Age'].fillna(train_data.Age.median(),inplace=True)

#impute the 'Embarked column with most common boarding port'
print("The most common boarding port  :", train_data['Embarked'].value_counts().idxmax())
train_data['Embarked'].fillna('S',inplace=True)


# In[ ]:


# New adjusted training data
print(train_data.info())
#train_data.isnull().sum()

#Top 5 rows of new train data
train_data.head()


# ###  Adding Additional Columns

#  As per the dataset description, both  SibSp and Parch defines family relations,so for simplicity sake, we will combine these two columns as travelling alone or not
# 

# In[ ]:


train_data['TAlone']=[0 if (train_data['SibSp'][i]+train_data['Parch'][i])>0  else 1  for i in range(len(train_data)) ]
train_data.drop(['SibSp','Parch'],axis=1,inplace=True)


# ######  Adding categorical variables 

# In[ ]:


# Adding categorical varibales for  pclass,Sex and Embarked 
train_Data=pd.get_dummies(train_data,columns=['Pclass','Sex','Embarked'])
train_Data.drop(['Name','Ticket','PassengerId','Sex_female'],inplace=True,axis=1)
final_train=train_Data.copy()


# In[ ]:


#Top 5 rows of fianl train data
final_train.head()


# #####  Applying the same changes to the test data
#         1.Imputing the missing values in any
#         2.Deleting the Cabin variable
#         3.Adding categorical variables
#             

# In[ ]:


#Checking the null values
test.isnull().sum()


# In[ ]:


#Imputing the missing values
test.Age.fillna(test.Age.median(),inplace=True)
test.Fare.fillna(test.Fare.value_counts().idxmax(),inplace=True)

# Combining Parch and SibSp variables into TAlone
test['TAlone']=[0 if (test['SibSp'][i]+test['Parch'][i])>0 else 1for i in range(len(test))]

#Adding categorical variables
test_data=pd.get_dummies(test,columns=['Pclass','Sex','Embarked'])

#Drop the un wanted variables
test_data.drop(['Cabin','PassengerId','Name','Ticket','Sex_female','Parch','SibSp'],axis=1,inplace=True)
test_data.isnull().sum()


# In[ ]:


final_test=test_data.copy()
final_test.head()


# # Logistic Regression using sklearn

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X=final_train.drop('Survived',axis=1)
Y=final_train['Survived'].values


# In[ ]:


X.head()


# In[ ]:


logreg=LogisticRegression()
logreg.fit(X,Y)
y_pred=logreg.predict_proba(X)


# In[ ]:


print('AUC score is' ,roc_auc_score(y_score=y_pred[:,1],y_true=Y))

