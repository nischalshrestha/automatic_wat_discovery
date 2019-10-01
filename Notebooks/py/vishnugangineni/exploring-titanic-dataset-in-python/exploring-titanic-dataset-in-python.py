#!/usr/bin/env python
# coding: utf-8

# #Exploring Titanic dataset in Python

# In this notebook, I will be exploring the Titanic dataset using pandas(for data ingestion and cleaning), matplotlib(for visualizations) and scikit-learn(for machine learning). 

# ### Importing data and creating dataframes

# In[ ]:


#Importing pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Creating train and test dataframes 
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# In[ ]:


#First 5 rows of the training data
train.head()


# In[ ]:


#First 5 rows of the test data
test.head()


# In[ ]:


#Merging train and test dataframes into a single dataframe
titanic=pd.concat([train,test],axis=0,ignore_index=True)


# In[ ]:


#Total number of values in each column
titanic.count()


# ### Exploratory Data Analysis

# This part would help us in identifying patterns within the data and thus in extracting, creating the most relevant features.

# In[ ]:


#Converting the unordered categorical 'Sex'
titanic.Sex=titanic.Sex.map({'male':1,'female':0})


# In[ ]:


#Total number of passengers and survivors by gender
train.groupby('Sex')['Survived'].agg({'Sex':['count'],'Survived':['sum']})


# 233 survivors out of 314 female passengers and
# 109 survivors out of 577 male passengers.

# In[ ]:


titanic.loc[train.index].info() #Training set Info


# In[ ]:


titanic.loc[train.index].describe() #Properties of the training set


# In[ ]:


#Creating a new feature 'has_Family'
#'has_Family' tells if a passenger is part of a family or not
titanic['has_Family']=((titanic.Parch!=0) | (titanic.SibSp!=0)).map({True:1,False:0})


# In[ ]:


titanic.has_Family.value_counts()


# 519 Passengers(from the entire dataset) are travelling with their families.

# In[ ]:


#Does travelling with family increase the survival chances?
titanic.loc[train.index].groupby('has_Family')['has_Family','Survived'].agg({'has_Family':['count'],'Survived':['sum']})


# From the training set,
# out of 354 people travelling with their families 179 survived. Out of 537 people travelling without families, 163 survived.

# In[ ]:


#Visualizing survival by has_Family
titanic.loc[train.index].Survived.hist(by=titanic.has_Family.map({0:'Without Family',1:'With Family'}),layout=(2,1),sharex=True)
plt.xticks([0,1],['Did not survive','Survived'])


# In[ ]:


#Visualizing survival by Pclass
train.Survived.hist(by=train.Pclass,layout=(3,1),sharex=True)
plt.xticks([0,1],['Did not survive','Survived'])


# In[ ]:


titanic.loc[train.index].Survived.hist(by=titanic.Embarked,layout=(3,1),sharex=True)
plt.xticks([0,1],['Did not survive','Survived'])


# In[ ]:


#Importing colormap
import matplotlib.cm as cm


# In[ ]:


#Scatterplot to visualize relationship between Age,Pclass and Survival
titanic.loc[train.index].plot(x='Age',y='Survived',c='Pclass',cmap=cm.hot,kind='scatter',figsize=(10,5))
plt.yticks([0,1])
plt.xticks(range(10,101,10))


# In[ ]:


#Visualizing age distribution
train[train.Survived==1].Age.hist(bins=10,normed=True)


# In[ ]:


#Embarked has two missing values -> Deleting those two rows
titanic.dropna(subset=['Embarked'],inplace=True)


# In[ ]:


#Imputing missing Age values with the mean Age
titanic.Age=titanic.Age.fillna(value=train.Age.mean())


# In[ ]:


#Since Cabin doesn't seem like an important feature, we're dropping it
titanic.drop('Cabin',axis=1)


# In[ ]:


#Using get_dummmies() method to convert Embarked into variables that can be used as features 
embarked_dummies=pd.get_dummies(titanic.Embarked,prefix='Embarked')


# Embarked is a non-binary unordered categorical.

# In[ ]:


#Adding embarked_dummies to the titanic dataframe
titanic=pd.concat([titanic,embarked_dummies],axis=1)


# In[ ]:


#Data with the new columns
titanic.head()


# ### Predicting Survivors using Logistic Regression

# In[ ]:


#Selecting our features
features=['Pclass','Sex','Age','has_Family','Embarked_C','Embarked_Q','Embarked_S']


# In[ ]:


#Training set
X_train=titanic[titanic.Survived.notnull()][features]  #Features
y_train=titanic[titanic.Survived.notnull()].Survived  #Response


# In[ ]:


#Test set
X_test=titanic[titanic.Survived.isnull()][features]


# In[ ]:


#Using Logistic Regression for classification
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)


# In[ ]:


#Prediction
y_pred_class=logreg.predict(X_test)


# Thank you for going through the notebook. Comments and suggestions are welcome :)
