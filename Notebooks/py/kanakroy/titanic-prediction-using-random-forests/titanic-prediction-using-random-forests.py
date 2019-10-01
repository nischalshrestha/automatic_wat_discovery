#!/usr/bin/env python
# coding: utf-8

# # Predicting Survival from Titanic Disaster using Random Forests

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# This analysis provides what sorts of people were likely to survive.

# ## Summary

# This project uses machine learning to predict if a given passenger survived from the Titanic disaster in following steps:-
# 1) Importing the Data and Initial Data Analysis
# 2) Data Visualization for Understanding the Data
# 3) Data Processing Stage
# 4) Predictive Modelling Stage
# 5) Data Visualization for Validation of the Predicted Test Data

# ## Importing the Data and Initial Data Analysis

# In[ ]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing training
df_train=pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


# Importing test data
df_test=pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


# Summary of train data
df_train.describe()


# In[ ]:


df_train.info()


# ## Data Visualization for understanding the Data

# In[ ]:


# NULL Heatmap for Train data
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)


# The heatmap shows that Age and Cabin has NULL values in it. However, we can impute the Age as the number of NULL values are less.

# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=df_train,palette='afmhot')


# The above Countplot shows that more females have survived than males.

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='hsv')


# The above Countplot shows that Class 3 Passengers have least Survival Rate

# In[ ]:


sns.countplot(x='Survived',hue='Parch',data=df_train,palette='rainbow')


# In[ ]:


sns.countplot(x='Survived',hue='SibSp',data=df_train,palette='rainbow')


# In[ ]:


sns.countplot(x='Survived',hue='Embarked',data=df_train,palette='rainbow')


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=df_train)


# In[ ]:


# Calculating Mean Age for each Passenger Class
df_train.groupby('Pclass', as_index=False)['Age'].mean()


# In[ ]:


sns.boxplot(x='Sex',y='Age',data=df_train)


# In[ ]:


# Calculating mean age across sex distribution
df_train.groupby('Sex', as_index=False)['Age'].mean()


# In[ ]:


sns.boxplot(x='Embarked',y='Age',data=df_train)


# In[ ]:


# Since it is evident from the above plots and analysis that Age has well-defined relation with respect to both Sex and Passenger Class.
# Computing mean Age across Sex and Passenger Class - This will be used for imputing the Age. 
df_train.groupby(['Sex','Pclass'])['Age'].mean()


# ## Data Processing Stage

# In[ ]:


# Age Imputation
def ImputeAge(column):
    Age = column[0]
    Sex = column[1]
    Pclass=column[2]

    if pd.isnull(Age):
        if Sex == 'male' and Pclass==1:
            return 41
        elif Sex == 'male' and Pclass==2:
            return 31
        elif Sex == 'male' and Pclass==3:
            return 26
        elif Sex == 'female' and Pclass==1:
            return 35
        elif Sex == 'female' and Pclass==2:
            return 29
        else:
            return 22
    else:
        return Age
    
df_train['Age'] = df_train[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)
df_test['Age'] = df_test[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)


# In[ ]:





# In[ ]:


def ImputeAge(column):
    Age = column[0]
    Sex = column[1]
    Pclass=column[2]

    if pd.isnull(Age):
        if Sex == 'male' and Pclass==1:
            return 41
        elif Sex == 'male' and Pclass==2:
            return 31
        elif Sex == 'male' and Pclass==3:
            return 26
        elif Sex == 'female' and Pclass==1:
            return 35
        elif Sex == 'female' and Pclass==2:
            return 29
        else:
            return 22
    else:
        return Age
    
df_train['Age'] = df_train[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)
df_test['Age'] = df_test[['Age','Sex','Pclass']].apply(ImputeAge,axis=1)


# In[ ]:


# NULL Heatmap to visualize that Age is correctly imputed.
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)


# In[ ]:


# Dropping Cabin from both Test and Train data as we do not have enough data across Cabin to predict Survival
df_train.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)


# In[ ]:


# Heatmap of the processed train data
sns.heatmap(df_train.isnull(),yticklabels=False, cbar=False,cmap='inferno',annot=True)


# In[ ]:


# Logic to replace the NULL Fare in df_test
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True) 


# In[ ]:


df_test.info()


# In[ ]:


# Converting categorical variables into indicator variables
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)


# In[ ]:


# Name and Ticket have no role in the model prediction
# Replacing Sex and Embarked columns with the new indicator variables
df_train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_train = pd.concat([df_train,sex,embark],axis=1)


# In[ ]:


# Repeating the above process for test
sex = pd.get_dummies(df_test['Sex'],drop_first=True)
embark = pd.get_dummies(df_test['Embarked'],drop_first=True)


# In[ ]:


df_test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df_test = pd.concat([df_test,sex,embark],axis=1)


# In[ ]:


df_train.drop(['PassengerId'],axis=1,inplace=True)
Passenger_ID = df_test['PassengerId'] # Saving for later
df_test.drop(['PassengerId'],axis=1,inplace=True)


# In[ ]:


# Fully processed train data
df_train.head()


# In[ ]:


# Fully processed test data
df_test.head()


# In[ ]:


# PassengerId
Passenger_ID


# ## Predictive Modelling Stage

# In[ ]:


# Using Train-Test Split to randomize the data for Predictive modelling
from sklearn.model_selection import train_test_split

x = df_train.drop('Survived', axis = 1)
y = df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Survived',axis=1),df_train['Survived'], test_size = 0.25,random_state=100)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


# In[ ]:


# Decision Tree Classifier
from sklearn import tree

treeclf = tree.DecisionTreeRegressor()
treeclf.fit(x_train,y_train)
acc_tree = round(treeclf.score(x_train, y_train) * 100, 2)
acc_tree


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

ranclf =RandomForestClassifier(n_estimators=20, max_depth=None,min_samples_split=2, random_state=0)
ranclf.fit(x_train,y_train)
acc_ranclf = round(ranclf.score(x_train, y_train) * 100, 2)
acc_ranclf


# In[ ]:


# Predicting Survival values for Test data.
survived=ranclf.predict(df_test)


# In[ ]:


# Feeding PassengerId and Survived into Test data
df_test['Survived']=survived
df_test['PassengerID']=Passenger_ID


# In[ ]:


df_test


# ## Data Visualization for Validation of the Predicted Test Data

# 1) Validating Survival across Sex

# In[ ]:


sns.countplot(x='Survived',hue='male',data=df_train,palette='afmhot')


# In[ ]:


sns.countplot(x='Survived',hue='male',data=df_test,palette='afmhot')


# The above Counplots show that the prediction of Survival across Sex is in sync with the train data.

# 2) Validating Survival across Passenger Class

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=df_train,palette='afmhot')


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=df_test,palette='afmhot')


# The above Counplots show that the prediction of Survival across Passenger Class is in sync with the train data.

# ## Exporting the results to external CSV file

# In[ ]:


df_test[['PassengerID', 'Survived']].to_csv('Titanic_LogRegression.csv', index=False)


# ## Checking accuracy of the algorithm used

# In[ ]:


acc_ranclf = round(ranclf.score(x_train, y_train) * 100, 2)
acc_ranclf

