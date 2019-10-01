#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#Importing all the needed libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Importing the Training set
train=pd.read_csv('../input/train.csv')


# In[4]:


#Visualizing the null values using HeatMaps
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='YlGnBu')


# In[5]:


#Visualizing Survivors based on their Passenger Classes
sns.countplot(x='Survived',hue='Pclass',data=train)


# In[6]:


#Plotting on the bases of the Age of the Passengers
sns.distplot(train['Age'].dropna(), kde=False, bins=30)


# In[7]:


#Counting the number of Passengers who had boarded with their siblings and/or their spouses
sns.countplot(x='SibSp', data=train)


# In[8]:


#Plotting Fare against the number of Passengers
train['Fare'].plot.hist(bins=40, figsize=(10,4))


# In[9]:


#Getting the Age of the Passengers on the basis of their Class, also the average Age per class
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)


# In[10]:


#Defing a  function that will impute the Age columns on the basis of the Pclass
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[11]:


#Calling the above defined function to impute the Age column
train['Age']=train[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[12]:


#Visualizing after the Imputation of the Age column
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


# In[13]:


#Dropping the Cabin column since it doesn't have any direct effect on the prediction
train.drop('Cabin', axis=1, inplace=True)


# In[14]:


#Checking the dataset
train.head()


# In[15]:


#Using HeatMap again to visualize the resultant set
plt.figure(figsize=(10,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


# In[16]:


#Dropping the remaining rows with null values in them
train.dropna(inplace=True)


# In[17]:


#Dealing with categorical column Sex, Embarked, making dummmies
sex=pd.get_dummies(train['Sex'], drop_first=True)
embark=pd.get_dummies(train['Embarked'], drop_first=True)
classes=pd.get_dummies(train['Pclass'])


# In[18]:


#Concatinating the newly created dummy columns with the existing dataframe
train=pd.concat([train,sex,embark], axis=1)
train=pd.concat([train,classes], axis=1)


# In[19]:


#Dropping the those columns which will not be used during training
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)
train.drop('Pclass', axis=1, inplace=True)


# In[20]:


# We are done with cleaning of the training set. We now need to do the same to the Test set


# In[21]:


#Importing the Test set
test=pd.read_csv('../input/test.csv')


# In[22]:


#Visulazing missing values using HeatMap
plt.figure(figsize=(18,12))
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='YlGnBu')


# In[23]:


#Using BoxPlot to see which age group belongs to which Pclass
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=test)


# In[24]:


#Defining a function which will be used to impute Age columns of the Test set
def impute_age2(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 42
        elif Pclass==2:
            return 26
        else:
            return 24
    else:
        return Age


# In[25]:


#Imputing the Age column by calling the above function        
test['Age']=test[['Age', 'Pclass']].apply(impute_age2, axis=1)


# In[26]:


# As we saw in the heatmap that some of the values of 'Fare' column are missing
#Imputing Fare column for missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(test.iloc[:, 8:9])
test.iloc[:, 8:9] = imputer.transform(test.iloc[:, 8:9])


# In[27]:


#Using HeatMap again to visualize the remaining missing values from the set
plt.figure(figsize=(10,7))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


# In[28]:


# The 'Cabin' column has too many values missing, so cannot perform imputation here.
#Dropping the Cabin column as done in the Training set
test.drop('Cabin', axis=1, inplace=True)


# In[29]:


#Visualizing again to see the effect after Imputation
plt.figure(figsize=(10,5))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')


# In[30]:


#Creating Dummy variables for categorical feaatures of the set and concatinating them to the Test set
sex2=pd.get_dummies(test['Sex'], drop_first=True)
embark2=pd.get_dummies(test['Embarked'], drop_first=True)
test=pd.concat([test,sex2,embark2], axis=1)
pclasses=pd.get_dummies(test['Pclass'])
test=pd.concat([test,pclasses], axis=1)


# In[31]:


#Saving PassengerId before dropping it so that we can add it to the resultant csv file later
result=test.iloc[:,0]


# In[32]:


#Dropping the redundant features which were converted to dummies earlier
test.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
test.drop('Pclass', axis=1, inplace=True)


# ## Building a Logistic Regression Model

# In[33]:


#Train Test Split
x_train=train.iloc[:,1:]
y_train=train.iloc[:,0:1]
x_test=test.iloc[:,:]


# ## Training and Predicting 

# In[34]:


from sklearn.linear_model import LogisticRegression
logisticReg=LogisticRegression()
logisticReg.fit(x_train,y_train)

y_pred= logisticReg.predict(x_test)


# ## Calculating accuracy of the model

# In[35]:


accuracy = round(logisticReg.score(x_train, y_train) * 100, 2)


# In[36]:


print(accuracy)


# ## Writing the predictions to a csv file

# In[38]:


df=pd.DataFrame(dict(PassengerId = result, Survived = y_pred)).reset_index()
df.drop('index', axis=1, inplace=True)

df.to_csv('logresult.csv', index=False)


# # Using Random Forest Classifier

# In[39]:


from sklearn.ensemble import RandomForestClassifier
ranFor = RandomForestClassifier(n_estimators = 70)


# # Training and Predicting

# In[40]:


ranFor.fit(x_train,y_train)
y_pred2= ranFor.predict(x_test)


# # Calculating accuracy of the model

# In[41]:


accuracy2 =round(ranFor.score(x_train, y_train)*100,2)


# In[42]:


print(accuracy2)


# # Writing the predictions to a csv file

# In[43]:


df=pd.DataFrame(dict(PassengerId = result, Survived = y_pred2)).reset_index()
df.drop('index', axis=1, inplace=True)

df.to_csv('rfcresult.csv', index=False)


# # Using Support Vector Machine

# In[44]:


from sklearn.svm import SVC
svc=SVC()


# # Training and Predicting

# In[45]:


svc.fit(x_train, y_train)

y_pred3=svc.predict(x_test)


#  ## Calculating accuracy of the model

# In[46]:


accuracy3=round(svc.score(x_train, y_train)*100,2)


# In[47]:


print(accuracy3)


# 
# ## Writing the predictions to a csv file

# In[48]:


df=pd.DataFrame(dict(PassengerId = result, Survived = y_pred2)).reset_index()
df.drop('index', axis=1, inplace=True)

df.to_csv('svcresult.csv', index=False)


# ### Obviously we see that the Random Forest Classifier gave more accurate predictions.

# In[ ]:




