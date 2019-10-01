#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
gender_sub = pd.read_csv('../input/gender_submission.csv')
test = pd.read_csv('../input/test.csv')


# **Training Dataset**

# In[ ]:


#showing the sample of train data 
train.head()


# In[ ]:


# describe the train dataset
train.describe()


# In[ ]:



#checking for null values in data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# We can see here in the heat map Age and Cabin columns have lots of null data.
# What we can do here is either drop the column or fill the null values with average age.
# We cant fill cabin values becouse there isn't any relation between cabin and other columns so we will drop it from the table.

# In[ ]:


# Count of survived and those who don't
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[ ]:


# Those who survived (male /female)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


# survived on basis of class
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


# column has so much null values
train=train.drop('Cabin',axis=1)


# In[ ]:


train.head()


# In[ ]:


sns.countplot(x='SibSp',data=train)


# Below Graph shows the relation between the age of passanger and there class 

# In[ ]:


# Average age and passanger class
plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# Above graph shows that Passangers having class 1 have average age of 37 similarly class 2 average age is 29 and class 3 have age of 24  years.

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
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


train.head()


# We Just filled all the null values with the average age of passangers.

# In[ ]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train = pd.concat([train,sex,embark],axis=1)


# Regression model cant predict on strings therefore we converted the string here to binaries

# In[ ]:


train.head()


# In[ ]:


plt.figure(figsize=(16, 10))
# this graph is showing that there is no null value in dataset
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Now dataset is ready for fitting in algorithm

# 

# **Testing dataset**
# 

# In[ ]:



sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Here, in testing set also we have null values. 
# What we have done with training set we will repeat the same with testing set. 

# In[ ]:


# droping the cabin

test = test.drop('Cabin',axis=1)
#here axis 1 specifies that we are searching for columns if it is 0 then rows.


# In[ ]:


test.head()


# * Now we have to convert Sex and Embarked columns from string to binaries.
# * Fill the age with average values

# In[ ]:


sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test= pd.concat([test,sex,embark],axis=1)


# In[ ]:


test.head()


# In[ ]:


plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')


# We can see here that there is slight deference in average age between training set and testing dataset. We will now impute age on the basis of this new graph.

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 28

        else:
            return 24

    else:
        return Age
test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


test.head()


# In[ ]:


plt.figure(figsize=(16, 10))
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Fare column is having a null value. Better we should fill it with average value rather than droping it.
# 
# For this we have to check is there is any relation between pclass.

# In[ ]:


plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Fare',data=test,palette='winter')
plt.ylim(0,100)


# In[ ]:


def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Fare):

        if Pclass == 1:
            return 60

        elif Pclass == 2:
            return 16

        else:
            return 10

    else:
        return Fare
test['Fare']=test[['Fare','Pclass']].apply(impute_fare,axis=1)


# Now our test set is also ready for fitting in algorithm.

# **MACHINE LEARNING**

# In[ ]:


X_train=train.drop('Survived',axis=1)
X_train.head()


# In[ ]:


y_train=train['Survived']
y_train.head()


# In[ ]:


y_test=gender_sub['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


X_test=test


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, predictions)


# In[ ]:


passid=np.array(list(range(892,1310)))
df = pd.DataFrame({'PassengerId':passid,'Survived':predictions})
df.to_csv('submission.csv',index=False)


# In[ ]:




