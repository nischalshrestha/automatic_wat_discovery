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
import re
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()

# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Check the head of training data set
df_train.head()


# In[ ]:


print (df_train.info())


# I will drop the 'Survived' from the training dataset and create a new DataFrame data that consists of training and test sets combined. To have data cohrence for reference will store the target variable of the training data for safe keeping.

# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets to create a new Merged_data
Merged_data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# Check info Merged_data 
Merged_data.info()


# In[ ]:


#Below gives visual indicator of columns which are having Null values
# Generate a custom diverging colormap
cmap = sns.diverging_palette(90, 980, as_cmap=True)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False, cmap=cmap)


# The above graph indicates that Age and Cabin have most null values in Merged data set. Even from Merged data set info we can see.
# 
# Out of total 1309 rows 
# Age has 1946 not null values
# 
# 

# In[ ]:


#Age distribution is shown in below graph
sns.set_style('whitegrid')
df_train['Age'].hist(bins=35)
plt.xlabel('Age')


# In[ ]:


#Fare distribution is shown in below graph
sns.set_style('whitegrid')
df_train['Fare'].hist(bins=40)
plt.xlabel('Fare')


# In[ ]:


sns.jointplot(x='Fare',y='Age',data=df_train)


# In[ ]:


#Below graph indicates that Female survived in grater proportion as compared to male.. 
sns.countplot(x="Survived",hue="Sex",data=df_train)


# In[ ]:


# Below number shows that 74% of woman survived where as only 18% of men survived
df_train.groupby(['Sex']).Survived.sum()/df_train.groupby(['Sex']).Survived.count()


# In[ ]:


df_train.groupby(['Sex','Pclass']).Survived.sum()/df_train.groupby(['Sex','Pclass']).Survived.count()


# In[ ]:


# Merged data counts by values which are grouped by Class and Sex
Merged_data.groupby(['Pclass','Sex']).count()


# In[ ]:


#LEt us impute values for Age values (Average of 1st class is 38 , average age 2nd class 29 and 3rd class is 24)
plt.figure(figsize=(10,5))
sns.boxplot(x='Pclass',y='Age',data=df_train)

# I will try to create function for imputation of age
# Fill the missing numerical variables
#Merged_data['Age'] = Merged_data.Age.fillna(Merged_data.Age.median())
#Merged_data['Fare'] = Merged_data.Fare.fillna(Merged_data.Fare.median())


# In[ ]:


# Function to fill the age of blank values
def Fill_Age(Cols):
    Age = Cols[0]
    Pclass = Cols[1]

    if pd.isnull(Age) :
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


Merged_data['Age']=Merged_data[['Age','Pclass']].apply(Fill_Age,axis=1)


# In[ ]:


# Cabin has too many blank values, Ticket and cabin are not useful so better to drop these columns
Merged_data.drop(['Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


Merged_data.head()


# In[ ]:


#This is method by which Age can be categorised in categories
Merged_data['CatAge'] = pd.qcut(Merged_data.Age, q=4, labels=False )
Merged_data['CatFare'] = pd.qcut(Merged_data.Age, q=5, labels=False )


# In[ ]:


#Let us also put numerical values for Sex and Embarked
Gender = pd.get_dummies(Merged_data['Sex'],drop_first=True,prefix='Gender')
Embarked = pd.get_dummies(Merged_data['Embarked'],drop_first=True,prefix='Embarked')


# In[ ]:


# LEt us concat the above dummy values with Merged DAta Frame
Modified_data =pd.concat([Merged_data,Embarked,Gender],axis=1)
Modified_data.head()


# In[ ]:


# Embarked , Age, Sex, Name and Fare
Modified_data.drop(['Embarked','Age','Sex','Name','Fare'],axis=1,inplace=True)
Modified_data.head()


# In[ ]:


cmap = sns.diverging_palette(90, 980, as_cmap=True)
sns.heatmap(Modified_data.isnull(),yticklabels=False,cbar=False, cmap=cmap)


# There is no blank values in the Modified data set. For now  passanger id should not have any relatation to survival so should drop that column as well.

# In[ ]:


Modified_data.drop(['PassengerId','SibSp'],axis=1,inplace=True)
Modified_data.head()


# In[ ]:


Modified_data.head()


# Going to use a Decision Tree Classifier... But before fitting the model, let me split the data back in training and test data set  which is 891 rows of training and rest test data set.
# 

# In[ ]:


data_train = Modified_data.iloc[:891]
data_test = Modified_data.iloc[891:]


# In[ ]:


X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


# Make predictions and store in 'Survived' column of 
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('Decision_Tree_Classification_Fare.csv', index=False)

