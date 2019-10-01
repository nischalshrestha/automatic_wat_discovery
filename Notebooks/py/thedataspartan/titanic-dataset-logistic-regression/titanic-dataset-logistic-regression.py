#!/usr/bin/env python
# coding: utf-8

# # Spartan Data Science
# 
# # Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print(train.columns.values)


# In[ ]:


print(test.columns.values)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


# As we can see from the above information of the training dataset 
# that there are a lot of missing values in the Cabin column. 


# In[ ]:


# While the Age column contains a very few missing values which can be 
# substituted by the average age value within each class i.e imputation.


# In[ ]:


# Moreover, only two values missing in the Embarked column.


# # Data Visualization

# In[ ]:


# The following Heatmap will reveal the missing values. 
# White lines indicate the missing values.
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="Blues")


# In[ ]:


# Checking how many survived vs. how many did not with respect to gender.
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


# Checking how many survived vs. how many did not with respect to class.
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


# Checking the distribution of age
sns.distplot(train['Age'].dropna(),kde=False,color='blue',bins=30)


# In[ ]:


# Checking the age groups of the people within each class. 
# Grouped into classes
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


# Plotting people who came in groups or alone
sns.countplot(x = 'SibSp', data = train)


# In[ ]:


# Plotting the Fare column
sns.countplot(x = 'Fare', data = train)


# In[ ]:


# A better representation for the above distribution using pandas
train['Fare'].hist(bins=30,figsize=(10,4))


# In[ ]:


# And lastly, distribution for Parch
sns.countplot(x = 'Parch', data = train)


# # Data Preprocessing
# We'll perform the following tasks:
# 1. Take care of all the missing values
# 2. Convert Categorical Values into Dummy Variables so that the Machine Learning Model can interpret them.
# 3. Take care of the Multicolinearity issue by dropping one column of the dummy variables from each set of dummy variables.
# 

# In[ ]:


# Imputing the Age Column
def AgeImputation(column):
    Age = column[0]
    Pclass = column[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(AgeImputation,axis=1)
test['Age'] = test[['Age','Pclass']].apply(AgeImputation,axis=1)


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap="Blues")


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="Blues")


# In[ ]:


# Dropping the Cabin column because it has too many missing values. Imputing wont give accurate representation for the data.
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="Blues")


# In[ ]:


# Lastly, dealing with the Embarked Column. 
# We're dropping the rows containing null values for any column column in the Training Set
train.dropna(inplace=True) 

# fill (instead of drop) the missing value of Fare with the mean of Fares
# so that there are exactly 418 rows (required for submission)
mean = test['Fare'].mean()
test['Fare'].fillna(mean, inplace=True) 


# In[ ]:


# All missing values have been taken care of.
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="Blues")


# In[ ]:


# All missing values have been taken care of.
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap="Blues")


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


# Let's see what are the current columns
train.head()


# In[ ]:


test.head()


# In[ ]:


# convert categorical variables into dummy/indicator variables
# drop_first drops one column to remove multi-colinearity i.e one or more columns predicting the other
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[ ]:


# dropping the Name and Ticket columns because they have no role in the model training and prediction
# dropping the Sex and Embarked columns to replace them with the new columns with dummy variables
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)


# In[ ]:


train.head()


# In[ ]:


# Since passenger id wont give any information about their survival
train.drop(['PassengerId'],axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


# Repeating the above process for test
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)


# In[ ]:


test.head()


# In[ ]:


# Since passenger id wont give any information about their survival
P_ID = test['PassengerId'] # Saving for later
test.drop(['PassengerId'],axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


train.head()


# In[ ]:


P_ID.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


P_ID


# # Preparing the Dataset for Machine Learning

# In[ ]:


from sklearn.model_selection import train_test_split

X = train.drop('Survived', axis = 1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                    train['Survived'], test_size = 0.30,
                                                    random_state=101)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


survived = logmodel.predict(test)


# In[ ]:


test['Survived'] = survived


# In[ ]:


test['PassengerId'] = P_ID


# In[ ]:


test.info()


# In[ ]:


test[['PassengerId', 'Survived']].to_csv('First_Logistic_Regression.csv', index=False)


# In[ ]:




