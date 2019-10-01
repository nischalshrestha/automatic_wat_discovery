#!/usr/bin/env python
# coding: utf-8

# # ** Titanic: Simple Approach**
# If you are just started in Machine Learning and come up with this Problem and looking for a solution then you are in the right place.
# This Notebook contains a simple approach to tackle the problem. The solution is not the best but it is the simplistic one from which you will get the intuition behind the problem. And can improve for further accuracy.
# 
# We'll be trying to predict a classification- **survival or deceased**.

# ## **Import Libraries**
# Let's import some libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# ## **The Data**
# 
# Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# # **Investigating Data Analysis**
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# From above figure we can see that roughly20 percent of the Age data is missing. We can use a function to fill the age data. The Cabin column missing a lot of data so we can drop that column. The problem with missing data is our model will not take such input.
# 
# *Let's visualize some more data! Get familiar with the data....*

# In[ ]:


plt.figure(figsize=(12, 7))
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12, 7))
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12, 7))
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[ ]:


train.info()
print("***************************************************************")
test.info()


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')


# # **Drop unnecessary Columns**
# 

# In[ ]:


train= train.drop(['PassengerId','Name','Ticket'], axis=1)
test= test.drop(['Name','Ticket'], axis=1)


# # **Converting Categorical Features **
# 

# ## Embarked Column

# In[ ]:


# Get dummy values for both train and test dataset
# Their are 3 values in embrked: C, Q, S
# drop_first = True: Drop C column as it will be redudant because we can identify the emarked column from S and Q.
embark_train = pd.get_dummies(train['Embarked'],drop_first=True)
emark_test = pd.get_dummies(test['Embarked'], drop_first=True)

# Drop Emarked column
train.drop(['Embarked'],axis=1,inplace=True)
test.drop(['Embarked'],axis=1,inplace=True)

# Concat new embark columns in respective datasets
train = pd.concat([train,embark_train],axis=1)
test = pd.concat([test, emark_test], axis=1)


# ## Cabin Column

# In[ ]:


# Drop Cabin attribute from both the dataset
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin", axis=1, inplace=True)


# ## Sex Column
# 
# Sex column contains Male and Female entries. We will just make one column Male (1- for male and 0- for female) entries.

# In[ ]:


sex_train = pd.get_dummies(train['Sex'],drop_first=True)
sex_test = pd.get_dummies(test['Sex'], drop_first=True)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

train = pd.concat([train,sex_train],axis=1)
test = pd.concat([test, sex_test], axis=1)


# # Fare
# only for test, since there is a missing "Fare" values
# 

# In[ ]:


test["Fare"].fillna(test["Fare"].median(), inplace=True)


# ## Age
# We can see the rich passengers in the higher classes tend to be older. We'll use these mean age values to impute based on Pclass for Age.

# In[ ]:


# Function to Impute Age
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


# In[ ]:


# Apply the above function to our training and testing datasets
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)

train['Age'] = train['Age'].astype(int)
test['Age']    = test['Age'].astype(int)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### **Now our Data is cleaned and we are ready to use a Model**

# # **Building a Model**

# ## Train-Test Split

# In[ ]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1)


# # **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
Prediction = lm.predict(X_test)

lm.score(X_train,y_train)


# 

# # **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train, y_train)

Prediction = log.predict(X_test)

log.score(X_train, y_train)


# 

# # **Support Vector Machines**
# 

# In[ ]:


from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)

Prediction_SVM = svc.predict(X_test)

svc.score(X_train, y_train)


# 

# # **KNeighbors Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

Prediction_KNC = knn.predict(X_test)

knn.score(X_train, y_train)


# 

# # **Gaussian Naive Bayes**
# 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Prediction_G = gaussian.predict(X_test)

gaussian.score(X_train, y_train)


# 

# # **Gradient Boosting Classifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gradient_boost = GradientBoostingClassifier(n_estimators=100)
gradient_boost.fit(X_train, y_train)

Prediction_GBC = gradient_boost.predict(X_test)

gradient_boost.score(X_train, y_train)


# 

# # **Random Forest**
# ### Best for the given cleaned Data

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
RFC_prediction = random_forest.predict(X_test)
random_forest.score(X_train, y_train)


# 

# # **For Submission File**

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": RFC_prediction
    })
submission.to_csv('Result_update.csv', index=False)


# 

# # **Score Value**
# 
# ### *The score we get is based on the Training Dataset, it is different when you use it on Test Dataset. After uploading the result.csv file the score is **0.75** which is notable good at the elementary level.*

# # Some suggestions to imrove further:
# - Can grab the tittles from the feature(Mr, Mrs, Dr, etc)
# - Cabin column can be a feature
# 
# 
# ### Any suggestions to improve the score are most welcome.
# 
# 

# ### *For any help feel free to comment.*
# 
# Source: Python for Data Science and Machine Learning Bootcamp(Udemy)
# 

# In[ ]:




