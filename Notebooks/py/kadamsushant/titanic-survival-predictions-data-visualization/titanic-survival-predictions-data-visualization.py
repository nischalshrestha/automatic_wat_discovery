#!/usr/bin/env python
# coding: utf-8

# My First Attempt in Data science  and Machine learning. I will be working on the Titanic data and with help of data visualization I will be updating the dataset with all logic and coments , Please feel free to comment and upvote if you find useful 
# 
# **Content **
# 1.  Import  neccessary libarary
# 2. Data Analysis & Visualization 
# 3. Data cleaning 
# 4. Value Imputation
# 5. Run differen model with cleaned data 
# 6. Submission
# 
# 
# ** Import  neccessary libarary**

# In[ ]:



import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
get_ipython().magic(u'matplotlib inline')


# Read files for training  &  test data 

# In[ ]:


titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")


# Display data for visual analysis 

# In[ ]:


titanic_train.head()


# ** Data Analysis**
# 
# Check for Correlation between Gender and survival 

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=titanic_train)
sns.factorplot('Sex','Survived', data=titanic_train,size=4,aspect=3)


# By looking at first graph it looks like  male survived more than females , but second graph talks of % of male & female survived and looks 
# like female had more probability to be survive , let see the percentage 

# In[ ]:


titanic_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# As confirmed from second graph female had more survival rate , now lets see Passaenger Class to survival rate 

# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=titanic_train)
sns.factorplot('Pclass','Survived', data=titanic_train,size=4,aspect=3)


# Based on the graph we can say higher your Passenger class the more your survival rate , lets check the percentage value for each class 

# In[ ]:


titanic_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# So your chances of survival rate  is high with higher Passenger class 
# 
# next up is where passenger  boraded , as mention C = Cherbourg, Q = Queenstown, S = Southampton , let's check 

# In[ ]:


sns.countplot(x='Survived',hue='Embarked',data=titanic_train)
sns.factorplot('Embarked','Survived', data=titanic_train,size=4,aspect=3)


# Let  calculate the percentage value 

# In[ ]:


titanic_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# **Data Cleaning **
# 
# First step is to check null values in data set  both training & test 
# 

# In[ ]:


sns.heatmap(titanic_train.isnull(),yticklabels= False,cbar= False,cmap='viridis')


# Looks like Age and Cabin has null value , lets check test data

# In[ ]:


sns.heatmap(titanic_test.isnull(),yticklabels= False,cbar= False,cmap='viridis')


# Looks like Age and Cabin has null value with one fare value missing.
# 
# Before we do something about lets check percentage of value missing for cabin and decide if we can drop the  column 

# In[ ]:





# In[ ]:


#For tarining data 
(float ( sum(pd.isnull(titanic_train['Cabin'])) ) / (sum(pd.isnull(titanic_train['Cabin'])) + sum(pd.notnull(titanic_train['Cabin']))))


# In[ ]:


#For test data
(float ( sum(pd.isnull(titanic_test['Cabin'])) ) / (sum(pd.isnull(titanic_test['Cabin'])) + sum(pd.notnull(titanic_test['Cabin']))))


# As  cabin has 78% values missing make sense for dropping , ticket number can also be dropped as we have class information in another column.

# In[ ]:


#drop values from both 

titanic_train = titanic_train.drop(['Cabin','Ticket'],axis=1)

titanic_test = titanic_test.drop(['Cabin','Ticket'],axis=1)


# In[ ]:


sns.heatmap(titanic_train.isnull(),yticklabels= False,cbar= False,cmap='viridis')


# In[ ]:


sns.heatmap(titanic_test.isnull(),yticklabels= False,cbar= False,cmap='viridis')


# Now Age column has null value lets impute values with below logic, we are looking at  Passenger class  to calculate the mean  and impute the value

# In[ ]:


def impute_age(col):
   age=col[0]
   pclass=col[1]
   
   if pd.isnull(age):
       
       if pclass == 1:
           return  titanic_train[titanic_train["Pclass"]==1].mean()["Age"]
       elif pclass == 2:
           return  titanic_train[titanic_train["Pclass"]==2].mean()["Age"]
       else:
           return  titanic_train[titanic_train["Pclass"]==3].mean()["Age"]
       
   else:
       return age
       


# In[ ]:


titanic_train['Age']= titanic_train[['Age','Pclass']].apply(impute_age,axis=1)
titanic_test['Age']= titanic_test[['Age','Pclass']].apply(impute_age,axis=1)


# Now we know that there is one null value in fare we can  choose to impute or drop the value 

# In[ ]:


titanic_train.dropna(inplace=True)


# Now lets check the null value  for training & test data

# In[ ]:


sns.heatmap(titanic_train.isnull(),yticklabels= False,cbar= False,cmap='viridis')


# In[ ]:


sns.heatmap(titanic_test.isnull(),yticklabels= False,cbar= False,cmap='viridis')


# Looks like we have no null value in training data , we will be  imputing  value for fare in test data  when we are creating category for Fare column
# ** Value Imputation**
# 
# Now next to other column conversion to  number , first Embarked 

# In[ ]:


titanic_full = [titanic_train,titanic_test]
for dataset in titanic_full:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# Next is Gender column 

# In[ ]:


titanic_full = [titanic_train,titanic_test]
for dataset in titanic_full:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# Also we can drop the Name column  

# In[ ]:


titanic_train.drop(['Name'],axis=1,inplace=True)
titanic_test.drop(['Name'],axis=1,inplace=True)



# Next up Age as its a continuos value need to convert to category like  below (age group can be adjusted , based on inetrnet information :) )
# * Baby: 0  
# * Child : 1 
# * Teenager: 2
# * Student: 3
# * Young Adult: 4
# * Adult: 5
# * Senior: 6

# In[ ]:


titanic_full = [titanic_train,titanic_test]
for dataset in titanic_full:    
    dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 11), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 64), 'Age'] = 5
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 6
    dataset['Age'] = dataset['Age'].astype(int)


# Now lets create category for Fare  , lets analyz  data 

# In[ ]:


titanic_train.describe()


# As we can see  we can create three category from it 

# In[ ]:


titanic_full = [titanic_train,titanic_test]
for dataset in titanic_full:
    dataset.loc[ pd.isnull(dataset['Fare']), 'Fare'] = 0
    dataset.loc[ dataset['Fare'] <= 7.89, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.89) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# Final Check before we built our Machine Learning Models 

# In[ ]:


titanic_train.head()


# **Run differen model with cleaned data **
# 
# 
# Lets prepare training & test data

# In[ ]:


X_train = titanic_train.drop(["Survived" ,"PassengerId"], axis=1)
Y_train = titanic_train["Survived"]
X_test  = titanic_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# **Logistic Regression**

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# **KNN**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# **Gaussian Naive Bayes**

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# **Random Forest**

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# **Decision Tree**

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# **Gradient Boosting Classifier**

# In[ ]:


gbk = GradientBoostingClassifier()
gbk.fit(X_train, Y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(gbk.score(X_train, Y_train) * 100, 2)
acc_gbk


# **Submission**
# 
# Based on the score I have selected Random forest for submission 

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# **Sources**
# 
# https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner
# 
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# All feedback is welcome 

# 
