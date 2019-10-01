#!/usr/bin/env python
# coding: utf-8

# **Titanic disaster problem**
# 
# This is my firts approach trying to solve titanic problem. It uses some models with titanic_train dataset to determine the best model, after that this choosen one is used to predict values with titanic_test.

# In[ ]:


#Import basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


#Metrics and library to implement cross validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#Read data sets
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')


# In[ ]:


#Inspect titanic train
titanic_train.info()


# In[ ]:


#Inspect titanic test
titanic_test.info()


# In[ ]:


#Missing data train
sns.heatmap(data=titanic_train.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


#Missing data test
sns.heatmap(data=titanic_test.isnull(),cmap='viridis',yticklabels=False,cbar=False)


# In[ ]:


#Explore data sets 

#Survived
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titanic_train)
sns.distplot(titanic_train['Survived'],kde=False,rug=True)


# In[ ]:


#Survived and Class
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titanic_train, hue='Pclass')


# In[ ]:


#Survived and Sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=titanic_train, hue='Sex')


# In[ ]:


#Survived and Fare
sns.set_style('whitegrid')
sns.jointplot(x='Survived',data=titanic_train, y='Fare')


# In[ ]:


#Age column in titanic_train
sns.set_style('whitegrid')
sns.distplot(titanic_train['Age'].dropna(),kde=False,bins=30)


# In[ ]:


#Age column in titanic_test
sns.set_style('whitegrid')
sns.distplot(titanic_test['Age'].dropna(),kde=False,bins=30)


# In[ ]:


#Fix missing data in Age column
age_first_class = titanic_train[titanic_train['Pclass']==1]['Age'].mean()
age_second_class = titanic_train[titanic_train['Pclass']==2]['Age'].mean()
age_third_class = titanic_train[titanic_train['Pclass']==3]['Age'].mean()


# In[ ]:


#Function to replace missing values in Age column with the in that passenger's
#class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return age_first_class
        elif Pclass == 2:
            return age_second_class
        else:
            return age_third_class
    else:
        return Age
    
    


# In[ ]:


#Replace missing values in column Age
titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(impute_age,axis=1)
titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


#Let's check the heatmap again 
#first titanic_train
sns.heatmap(data=titanic_train.isnull(),cmap='coolwarm',cbar=False,yticklabels=False)


# In[ ]:


#titanic_test
sns.heatmap(data=titanic_test.isnull(),cmap='coolwarm',cbar=False,yticklabels=False)


# In[ ]:


#Remove cabin column
titanic_train.drop('Cabin',axis=1, inplace=True)
titanic_test.drop('Cabin',axis=1, inplace=True)


# In[ ]:


#Get dummies of categorial columns
sex_train = pd.get_dummies(titanic_train['Sex'],drop_first=True)
embark_train =pd.get_dummies(titanic_train['Embarked'],drop_first=True) 
sex_test = pd.get_dummies(titanic_test['Sex'],drop_first=True)
embark_test = pd.get_dummies(titanic_test['Embarked'],drop_first=True)


# In[ ]:


#Remove categorical columns and other columns that are useless
titanic_train.drop(['PassengerId','Embarked','Sex','Ticket','Name'],axis=1,inplace=True)
titanic_test.drop(['Embarked','Ticket','Name','Sex'],axis=1,inplace=True)

#Concat dummies to the dataset
titanic_train = pd.concat([titanic_train,sex_train,embark_train],axis=1)
titanic_test = pd.concat([titanic_test,sex_test,embark_test],axis=1)


#Fill null value in Fare column
titanic_test['Fare'].fillna(titanic_train['Fare'].median(),inplace=True)


# In[ ]:


#Define training and testing sets
X_train = titanic_train.drop("Survived", axis=1)
Y_train = titanic_train['Survived']


# In[ ]:


#Cross validation
#Split trainint set into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, 
                                                    test_size = 0.3, 
                                                    random_state = 6)


# In[ ]:


#Logistic Regression
logModel = LogisticRegression(random_state=0)
logModel.fit(X_train,y_train)

y_pred = logModel.predict(X_test)
metrics.accuracy_score(y_test,y_pred)


# In[ ]:


#SVM 
svc = SVC()
svc.fit(X_train,y_train)

y_pred_1 = svc.predict(X_test)
metrics.accuracy_score(y_test,y_pred_1)


# In[ ]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

y_pred_2 = dtree.predict(X_test)
metrics.accuracy_score(y_test,y_pred_2)


# In[ ]:


#Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred_3 = rfc.predict(X_test)
metrics.accuracy_score(y_test,y_pred_3)


# In[ ]:


#KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

y_pred_4 = knn.predict(X_test)
metrics.accuracy_score(y_test,y_pred_4)


# In[ ]:


#Predict values using real test set
X_test = titanic_test.drop("PassengerId",axis=1)

#Logistic Regression got the best score
from sklearn.linear_model import LogisticRegression

y_pred = logModel.predict(X_test)


# In[ ]:


#Submission

submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],
                           "Survived":y_pred})

#np.savetxt('predictions.csv',predictions.astype(np.int),fmt='%d', delimiter=",",)

submission.to_csv('titanic.csv',index=False)


# **Final Thoughts**
# 
# My next step is to understand the techniques to avoid overfitting and how to use them in this model. Please let me know your suggestions, all of them will be appreciated
# Thanks
