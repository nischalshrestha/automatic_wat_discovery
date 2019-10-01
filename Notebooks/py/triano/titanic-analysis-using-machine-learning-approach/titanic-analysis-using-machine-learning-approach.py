#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hello everyone !
# 
# This is my first project in kaggle. i would like to sharing about implementation of the machine learning to predict titanic classification, especially using Logistic Regression, Decision Tree, and Random Forest  algorithm. Lets Check it out.

# # 1. Import python packages and dataset is needed
# 

# In[ ]:


#import python packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score 
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#import dataset from draft environment
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.columns, test.columns


# **Information of variables :**
# * **PassengerId** = ID number of each Passenger 
# * **Survived**       = Whether the passenger survived or not ( 0 = no, 1 =yes) 
# * **Pclass**           = Passanger class indicates the class of that person aboard the ship. (1 (1st)= Upper,  2(2nd) = Middle, 3(3rd) = lower)
# * **Name**            = The name of Passenger
# * **Sex**                = sex
# * **Age**                = Age in years
# * **SibSp**            = The number of Sibling/Spouces they had.
# * **Parch**             = Parch indicates Parents with children.
# * **Ticket**            = Ticket name/Number.
# * **Fare**               = How much the passenger should be paid
# * **Cabin**             = Cabin name of that Passenger.
# * **Embarked**      = Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) 

# In order to see the descriptive statistics of Titanic data, you just write the code is following 

# In[ ]:


#Descriptive statistic of Titanic data
train.describe()


# In[ ]:


#Check data type
train.dtypes


# # 2.  Check missing value

# In[ ]:


#titanic info
train.info()


# In[ ]:


#check missing value
train.isnull().sum()


# ## 2.1. Age missing value

# In[ ]:


sum(pd.isnull(train['Age']))


# In[ ]:


# proportion of "Age" missing
round(177/(len(train["PassengerId"])),4)


# ## 2.2. Cabin Missing Value

# In[ ]:


# proportion of "cabin" missing
round(687/len(train["PassengerId"]),4)


# ## 2.3. Embarked Missing Value

# In[ ]:


# proportion of "Embarked" missing
round(2/len(train["PassengerId"]),4)


# 
# # 3. Data Preprocessing

# If we see from the chceking missing value, there are 3 variables that have NA value, that is Age, Cabin, and Embarked variables. how to fix it ?
# 
# in this analysis, we will be filling several of  columns that have NA value with existing value, foe example :
# 1. in order to fillin the empty columns on the Age variable can be filled with the median value of the Age variable. what is the median value of Age ?

# In[ ]:


# median age is 28 (as compared to mean which is ~30)
train["Age"].median(skipna=True)


# the median value is 28. so that, we'll filling each empty column with that value. 
# 
# 2. In order to fillin the empty columns on the Embarked variable can be filled with S (Southampton). Since the average of passengers boards a ship in that city. 
# 
# 3. We will remove several of a variable which not used, that is Cabin since that variable has much missing value. so that, we will not use that variable. 
# 
# 4. Remove SibSp and Parch, then combine those variables to be one group. We will give the name of variables are **TravelBuds**.
# 
# 5. The last, remove  variables that are not used. (PassengerId, Name, and Ticket)

# ### 3.1. Preprocessing Data Train

# In[ ]:


#final adjustment
train_data = train
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


## Create categorical variable for traveling alone
train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)


# In[ ]:


train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)


# In[ ]:


#create categorical variable for Pclass

train2 = pd.get_dummies(train_data, columns=["Pclass"])


# In[ ]:


train3 = pd.get_dummies(train2, columns=["Embarked"])


# In[ ]:


train4=pd.get_dummies(train3, columns=["Sex"])
train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)
train4.head(5)
final_train = train4

final_train.head()


# ### 3.2. Preprocessing Data Test 

# In[ ]:


#final adjustment
test_data = test
test_data["Age"].fillna(28, inplace=True)
test_data["Embarked"].fillna("S", inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

## Create categorical variable for traveling alone

test_data['TravelBuds']=test_data["SibSp"]+test_data["Parch"]
test_data['TravelAlone']=np.where(test_data['TravelBuds']>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)
test_data.drop('TravelBuds', axis=1, inplace=True)

#create categorical variable for Pclass

test2 = pd.get_dummies(test_data, columns=["Pclass"])

test3 = pd.get_dummies(test2, columns=["Embarked"])

test4=pd.get_dummies(test3, columns=["Sex"])

test4.drop('PassengerId', axis=1, inplace=True)
test4.drop('Name', axis=1, inplace=True)
test4.drop('Ticket', axis=1, inplace=True)
test4.head(5)

final_test=test4
final_test.head()


# # 4. Exploratory Data

# In[ ]:


ax = train["Age"].hist(bins=15, color='green', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()


# In[ ]:


#Explor Age Variable
plt.figure(figsize=(10,5))
train['Age'].plot.hist(bins=35)


# In[ ]:


plt.figure(figsize=(10,5))
train[train['Survived']==0]['Age'].hist(bins=35,color='blue',
                                       label='Survived = 0', 
                                        alpha=0.6)
train[train['Survived']==1]['Age'].hist(bins=35,color='red',
                                       label='Survived = 1',
                                       alpha=0.6)
plt.legend()
plt.xlabel("The Number of Age")


# In[ ]:


sns.countplot(x='Embarked',data=train,palette='Set2')
plt.show()


# # 5. Machine Learning Models

# In[ ]:


#Import Packages for Machine Learning models 
## the packages is Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cross_validation import train_test_split


# ## 5.1. Logistic Regression

# In[ ]:


x = final_train.drop('Survived', axis=1)
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.20)

Logmodel = LogisticRegression()
Logmodel.fit(X_train,y_train)


# In[ ]:


pred_LR = Logmodel.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred_LR))
print('\n')
print(classification_report(y_test,pred_LR))


# In[ ]:


Accuracy_LR = print ('1. Accuracy_L.Regression_Classifier :', 
                     accuracy_score(y_test,pred_LR)*100)


# ## 5.2. Decision Tree Classifier 

# In[ ]:


x = final_train.drop('Survived', axis=1)
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.20)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


pred_tree = dtree.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,pred_tree))
print('\n')
print(classification_report(y_test,pred_tree))


# In[ ]:


Accuracy_DT = print ('2. Accuracy_D.Tree_Classifier :', 
                     accuracy_score(y_test,pred_tree)*100)


# ## 5.3. Random Forest Classifier

# In[ ]:


x = final_train.drop('Survived', axis=1)
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.20)

Rfc = RandomForestClassifier(n_estimators = 300 )
Rfc.fit(X_train,y_train)


# In[ ]:


Pred_Rfc =Rfc.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,Pred_Rfc))
print ('\n')
print(classification_report(y_test,Pred_Rfc))


# In[ ]:


Accuracy_RF = print ('3. Accuracy_R.Forest_Classifier :', 
                     accuracy_score(y_test,Pred_Rfc)*100)


# # Conculsion :
# 
# From the Analisys we can see that there are several of acc model from 
# each machine learning that have been made. Choose the highest value of accuracy models.  
# 
# noted : (those models are not consistant) 
