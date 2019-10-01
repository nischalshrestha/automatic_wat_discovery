#!/usr/bin/env python
# coding: utf-8

# Titanic: Machine Learning from Disaster

# 
# 
# **1. Question/ Problem Definition**
# -------------------------------
# 
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# In this challenge, we complete the analysis to see what sorts of people were likely to survive. 

# In[ ]:


# This section updated as and when required #

#Data analysis and wrangling
import numpy as np
import pandas as pd

#Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR




# **2. Acquire Data**
# -------------------------------
# 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


# In[ ]:


print(train.head(5))


# **3. Analyse Data**
# -------------------------------
# 

# In[ ]:


print(train.columns.values)
print('-'*50)
train.info()
print('-'*50)
test.info()





# Summary:
# 
# Total columns=10 excluding Survived ['PassengerId' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' Ticket' 'Fare' 'Cabin' 'Embarked']
# 
# Numerical Columns: PassengerId, Pclass, Age, SibSp, Parch, Fare, Survived
# 
# Alphanumeric Columns: Name, Sex, Ticket, Cabin, Embarked
# 
# Total Data entries=891(Test)+418(Train)=1309
# 
# Columns with null values[Columns whose count!= Total count]= Age, Cabin,Embarked(Only 2 in Training Dataset), Fare(Only 1 in Test Dataset)
# 
# 

# In[ ]:


#For Numerical Features
print ("N Features")
print (train.describe())
print('-'*50)
#For Categorical Features
print ("C Features")
print (train.describe(include=['O']))


# Summary:
# 
# Total Numerical Features=6(+1=Survived)
# 
# Total Categorical Features=5
# 
# //The above info is inconsequential now but will be important during Prediction Phase// 
# 
# 
# 
# 
# 
# 
# -> Numerical Data Summary
# 
#      Around 38% samples survived representative of the actual survival rate at 32%
# 
#      Nearly 70% of the passengers did not have siblings and/or spouse aboard
# 
#      Most passengers (> 75%) did not travel with parents or children.
# 
#      Fares varied significantly with few passengers (<1%) paying as high as $512
# 
#      More than 70-75% travelled in Pclass=3
# 
# 
# -> Categorical Data Summary
# 
#      Names are all unique
# 
#      There are more male passengers than female passengers(Male Passenger=577/891)
# 
#       Embarked takes three possible values. S port used by most passengers (top=S)
# 
#       Ticket feature has high ratio (22%) of duplicate values (unique=681)

# In[ ]:


#Check unique values in all columns
for column in train:
    uni=train[column].unique()
    print("No of unique values of ", column, ":", len(uni), "\n" )
   


# Always good to know the unique values in order for better data prediction

# Apply Decision tree for predicting survival:
# 
# Though obvioulsy, applying now won't be the most accurate but still we need to know the base % that the entire dataset can predict. 

# In[ ]:


X=train.dropna(axis=0)#Null values create problem in predicting data
y=X['Survived'].copy()
X=X.drop(labels=['PassengerId','Survived'],axis=1)#Dropping PassengerId as well as it is inconsequential
X=pd.get_dummies(X,columns=['Sex',  'Ticket' ,'Cabin', 'Embarked','Name'])
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=7)
#model= tree.DecisionTreeClassifier()
model=RandomForestClassifier(random_state=7)
model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print("Accuracy of RandomForest(Will vary according to random state)=",accuracy*100,'%')


# The accuracy score of 65% is a good enough score considering that it was our first attempt to predict without any sort of feature engineering or manipulation of values. 
# 
# Now let's see how we can improve our score by checking how the different features are correlated to survival

# **4. Correlating/Creating Data**
# -------------------------------
# 

# Correlating data
# Let's check how each feature correaltes with 'Survival'

# In[ ]:


PClassSurvived=train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
print(PClassSurvived)
sns.barplot(x='Pclass', y='Survived', data=PClassSurvived)


# In[ ]:


AgeSurvived= train[['Age','Survived']].groupby('Age',as_index=False).mean().sort_values(by='Survived',ascending=False)
print(AgeSurvived)
fig, axis=plt.subplots(1,1,figsize=(18,8))
AgeSurvived["Age"] = AgeSurvived["Age"].astype(int)
sns.barplot(x='Age',y='Survived',data=AgeSurvived)


# In[ ]:


SibSpSurvived= train[['SibSp','Survived']].groupby('SibSp',as_index=False).mean().sort_values(by='Survived',ascending=False)
print(SibSpSurvived)
sns.barplot(x='SibSp',y='Survived',data=SibSpSurvived)


# In[ ]:


ParchSurvived= train[['Parch','Survived']].groupby('Parch',as_index=False).mean().sort_values(by='Survived',ascending=False)
print(ParchSurvived)
sns.barplot(x='Parch',y='Survived',data=ParchSurvived)


# In[ ]:


SexSurvived= train[['Sex','Survived']].groupby('Sex',as_index=False).mean().sort_values(by='Survived',ascending=False)
print(SexSurvived)
sns.barplot(x='Sex',y='Survived',data=SexSurvived)


# In[ ]:


EmbarkedSurvived= train[['Embarked','Survived']].groupby('Embarked',as_index=False).mean().sort_values(by='Survived',ascending=False)
print(EmbarkedSurvived)
sns.barplot(x='Embarked',y='Survived',data=EmbarkedSurvived)


# Summary:
# 
# PClass, Sex, Age and Embarked can be readily seen as features which are directly related to Survived
# 
# PClass- PClass 1 passengers  are most likely to survive
# 
# Sex- Female are more likely to survive than Male(74%  have survived)
# 
# Age- Infants(Age<2) and Old people(=80) have survived
# 
# Embarked- Passengers who embarked from C have a higher rate of survival
# 
# SibSp and Parch are features that don't really show any strong relation(Combine them?)
# 
# Name, Ticket, Fare and Cabin though might be related are too dispersed to currently show and relation(Categorise them?)

# *Feature Engineering*

# In[ ]:


#Let's combine SibSp and Parch simply as a new column family members
train['family_size']=train['SibSp']+ train['Parch'] + 1
test['family_size']=test['SibSp']+ test['Parch'] + 1
FSurvived= train[['family_size','Survived']].groupby('family_size',as_index=False).mean().sort_values(by='Survived',ascending=False)
print(FSurvived)
sns.barplot(x='family_size',y='Survived',data=FSurvived)


# Passengers with 2-4 family members have a much higher chance of survival

# In[ ]:


#Extract Title from Names and group them
train['Title']=train['Name'].str.extract('([A-Za-z]+)\.',expand=False)
test['Title']=test['Name'].str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])


# In[ ]:


#Let's do combining of the titles
train['Title']=train['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

test['Title']=test['Title'].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

TSurvived=train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',ascending=False)
print(TSurvived)
sns.barplot(x='Title',y='Survived',data=TSurvived)


# Though we knew female have a better chance of survival than male, quite a few other interesting aspects have come out
# 
# 1. Married women have a slightly(Around 9%) better chance of survival than single women(Maybe because their husbands save them?)
# 
# 2. Having a rare title doesn't really help survival rate(In times of panic, ppl hardly care about titles)

# In[ ]:


TicketSurvived=train[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Ticket',ascending=False)
print(TicketSurvived)


# In[ ]:


print(train[['Cabin']].info())
print('-'*50)
print(test[['Cabin']].info())


# Ticket value is dispersed and outside of our domain knowledge to categorise
# Cabin has too many missing values for it to be effective.
# 
# Hence, it's better to drop them both

# In[ ]:


train=train.drop(['Cabin','Ticket'],axis=1)
test=test.drop(['Cabin','Ticket'],axis=1)


# And while we are it, it is easy to notice that Embarked has 2 null values in test dataset and Fare has 1 Null value in test dataset. 
# 
# Let's drop the Embarked and predict the Fare.(Note: You can't drop the Fare column from test dataset as it will tamper with the row count)
# 
# (Additionally you could also predict these 3 missing values as was done by the wonderfully smart Megan Risdal https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic  )
# 
# 
# 

# In[ ]:


train=train.dropna(axis=0,subset=['Embarked'])
test['Fare']=test['Fare'].fillna(test['Fare'].mean())
test.info()


# **5. Completing Data**
# -------------------------------
# 

# 'Age' despite having lot of missing values is a very important factor in determining Survived. So we need to replace the missing data. You could, of course, replace it with the mean. 
# 
# 

# In[ ]:


train['Age']=train['Age'].fillna(train['Age'].mean())
test['Age']=test['Age'].fillna(test['Age'].mean())


# **6. Categorising/Converting Data**
# -------------------------------
# 

# Now after Correlating, Creating and Cleaning data, let us start categorise the data which is too dispersed(Age,Fare)

# In[ ]:


train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4   

test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4   

print(train['Age'].unique())


# In[ ]:


train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] = 3
train['Fare'] = train['Fare'].astype(int)
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3
test['Fare'] = test['Fare'].astype(int)

print(train['Fare'].unique())


# Now let's convert categorical data to numeric data for better prediction namely 'Sex','Title' and 'Embarked'
# 

# In[ ]:


train['Sex']=train['Sex'].map({'female':0,'male':1}).astype(int)
train['Title']=train['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test['Sex']=test['Sex'].map({'female':0,'male':1}).astype(int)
test['Title']=test['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
print(train.dtypes)


# **7. Predicting**
# -------------------------------
# 

# We made it !

# In[ ]:


y=train['Survived'].copy()
X=train.drop(labels=['PassengerId','Survived','Name'],axis=1)#Dropping PassengerId, Name it is inconsequential
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=7)
#model= tree.DecisionTreeClassifier()
model=RandomForestClassifier(random_state=15)
model.fit(X_train,y_train)
accuracy=model.score(X_test,y_test)
print("Accuracy of RandomForest(Will vary according to random state)=",accuracy*100,'%')


# Accuracy of around 84%. Not bad, not bad at all! 
# 

# **7. Submission**
# -------------------------------
# 

# In[ ]:


XTest=test.drop(labels=['PassengerId','Name'],axis=1)
Y_pred=model.predict(XTest)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)




# So that's it folks
# 
# Though I would be changing one or two things here and there, now and then to increase the score, this is more or less my final submission.
# 
# Was fun finishing my first competition on Kaggle. Learned a lot. Hope to learn more.
# 
# Also, all comments on how to improve this are welcome!
# 
# 
# 
# ***"May the Force be with you"*** 
# 
