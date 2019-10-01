#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load Test and Train DATA
train_data= pd.read_csv("../input/train.csv")
test_data= pd.read_csv("../input/test.csv")

# lenght of train data
#train_data_len=len(train_data)

# concatenate train and test data to execute data clearning operations effectively
#full_data = pd.concat([train_data,test_data], axis=0).reset_index(drop=True)


# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


# Compute missing and null values
#def find_null_cols():
 #   for cols in train_data:
  #      l_null_cnt=train_data[cols].isnull().values.sum()
   #     print ("Null values in %s : %i " %(cols, l_null_cnt)) 
print(train_data.isnull().sum())


# In[ ]:


print(test_data.isnull().sum())


# In[ ]:


# Find out correlation among features
plt.figure(figsize=(10,10))
sns.heatmap(train_data.corr(), square=True, annot=True, cmap='Reds',linecolor="black", linewidths=0.20)
plt.show()


# In[ ]:


# this is correlation between Pclass and Fare
# there is positive correlation between siblings and parch


# AGE:

# In[ ]:


# Fill missing values in Age column with Mean
train_data["Age"]=train_data.Age.fillna(train_data.Age.mean())
test_data["Age"]=test_data.Age.fillna(test_data.Age.mean())

# Cover to Integer
train_data["Age"]=train_data["Age"].astype(int)
test_data["Age"]=test_data["Age"].astype(int)

sns.distplot(train_data["Age"])
plt.show()


# In[ ]:


# Introducing a new column - Age_Group

train_data.loc[train_data["Age"] <=16, "Age_Group"]=1
train_data.loc[(train_data["Age"] > 16) & (train_data["Age"] <= 30), 'Age_Group'] = 2
train_data.loc[(train_data["Age"] > 30) & (train_data["Age"] <= 50), 'Age_Group'] = 3
train_data.loc[(train_data["Age"] > 50) & (train_data["Age"] <= 80), 'Age_Group'] = 4


test_data.loc[test_data["Age"] <=16, "Age_Group"]=1
test_data.loc[(test_data["Age"] > 16) & (test_data["Age"] <= 30), 'Age_Group'] = 2
test_data.loc[(test_data["Age"] > 30) & (test_data["Age"] <= 50), 'Age_Group'] = 3
test_data.loc[(test_data["Age"] > 50) & (test_data["Age"] <= 80), 'Age_Group'] = 4

train_data["Age_Group"]=train_data["Age_Group"].astype(int)
test_data["Age_Group"]=test_data["Age_Group"].astype(int)

train_data=train_data.drop("Age", axis=1)
test_data=test_data.drop("Age", axis=1)

# plot new Age Values
train_data['Age_Group'].hist(bins=70)


# In[ ]:


sns.factorplot(x="Age_Group",y="Survived",data=train_data)
plt.show()


# PCLASS

# In[ ]:


pclass_data=train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() 
print (pclass_data)


# In[ ]:


sns.barplot(x="Pclass",y="Survived", data=pclass_data)
plt.show()


# In[ ]:


sns.factorplot('Pclass', 'Survived', hue='Sex', col = 'Embarked', data=train_data)
plt.show()


# SEX:

# In[ ]:


sex_data =train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
sex_data


# In[ ]:


sns.barplot(x="Sex",y="Survived", data=sex_data)
plt.show()


# In[ ]:


train_data["Sex"]= train_data["Sex"].map({"male": 1, "female": 2})
test_data["Sex"]= test_data["Sex"].map({"male": 1, "female": 2})


# SIBSP and PARCH:

# In[ ]:


# Family size: SibSp and Parch have strong correlation so lets combine the features
for i in train_data:
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']+1

for i in test_data:
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']+1

# drop Parch & SibSp
train_data = train_data.drop(['SibSp','Parch'], axis=1)
test_data  = test_data.drop(['SibSp','Parch'], axis=1)

family_data=train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
family_data


# In[ ]:


sns.barplot(x="FamilySize",y="Survived", data=family_data)
plt.show()


# In[ ]:


sns.violinplot(x='Pclass', y='Fare', hue='Survived', data=train_data, split=True)
plt.show()


# EMBARKED:

# In[ ]:


# Embarked
sns.barplot(x="Embarked",y="Survived", data=train_data)
plt.show()


# In[ ]:


train_data["Embarked"].isnull().sum()


# In[ ]:


train_data["Embarked"] = train_data["Embarked"].fillna("S")

train_data["Embarked"]=train_data["Embarked"].map({"S": 1, "C" :2, "Q" :3})
test_data["Embarked"]=test_data["Embarked"].map({"S": 1, "C" :2, "Q" :3})

train_data["Embarked"]=train_data["Embarked"].astype(int)
test_data["Embarked"]=test_data["Embarked"].astype(int)


# FARE:

# In[ ]:


test_data["Fare"].isnull().values.sum()


# In[ ]:


test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

# convert from float to int
train_data['Fare'] = train_data['Fare'].astype(int)
test_data['Fare']  = test_data['Fare'].astype(int)


# In[ ]:


train_data_train= train_data[0:620]
train_data_test=train_data[620:]


# In[ ]:


# extract this column for submiting the result
test_data_passenger=test_data["PassengerId"]


# In[ ]:


# drop unecessary features
drop_features=["Name","Ticket","Cabin","PassengerId"]

train_data_train=train_data_train.drop(drop_features, axis=1)
train_data_test=train_data_test.drop(drop_features, axis=1)
test_data=test_data.drop(drop_features, axis=1)


# In[ ]:


x_train_data_train=train_data_train.drop("Survived",axis=1)
y_train_data_train=train_data_train["Survived"]


# In[ ]:


x_train_data_test=train_data_test.drop("Survived",axis=1)
y_train_data_test=train_data_test["Survived"]


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix


reg_model = LogisticRegression()
reg_model.fit(x_train_data_train,y_train_data_train)
predict = reg_model.predict(x_train_data_test)

print('Training Score: %f ' %(reg_model.score(x_train_data_train, y_train_data_train)))
print('Testing Score : %f ' %(accuracy_score(y_train_data_test, predict)))

print('Confusing Metrix :')
print(confusion_matrix(y_train_data_test, predict))


# SVM

# In[ ]:


from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
svc_model = SVC()
svc_model.fit(x_train_data_train, y_train_data_train)
predict= svc_model.predict(x_train_data_test)
print('Training Score: %f ' %(svc_model.score(x_train_data_train, y_train_data_train)))
print('Testing Score : %f ' %(accuracy_score(y_train_data_test, predict)))

print('Confusing Metrix :')
print(confusion_matrix(y_train_data_test, predict))


# Random Forrest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

rf_model = RandomForestClassifier(n_estimators=45,n_jobs =1, max_features = 'auto',min_samples_leaf=5, random_state=1)

rf_model.fit(x_train_data_train, y_train_data_train)
predict= rf_model.predict(x_train_data_test)
print('Training Score: %f ' %(rf_model.score(x_train_data_train, y_train_data_train)))
print('Testing Score : %f ' %(accuracy_score(y_train_data_test, predict)))

print('Confusing Metrix :')
print(confusion_matrix(y_train_data_test, predict))


# Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

dt_model = DecisionTreeClassifier()
dt_model.fit(x_train_data_train, y_train_data_train)
predict= dt_model.predict(x_train_data_test)
print('Training Score: %f ' %(dt_model.score(x_train_data_train, y_train_data_train)))
print('Testing Score : %f ' %(accuracy_score(y_train_data_test, predict)))

print('Confusing Metrix :')
print(confusion_matrix(y_train_data_test, predict))


# In[ ]:


# Time for predicting the given test data

actual_predict= rf_model.predict(test_data)

sub = pd.DataFrame({"PassengerId": test_data_passenger,"Survived": actual_predict})

submission=sub.to_csv("submission_rf4.csv",index=False)


# In[ ]:




