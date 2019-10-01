#!/usr/bin/env python
# coding: utf-8

# 

# For the prediction doing the following things
# 
#  - Load data
#  - Clean data
#  - select useful data
#  - predict the data
# 

# In[ ]:


#importing the data & liberary
import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#for linear regression
from sklearn import linear_model
#for plotting the data
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#checking the data
train.head()
#useful colums are
#pclass ,sex,age,fare,embarked


# In[ ]:


#checking the useful data
train.describe()


# In[ ]:


#cleaning the data
#pclass ,sex,age,fare,embarked
train["Age"].isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Age"].isnull().sum()


# In[ ]:


train["SibSp"].isnull().sum()


# In[ ]:


train["Embarked"].isnull().sum()
train["Embarked"]=train["Embarked"].fillna("S")


# In[ ]:


#ploting the data for embarked
sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

sns.countplot(x='Embarked', data=train, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)


# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

plt.show()


# In[ ]:


#checking for the  fare

train["Fare"].plot(kind='hist',bins=100)


# In[ ]:


#Age

train['Age'].plot(kind='hist',bins=70)
age_perc = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=age_perc,ax=axis3)


# In[ ]:


sns.barplot(x='Age', y='Survived', data=age_perc)


# In[ ]:


#Sex
sex_perc = train[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean()
sns.barplot(x='Sex', y='Survived', data=sex_perc)

plt.show()


# In[ ]:


#now the sex and embarked change to the numerical values

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"]=="female","Sex"]=1

train.loc[train["Embarked"]=="S","Embarked"] =0
train.loc[train["Embarked"]=="C","Embarked"] =1
train.loc[train["Embarked"]=="Q","Embarked"] =2


# In[ ]:


pd.to_numeric(train["Sex"])
pd.to_numeric(train["Embarked"])


# In[ ]:


train.describe()


# In[ ]:


#taking the useful data
new_col = ["Age","Sex","Embarked","Pclass","Fare","SibSp"]
x_train = train[new_col]
y_train = train["Survived"]


# In[ ]:


#apply the linear regression
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)


# In[ ]:


#checking the coffiecient
reg.coef_
reg.score(x_train,y_train)


# In[ ]:


#cleaning the test data
test["Age"].isnull().sum()
test["Age"] = test["Age"].fillna(train["Age"].median())
test["Age"].isnull().sum()


# In[ ]:


#cleanig the Embarked
test["Embarked"].isnull().sum()
test["Embarked"]=test["Embarked"].fillna("S")


# In[ ]:



test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"]=="female","Sex"]=1

test.loc[test["Embarked"]=="S","Embarked"] =0
test.loc[test["Embarked"]=="C","Embarked"] =1
test.loc[test["Embarked"]=="Q","Embarked"] =2


# In[ ]:


pd.to_numeric(train["Sex"])
pd.to_numeric(train["Embarked"])


# In[ ]:


x_test = test[new_col]
#"Age","Sex","Embarked","Pclass","Fare"
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
x_test = test[new_col]


# In[ ]:


test["SibSp"].isnull().sum()


# In[ ]:


a=reg.predict(x_test)
a = a.round()
l=[]
for i in a:
    i = int(i)
    l.append(i)


# In[ ]:


reg.intercept_


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": l
    })
submission.to_csv('titanic.csv', index=False)

