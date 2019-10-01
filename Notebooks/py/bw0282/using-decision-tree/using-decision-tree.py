#!/usr/bin/env python
# coding: utf-8

# Load Pacakge

# In[62]:


get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import os
print(os.listdir("../input"))


# Load Data

# In[63]:


train = pd.read_csv("../input/train.csv", index_col="PassengerId")
test = pd.read_csv("../input/test.csv", index_col = "PassengerId")


# In[64]:


print(train.shape)
print(test.shape)
print("train has {} data | test has {} data".format(train.shape[0], test.shape[0]))


# In[65]:


train.head()


# In[66]:


train.head()


# Explore Data

# In[67]:


print("train has {} columns".format(len(train.columns)))
print("test has {} columns".format(len(test.columns)))
print("Target data is Survived")


# Unique Data

# In[68]:


# check unique data in train
for i in train.columns:
    print("{} has {} unique data".format(i, len(train[i].unique())))


# In[69]:


# check unique data in test
for i in test.columns:
    print("{} has {} unique data".format(i, len(test[i].unique())))


# Missing Data

# In[70]:


#check missing data in train
for i in train.columns:
    print("{0} has {1:.2f}% missing data".format(i, (len(train[train[i].isnull()]) / train.shape[0]) *100)) 


# Define Data Type

# In[71]:


cat_data = ["Pclass","Sex","Ticket","Cabin","Emabarked"]
num_data = ["Age","Fare","SibSp","Parch"]
cat_1 = [14,15,16,17,18,24,25,26,27,30,31,32,33,34,35,36,37]


# In[72]:


figure, axe = plt.subplots(nrows =1,ncols =1)
figure.set_size_inches(20,4)
sns.countplot(train["Age"])


# In[73]:


#replace NoN data with Mode data
train.loc[train["Age"].isnull(),"Age"] = train["Age"].mode()[0]


# In[74]:


test.loc[test["Age"].isnull(),"Age"] = test["Age"].mode()[0]


# In[75]:


train.loc[train["Age"].isnull(),"Age"]


# In[76]:


#Check deviation of Age
sns.distplot(train["Age"])


# This distribution chart does not follow standard distribution(usually age data follows standard distribution)
# 
# So It could be non-useful feature 
# 
# hypothesis: NoN data belong to range from 10 to 30

# In[77]:


train["Age"].describe()


# Cabin

# In[78]:



#Too many missing data, it means Cabin data would not be good feature to predict Target label
len(train[train["Cabin"].isnull()])


# Embarked

# In[79]:


# replace NaN data with mdoe data
train.loc[train["Embarked"].isnull(),"Embarked"]  = train["Embarked"].mode()[0]


# **Fare Data**

# In[80]:


#replace NaN data with mean 
test.loc[test["Fare"].isnull(),"Fare"] = test["Fare"].mean()


# **Preprocessing**

# Encode Embarked

# In[81]:


#train
le= LabelEncoder()
le.fit(train["Embarked"])
Embarked = le.transform(train["Embarked"])
# One hot encoding
Embarked= np.eye(3)[Embarked]
Embarked = pd.DataFrame(Embarked,columns =["Embarked_C","Embarked_Q","Embarked_S"])


# In[82]:


train.reset_index(inplace=True)


# In[83]:


train = pd.concat([train,Embarked], axis =1)
train.set_index("PassengerId",inplace=True)
train.head()


# In[84]:


#test
le= LabelEncoder()
le.fit(test["Embarked"])
Embarked = le.transform(test["Embarked"])
# One hot encoding
Embarked= np.eye(3)[Embarked]
Embarked = pd.DataFrame(Embarked,columns =["Embarked_C","Embarked_Q","Embarked_S"])


# In[85]:


test.reset_index(inplace=True)
test = pd.concat([test,Embarked], axis =1)
test.set_index("PassengerId",inplace=True)


# Encode Sex

# In[86]:


#Encoding sex data in train
le.fit(train["Sex"])
sex = le.transform(train["Sex"])
train["Sex"] = sex.reshape(-1,1)


# In[87]:


#Encoding sex data in test
le.fit(test["Sex"])
sex = le.transform(test["Sex"])
test["Sex"] = sex.reshape(-1,1)


# Explore correlation between each feature and tarket label

# In[88]:


train.corr()


# High Corrleation with Survied : Sex, Pclass 

# **Dive into two feature**

# Sex

# In[89]:


# 1 : male, 0: femle
sex_corr = train[["Sex","Survived"]]


# In[90]:


grouped = sex_corr.groupby("Sex")["Survived"].aggregate({"sum_of_survior":"sum"})


# In[91]:


grouped["count_of_sex"] = sex_corr.groupby("Sex")["Survived"].aggregate({"count_of_sex":"count"})


# In[92]:


grouped["s_rate"] = grouped["sum_of_survior"] / grouped["count_of_sex"]


# In[93]:


grouped


# In[94]:


figure, (axe1,axe2) = plt.subplots(nrows = 1, ncols =2)
figure.set_size_inches(14,4)
sns.barplot(grouped.index,grouped["sum_of_survior"],ax = axe1)
sns.barplot(grouped.index,grouped["s_rate"],ax = axe2)


# Pclass

# In[95]:


pclass = train[["Pclass","Survived"]]
grouped = pclass.groupby("Pclass")["Survived"].aggregate({"sum_of_survivor":"sum"})
grouped["count_of_class"] = pclass.groupby("Pclass")["Survived"].aggregate({"count_of_class":"count"})
grouped["s_rate"] = grouped["sum_of_survivor"] / grouped["count_of_class"]
grouped


# In[96]:


figure, (axe1,axe2) = plt.subplots(nrows = 1, ncols =2)
figure.set_size_inches(14,4)
sns.barplot(grouped.index,grouped["sum_of_survivor"],ax = axe1)
sns.barplot(grouped.index,grouped["s_rate"],ax = axe2)


# high pclass customer must pay high fare that means Fare data would be same with Plcass pattern
# Graphs show that Plcass is key feature of this data

# Age

# In[97]:


age = train.groupby("Age")["Survived"].aggregate({"sum_of_survivor":"sum"})
age["count_of_age"] = train.groupby("Age").size().values
age["s_rate"] = age["sum_of_survivor"] / age["count_of_age"]
age


# In[98]:


figure, (axe1,axe2) = plt.subplots(nrows = 1, ncols =2)
figure.set_size_inches(14,4)
sns.pointplot(age.index,age["sum_of_survivor"],ax = axe1)
sns.pointplot(age.index,age["s_rate"],ax = axe2)


# In[99]:


train.head()


# Sibsp

# In[100]:


sns.barplot(train["SibSp"],train["Survived"])


# when size is big, survived decreaes

# Parch

# In[101]:


sns.barplot(train["Parch"],train["Survived"])


# Embarked

# In[102]:


sns.barplot(train["Embarked"],train["Survived"])


# **Feature engineering**

# hypothesis : Large family is hard to survive because hard to find family and move

# In[103]:


train["Family_size"] = train["Parch"] + train["SibSp"]
test["Family_size"] = test["Parch"] + test["SibSp"]
sns.pointplot(train["Family_size"],train["Survived"])


# Hypothesis makes sense

# Train

# In[104]:


feature_names = ["Pclass", "Sex", "Fare","Family_size",
                 "Embarked_C", "Embarked_S", "Embarked_Q"]
feature_names


# In[105]:


x_train = train[feature_names]

print(x_train.shape)
x_train.head()


# In[106]:


x_test = test[feature_names]

print(x_test.shape)
x_test.head()


# In[107]:


label_name = "Survived"
y_train = train[label_name]

print(y_train.shape)
y_train.head()


# In[108]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=71,max_leaf_nodes=6,min_impurity_decrease=0.002818)
model


# In[109]:


model.fit(x_train, y_train)


# Score

# In[110]:


y_predict = cross_val_predict(model,x_train,y_train)
accuracy = accuracy_score(y_predict,y_train,)
print("accuracy = {0:.2f}".format(accuracy))


# Predict

# In[111]:


predictions = model.predict(x_test)


# Submit

# In[113]:


submit = pd.read_csv("../input/gender_submission.csv", index_col="PassengerId")

print(submit.shape)
submit.head()


# In[114]:


submit["Survived"] = predictions

print(submit.shape)
submit.head()


# In[115]:


submit.to_csv("submit.csv")


# Kaggle Score:0.78468

# In[ ]:




