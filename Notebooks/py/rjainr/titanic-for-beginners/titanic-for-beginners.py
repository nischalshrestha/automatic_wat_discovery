#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction

# #### About the Problem :-
# 
# This is a Binary classification problem where we have to predict only Yes(1, true) or No(0, false). 
# So these are the following steps which we will follow - 
# 1. Data Analysis - 
#  a. Check the type and  number of Features (Inputs) and trying to remove any unwanted features which             may not help in prediction.
#  b. Join train and test dataset and try to fill Null values 
#                    
# 2. Visualisation - 
# a. Plot Graphs.
# b. Try to find Co-relation between different features and Survival.
#                    
# 3. Feature Engineering - 
# a. Adding additional features which may help in Survival prediction.
# 
# 4. Modelling - 
# a.Try to predict output using various Models.

# #### Lets Import some Libraries !!

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import numpy.random as rnd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Loading Train and Test Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
train.head()


# In[ ]:


print(test.shape)
test.head()


# In[ ]:


len_train = len(train)


# #### Concatenate both Train and Test datasets for data cleaning

# In[ ]:


dataset = pd.concat([train, test], ignore_index=True)
print(dataset.shape)
dataset.head()


# In[ ]:


dataset.describe()


# ### Visualisation
# Lets plot some graph to get better Insights.

# #### Sex vs Survival

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train)


# As we see, females have got a better chance of Survival 

# #### PClass vs Survival

# In[ ]:


sns.barplot(x= 'Pclass', y='Survived', data=train)


# #### Embarked vs Survival

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=train)


# Those who embarked at port C had better Survival chances

# #### Age vs Survival

# In[ ]:


sns.violinplot(x='Survived', y= 'Age', data=train)


# As we see from the Violin plot, younger children and old people had more Survival rate. 

# #### Filling Null values

# In[ ]:


dataset.isnull().sum()


# Age has 267 Null values.
# Survived - Since we have concatenated test and train data, the 418 null values are from test set which need not be filled.
# Embarked - Since there are only 2 null values, we will fill it with the most Embarked one.
# Since cabin has too many empty values, lets drop that column.
# Since Ticket has not much Corelation with Survival, we will drop that too.
# 

# Filling Age null Values with random numbers generated between its mean and standard deviation.

# In[ ]:


age_mean = dataset['Age'].mean()
age_std = dataset['Age'].std()
nan_count = dataset['Age'].isnull().sum()
dataset['Age'][np.isnan(dataset['Age'])] = rnd.randint(age_mean - age_std, age_mean + age_std, size= nan_count)


# In[ ]:


dataset = dataset.drop(['Cabin', 'Ticket'], axis =1)


# In[ ]:


dataset['Fare'][np.isnan(dataset['Fare'])] = dataset['Fare'].mean()


# In[ ]:


top = dataset['Embarked'].describe().top
dataset['Embarked'] = dataset['Embarked'].fillna(top)


# #### converting Categorical data to Numerical 

# In[ ]:


dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})


# In[ ]:


dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q':2})


# In[ ]:


print(dataset.isnull().sum())
dataset.head()


# ### Feature Engineering

# ##### Extracting titles from name

# In[ ]:


title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset['Title'] = pd.Series(title)
dataset['Title'].head()


# In[ ]:


dataset['Title'].describe()


# In[ ]:


sns.countplot(x="Title",data=dataset)


# In[ ]:


g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# In[ ]:


dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don',
                                             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1,
                                         "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[ ]:


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[ ]:


g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# In[ ]:


dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# In[ ]:


dataset.head()


# ### Modelling

# Seperating the train and test data from the concatenated dataframe.

# In[ ]:


train = dataset[:len_train]
test = dataset[len_train:]
test.drop(['Survived'], axis=1, inplace=True)


# In[ ]:


Y_train = train['Survived'].astype(int)
X_train = train.drop(['Survived'], axis=1)
X_train.drop(labels=["PassengerId"], axis=1, inplace=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X_train.head()


# In[ ]:


clf_log = LogisticRegression()
clf_log.fit(X_train, Y_train)
acc_log = round(clf_log.score(X_train, Y_train)*100, 2)
acc_log


# In[ ]:


clf_rnd = RandomForestClassifier()
clf_rnd.fit(X_train, Y_train)
acc_rnd = round(clf_rnd.score(X_train, Y_train)*100, 2)
acc_rnd


# In[ ]:


clf_svc = LinearSVC()
clf_svc.fit(X_train, Y_train)
acc_svc = round(clf_svc.score(X_train, Y_train)*100, 2)
acc_svc


# In[ ]:


clf_knc = KNeighborsClassifier()
clf_knc.fit(X_train, Y_train)
acc_knc = round(clf_knc.score(X_train, Y_train)*100, 2)
acc_knc


# In[ ]:


clf_gc = GaussianNB();
clf_gc.fit(X_train, Y_train)
acc_gc = round(clf_gc.score(X_train, Y_train)*100, 2)
acc_gc


# After checking accuracies of various Algorithms, we see that RandomForestClassifier has highest accuracy (Infact i think it has overfit the data, we should decrease the variance by  adding more features and  normalize the data) . Will do that soon.

# #### Lets predict the test data output using random forest classifier

# In[ ]:


y_rnd = clf_rnd.predict(test)


# Add the result y_rnd to submission dataframe and submit the created file on Kaggle.

# In[ ]:



submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": y_rnd
})


# In[ ]:


submission.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)


# #### Hope this Notebook has helped you in getting started with kaggle. Please upvote if you have liked it.

# In[ ]:




