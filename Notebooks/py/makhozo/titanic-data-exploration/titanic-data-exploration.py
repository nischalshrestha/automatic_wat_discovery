#!/usr/bin/env python
# coding: utf-8

# <font size=10>**Titanic - Data Exploration** </font>
# 
# 
# This kernel is to become familiar with the Kaggle interface and be able to generate and submit the allgender solution.

# <font size=6>**Exploring the Data** </font>
# 
# Import helpful libraries (https://github.com/kaggle/docker-python)
# 
# 

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
get_ipython().magic(u'matplotlib inline')

from numpy import mean;
from numpy import median;

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
original_train = pd.read_csv('../input/train.csv')
data_train = original_train

original_test = pd.read_csv('../input/test.csv')
data_test = original_test



# Any results you write to the current directory are saved as output.
data_train.head()


# In[3]:


data_train.describe()


# Create some bar plots to look at some variable descriptive statistics.

# In[4]:


# Learning sns.barplot:
# sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train);
# sns.set_style("whitegrid")
# sns.barplot(x="Pclass", y = "Survived", data=data_train);
# sns.barplot(x="Survived", y = "Sex", data=data_train, order=["female", "male"]);
# sns.barplot(x="Pclass", y="Survived", data=data_train, estimator=mean, ci=68, capsize=0.2, palette = "Blues_d");
# sns.barplot(x="Survived", y = "Pclass", orient="h", data=data_train);


# In[5]:


sns.set_style("whitegrid");
fig, ax =plt.subplots(1,2)
sns.barplot(x="SibSp", y="Survived", palette = "Blues_d", ci=None, data=data_train, ax=ax[0]);
sns.barplot(x="Parch", y="Survived", palette = "Blues_d", ci=None, data=data_train, ax=ax[1]);
# fig.show()
fig.tight_layout() 
fig, ax =plt.subplots(1,2)
# sns.barplot(x="Parch", y="Survived", palette = "Blues_d", ci=None, data=data_train);
sns.barplot(x="Pclass",  hue = "Sex", y="Survived", palette = "Blues_d", ci=None, data=data_train, ax=ax[0]);
sns.barplot(x="Survived", y="Age", palette = "Blues_d", orient="v", data=data_train, ci= None, ax=ax[1]);

fig.tight_layout() 
fig, ax =plt.subplots(1,2)
sns.barplot(x="Embarked",  y = "Survived", palette = "Blues_d", data=data_train, ax=ax[0]);
sns.distplot(data_train['Age'].dropna(how='any'),ax=ax[1]);
fig.tight_layout() 

fig, ax = plt.subplots()
for a in [data_train[data_train["Survived"]==1]["Age"].dropna(how='any'),             #yellow
          data_train[data_train["Survived"]==0]["Age"].sample(342).dropna(how='any')]: #blue
    sns.distplot(a, bins=range(1, 81, 10), ax=ax, kde=False)
    
fig.tight_layout() 
# ax.set_xlim([0, 100])
# 342+549 #549 did not survive


# Look at some of the unique values in the data

# In[7]:



data_train["Parch"].describe()
# print("Unique PClass values",data_train.Pclass.unique())
for column in data_train.drop(["PassengerId", "Name", "Ticket"],axis=1): #axis=1 specifics columns; axis=0 specifies rows
    print("Unique ", column, " values",data_train[column].unique())



# In[8]:


data_train.describe()


# <font size=6>**Transforming the Data** </font>
# 
# First let's modify some of the variables for our needs:
# * "Cabin" is all over the place. Let's use only the first character, since it *might* be important.
# * We can safely drop Passenger ID as it is highly unlikely it is correlated to any existing variables.
# * As interesting as it could be for text analysis, let's drop Name and Ticket too.
# * Convert 'Embarked' and 'Cabin' and 'Gender' into dummy variables

# In[110]:


#Reload data just in case
data_train = original_train
data_test = original_test



data_train["Cabin"] = data_train["Cabin"].str[0]
data_test["Cabin"] = data_test["Cabin"].str[0]
data_train = data_train.drop(columns=["PassengerId"])
# data_test = data_test.drop(columns=["PassengerId"])
data_train = data_train.drop(columns=["Name", "Ticket"])
data_test = data_test.drop(columns=["Name", "Ticket"])
# funtemp = data_train["Embarked"] == "C"
data_train = pd.get_dummies(data_train, columns=["Embarked", "Cabin", "Sex"], prefix=["embark", "cabin", "g"])
data_test = pd.get_dummies(data_test, columns=["Embarked", "Cabin", "Sex"], prefix=["embark", "cabin", "g"])
data_test["cabin_T"] = 0 #because this variable never appeared in the test set, which removed it from the dummies function
data_test.head()


# # Missing Values
# 
# Notice how there are lots of missing values in the dataset. Let us investigate so we can impute new values in their place.
# 
# First investigate existing correlations between variables; this automatically ignores records with missing values.

# In[111]:



corr = original_train.corr()
sns.heatmap(corr, cmap="Blues_r",annot=True, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# After some experimentation, Pclass and Parch are highly correlated with Age and not each other; we use them to impute the missing ages. SibSp was also a good candidate, but including them as another subgroup creates groups made of entirely NaN ages, which we cannot take the mean of. Therefore, let us impute as follows:
# * Group all observations by Pclass and Parch (that is, 3 * 7 possible groups for the 851 observations to be in).
# * Find the mean age of each of the groups. This mean calculation ignores all missing values of Age.
# * Replace all missing values with the mean of that observation's group.
# 
# Note this all happens in the first line of code.

# In[112]:



data_train["Age"] = data_train["Age"].fillna(data_train.groupby(["Pclass","Parch"])["Age"].transform("mean")) 
data_test["Age"] = data_test["Age"].fillna(data_train.groupby(["Pclass","Parch"])["Age"].transform("mean")) 
    # Note - test set imputed FROM TRAINING set

sns.distplot(data_train.Age);

#Check: no records exist where Age = NaN
print(data_train[data_train["Age"].isnull()])
print(data_test[data_test["Age"].isnull()])


# # Binning
# 
# Let's consider the effect of binning 'Age' and 'Fare' into fewer categories.
# 
# [Future project to analyze what the best bins are, and if it's even effective]

# In[113]:


# data_train["Age"] = original_train["Age"]
# data_train["Fare"] = original_train["Fare"]

age_describe =  original_train["Age"].describe()
agebins = (0, age_describe[4], age_describe[5], age_describe[6], age_describe[7])

group_names = ["Child", "Young Adult", "Adult", "Senior"]
data_train["Age"] = pd.cut(data_train["Age"], agebins, labels=group_names)
data_test["Age"] = pd.cut(data_test["Age"], agebins, labels=group_names) # Note - test set imputed FROM TRAINING set
print(data_train["Age"].value_counts())
print(data_test["Age"].value_counts())

fare_describe =  original_train["Fare"].describe()
farebins = (0, fare_describe[4], fare_describe[5], fare_describe[6], fare_describe[7])

group_names = ["Budget", "Cheap", "Regular", "Expensive"]
data_train["Fare"]  = pd.cut(data_train["Fare"], farebins, labels=group_names)
data_test["Fare"]  = pd.cut(data_test["Fare"], farebins, labels=group_names)
print(data_train["Fare"].value_counts())
print(data_test["Fare"].value_counts()) # Note - test set imputed FROM TRAINING set



# In[114]:


data_train.head()


# In[115]:


data_train = data_train.drop(columns=["Fare", "Age"])
data_test = data_test.drop(columns=["Fare", "Age"])
data_train.head()


# <font size=6>**Data Mangling to Get a Classifer Out the Door** </font>

# In[116]:


X_train = data_train.drop("Survived", axis=1)
Y_train = data_train["Survived"]
X_test  = data_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape
X_test.shape


# In[117]:


# Y_train.describe()
# X_train.describe()
X_test.describe()


# In[129]:


# Random Forest
# https://www.kaggle.com/startupsci/titanic-data-science-solutions


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# wait that's actually decent even though it's training data

#  <font size=6>**Getting a Submission Out the Door** </font>

# In[143]:



submission = pd.DataFrame({
        "PassengerId": original_test["PassengerId"],
        "Survived": Y_pred
    })
submission.head()
# submission.to_csv('../output/submission.csv', index=False)
submission.to_csv('submission24Apr2018.csv', index = False)


# In[140]:



original_gendersubmission =  pd.read_csv('../input/gender_submission.csv')
original_gendersubmission.head()

