#!/usr/bin/env python
# coding: utf-8

# # Introduction to data science with python

# A classic machine learning workflow arround Titanic data,
#  1. Exploring data
#  1. Cleaning and feature Engineering
#  1. Simple ML algorithms
#  1. Improving the score

# ## 1. Exploring the data

# In[5]:


# Data libs
import numpy as np
import pandas as pd
# ML libs
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Plot libs
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objs as go
get_ipython().magic(u'matplotlib inline')
pyo.init_notebook_mode(connected=False)


# In[6]:


train, test = pd.read_csv("../input/train.csv"), pd.read_csv("../input/test.csv")
test_ids = test[["PassengerId"]]


# In[7]:


train.head()


# We will plot various features with their relation to survival rate to have an idea of correlations

# In[8]:


fig, axs = plt.subplots(ncols=3, figsize=(16,5))
sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=train, ax=axs[0]);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train, ax=axs[1]);
sns.violinplot(x="Survived", y="Age", hue="Sex", data=train, ax=axs[2]);


# We can already see some (strong) correlation between sex, age, Pclass, embarked and survival rate

# In[9]:


data_age = [train[train.Survived == 1].Age.dropna(), train[train.Survived == 0].Age.dropna()]
labels = ["Survived", "Not survived"]
fig = ff.create_distplot(data_age, labels, bin_size=2, show_rug=False)
pyo.iplot(fig)


# Correlations values

# In[10]:


corr = train.corr().abs().Survived.sort_values(ascending=False)[1:]
data = [go.Bar(
            x=corr.index.values,
            y=corr.values
    )]

pyo.iplot(data, filename='basic-bar')


# ## Cleaning & feature engineering
# 
# The data is not yet ready to be feed to machine learning algorithms.

# In[11]:


train.head()
# Keep train as read only
data = train.copy()


# ### Feature engineering
# 
# The name is not as it is exploitable, a popular idea on this dataset is to extract the title of the name as some are rare and their survival rate may be higher.

# In[12]:


data.Title = None
test.Title = None

title_cats = {
    "Mr": ["Mr."],
    "Miss": ["Miss.", "Ms.", "Mlle."],
    "Mrs": ["Mrs.", "Mme."],
    "Rare_M": ["Master.", "Don.", "Rev.", "Dr.", "Major.", "Sir.", "Col", "Capt.", "Jonkheer."],
    "Rare_F": ["Lady.", "the Countess.", "Dona."]
}

data["LastName"] = train["Name"].str.split(",", expand=True)[1].str.strip()
test["LastName"] = test["Name"].str.split(",", expand=True)[1].str.strip()

for c_title, l_title in title_cats.items():
    for title in l_title:
        data.loc[data["LastName"].str.startswith(title), "Title"] = c_title
        test.loc[test["LastName"].str.startswith(title), "Title"] = c_title

le = LabelEncoder().fit(data["Title"].append(test["Title"]))
data["Title"] = le.transform(data["Title"])
test["Title"] = le.transform(test["Title"])


# In[13]:


data["FamilySize"] = train["Parch"] + train["SibSp"] + 1
data["Alone"] = 0
data.loc[data["FamilySize"] == 1, "Alone"] = 1

test["FamilySize"] = test["Parch"] + test["SibSp"] + 1
test["Alone"] = 0
test.loc[test["FamilySize"] == 1, "Alone"] = 1


# In[14]:


train[train["Cabin"].notnull()][["Cabin"]].sample(10)


# In[15]:


le = LabelEncoder()

data.loc[train["Cabin"].notnull(), "Cabin"] = train.loc[train["Cabin"].notnull(), "Cabin"].str[0]
data.loc[train["Cabin"].isnull(), "Cabin"] = ""

test.loc[test["Cabin"].notnull(), "Cabin"] = test.loc[test["Cabin"].notnull(), "Cabin"].str[0]
test.loc[test["Cabin"].isnull(), "Cabin"] = ""

le.fit(data["Cabin"].append(test["Cabin"]))

data["Cabin"] = le.transform(data["Cabin"])
test["Cabin"] = le.transform(test["Cabin"])


# Let see the correlation between our engineeered feature and the Survived column

# In[16]:


data[["Title", "FamilySize", "Alone", "Survived", "Cabin"]].corr().abs()[["Survived"]]


# ### Cleaning data

# #### Preparing data for ML algorithms

# In[17]:


data.head()


# In[18]:


fig, axs = plt.subplots(ncols=2, figsize=(16,5))
sns.distplot(data[data["Age"].notnull()]["Age"], ax=axs[0])
sns.distplot(data[data["Fare"].notnull()]["Fare"], ax=axs[1])


# Age and Fare are mostly gaussian

# In[19]:


data["Sex"] = LabelEncoder().fit_transform(data["Sex"])
data.loc[train["Embarked"].notnull(), "Embarked"] = LabelEncoder().fit_transform(train.loc[train["Embarked"].notnull(), "Embarked"])
data["Age"] = pd.qcut(train["Age"], q=5, labels=False)
data["Fare"] = pd.qcut(train.loc[train["Fare"] != 0, "Fare"], q=5, labels=False)

data.loc[train["Age"].isnull(), "Age"] = None
data.loc[train["Fare"] == 0, "Fare"] = None

test["Sex"] = LabelEncoder().fit_transform(test["Sex"])
test.loc[test["Embarked"].notnull(), "Embarked"] = LabelEncoder().fit_transform(test.loc[test["Embarked"].notnull(), "Embarked"])
test["Age"] = pd.qcut(test["Age"], q=5, labels=False)
test["Fare"] = pd.qcut(test.loc[test["Fare"] != 0, "Fare"], q=5, labels=False)

test.loc[test["Age"].isnull(), "Age"] = None
test.loc[test["Fare"] == 0, "Fare"] = None


# In[20]:


data.head()


# One Hot Encoding for non hierarchical categories

# In[21]:


ohe_columns = ["Cabin", "Embarked", "Title"]
data = pd.get_dummies(data, prefix=ohe_columns, columns=ohe_columns, drop_first=True)
test = pd.get_dummies(test, prefix=ohe_columns, columns=ohe_columns, drop_first=True)


# In[22]:


set(data.columns) - set(test.columns)


# In[23]:


test["Cabin_8"] = 0


# In[24]:


data.drop(["PassengerId", "Name", "LastName", "SibSp", "Parch", "Ticket"], axis=1, inplace=True)
test.drop(["PassengerId", "Name", "LastName", "SibSp", "Parch", "Ticket"], axis=1, inplace=True)


# #### Dealing with empty data

# A very simple strategy is to use the median value for all missing values. A not so simple is to use another ML algorithm to guess age

# In[25]:


data.loc[data["Age"].isnull() & data["Fare"].isnull(), "Age"] = data["Age"].value_counts().idxmax()
data.loc[data["Age"].isnull() & data["Fare"].isnull(), "Fare"] = data["Fare"].value_counts().idxmax()

test.loc[test["Age"].isnull() & test["Fare"].isnull(), "Age"] = test["Age"].value_counts().idxmax()
test.loc[test["Age"].isnull() & test["Fare"].isnull(), "Fare"] = test["Fare"].value_counts().idxmax()


# In[26]:


clf = GradientBoostingClassifier() 
X_train = data[data["Fare"].notnull()].copy()

X_Age_nn = X_train[X_train["Age"].notnull()].drop("Age", axis=1)
y_Age_nn = X_train[X_train["Age"].notnull()]["Age"]
X_Age_n = X_train[X_train["Age"].isnull()].drop("Age", axis=1)
X_Age_train, X_Age_test, y_Age_train, y_Age_test = train_test_split(X_Age_nn, y_Age_nn, test_size=0.80)
clf.fit(X_Age_nn, y_Age_nn)
predictions = clf.predict(X_Age_test)
print(metrics.accuracy_score(predictions, y_Age_test))

data.loc[data["Fare"].notnull() & data["Age"].isnull(), "Age"] = clf.predict(X_Age_n)


# In[27]:


clf = GradientBoostingClassifier() 
X_train = test[test["Fare"].notnull()].copy()

X_Age_nn = X_train[X_train["Age"].notnull()].drop("Age", axis=1)
y_Age_nn = X_train[X_train["Age"].notnull()]["Age"]
X_Age_n = X_train[X_train["Age"].isnull()].drop("Age", axis=1)
X_Age_train, X_Age_test, y_Age_train, y_Age_test = train_test_split(X_Age_nn, y_Age_nn, test_size=0.80)
clf.fit(X_Age_nn, y_Age_nn)
predictions = clf.predict(X_Age_test)
print(metrics.accuracy_score(predictions, y_Age_test))

test.loc[test["Fare"].notnull() & test["Age"].isnull(), "Age"] = clf.predict(X_Age_n)


# In[ ]:


clf = GradientBoostingClassifier() 
X_train = data.copy()

X_Fare_nn = X_train[X_train["Fare"].notnull()].drop("Fare", axis=1)
y_Fare_nn = X_train[X_train["Fare"].notnull()]["Fare"]
X_Fare_n = X_train[X_train["Fare"].isnull()].drop("Fare", axis=1)
X_Fare_train, X_Fare_test, y_Fare_train, y_Fare_test = train_test_split(X_Fare_nn, y_Fare_nn, test_size=0.80)
clf.fit(X_Fare_nn, y_Fare_nn)
predictions = clf.predict(X_Fare_test)
print(metrics.accuracy_score(predictions, y_Fare_test))

data.loc[data["Fare"].isnull(), "Fare"] = clf.predict(X_Fare_n)


# In[ ]:


clf = GradientBoostingClassifier() 
X_train = test.copy()

X_Fare_nn = X_train[X_train["Fare"].notnull()].drop("Fare", axis=1)
y_Fare_nn = X_train[X_train["Fare"].notnull()]["Fare"]
X_Fare_n = X_train[X_train["Fare"].isnull()].drop("Fare", axis=1)
X_Fare_train, X_Fare_test, y_Fare_train, y_Fare_test = train_test_split(X_Fare_nn, y_Fare_nn, test_size=0.80)
clf.fit(X_Fare_nn, y_Fare_nn)
predictions = clf.predict(X_Fare_test)
print(metrics.accuracy_score(predictions, y_Fare_test))

test.loc[test["Fare"].isnull(), "Fare"] = clf.predict(X_Fare_n)


# ## Let's do it

# In[ ]:


clf = GradientBoostingClassifier() 
X = data.drop("Survived", axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
metrics.accuracy_score(predictions, y_test)


# In[ ]:


predictions = clf.predict(test)
final = test_ids.copy()
final["Survived"] = predictions
final.set_index("PassengerId", inplace=True)
final.to_csv("final.csv")

