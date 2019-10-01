#!/usr/bin/env python
# coding: utf-8

# Due Credits to:
# 
# Titanic Best Working Classfier, authored by Sina, for comprehensive features engineering 
# 
# Check out that kernel here: https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier
# 

# We all get the idea of Titanic. It has been in books, movies, documentaries and etc. Ship hit iceberg, met its demise. But could there be more to the story? What about the survivors of Titanic? Are there properties that certain passengers had that helped in surviving the crash? This notebook would look deeper beyond the tip of the ice berg to investigate important attributes that could have boosted survival rates.
# 
# **Disclaimer**
# This notebook does not go too far as to implement best practices such as cross-validating across different models and ensembling.
# Rather, the aim of this notebook is for the author to familiarize with features engineering, data wrangling, implementing random forest and gradient boosting model.
# 
# Of course, any further suggestions for improvement is always welcomed. Feel free to contact @: 
# alanchn31@gmail.com

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


full_data = [df_train,df_test]


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# **Features Engineering**

# We can combine Parch and SibSp into 1 column, known as number of family members (Familymem). 
# In addition, drop SibSp and Parch columns:

# In[ ]:


for data_set in full_data:
    data_set["Familymem"] = data_set["SibSp"] + data_set["Parch"]
    data_set.drop(columns = ["SibSp","Parch"])


# In[ ]:


df_train.head()


# Regex should be used to extract title out of Name.

# In[ ]:


import re


# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(df_train['Title'], df_train['Sex']))


# In[ ]:


print(pd.crosstab(df_test['Title'], df_test['Sex']))


# In[ ]:


df_test[df_test["Title"]=="Dona"]


# In[ ]:


df_test.loc[df_test["Title"]=='Dona',"Title"] = "Mrs"


# In[ ]:


print(pd.crosstab(df_test['Title'], df_test['Sex']))


# **Data Wrangling:**

# Create Dictionary to map each category of Embarked destination  and Sex to a numeric category:

# In[ ]:


for dataset in full_data:
    encoded_cols = {"Embarked":     {"C": 1, "S":2, "Q":3}, "Sex": {"male":1,"female":0} , 
                    "Title": {"Capt":1,"Col":2,"Countess":3,"Don":4,"Dr":5,"Jonkheer":6,
                             "Lady": 7, "Major": 8, "Master": 9, "Miss": 10, "Mlle": 11,
                             "Mme": 12,"Mr": 13, "Mrs": 14, "Ms": 15, "Rev": 16, "Sir": 17} }
    dataset.replace(encoded_cols, inplace=True)


# In[ ]:


df_train.head()


# **Imputing Missing Values:**

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# Cabin column should be dropped as it is subjective to interpretation (which makes it difficult to impute) and there are a large number of missing observations in both training and test sets.
# 
# Missing Value for Fare in test data can be imputed by mean.
# 
# This leaves Age which we must find a way to impute.

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(x="Title",y="Age",data=df_train,ax=ax)
plt.tight_layout()


# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(x="Pclass",y="Age",data=df_train,ax=ax)
plt.tight_layout()


# In[ ]:


df_train.groupby(['Pclass','Title'])["Age"].mean()


# Generally, the higher the ticket class, the older the person. Title helps to differentiate a person's Age as well. These are 2 important pieces of information that can allow us to impute Age.

# In[ ]:


for data_set in full_data:    
    data_set["Age"] = data_set.groupby(['Pclass','Title'])["Age"].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


df_test.groupby(['Pclass','Title'])["Age"].mean()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test[df_test["Age"].isnull()]


# In[ ]:


df_test[df_test["Title"]==15]


# Unable to impute 1 value (Class 3 with Title Ms), due to it being unique in the group. We will have to impute it by age of the person with title 'Ms' in our training data

# In[ ]:


df_test["Age"].fillna(28,inplace=True)


# In[ ]:


df_train["Age"].describe()


# In[ ]:


df_test["Age"].describe()


# Great! We are left with Embarked and Fare to impute for training and test set respectively

# For simplicity's sake, let's drop the rows with missing values of Embarked

# In[ ]:


df_train = df_train.dropna(subset=["Embarked"])


# To impute the missing value of Fare in test data, we can just use mean

# In[ ]:


df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].mean())


# In[ ]:


df_test.isnull().sum()


# We can drop the Cabin columns for both training and test sets, given the substantial number of rows with missing values, 
# along with the columns we do not need

# In[ ]:


train_id = df_train["PassengerId"]
test_id = df_test["PassengerId"]


# In[ ]:


for data_set in full_data:
    data_set.drop(columns=["PassengerId","Cabin","Name","SibSp","Parch","Ticket"],inplace=True)


# In[ ]:


df_train.drop(columns=["PassengerId","Cabin","Name","SibSp","Parch","Ticket"],inplace=True)


# In[ ]:


df_train.head()


# **Exploratory Data Analysis**

# Counts for Ports of Embarkation:

# In[ ]:


sns.countplot(x = "Embarked", data = df_train)


# It seems that most people were heading for Southampton, followed by Cherbourg and then Queenstown

# Counts for Cabin Number:

# In[ ]:


sns.countplot(x = "Familymem", data = df_train)


# Most passengers boarded with less than 2 family members.
# Majority of passengers boarder without any family members.

# In[ ]:


sns.countplot(x = "Sex", data = df_train)


# In[ ]:


sns.countplot(x = "Pclass", data = df_train)


# Most passengers were in 3rd class with majority of overall passengers being Male

# **Training our Models:**

# In[ ]:


train_x = df_train.drop(columns=["Survived"])
train_y = df_train["Survived"]


# Let's start our training with a random forest model!
# 
# Credits to: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 for its guidance towards cross-validation for a random forest model.
# 

# The most important settings for a random forest are the number of trees in the forest (n_estimators) and the number of features considered for splitting at each leaf node (max_features)

# Let's try adjusting these hyperparameters:
# 
# 1) n_estimators = number of trees in the foreset
# 
# 2) max_features = max number of features considered for splitting a node
# 
# 3) max_depth = max number of levels in each decision tree
# 
# 4) min_samples_split = min number of data points placed in a node before the node is split
# 
# 5) min_samples_leaf = min number of data points allowed in a leaf node
# 
# 6) bootstrap = method for sampling data points (with or without replacement)

# To do so, let's set up a parameter grid to sample:

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# The benefit of a random search is that we are not trying every combination, but selecting at random to sample a wide range of values.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_x, train_y)


# In[ ]:


rf_random.best_params_


# In[ ]:


rf_random.best_score_


# Our best cross-validation score is around 83.1%

# In[ ]:


clf = RandomForestClassifier(bootstrap=False,max_depth=30,max_features="sqrt",min_samples_leaf=2,min_samples_split=10,n_estimators=800)


# In[ ]:


clf.fit(train_x,train_y)


# In[ ]:


preds_rf = clf.predict(df_test)


# In[ ]:


preds_rf


# In[ ]:


rf_submit = pd.DataFrame({"Survived": preds_rf,"PassengerId": test_id})


# In[ ]:


rf_submit = rf_submit[["PassengerId","Survived"]]


# In[ ]:


rf_submit.to_csv("submission_rf.csv",index=False)


# Test accuracy is around 77.5% after submission

# In[ ]:


importances = clf.feature_importances_
features = ['Pclass','Sex','Age','Fare','Embarked','Familymem','Title']


# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
plt.title('Feature Importances based on trees generated by Random Forest Classifier')
sns.barplot(x=features,y=importances,ax=ax)
plt.tight_layout()


# Next model that can be train is an XGBoost Classifier

# In[ ]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(train_x, train_y)


# In[ ]:


preds_xgb = xgb_model.predict(df_test)


# In[ ]:


preds_xgb


# In[ ]:


xgb_submit = pd.DataFrame({"Survived": preds_xgb,"PassengerId": test_id})
xgb_submit = xgb_submit[["PassengerId","Survived"]]
xgb_submit.to_csv("submission_xgb.csv",index=False)


# XGBoost improved test accuracy to ~78.9%

# Areas of Improvement:
# 
# * IsAlone (1 if passenger is alone and 0 if passenger is not) could be included in features engineering
# 
# * Cross-Validation could have been performed across different models such as RandomForest, AdaBoost.
# 
# * Ensembling could also be explored to boost accuracy score.

# 

# 

# 
