#!/usr/bin/env python
# coding: utf-8

# # **Titanic Machine Learning **
# 
# This is my first machine learning competition and I am trying to structure this notebook in order to:
# 
# 1) Have a starting point for each future competition (libraries, techniques, code..);
# 
# 2) Help people like me (self-learners without a technical background) understanding how a machine learning competition works, how to make data analysis and predictions using ML techniques.
# 
# This notebook is a work in progress: I have put it together as a starting point to work on in my free time.
# 
# Have fun!

# # **Module Importing**

# As the first step all the necessary libraries will be imported; this list will be updated as we are going forward.

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # **Data importing**

# The code below will import the data and create Pandas Dataframes to manage them.
# 
# The combined variable create a dataframe that is the union of the train and test dataframes.

# In[37]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combined = [train, test]


# # **First review of the data**

# In[5]:


train.head()


# In[4]:


train.describe()


# # **How many passengers survives?**

# From the 'describe' function above we can see (row 'mean', column 'Survived') that the mean of the passenger survived is 0.383838.
# 
# We can now analyze the survival rate using as a criteria the Passenger Class and the Sex.

# In[65]:


sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train)


# What comes out is that the higher the class of the passenger, the more its possibilities to survive.
# 
# Also, women survival rate is greater than men' ones.
# 
# This aspect is strictly correlated to the maritime tradition of evacuating women and children first.
# 
# In fact, if we group the data above for age, what we see is that children have a higher survival rate.

# In[6]:


group_by_age = pd.cut(train["Age"], np.arange(0, 90, 10))
age_grouping = train.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()


# # **Plotting Age Distribution and its relation with the Passengers Class**

# We will now use a swarmplot to see the relations between Age, Class and Sex.

# In[40]:


sns.swarmplot(x="Pclass", y="Age", hue="Sex", data=train)
plt.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
plt.title("Age distribution vs Class", fontsize=15)


# # **Using a facet grid to create a box plot of Age Distribution**

# An alternative to the swarmplot could be using a box plot, as shown below.
# 
# Thanks to these two plots, we discovered that 1st class passengers seems older: probabily, according to the age, they can afford a expensive ticket.

# In[9]:


g = sns.FacetGrid(train, col="Pclass")
g.map(sns.boxplot, "Sex", "Age", palette="Set3")


# # **Machine Learning**

#  As a starting point, we will applying all the following models and then compare the results.
# 
# - Logistic Regression
# - KNN or k-Nearest Neighbors
# - Support Vector Machines
# - Naive Bayes classifier
# - Decision Tree
# - Random Forest
# - Perceptron
# - Artificial neural network
# - RVM or Relevance Vector Machine

# # **Preparing the Data**

# **Categorical features analysis**

# Categorical features needs to be converted in order to apply our models.
# 
# This needs to be done because Machine Learning algorithms cannot elaborate strings.
# 
# 'Sex' is the first feature we will convert.

# In[38]:


for dataset in combined:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head()


# **Research of NaN values**

# NaN values (not a number) needs to be replaced/dropped or the Machine Learning algorithms will not work.
# In the two lines below, we will check which columns has NaN values (if True, there is at least a NaN record in the column)

# In[93]:


train.isnull().any()


# In[115]:


test.isnull().any()


# Let's now replace the NaN values with the mean of the value of the column.

# In[39]:


train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())


# Let's check the data again to see if the substitution is OK.

# In[95]:


train.isnull().any()


# In[161]:


test.isnull().any()


# We will now work on the 'Cabin' column.
# 
# Let's starting filling NaN values

# In[40]:


train['Cabin'].fillna('U', inplace=True)
train['Cabin'] = train['Cabin'].apply(lambda x: x[0])
train['Cabin'].unique()


# In[41]:


test['Cabin'].fillna('U', inplace=True)
test['Cabin'] = test['Cabin'].apply(lambda x: x[0])
test['Cabin'].unique()


# In[42]:


replacement = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

train['Cabin'] = train['Cabin'].apply(lambda x: replacement.get(x))
train['Cabin'] = StandardScaler().fit_transform(train['Cabin'].values.reshape(-1, 1))
train.head()['Cabin']


# In[43]:


test['Cabin'] = test['Cabin'].apply(lambda x: replacement.get(x))
test['Cabin'] = StandardScaler().fit_transform(test['Cabin'].values.reshape(-1, 1))
test.head()['Cabin']


# We can apply the same logic developed for the Cabin column to the Embarked column
# The possible values are:
# C = Cherbourg, Q = Queenstown, S = Southampton
# We will check first if we find any NaN value (that we will replace with N) and then we will transform this feature in a list of numbers

# In[44]:


train['Embarked'].fillna('N', inplace=True)
train['Embarked'] = train['Embarked'].apply(lambda x: x[0])
train['Embarked'].unique()


# In[45]:


replacement = {
    'S': 0,
    'C': 1,
    'Q': 2,
    'N': 3
}

train['Embarked'] = train['Embarked'].apply(lambda x: replacement.get(x))
train['Embarked'] = StandardScaler().fit_transform(train['Embarked'].values.reshape(-1, 1))
train.head()['Embarked']


# In[46]:


test['Embarked'] = test['Embarked'].apply(lambda x: replacement.get(x))
test['Embarked'] = StandardScaler().fit_transform(test['Embarked'].values.reshape(-1, 1))
test.head()['Embarked']


# We will now work with the Fare column (filling the NaN values with the mean of the other columns.

# In[47]:


train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# # **New Features**

# Adding new features can help the model (more data is better is a principle that can be generally applied). 
# 
# We will add now create two new features: Family Size and Age*Class.

# **Family Size**

# In[48]:


def process_family_train():
    
    # introducing a new feature : the size of families (including the passenger)
    train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
    
    # introducing other features based on the family size
    train['Singleton'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    train['SmallFamily'] = train['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    train['LargeFamily'] = train['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    return train


# In[49]:


train = process_family_train()
train.head()


# In[50]:


def process_family_test():
    
    # introducing a new feature : the size of families (including the passenger)
    test['FamilySize'] = test['Parch'] + test['SibSp'] + 1
    
    # introducing other features based on the family size
    test['Singleton'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    test['SmallFamily'] = test['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    test['LargeFamily'] = test['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    
    return test


# In[51]:


test = process_family_test()
test.head()


# # **Interaction terms**

# **Age * Class**
# 
# This is an interaction term, since age and class are both numbers we can just multiply them.

# In[52]:


train['Age*Class']=train['Age']*train['Pclass']
test['Age*Class']=train['Age']*train['Pclass']


# **FamilySize*Class**

# In[53]:


train['FamilySize*Pclass']=train['FamilySize']*train['Pclass']
test['FamilySize*Pclass']=train['FamilySize']*train['Pclass']


# **Singleton*Class**

# In[54]:


train['Singleton*Pclass']=train['Singleton']*train['Pclass']
test['Singleton*Pclass']=train['Singleton']*train['Pclass']


# **SmallFamily*Class**

# In[55]:


train['SmallFamily*Pclass']=train['SmallFamily']*train['Pclass']
test['SmallFamily*Pclass']=train['SmallFamily']*train['Pclass']


# # Other Features

# **Title**

# We will not use the 'Name' column, but we can at least extract the Title from the name. 
# 
# There are quite a few titles going around, but I want to reduce them all to Mrs, Miss, Mr and Master.  

# In[56]:


# Grab title from passenger names

train["Name"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
train.head()


# In[57]:


# Show title counts by sex

train.groupby(["Sex", "Name"]).size().unstack(fill_value=0)


# In[58]:


# Titles with very low cell counts to be combined to "rare" level

rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
train.replace(rare_titles, "Rare title", inplace=True)

# Also reassign mlle, ms, and mme accordingly

train.replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)


# In[59]:


# Show title counts by sex

train.groupby(["Sex", "Name"]).size().unstack(fill_value=0)


# In[60]:


#Now we can create a method to map/replace the titles

replacement = {
    'Master': 0,
    'Miss': 1,
    'Mr': 2,
    'Mrs': 3,
    'Rare title': 4
}

train['Name'] = train['Name'].apply(lambda x: replacement.get(x))
train['Name'] = StandardScaler().fit_transform(train['Name'].values.reshape(-1, 1))
train.head()['Name']


# In[33]:


train.head()


# Let's apply the same logic to the test dataframe

# In[61]:


test["Name"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)
test.head()


# In[62]:


test.groupby(["Sex", "Name"]).size().unstack(fill_value=0)


# In[63]:


rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
test.replace(rare_titles, "Rare title", inplace=True)

test.replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)


# In[64]:


# Show title counts by sex

test.groupby(["Sex", "Name"]).size().unstack(fill_value=0)


# In[65]:


#Now we can create a method to map/replace the titles

replacement = {
    'Master': 0,
    'Miss': 1,
    'Mr': 2,
    'Mrs': 3,
    'Rare title': 4
}

test['Name'] = test['Name'].apply(lambda x: replacement.get(x))
test['Name'] = StandardScaler().fit_transform(test['Name'].values.reshape(-1, 1))
test.head()['Name']


# In[39]:


test.head()


# At this point, we will shape the dataframes dropping a few columns

# In[66]:


train_df = train.drop(['Ticket', 'PassengerId'], axis=1)
test_df = test.drop(['Ticket'], axis=1)
combined = [train_df, test_df]
train_df.shape, test_df.shape


# In[67]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[111]:


X_train.head()


# In[69]:


X_test.head()


# # **Machine Learning Algorithms**

# # **Logistic Regression**

# In[68]:


regr = LogisticRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)
acc_log = round(regr.score(X_train, Y_train) * 100, 2)
acc_log


# # **Support Vector Machines**

# In[69]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# # **k-Nearest Neighbors**

# In[70]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# # **Gaussian Naive Bayes**

# In[71]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# # **Perceptron**

# In[72]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# # **Linear SVC**

# In[73]:


linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# # **Stochastic Gradient Descent**

# In[74]:


sgd = SGDClassifier(max_iter=16050)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# # **Decision Tree**

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# # **Random Forest**

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# # **Evaluation of the models**

# In[77]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# # **Submission**

# In[129]:


subm = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
subm.to_csv('subm.csv', index=False)

