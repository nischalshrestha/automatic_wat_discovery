#!/usr/bin/env python
# coding: utf-8

# # Using Machine Learning to Predict Titanic Survivors
# 
# This is my first submission for a kaggle competition. I will try to keep things simple and will explain things on the go. In  this notebook I will build a machine learning model using sklearn to predict the outcomes of each passenger aboard titanic. I will build this model step by step. This kernel will help people who are getting started with data visualization, analysis and machine learning.
# 
# 
# #### If you like my work. Please, leave an upvote, the upvote will me to help to contribute more  and more to the community.
# 
# #### Please leave your valuable  suggestions in the comments section.
# 
# #### Follow me for even better kernels than this one.
# 
# 
# 
# ## What type of problem is this one?
# 
# Since we have to classify passengers as either survived, or not survived. Hence, This is a supervised classification machine learning problem.
# 
# 
# ## Content 
# 
# The file 'train.csv' contains 12 columns and - rows. Each row containes details of individual passenger onboard. The columns  are: 
# 
# 1. PassengerId - type should be integers
# 2. Survived - Survived or Not
# 3. PclassClass - of Travel
# 4. Name - Name of Passenger
# 5. Sex - Gender
# 6. Age - Age of passenger
# 7. SibSp - Number of Sibling/Spouse abord
# 8. Parch - Number of Parent/Child abord
# 9. Ticket
# 10. Fare
# 11. Cabin
# 12. EmbarkedThe port in which a passenger has embarked. C - Cherbourg, S - Southampton, Q = Queenstown
# 
# 
# ## Version
# 
# Version 1.0
# 
# 
# 
# 
# ## Index of Content
# 
# 1. Importing packages
# 2. Importing dataset
# 3. Analysing dataset
# 4. Assumptions based on data analysis done so far
# 5. Actions based on assumptions
# 6. Visualizing by plotting data
# 7. Feature Engineering
# 8. Creating New Features
# 9. Build Model and Make Predictions
# 10. Compare Model Performances
# 

# 
# 
# 
# ## 1. Importing Modules

# In[ ]:


# data analysis and wrangling
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
get_ipython().magic(u'matplotlib inline')

# machine learning
#Let's import them as and when needed

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input/"))


# ## 2. Importing dataset
# 
# We will use Python Pandas package to import the dataset and play with it.

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ## 3. Analysing the dataset
# 
# We will use pandas for this.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# The data is not fit to feed into machine learning model. We have to clean it. 

# In[ ]:


train_data.info()
print('-'*50)
test_data.info()


# After seeing these head of the dataset. We can comment on categorical features, numerical features, data types of features, typos in features and null or empty features.
# 
# #### Categorical features: 
# A categorical variable (sometimes called a nominal variable) is one that has two or more categories, but there is no intrinsic ordering to the categories. This is further classified as nominal, ordinal, ratio, or interval based.
# -  Categorical: Survived, Sex and Embarked
# -  Ordinal: Pclass
# 
# #### Numerical Features:
# As the name suggest, these values are numerical in nature and changes from sample to sample. This is further classified as discrete, continuous, or timeseries based.
# -  Continuous: Age, Fare
# -  Discrete: SibSp, Parch
# 
# #### Data types of features:
# 
# -  Seven features are integer or floats. Six in case of test dataset
# -  Five features are strings (object)
# -  MIXED DATA TYPES: Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric
# 
# #### Featurs with error and typos:
# -  Name feature may contain errors or typos ass there are several ways to write a name
# 
# #### Null or Empty Features:
# -  Age, Cabin and Embarked containes a lot of null values for the training dataset

# In[ ]:


train_data.describe()


# Points to be taken from these numerical features:
# 
# -  Total samples are 891 i.e. 40% of the actual number of passengers on board the Titanic(2224)
# -  Survived is a categorical feature with 0 or 1 values.
# -  Most expensive ticket is $512

# In[ ]:


train_data.describe(include=['O'])


# Points to be taken from these categorical features:
# 
# -  No person with same name.
# -  Sex variable have two possible values with 65% male (top=male, freq= 577/count=891)
# -  Cabin values have several duplicates as many passengers shared cabin.
# -  S port is used most among the three possible values.
# -  Ticket feature has 681 unique values i.e.: 22% duplicate ratio

# ## 4. Assumptions based on data analysis done so far:
# 
# #### Based on our data analysis so far. We can state that:
# 
# -  Ticket Feature contains 22% duplicates and can be dropped. As there may not be a correlation b/w Ticket and Survival
# -  Cabin feature can be dropped as it is highly incomplete and contains null values
# -  We can drop PassengerId as it does not contribute to survival
# -  We can create a new feature called Title for Name feature
# -  We can create a new feature called FamilySize based on Parch and SibSp to get total count of family members on board
# -  We should complete Age feature as it is directly correlated to survival
# -  Women might have been more likely to have survived
# -  Children below some certain age were also more likely to have survived
# -  The upper class passengers were more likely to have survived

# ## 5. Actions based on assumptions

# In[ ]:


#drop the Cabin and Ticket columns in both dataset. We also don't need the PassengerId in training dataset.

train_data.drop(labels = ['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(labels = ['Ticket', 'Cabin'], axis=1, inplace=True)


# Let's see the number of null values now.

# In[ ]:


print('Training Data')
print(pd.isnull(train_data).sum())
print("-"*50)
print('Testing Data')
print(pd.isnull(test_data).sum())


# #### Age columns seems to have null values.
# 
# We will look at the distribution of Age column to see if it's skewed or symmetrical. This will help us to determine what values to replaec wit NaN values.

# In[ ]:


sns.distplot(train_data['Age'].dropna())


# As it is evident from the graph. The distribution is slightly skewed right. That's why we will fill the null values with median for better accuracy.
# 
# I know you have questions here. Ask me in the comments section. :)

# In[ ]:


train_data['Age'].fillna(train_data['Age'].median(), inplace= True)
test_data['Age'].fillna(test_data['Age'].median(), inplace= True)


# Since, We know from previous steps that "S" is the most Embarked port. Let's fill the null values in Embarked with "S" port.

# In[ ]:


train_data['Embarked'].fillna("S", inplace = True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)


# Let's see the status of null values now.

# In[ ]:


print('Training Data')
print(pd.isnull(train_data).sum())
print("-"*50)
print('Testing Data')
print(pd.isnull(test_data).sum())


# Yes!! We got rid of missing values. Now let's see our cleaned data.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ## 6. Visualizing by Plotting Data
# 
# Visualizing the data is important to see the trends and general associations of Variables. We can make different kinds of graphs for the features we want to work with.

# In[ ]:


plt.figure(figsize=(8,4))
plt.tight_layout()
sns.barplot(x='Sex', y='Survived', data=train_data)
plt.title('Distribution of Survival based on Gender')
plt.show()


# Clearly, Women were the top survivors.
# 
# Also, Gender is a good feature to use for our machine learning model. But, we have to engineer it before feeding into our model.

# In[ ]:


plt.figure(figsize=(8,4))
plt.tight_layout()
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('Distribution of Survival based on Class')
plt.show()


# So, The first class people were more likely to survive. 

# In[ ]:


plt.figure(figsize=(12,6))
plt.tight_layout()
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_data)
plt.title('Distribution of Survival based on Gender and Class')
plt.show()


# So, Class also plays an important role in surival of the passengers. 1st Class were more likely to survive than other classes passengers.

# In[ ]:


g = sns.FacetGrid(train_data, col="Survived", margin_titles=True)
g.map(plt.hist, "Age", color="steelblue", bins=20)


# In[ ]:


sns.swarmplot(x="Survived", y="Age", hue="Sex", palette=["r", "c", "y"], data=train_data)


# As evident from the these distributions that younger people were more likely to survive than those of older people. Also, There were more females who survived.
# 
# Let's drap a pairplot to see possible relatins between all the features.

# In[ ]:


sns.pairplot(train_data)


# ## 7. Feature Engineering
# 
# Categorical features needs to be represented as numerical values before we feed it into the machine learning model. "Sex" and "Embarked" columns needs to be engineered.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# We can change Sex to binary, as either 1 for female or 0 for male. We do the same for Embarked. 

# In[ ]:


train_data['Sex'] = train_data['Sex'].map( {'female': 1, 'male': 0}).astype(int)


# In[ ]:


train_data.head()


# In[ ]:


test_data['Sex'] = test_data['Sex'].map( {'female': 1, 'male': 0}).astype(int)


# In[ ]:


test_data.head()


# Let's do the same with Embarked column. Since, We already took care of the NaN values then we just have to convert the categorical port feature to numerical port feature.

# In[ ]:


train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ## 8. Create New Features
# 
# Sometimes in a dataset individual features might be of no use. But, if you create a new features using them, then you might be able to get some more insights into the problem. But, who knows. So, Let's try it out.
# 
# After reading on kaggle discussions, I came to know that we can combine "SibSp" and "Parch" and make a new feature named "FamilySize". Since, People who have had family might have risked their life to search for their family.
# 
# Also, We can make a new feature for people who were alone on the ship. We can name it as "Single" or "IsAlone" or whatever you like.

# In[ ]:


train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[ ]:


train_data['Single'] = train_data.FamilySize.apply(lambda x: 1 if x == 1 else 0)
test_data['Single'] = test_data.FamilySize.apply(lambda x: 1 if x == 1 else 0)


# Let's have a look at our dataset.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# Everything looks out numeric except the name column. Let's engineer the name column also.
# 
# Since,name column is not used. We can extract the title from the names and then encode them in to numeric values. Then we can use name column to predict the outcome.

# In[ ]:


train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) ##Regular Expression <3
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# Let's replace title with common name and the unique ones as unique.
# 
# Let's also combine the dataset into an array so that we can perform operations on both dataset using a loop.

# In[ ]:


title_list = list(set(train_data['Title']))
title_list


# In[ ]:


mix = [train_data, test_data] ## Just so save ourself some time and repeated code.

for dt in mix:
    dt['Title'] = dt['Title'].replace(['Dr', 'Col', 'Sir', 'Countess', 'Jonkheer', 'Lady', 'Don', 'Capt', 'Major', 'Rev',                                       ], 'Unique')
    dt['Title'] = dt['Title'].replace('Mlle', 'Miss')
    dt['Title'] = dt['Title'].replace('Ms', 'Miss')
    dt['Title'] = dt['Title'].replace('Mme', 'Mrs')


# In[ ]:


map_title = {'Mrs': 1, 'Miss': 2, 'Mr': 3, 'Master': 4, 'Unique': 5}

for dt in mix:
    dt['Title'] = dt['Title'].map(map_title)
    dt['Title'] = dt['Title'].fillna(0)
    
train_data.head()


# Nice!
# 
# We can now drop "Name", "SibSp" and "Parch" column.

# In[ ]:


train_data = train_data.drop(['Name', 'SibSp', 'Parch'], axis=1)
test_data = test_data.drop(['Name', 'SibSp', 'Parch'], axis=1)


# In[ ]:


train_data.head()


# ## 9. Build Model and Make Prediction
# 
# We are all set to build our model. I will be using different models and will select the one with the best accuracy.
# 
# 
# 
# #### Import all the sklearn models to test

# In[ ]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# to evaluate model performance, we can use the accuracy_score function.


# Let's define features in Training/Test set

# In[ ]:


X_train = train_data.drop(labels=['Survived'], axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId',axis=1).copy()


# In[ ]:


X_train.shape,  y_train.shape, X_test.shape


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
accuracy_log = logreg.score(X_train, y_train)
print("The accuracy for the Logistic Regression is: " + str(accuracy_log))


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
accuracy_svc = svc.score(X_train, y_train)
print("The accuracy for Support Vector Machines is:" + str(accuracy_svc))


# In[ ]:


# K Nearest Neighbours

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
accuracy_knn = knn.score(X_train, y_train)
print("The accuracy of KNN is: " + str(accuracy_knn))


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
accuracy_gnb = gaussian.score(X_train, y_train)
print("The accuracy of Gaussian Naive Bayes is: " + str(accuracy_gnb))


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
accuracy_lsvc = linear_svc.score(X_train, y_train)
print("The accuracy of linear SVC is: " + str(accuracy_lsvc))


# In[ ]:


#Decision Tree

d_tree = DecisionTreeClassifier()
d_tree.fit(X_train, y_train)
Y_pred = d_tree.predict(X_test)
accuracy_d_tree = d_tree.score(X_train, y_train)
print("The accuracy of Decision Tree is: " + str(accuracy_d_tree))


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100) #tried with 80, 90, 100 no difference
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
accuracy_r_forest = random_forest.score(X_train, y_train)
print("The accuracy of Random Forest is: " + str(accuracy_r_forest))


# ## 10. Compare Model Performances
# 
# We have done predictions with many models. Now, we should see which model performed the best.

# In[ ]:


model_performance = pd.DataFrame({
    "Model": ["Logistic Regression", "Support Vector Machines", "K Nearest Neighbours", "Gaussian Naive Bayes",
             "Linear SVC", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_log, accuracy_svc, accuracy_knn, accuracy_gnb, accuracy_lsvc, accuracy_d_tree, accuracy_r_forest]
})

model_performance.sort_values(by="Accuracy", ascending=False)


# It's is clear that both Decision Tree and Random Forest score the same. We choose to use the Random Forest as they tend not to overfit as decision tree.

# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": Y_pred
})

#submission.to_csv("../output/titanic.csv", index=False)


# ### References: 
# * [startupsci](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# * [Blog](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/)
