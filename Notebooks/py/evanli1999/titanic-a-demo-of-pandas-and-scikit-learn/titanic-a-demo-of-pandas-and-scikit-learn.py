#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# This is a demonstration the Pandas and Scikit-learn libraries. Our goal is to create a model that predicts which passengers onboard the titanic survived the disaster. 

# # Obtaining Data
# 
# First, let's import the tools we will be using, as well as loading in the data provided: 

# In[126]:


import numpy as np
import re
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


original = pandas.read_csv("../input/train.csv")


# # Features

# In[79]:


print(original.columns.values)


# Let's break down immedieately signifant features, from an intuitive point of view: 
# 
# **Pclass, Sex, Age**
# 
# Immedieately, we know that this is probably important: the rich, young, and female were prioritized spots on lifeboats. 
# 
# **Name**
# 
# Initally, it is unclear how we can use the passenger's name in our model. However, we notice that the *title* of the passenger is also included here. Perhaps that will be useful. 
# 
# **Parch, Sibsp**
# 
# Perhaps large families were priortized for spots on lifeboats? 
# 
# ** Fare**
# 
# This is a great indication of the passenger's socio-economic status. Again, the rich were favoured for spots on the lifeboats. 
# 
# ** Cabin** 
# 
# Does the first letter represent the level their rooms were on? If so, this could allude to their social status. However, since we are already given ticket fare and class, cabin number does not seem to offer anything new. 

# # Missing and Restoring Data
# 
# Some passengers are missing some data entries. In particular, age, cabin, and embarked are missing values. While age and embarked all have a good number of entries, we have less than a quarter of the cabin numbers. 
# 
# **Age**
# 
# The simplest solution is to simply use the mean value of the column. It is indeed possible to use the other features of a passenger to make a better prediction of their age, but for this demonstration I will limit to using the mean: 

# In[128]:


original['Age'] = original['Age'].replace(np.nan, original['Age'].mean(), regex=True)


# **Embarked**
# 
# We replace missing values with M(issing):

# In[130]:


original['Embarked'] = original['Embarked'].replace(np.nan, "M", regex=True)
#We fill in missing values with M

original.sample(5)


# # Extracting Titles
# 
# Titles will be useful to us! Create a new column in our dataframe only for the column. Note that they are preceeded by ", ", and ends with ".": 

# In[131]:


name = original['Name']
titles = []

for i in range(len(name)):
	s = name[i]
	title = re.search(', (.*)\.', s)
	title = title.group(1)
	titles.append(title)

original['Titles'] = titles


# We now have a new column in our dataframe called Titles: 

# In[98]:


original.sample(5)


# Credit to Manav Sehgal for totalling the number of occurances of the titles we have just generated. We notice some very rare titles (e.g. Countess). Let's group these titles together. Also, we can group together titles that have equvilent meanings, or ones that may have simply been spelling errors (e.g. Mme and Mrs, Mlle and Miss, Ms and Miss ... etc): 

# In[133]:


original['Titles'].replace(['Sir', 'Rev', 'Major', 'Lady', 'Jonkheer', 'Dr', 'Don', 'Countess', 'Col', 'Capt'], 'Name')

original['Titles'].replace(['Ms', 'Mme', 'Mlle'], ['Miss', 'Mrs', 'Miss'])


# # One Hot Encoding
# 
# Some features are given to us in numerical form, such as age, may not have the same ordinal significance that their numerical value may imply.  Hence, we categorize these features usling the Sklearn LabelBinarizer: 

# In[135]:


Sex_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Sex))
Ticket_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Ticket))
Embarked_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Embarked))
Titles_binarized = pandas.DataFrame(preprocessing.LabelBinarizer().fit_transform(original.Titles))


# # Collecting Variables
# 
# We have managed to isolate some variables that seem reasonable for our purpose, Let's collect the features that we will be using into a new dataframe. Also, we know the result of our prediction: whether or not the particular passenger survives or not (1 and 0, respectively). While we're here, let's split our data into two sets, one for training and one for testing our model. We use Sklearn's train_test_split to do so:

# In[136]:


prediction_params = pandas.concat([Sex_binarized, Ticket_binarized, Embarked_binarized, Titles_binarized, original.Pclass, original.Age, original.SibSp, original.Fare, original.Parch], axis=1)
prediction_result = original.Survived

x_train, x_test, y_train, y_test = train_test_split(prediction_params, prediction_result, test_size = 0.15, random_state = 10)


# # Creating our Models: 
# 
# This is an obvious classification problem. Our first instinct would be to use a **logistic model**. This does surprisingly well: 

# In[137]:


logistic_model = linear_model.LogisticRegression().fit(x_train, y_train.values.ravel())
logistic_prediction = logistic_model.predict(x_test)

accuracy_score(logistic_prediction, y_test)


# Another obvious one is using a **decision tree**: 

# In[43]:


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_prediction = dt.predict(x_test)

accuracy_score(dt_prediction, y_test)


# **Random Forests**:  

# In[44]:


rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)

accuracy_score(rf_prediction, y_test)


# Finally, we can consider the use of a perceptron classifier. We use a relu activation function, and a single hidden layer about 1/5 the size of the input layer. 

# In[139]:


clf = MLPClassifier(activation = 'relu', solver='lbfgs', hidden_layer_sizes=(150), random_state=10)
clf.fit(x_train, y_train)
neural_prediction = clf.predict(x_test)

accuracy_score(neural_prediction, y_test)


# # Conclusion
# 
# While some have achieved more accurate results using a decision tree or a random forest sampling, it would seem that our logistic regression produced the most accurate result. We submit at another day  because I have a calculus exam tomorrow and am very sleep deprived.. 
