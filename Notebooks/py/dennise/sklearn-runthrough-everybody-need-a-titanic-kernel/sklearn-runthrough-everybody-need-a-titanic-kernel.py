#!/usr/bin/env python
# coding: utf-8

# Welcome to my Titanic Kernel!
# 
# I am using this Kernel as excercise for Udemy course python-for-data-science-and-machine-learning-bootcamp (www.udemy.com/python-for-data-science-and-machine-learning-bootcamp)
# 
# After the excercise on linear regression which was very unstructured and messy in the end I'll apply a very structured approach to this Kernel. (https://www.kaggle.com/dennise/learning-ml-algorithms-on-this-dataset)
# 
# I'll document my work step-by-step as learning excercise and look-up opportunity for myself
# 
# Hope other learners will find it useful
# 
# Happy to receive your votes, comments, questions and suggestions

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


test=pd.read_csv("../input/test.csv")
train=pd.read_csv("../input/train.csv")
#gender=pd.read_csv("../input/gender_submission.csv")
#gender file is only an example how a submission should look like format-wise


# 
# Complete project workflow (Found in Kaggle discussion forum, thanks)
# 
# 1. Prepare Problem
#     a) Load libraries
#     b) Load dataset
# 
# 2. Summarize Data
#     a) Descriptive statistics
#     b) Data visualizations
# 
# 3. Prepare Data
#     a) Data Cleaning
#     b) Feature Selection
#     c) Data Transforms
# 
# 4. Evaluate Algorithms
#     a) Split-out validation dataset
#     b) Test options and evaluation metric
#     c) Spot Check Algorithms
#     d) Compare Algorithms
# 
# 5. Improve Accuracy
#     a) Algorithm Tuning
#     b) Ensembles
# 
# 6. Finalize Model
#     a) Predictions on validation dataset
#     b) Create standalone model on entire training dataset
#     c) Save model for later use
# 
# 

# In[ ]:


# Step 2a: Descriptive statistics
# Lets only focus on the available train-data as if we would have had one set that we split into test_train and that the model is then put to production
train.info()


# Observations:
# * Age missing for 180 passengers
# * Cabin missing for many observations - lets be careful what this means)
# * 2 observations for Embarked missing (What is "Embarked?)
# * 5 qualitative / categorical features: 
#     * Name (lets drop this)
#     * Sex (translate to 1 and 0)
#     * Ticket (Ticket number) - Why string? Relevance of number? Has to do with buying date?
#     * Cabin (Number of cabin + careful little observations) - Could be useful to check location of cabin and construct feature e.g. on which deck the cabin is or how close to exit,...
#     + Embarked (Port) - How would this be relevant? Not by itself. Lets drop it for now.
# * Rest quantitative features:
#     * Passenger ID (lets drop)
#     * Survived (quantitative? 1 and 0 I guess)
#     * Pclass (1,2 or 3) - ordered, fine
#     * Age (fine, but lots of data missing)
#     * SibSp (# of siblings / spouses on board) = Family size horizontal
#     * Parch (# of parents / children aboard) = Family size vertical - symmetric kids / parents
#     * Fare - correlated with Pclass?
# 
# All changes come in step 3

# In[ ]:


train.describe()


# Observations:
# * Survival rate of 38%
# * Class, as expected, mostly in 3rd class
# * Age
# * Siblings: Most travelling alone. mean of 0.5. With outliers (8 is max)
# * Same for parents/children. Mostly travel alone, some outliers
# * Fare: Probably strong correlated with PClass. Standard deviation higher than average, speaking for a very skewed distribution

# In[ ]:


# Lets look into concrete data
train.head()


# Observations:
# * The few existing cabin numbers seem to follow a potentially interesting pattern. Lets check it out more in detail what "C" means and what else exists.
# * Based on names siblings and kids could potentially be identified
#     * Maybe to extract last names could be interesting
# * Tickets could also have kind of numbering system, lets see
# 

# In[ ]:


train.Cabin.dropna()[5:40]


# Observations:
# * Some individuals have booked multiple cabines, mostly neighbouring each other. 
#     * Probably families
#     * Yes, e.g. 27 and 88 live together

# In[ ]:


train[(train.PassengerId==27)|(train.PassengerId==88)]


# Hmm. Two gentlemen from different ports. In 3rd class. With unknown age.
# You can actually research all these passengers: 
# https://www.encyclopedia-titanica.org/titanic-victim/emir-farres-chehab-shihab.html
# 
# Also missing age could be filled this way. Emir was 26
# Selman was 20
# 
# Age is missing for 180 people. Will do this when I'll do a webscraping excercise.
# 
# Anyhow - column is messy. I would try to cut the first letter that appears out of the column to have labels "A","B" etc. Maybe that means something.

# In[ ]:


# But for the fun of it:
train.at[26,"Age"]=26
train.at[87,"Age"]=20


# In[ ]:


sns.countplot(train.Pclass)


# In[ ]:


sns.distplot(train.Age.dropna())
# Many babies on board!
# Left-skewed
# not many kids/teenagers
# young, affluent people


# In[ ]:


sns.countplot(train.Sex)


# 2/3 man and 1/3 woman

# In[ ]:


sns.countplot(train.SibSp)


# In[ ]:


# For family size lets make a short sensecheck
train[train.SibSp==8]
# only 7 people have 8 siblings. One is missing (probably in test). But a family. No one survived.
# All on same Ticket nr -> so these numbers are not unique


# In[ ]:


sns.countplot(train.Parch)


# Number of parents / children on board

# In[ ]:


sns.heatmap(train.corr(),cmap="coolwarm")


# Observations:
# * Correlation between Fare and Survival
# * Correlation between Fare and Class
# * Most probably Fare and Class indicate the same for Survival so probably best to only include one
# * Age and class also a bit correlated, probably older people, higher class
# * Parch and SibSp correlated - Family size

# In[ ]:


sns.boxplot(data=train,y="Fare",x="Pclass", hue="Survived")


# Observations:
# * Extreme Outlier in Fare - maybe exclude them
# * Probably to plot age like this is interesting - age in categories could be interesting - babies / kids / teens / young adults / adults / seniors

# In[ ]:


sns.boxplot(data=train,y="Age",x="Pclass", hue="Survived")


# Observations:
# * Survived people are generally younger, in all classes

# In[ ]:


sns.boxplot(data=train,y="Fare",x="Pclass", hue="Sex")


# Observations:
# * Females in class 1 and 3 paid more than man!?

# In[ ]:


sns.boxplot(data=train,y="Age",x="Pclass", hue="Sex")


# Observations:
# * Women on average younger than man, especially in first and third class

# In[ ]:


sns.countplot(train["Sex"],hue=train["Survived"])


# Wow this is pretty clear.
# Probably even clearer if we do not count babies / kids as male or female.
# 
# Ok I think its time to go to the next step:
# 
# Prepare Data:
# a) Data Cleaning 
# b) Feature Selection 
# c) Data Transforms

# In[ ]:


#Lets check the fare outlier
train.sort_values("Fare",ascending=False)
# Actually the 15 people with fare = 0 worry me more than the 3 that paid 500. I leave it like this for now


# In[ ]:


# But the passenger ID I dont need
# Wait a moment, before I delete: I need to do operations both on test and train. Therefore, lets combine them. I can separate them later by whether there is a Survived value or not
data=pd.concat([train, test])
data.info()


# In[ ]:


data.drop("PassengerId",inplace=True, axis=1)


# In[ ]:


data.info()


# In[ ]:


data.drop("Name",inplace=True, axis=1)


# In[ ]:


data.drop("Embarked",inplace=True, axis=1)


# In[ ]:


data.Sex2=np.nan
data.loc[data["Sex"]=="male","Sex2"]=1
data.loc[data["Sex"]=="female","Sex2"]=0


# In[ ]:


data.head()


# In[ ]:


data.drop("Ticket",inplace=True,axis=1)


# In[ ]:


# So far so good. What were other ideas?


# * Age missing for 180 passengers. For a regression these should not be considered. I do not see a meaningful way to "fill" them. Average / Mean / mode?
#     * But lets categorize age
#         * 0-2 infant (0)
#         * 3-6 baby (1)
#         * 7-10 kid (2)
#         * 10-15 teen (3)
#         * 16-30 young adult (4)
#         * 30-50 adult (5)
#         * older than 50 senior (6)
# *Cabin missing for many observations - lets be careful what this means)
#     * Cabins are ordered by decks, so the letters are important (https://www.encyclopedia-titanica.org/titanic-store/titanic-deckplans.html)

# In[ ]:


data["Age_cohort"]=np.nan


# In[ ]:


data["Age_cohort"][(data["Age"]>=0)&(data["Age"]<=2)]=0
data["Age_cohort"][(data["Age"]>2)&(data["Age"]<=6)]=1
data["Age_cohort"][(data["Age"]>6)&(data["Age"]<=10)]=2
data["Age_cohort"][(data["Age"]>10)&(data["Age"]<=16)]=3
data["Age_cohort"][(data["Age"]>16)&(data["Age"]<=30)]=4
data["Age_cohort"][(data["Age"]>30)&(data["Age"]<=50)]=5
data["Age_cohort"][(data["Age"]>50)]=6


# In[ ]:


data.head(30)


# In[ ]:


sns.countplot(data.Age_cohort)


# In[ ]:


# Ok lets see if it makes so much sense to split up the kids so much
# But now lets get the decks out of the room numbers
data["Cabin"].fillna(" ", inplace=True)
data["Deck"]=""
data["Deck"]=data["Cabin"].apply(lambda x:str(x)[0])
data.head()


# In[ ]:


sns.countplot(data.Deck)


# In[ ]:


sns.countplot(data["Deck"],hue=data["Survived"])


# This looks a bit like Cabin-numbers are mainly known for people who survived.

# In[ ]:


#Let's engineer a new feature of only adult sex to account for male and female kids being treated rather as "kids" than as "female" or "male" in prioritization of life boats
data["Sex_adults"]=np.nan
data["Sex_adults"][(data["Sex"]==1)&(data["Age"]>16)]=1
data["Sex_adults"][(data["Sex"]==0)&(data["Age"]>3)]=0
data.head(20)


# In[ ]:


sns.countplot(data["Age_cohort"],hue=data["Survived"])


# The difference in survival rate between 6-10 and 10-16 year olds and then to 16+ looks strange. 

# In[ ]:


train=data[(data["Survived"]==0)|(data["Survived"]==1)]
train.info()


# In[ ]:


# Make one version of a full dataset
del train["Sex"]
del train["Sex_adults"]
del train["Deck"]
del train["Cabin"]
train.head()


# In[ ]:


train.dropna(axis=0,how="any",inplace=True)
train.info()


# In[ ]:


# Now lets check the first classification algorithm
# Make 1 version of cleaned data
sns.heatmap(train.isnull(),cbar=None,cmap="viridis")


# In[ ]:


test=data[data["Survived"].isnull()]
del test["Sex"]
del test["Sex_adults"]
del test["Deck"]
del test["Cabin"]
del test["Survived"]
test.info()


# In[ ]:


X_train=train.drop("Survived", axis=1)
y_train=train["Survived"]


# In[ ]:





# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


logmodel.coef_


# In[ ]:


X_train.columns


# In[ ]:





# In[ ]:


predictions=logmodel.predict(X_train)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_train,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train,predictions))


# A confusion matrix is:
#     https://en.wikipedia.org/wiki/Confusion_matrix

# In[ ]:


sns.heatmap(test,cmap="viridis",cbar=False)


# In[ ]:


test.info()


# In[ ]:


test.fillna(test.mean(),inplace=True)
test.info()


# In[ ]:


predictions=logmodel.predict(test).astype("int")


# In[ ]:


type(predictions[0])


# In[ ]:


solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)


# Scores: 0.76555

# # Todos
# - Only age or age_cohort as co-linear
# - Dummy variables for decks
# - Probably smarter way to fill the missing age data

# In[ ]:


#Now lets try a KNN approach
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=100)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
features=scaler.transform(X_train)
X_train_scaled = pd.DataFrame(features,columns=X_train.columns)

scaler.fit(test)
features=scaler.transform(test)
test_scaled = pd.DataFrame(features,columns=test.columns)

X_train_scaled.head()
test_scaled.head()


# In[ ]:


knn.fit(X_train_scaled,y_train)


# In[ ]:


predictions=knn.predict(X_train)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predictions))


# In[ ]:


print(classification_report(y_train, predictions))


# In[ ]:


predictions=knn.predict(test_scaled).astype("int")
solution=pd.DataFrame({"PassengerId":list(range(892,1310)), "survived":predictions})
solution.to_csv("solution.csv", index=False)


# In[ ]:





# KNN results:
# - with k=5 74%
# - with k=2 73%
# - with k=10 75%
# - with k=20 76%
# - with k=30 76,55
# - with k=50 0.77511
