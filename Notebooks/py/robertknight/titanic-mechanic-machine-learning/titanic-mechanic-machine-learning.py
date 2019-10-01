#!/usr/bin/env python
# coding: utf-8

# Introduction
# ----------------
# 
# My first kaggle entry, just joining the other 17,000 already uploaded! My thinking is to explore the data looking how different features affect survival rates. Using intuition and knowledge of the scenario, we can say that gender, age and whether or not you are with family will make a huge difference to your survival rate. But let's confirm these assumptions.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Numpy and Pandas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Machine Learning
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn import metrics

# graphical modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# Import
# ----------
# 
# First we need to import the data set, and assign them to test, train and full (test and train combined). After printing the head of the data frame to see how the data is formated.

# In[ ]:


# import test and train and combine to full data set 
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

titanic = train.append(test, ignore_index = True)

# printing the first row and structure
print("The number of records are as follows:")
print("Training set: {}".format(train["Age"].count()))
print("Test set: {}".format(test["Age"].count()))
print("\nBelow see an example record \n")
print(titanic.iloc[1])
print("\nBelow see the data type for each column/variable \n")
print(titanic.dtypes)


# Missing Values
# --------------
# 
# Before looking into any relationships in the data, I would like to observe if there are missing values and attempt to impute these (if reasonable) for features of importance.

# In[ ]:


# Return missing values in dataframe for training set
print("Total Records: {}\n".format(titanic["Age"].count()))

print("Training set missing values")
print(train.isnull().sum())
print("\n")

print("Testing set missing values")
print(test.isnull().sum())


# Whilst some useful information could potentially have been gathered from Cabin (as this may have been and implication of location of the persons in relation to the lifeboats) the bulk of values are missing and so it may be more useful to ignore this feature in any predictive models.
# 
# However, I have already hypothesised that Age will be important. In order to build an accurate model later I would like to try using simple imputation to produce a preprocessed dataframe. This is so that I can use age as a feature and use the data from other features in records where Age is Null in formulating my model.

# In[ ]:


# replace each Null Age with the mean
train["Age"].fillna(train["Age"].mean(), inplace=True)

# apply the same preprocessing to test set
test["Age"].fillna(test["Age"].mean(), inplace=True)
test["Fare"].fillna(test["Fare"].mean(), inplace=True)

# confirm values are no longer null
print("Training set missing values after pre-processing")
print(train.isnull().sum())
print("\n")


# Feature relationships
# -------------------------
# 
# **Sex & Age**
# 
# Now I would like to look deeper at how the different features are related. Some of these features are obviously going to have no causal relationship to survival, for example what embark point the passenger is will likely have no affect on their survival, so I would like to ignore these.
# 
# However, I have already hypothesised that Sex and Age will play a key role in survival rates but visualising this relationship to test that seems like a good idea.

# In[ ]:


# Seaborn Violin Plot would be a nice way to look at this relationship
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train,
               split=True, cut =0, inner="stick", palette="Set1")


# So quite clearly my assumption about Sex and Age was a bit off. It is clear that males under 18 had a much higher survival rate, and young males has a relatively poor survival rate. However, females are much harder to understand and is much more of an even distribution. A deeper dive is needed to establish what features are clearly important to survival.
# 

# **Class & Fare**
# 
# We also know that aboard the titanic there were many different classes of people, having more or less influence in the disaster situation when trying to get a lifeboat. So let's take a dive into this relationship via a swarm plot.

# In[ ]:


# swarm plot of embarked and fare
sns.swarmplot(x="Pclass", y="Fare", hue="Survived", data=train, palette="dark")


# To predict survival for passengers will be a multi feature classification problem, so it is worth considering all of the features and then trying combinations of the features that seem to have clear relationships on survival rates to see which produce the best fitting model.

# Feature Engineering
# -------------------
# There are a number of features that can be engineered from the existing data. If we consider what assumptions we have made around women and children, there are some fairly obvious features to extract.
# 
# **Children**
# 
# Let us assume that all passengers under 18 are children, and assign within Child in the data frame. 

# In[ ]:


# create Child column into dataframe for train and test using a list comprehension
train["Child"] = ["Child" if int(x) < 18 else "Adult" for x in train["Age"]]
test["Child"] = ["Child" if int(x) < 18 else "Adult" for x in test["Age"]]


# **Family Size**
# 
# From the children/parent and sibling/spouse attributes we can quite easily establish a family size feature. My thinking is larger families would have found it difficult to find lifeboats, and maybe being in a smaller unit would have been better. 
# 
# Below I can produce a family size feature using Parch and SibSp, and adding 1 (to include the person themselves).

# In[ ]:


# Family size column
train["Family_size"] = train["Parch"].astype(np.int64) + train["SibSp"].astype(np.int64) + 1 
test["Family_size"] = test["Parch"].astype(np.int64) + test["SibSp"].astype(np.int64) + 1 

print("See below a record for a child with sibling and parent, we know have a Child and Family Size indicator: \n")
print(train.iloc[10])


# Then we can plot this into a heatmap with child/adult to observe how family size affects survival rate when a child or adult. 

# In[ ]:


# Produce heatmap
family = train.pivot_table(values="Survived", index = ["Child"], columns = "Family_size")

# Draw a heatmap with the numeric values in each cell
htmp = sns.heatmap(family, annot=True, cmap="YlGn")


# We clearly see that both of these features have a clear affect on survival. Children in families of 2-4 in size had significantly better survival rates than those who were alone or in large families. Adults in these family sizes also experience some of this benefit over larger or smaller families, but less so, which makes sense if you consider that both parents might not have been allowed to board the life boat in the disaster.
# 
# **Title**
# 
# Next I would like to look at Title, and how this can be used to extract further features such as Mothers more effectively. To do this we will need to extract the Title from the Name field. Whilst we are at it lets print out the values so we can see what titles are aboard the Titanic.

# In[ ]:


# Use regex and str.extract method to extract title from name for test and train
train["Title"] = train["Name"].str.extract("\,\s(.*?)\." , expand=True)
train["Title"].str.strip(" ")
test["Title"] = test["Name"].str.extract("\,\s(.*?)\." , expand=True)
test["Title"].str.strip(" ")

# Print list of values and the count for that data frame series
train["Title"].value_counts(ascending = False)


# Next, lets combine all of the titles with only 1 count into a variable "Vip" as they seem to relate to people who would likely be highly esteemed passengers aboard the ship.

# In[ ]:


# roll-up titles
train["Title"] = [x if x in ["Miss", "Mr", "Mrs", "Master", "Dr", "Rev"] else "Vip" for x in train["Title"] ]
test["Title"] = [x if x in ["Miss", "Mr", "Mrs", "Master", "Dr", "Rev"] else "Vip" for x in test["Title"] ]

# Seaborn Plot to show survival based on Title
bar = sns.barplot("Title", "Survived", data = train, palette="Greys")
bar.set_ylabel("Chance of Survival")


# From this we can see the chance of survival is a lot grater if you are female, a young male, doctor or Vip. The confidence interval is a lot wider on Vip and Doctor is a lot larger however. Clearly Title is going to be a very important feature for any classification algorithm we might implement later. 

# **Mothers**
# 
# Now using the Age, Sex, Title and Parch we can define a Mother feature much like the child. The assumption being that a female with Age greater than 18, Title Mrs and Parch is greater than 0, that the person in question is a mother. Obviously this is not perfect as there could be unmarried mothers, or mothers under the age of 18. But roughly speaking this should create a feature that can be of use in the classification step.

# In[ ]:


# create mothers
def mother(row):
  if row["Child"] == "Adult" and row["Sex"] == "female" and row["Title"] == "Mrs" and row["Parch"] > 0:
    return "Mother"
  else:
    return "Not Mother"

train["Mother"] = train.apply(mother, axis=1)
test["Mother"] = test.apply(mother, axis=1)

print("See below and example record for who we believe to be a mother:\n")
print(train.iloc[25])


# Classification
# =======
# 
# Now I think I have enough features to start training a classifier which can hopefully prove to be fairly accurate at predicting survival on our test set population that has been preprocessed in the same way as the training set. 
# 
# As this is my first kernel, I would like to try a number of classification algorithms starting from the simpler ones such as Logistic Regression and SVM. Moving towards using Random Forest and maybe even Neural Networks and Deep Learning if I can. 
# 
# If I can gather accuracy stats from the test set I can choose the best model and finish the Kernel with a maximum achieved accuracy after some hyper parameter tuning.

# *NOTE: I have realized that the est.csv data set is not a test set but rather a set to use at the end to predict on as there are no survived labels to test against. Therefore I will split the train set into a train and test set to enable validating the accuracy of my model against a data set that it was not trained on.*

# **Logistic Regression**
# 
# I am splitting the train data into training and test sets with a 80/20 split and checking the accuracy.

# In[ ]:


# format using patsy to get a matrix to pass into the LR model
y, X = dmatrices('Survived ~ Title + Age + Sex + Child + Family_size + Mother + Fare',
                  train, return_type="dataframe")
# flatten y into a 1-D array
y = np.ravel(y)

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# instantiate a logistic regression model, and fit with X and y
LRmodel = LogisticRegression()
LRmodel = LRmodel.fit(X_train, y_train)

# check the accuracy on the training set
LRmodel.score(X_train, y_train)


# According to this we have an 83% accuracy on the training set. Let's compare this with the Null Error Rate, which is the percentage we would get if we only predicted survived or died. 

# In[ ]:


# what percentage had affairs?
y.mean()


# If 38.4% survived, this means that the null error rate for predicting death for every passenger is 61.6%, so the prediction accuracy on the training set is significantly higher.
# 
# Now let's look at predictions in the test set. 

# In[ ]:


# check the accuracy on the test set
predicted = LRmodel.predict(X_test)
print(metrics.accuracy_score(y_test, predicted))


# Here we see we have a similar accuracy level to the accuracy in the training set, implying that the model has high training accuracy because of over fitting. 
# 
# Lets use *Cross Validation* to check more thoroughly the accuracy of the model by applying it to smaller units and averaging the accuracy response.

# In[ ]:


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

print("List of Scores for CV Folds:")
[print(score) for score in scores]

print("\nMean Accuracy")
print(scores.mean())


# We can see that the accuracy across the folds is pretty close to 83% again, implying that we have no over fitted the LRmodel.
# 
# Now let's use this model to predict on the "test" set provided to see the percentage of people we think survived based on the features we have defined. 

# In[ ]:


# add Survived column to test dataframe (blank)
test["Survived"] = ""

# format using patsy to get a matrix to pass into the LR model
yt, Xt = dmatrices('Survived ~ Title + Age + Sex + Child + Family_size + Mother + Fare',
                  test, return_type="dataframe")
# flatten y into a 1-D array
yt = np.ravel(yt)

# predict on the data
test["Survived"] = LRmodel.predict(Xt)

print("Chance of Survival for Passengers in Test Data:")
print(test["Survived"].mean())


# **Random Forest Classifier**
# 
# Now lets look at Random Forest, which should allow for us to more easily observe feature importance and should handle non-linear relationships between features more efficiently than a simple logistic regression.

# In[ ]:


# instantiate Random Forest Classifier and train
RFCmodel = RandomForestClassifier(n_estimators =1000)
RFCmodel = RFCmodel.fit(X_train, y_train)

# print score for training set
print("Train data accuracy:")
print(RFCmodel.score(X_train, y_train))

# check the accuracy on the test set
predicted = RFCmodel.predict(X_test)
print("\nTest data accuracy:")
print(metrics.accuracy_score(y_test, predicted))


# So initially the model looked to be very accurate on the train data set, but clearly there is some overfitting and the RFC is probably sensitive to noise in the data. 
# 
# I think we can optimize this algorithm further by looking at hyper parameters. In particular, min_sample_leaf will affect the RFC susceptibility to noise. So by looking at different values for this and selecting a value which produces the highest test data accuracy without having a huge affect on the train data accuracy, we should have a model that is less susceptible to noise.
# 
# I will right a script to return the 5 highest test accuracy figures and we will pick the size that has the smallest difference between test and train accuracy for the model going forward.

# In[ ]:


leafsizes = []

for x in range(1,110,2):
    RFCmodel = RandomForestClassifier(n_estimators =100, min_samples_leaf= x)
    
    RFCmodel = RFCmodel.fit(X_train, y_train)
    RFC_acc = (RFCmodel.score(X_train, y_train))
    
    predicted = RFCmodel.predict(X_test)
    RFC_test_acc = metrics.accuracy_score(y_test, predicted)
    
    diff_mag = (((RFC_acc - RFC_test_acc)**2)**0.5)
    
    leafsizes.append(("leaf {0}".format(x), RFC_acc, RFC_test_acc, diff_mag))

leafsizes = list(reversed(sorted(leafsizes, key=lambda tup: tup[2])))
    
for i in range(5):
    print(leafsizes[i])


# So we see at around min_sample_leaf = 15 we get quite a closely accuracy but with very low difference between test and train.
# 
# Lets use cross validation as with the LR to see if this model is actually less accurate and that we should stick to the simpler but more effective LR model.

# In[ ]:


# assign RFC model with the chosen  min_samples_leaf
RFCmodel = RandomForestClassifier(n_estimators =100, min_samples_leaf= 15)
RFCmodel = RFCmodel.fit(X_train, y_train)

# evaluate the model using 10-fold cross-validation
scores = cross_val_score(RandomForestClassifier(n_estimators =100, min_samples_leaf= 15), X, y, scoring='accuracy', cv=10)

print("List of Scores for CV Folds:")
[print(score) for score in scores]

print("\nMean Accuracy")
print(scores.mean())

print(RFCmodel.feature_importances_)


# Submissions
# =======
# 
# First submission - I believe the LR model to be more accurate. Lets see how I do!
# 

# In[ ]:


# run model and place values in test dataframe
test["Survived"] = LRmodel.predict(Xt)

# produce submission format
submission_lr = pd.DataFrame()

submission_lr["PassengerId"] = test["PassengerId"]
submission_lr["Survived"] = test["Survived"]

print("Check format:\n")
print(submission_lr.head())

submission_lr.to_csv("Submission_lr.csv", index = False)


# The competition entry for LR was not as high as the tests (around 78%) so now I would like to utilize the RFC and see if that can do any better upon submission. 

# In[ ]:


# run model and place values in test dataframe
test["Survived"] = RFCmodel.predict(Xt)

# produce submission format
submission_rfc = pd.DataFrame()

submission_rfc["PassengerId"] = test["PassengerId"]
submission_rfc["Survived"] = test["Survived"]

print("Check format:\n")
print(submission_rfc.head())

submission_rfc.to_csv("Submission_rfc.csv", index = False)


# I will submit this csv to see how it fairs in comparison to LR.
# 
# LR is still marginally better but both models are sitting around 78%
