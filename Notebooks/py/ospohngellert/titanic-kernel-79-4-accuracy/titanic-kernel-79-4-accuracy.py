#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Solution
# 
# ## Table of Contents
# 
# * [Abstract](#Abstract)
# * [Data Analysis](#Data-Analysis)
#     * [Initial Impressions](#Initial-Impressions)
#     * [Missing Ages](#Missing-Ages)
#         * [Conclusion on Age](#Conclusion-on-Age)
#     * [Fare](#Fare)
# * [Data Cleansing](#Data-Cleansing)
#     * [Initial Cleansing](#Initial-Cleansing)
#     * [Assessing Correlation](#Assessing-Correlation)
#     * [Feature Extraction](#Feature-Extraction)
#     * [Dealing with Colinearity](#Dealing-with-Colinearity)
# * [Running the Algorithms](#Running-the-Algorithms)
#     * [Algorithms](#Algorithms)
#     * [Testing Without Colinearity](#Testing-Without-Colinearity)
#     * [Testing With Colinearity](#Testing-With-Colinearity)
#     * [Dealing with Overfitting](#Dealing-with-Overfitting)
#     * [Final Modeling](#Final-Modeling)
# * [Conclusion and Final Thoughts](#Conclusion-and-Final-Thoughts)
# 
# 
# ## Abstract
# 
# The purpose of this notebook is to provide a solution to the titanic problem on kaggle. In order to do this, the data was imported, and then trends were analyzed in an attempt to extract the useful features. The most notable part of the data cleansing was using number of siblings, name, and number of parents to more accurately impute missing age values. Through this, 79.4% accuracy was achieved.

# In[ ]:


import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
import numpy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import scipy.stats as stats
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# Below we import train and test data into two separate data frames.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## Data Analysis

# First, let's take a look at a general summary of the data.

# In[ ]:


train.describe()


# ### Initial Impressions
# 
# * Age has missing values (714 is its count, meaning there are some na)
# * Fare has some values that are skewing the mean, as the mean is greater than the 75% value.
# * Only 38% of people in the data actually survived
# 
# ### Missing Ages
# The first thing we will look at is the missing ages. How does the age data look, and what should be our strategies for imputing this? First let's find what proportion of the ages are missing.

# In[ ]:


sum(pd.isnull(train.Age))/len(train)


# This is a large chunk of the data, so we would like to impute these ages as accurately as possible. Let's analyze the ages we have to see if we can find significant trends to impute our unknown ages.

# In[ ]:


ageNoNa = train.Age[pd.isnull(train.Age) == False]
plt.hist(ageNoNa)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for known training data.")
plt.show()
print("Mean age: " + str(numpy.mean(ageNoNa)))
print("Median age: " + str(numpy.median(ageNoNa)))
print("Standard debiation: " + str(numpy.std(ageNoNa)))
print("Normality: " + str(stats.normaltest(ageNoNa)))


# The data here is fairly normal, but the mean isn't exactly desirable as a fill in, as it does not encompass the range of the data. Next let's look into which other factors in the data can be used to make an educated guess on the missing ages. The first thing we will look at is using the number of siblings someone has, and the number of parents/children someone has.

# In[ ]:


ageHasSibs = train[(train.SibSp > 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageHasSibs)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have siblings.")
plt.show()
print("Mean age: " + str(numpy.mean(ageHasSibs)))
print("Median age: " + str(numpy.median(ageHasSibs)))


# In[ ]:


ageHasNoSibs = train[(train.SibSp == 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageHasNoSibs)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who don't have siblings.")
plt.show()
print("Mean age: " + str(numpy.mean(ageHasNoSibs)))
print("Median age: " + str(numpy.median(ageHasNoSibs)))


# Though the average age for people who have siblings was lower than for those who don't, it is not a significant enough difference to be noticeable. Though it is less likely to be indicative, let's look at people who have more than one parent/child, versus those who don't.

# In[ ]:


ageHasParch = train[(train.Parch > 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageHasParch)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have parents/children.")
plt.show()
print("Mean age: " + str(numpy.mean(ageHasParch)))
print("Median age: " + str(numpy.median(ageHasParch)))


# In[ ]:


ageHasNoParch = train[(train.Parch == 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageHasNoParch)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who don't have siblings.")
plt.show()
print("Mean age: " + str(numpy.mean(ageHasNoParch)))
print("Median age: " + str(numpy.median(ageHasNoParch)))


# Surprisingly, this is a much better indicator. However, because of the large distribution of ages of people with Parch, simply using Parch may not be the best indicator. One thing we can look into is using the names, specifically their title. Let's look at the ages of people who have the word "Master", "Miss", "Mr.", and "Mrs." in their names.

# In[ ]:


ageOfMaster = train[train.Name.str.contains('Master.') & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMaster)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Master.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMaster)))
print("Median age: " + str(numpy.median(ageOfMaster)))


# In[ ]:


ageOfMiss = train[train.Name.str.contains('Miss.') & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMiss)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Miss.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMiss)))
print("Median age: " + str(numpy.median(ageOfMiss)))


# In[ ]:


ageOfMr = train[train.Name.str.contains('Mr.') & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMr)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Mr.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMr)))
print("Median age: " + str(numpy.median(ageOfMr)))


# In[ ]:


ageOfMrs = train[train.Name.str.contains('Mrs.') & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMr)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Mrs.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMrs)))
print("Median age: " + str(numpy.median(ageOfMrs)))


# There is a clear difference between people who have the title "Miss" or "Master", and those who have the titles "Mr" or "Mrs". Master has a much clearer difference than Miss though, so we should look to combine Miss with other factors to see if we can get some even better differences.

# In[ ]:


ageOfMissWithParch = train[train.Name.str.contains('Miss.') & (train.Parch > 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMissWithParch)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Miss and have Parents/Children.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMissWithParch)))
print("Median age: " + str(numpy.median(ageOfMissWithParch)))


# In[ ]:


ageOfMissWithoutParch = train[train.Name.str.contains('Miss.') & (train.Parch == 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMissWithoutParch)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Miss and don't have Parents/Children.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMissWithoutParch)))
print("Median age: " + str(numpy.median(ageOfMissWithoutParch)))


# In[ ]:


ageOfMrsWithParch = train[train.Name.str.contains('Mrs.') & (train.Parch > 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMrsWithParch)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Mrs and have Parents/Children.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMrsWithParch)))
print("Median age: " + str(numpy.median(ageOfMrsWithParch)))


# In[ ]:


ageOfMrsWithoutParch = train[train.Name.str.contains('Mrs.') & (train.Parch == 0) & (pd.isnull(train.Age) == False)].Age
plt.hist(ageOfMrsWithoutParch)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age distribution for people who have the title Mrs and don't have Parents/Children.")
plt.show()
print("Mean age: " + str(numpy.mean(ageOfMrsWithoutParch)))
print("Median age: " + str(numpy.median(ageOfMrsWithoutParch)))


# It appears that the difference is extremely notable for Miss, but negligable for Mrs. One thing we have to note is that the number of samples when we get this specific is much lower, so we have to make sure that we are not overfitting to it. This is something we should be looking for if we get high variance on the test set. I won't be doing similar analysis on the test set now, as this would not emulate a real world scenario, where we may not have a test set on hand.
# 
# #### Conclusion on Age
# 
# Based on the previous analysis, we will use a function that takes into account both name and number of siblings in order to impute the age. The titles "Master" and "Miss" will be used to identify potential children, and the rest will be adults.
# 
# ### Fare
# 
# The first thing we should look into with fare is its distribution.

# In[ ]:


fare = train.Fare
plt.hist(fare)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price distribution for known training data.")
plt.show()
print("Mean fare: " + str(numpy.mean(fare)))
print("Median fare: " + str(numpy.median(fare)))
print("Standard deviation: " + str(numpy.std(fare)))
print("Normality: " + str(stats.normaltest(fare)))


# It is clear here that the data is not even close to normally distributed, and that the majority of the values are under 50 dollars. So, we should look at the following:
# 
# 1. Do people who paid greater than 50 dollars survive at a higher rate?
# 2. Is there a way to make the data more normally distributed?
# 
# Based on the data above, data within 3 standard deviations of the mean is fares below 181 dollars. So, we will test to see if removing those data points helps, and if the fares above 181 dollars differ significantly to those near 181 dollars.

# In[ ]:


fareOver181 = train[train.Fare > 181]
fareBetween100And181 = train[(train.Fare > 100) & (train.Fare < 181)]
fareUnder181 = train[(train.Fare < 181)].Fare
plt.hist(fareUnder181)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price distribution for fares under 181.")
plt.show()
print("Survival rate for those whose fare is > 181: " + str(sum(fareOver181.Survived) / len(fareOver181.Survived)))
print("Survival rate for those whose fare is < 181 and > 100: " + str(sum(fareBetween100And181.Survived) / len(fareBetween100And181.Survived)))

print("Normality: " + str(stats.normaltest(fareUnder181)))


# It appears that those who pay a higher rate have a significantly higher survival rate, so we may not want to lose that data. However, data nearby it has a similar survival rate, so it appears fine to eliminate it as an outlier. Furthermore, though this is an improvement, it certainly does not make the data normal. Next, let's try applying some transformations to the data and checking the normality.
# 
# #### Square Root

# In[ ]:


rootFareUnder181 = fareUnder181.map(lambda x: math.sqrt(x))
plt.hist(rootFareUnder181)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price distribution for square roots of fares under 181.")
plt.show()
print("Normality: " + str(stats.normaltest(rootFareUnder181)))


# #### Log

# In[ ]:


logFareUnder181 = fareUnder181.map(lambda x: math.log(x, 10) if x != 0 else x)
plt.hist(rootFareUnder181)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Price distribution for logs of fares under 181.")
plt.show()
print("Normality: " + str(stats.normaltest(logFareUnder181)))


# The log of the filtered data is a significant improvement in terms of normality, while potentially offering little performance hit. Therefore, this will be implemented below.

# ## Data Cleansing
# 
# The first step for cleansing the data is creating the functions we will use, most importantly the age function.

# In[ ]:


def ageFunc(x):
    age = x['Age']
    name = x['Name']
    sibs = x['SibSp']
    if math.isnan(age):
        if "Master." in name:
            x['Age'] = 5
        elif "Miss." in name and sibs > 0:
            x['Age'] = 11
        elif "Miss." in name and sibs == 0:
            x['Age'] = 27
        elif "Mr." in name:
            x['Age'] = 32
        elif "Mrs." in name:
            x['Age'] = 36
        else:
            x['Age'] = 29
    return x
    
def embarkedFunc(x):
    vals = {'Q': 1, 'S': 2, 'C': 3}
    return vals.get(x, 0)


# ### Initial Cleansing
# 
# Next, we will create a function that will cleanse an input data frame, so it can be reused on the training and test sets. The most notable parts of this cleansing are the following:
# * Text data that has discrete values are converted to factors (0:n-1)
# * We use whether or not there is a letter (not number) in the ticket
# * We use whether or not a cabin was listed

# In[ ]:


def cleanData(frame, isTest):
    averageFare = numpy.mean(frame.Fare)
    frame = frame.apply(ageFunc, axis='columns')
    frame.Fare = frame.Fare.map(lambda x: x if not numpy.isnan(x) else averageFare)
    out = frame.query('Fare < 181').copy() if not isTest else frame.copy()
    out.Sex = out.Sex == 'female'
    out.Fare = out.Fare.map(lambda y: math.log(y, 10) if y != 0 else y)
    out.Ticket = out.Ticket.map(lambda x: not pd.isnull(re.search("[a-zA-Z]", x)))
    out['emS'] = out.Embarked == 'S'
    out['emC'] = out.Embarked == 'C'
    out['class1'] = out.Pclass == 1
    out['class2'] = out.Pclass == 2
    out.Embarked = out.Embarked.map(embarkedFunc)
    out.Cabin = out.Cabin.map(lambda x: not pd.isnull(x))
    return out.iloc[:,out.columns.get_level_values(0).isin({"Survived", "PassengerId", "Fare", 'class1', 'class2', "Sex", "Age", "SibSp", "Parch", "Cabin", "Ticket", 'emS', 'emC'})]

train = cleanData(train, False)
test = cleanData(test, True)


# ### Assessing Correlation
# 
# Now that we have cleaned the data, let's look into the colinearity of each of the variables, and see if we can eliminate some that are heavily correlated.

# In[ ]:


train.corr()


# Based on this there are two major groups of colinear features: the group Cabin, Fare, and PClass, and the second group SibSp and Parch. These both make sense, and we will handle the two differently.
# 
# The most suprising thing about this correlation is the lack of correlation between age and family factors in survival. We will also look further into this to see if we can use feature extraction to create a better correlation.
# 
# ### Feature Extraction
# 
# We will first look into using bins of ages to create a better correlation between age and survival. First, let's look at the survival rates of different age ranges.

# In[ ]:


ageLessEq15 = train[(train.Age <= 15)].Survived
ageBetween16and35 = train[(train.Age > 15) & (train.Age <= 35)].Survived
ageBetween36and60 = train[(train.Age > 35) & (train.Age <= 60)].Survived
ageGreater60 = train[(train.Age > 60)].Survived
print("<= 15 survival: " + str(sum(ageLessEq15) / len(ageLessEq15)))
print("16 - 35 survival: " + str(sum(ageBetween16and35) / len(ageBetween16and35)))
print("36 - 60 survival: " + str(sum(ageBetween36and60) / len(ageBetween36and60)))
print("> 60 survival: " + str(sum(ageGreater60) / len(ageGreater60)))


# It seems like people below the age of 15 (children) had a significantly better than average chance of survival. Further, it seems the elderly had a low survival rate. So, let's see if adding an isChild and an isSenior variable has a good correlation with survival.

# In[ ]:


train['isChild'] = train.Age <= 15
train['isSenior'] = train.Age > 60
train.corr()


# isChild provided a slightly better correlation, but isSenior did not. This is probably due to the relative sizes of the two groups as displayed below:

# In[ ]:


print("isChild Size: {}".format(len(ageLessEq15)))
print("isSenior Size: {}".format(len(ageGreater60)))


# Based on this, the isSenior variable should not be used, so we will remove it.

# In[ ]:


train = train.drop("isSenior", axis=1, errors='ignore')
test["isChild"] = test.Age <= 15


# Now let's look at Siblings and Parch

# In[ ]:


plt.hist(train.SibSp)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Siblings")
plt.show()

plt.hist(train.Parch)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distributions of Parch")
plt.show()


# The distributions of these two variables is very similar, which is further suported by the 0.4 correlation between the two variables. In another kernel, I saw the idea to sum these two together into a variable called family size. We will try that and see the distribution and correlation with survival.

# In[ ]:


train["FamilySize"] = train.SibSp + train.Parch
plt.hist(train.FamilySize)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distributions of Parch")
plt.show()
train.corr()


# Unfortunately, family size is even less correlated than the other two. In the end, the best approach may be to just use Parch. For now, we will drop family size.

# In[ ]:


train = train.drop("FamilySize", axis=1, errors='ignore')


# ### Dealing with Colinearity
# 
# The way we will deal with colinearity is testing the algorithms. We will apply each algorithm to the data with and without colinear features. The features we will use regardless of colinearity are: Parch and Fare. isChild would be used instead of Age, however this is a derived feature.

# ## Running the Algorithms
# 
# Next we will run a few different algorithms, splitting the training data into train and verify data so that we can have a preliminary way of seeing which algorithm is performing best.
# 
# ### Algorithms
# 
# We will be trying four different algorithms on the data: an SVM, a Neural Network, a Random Forest, and a best two out of three vote of the three algorithms. Note that for the SVM we will use a Gaussian kernel, as we do not have many features. We will test it on 50 different 80/20 splits of the data for train/validation. We will also do this using the colinear version of the data and the non colinear version. 

# In[ ]:


def createSplitNoColinear(t):
    train, cv = train_test_split(t, test_size = 0.2)
    X_train = preprocessing.scale(numpy.transpose([train.isChild, train.Sex, train.Parch, train.Fare, train.emS, train.emC, train.Ticket]))
    Y_train = numpy.transpose(train.Survived)
    X_cv = preprocessing.scale(numpy.transpose([cv.isChild, cv.Sex, cv.Parch, cv.Fare, cv.emS, cv.emC, cv.Ticket]))
    Y_cv = numpy.transpose(cv.Survived) 
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_cv": X_cv,
        "Y_cv": Y_cv
    }

def createSplitColinear(t):
    train, cv = train_test_split(t, test_size = 0.2)
    X_train = preprocessing.scale(numpy.transpose([train.isChild, train.Sex, train.Parch, train.Fare, train.emS, train.emC, train.Ticket, train.class1, train.class2, train.Age, train.SibSp, train.Cabin]))
    Y_train = numpy.transpose(train.Survived)
    X_cv = preprocessing.scale(numpy.transpose([cv.isChild, cv.Sex, cv.Parch, cv.Fare, cv.emS, cv.emC, cv.Ticket, cv.class1, cv.class2, cv.Age, cv.SibSp, cv.Cabin]))
    Y_cv = numpy.transpose(cv.Survived) 
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_cv": X_cv,
        "Y_cv": Y_cv
    }


def testData(split, x):
    layer_sizes = [len(split['X_train'][0]) * 3, len(split['X_train'][0])]
    neural =  MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(layer_sizes[0], layer_sizes[1], 1), random_state=1)
    forest = RandomForestClassifier(max_depth=5, random_state=0)
    machine = svm.SVC(kernel='rbf')
    neural.fit(split['X_train'], split['Y_train'])
    forest.fit(split['X_train'], split['Y_train'])
    machine.fit(split['X_train'], split['Y_train'])
    nSurvived = neural.predict(split['X_cv'])
    fSurvived = forest.predict(split['X_cv'])
    sSurvived = machine.predict(split['X_cv'])
    vSurvived = list(map(lambda x: int(sum(x) > 2), zip(nSurvived, fSurvived, sSurvived)))
    x['neural'] = sum(nSurvived == split['Y_cv'])/len(nSurvived)
    x['forest'] = sum(fSurvived == split['Y_cv'])/len(fSurvived)
    x['svm'] = sum(sSurvived == split['Y_cv'])/len(sSurvived)
    x['vote'] = sum(vSurvived == split['Y_cv'])/len(vSurvived)
    return x


# ### Testing Without Colinearity

# In[ ]:


TRIALS = 50
predictions = pd.DataFrame({'neural': range(TRIALS), 'forest': range(TRIALS), 'svm': range(TRIALS), 'vote': range(TRIALS)}, dtype=float)
predictions = predictions.apply(lambda x: testData(createSplitNoColinear(train), x), axis=1)
predictions.describe()


# Interestingly enough, the neural network performed the worst out of all the algorithms in all cases, while the SVM performed the best. Next, we will try using all of the data, instead of eliminating data that is colinear.
# 
# ### Testing With Colinearity

# In[ ]:


TRIALS = 50
predictions = pd.DataFrame({'neural': range(TRIALS), 'forest': range(TRIALS), 'svm': range(TRIALS), 'vote': range(TRIALS)}, dtype=float)
predictions = predictions.apply(lambda x: testData(createSplitColinear(train), x), axis=1)
predictions.describe()


# Again, the neural network perfored extremely poorly, and the SVM performed the best. The forest performed even better with the colinear data than it did without it, so we will include everything when training for the real test data.
# 
# The neural network performed pretty poorly in general this could be because we are overfitting the data. We should look into adjusting the regularization parameter to find the optimal value.
# 
# ### Dealing with Overfitting
# 
# We will now find the optimal regularization parameter for the colinear data, as the algorithms performed better on the colinear data in general.

# In[ ]:


def testCVNeural(split, x, reg):
    layer_sizes = [len(split['X_train'][0]) * 3, len(split['X_train'][0])]
    for a in reg:
        neural =  MLPClassifier(solver='lbfgs', alpha=a, hidden_layer_sizes=(layer_sizes[0], layer_sizes[1], 1), random_state=1)
        neural.fit(split['X_train'], split['Y_train'])
        nSurvived = neural.predict(split['X_cv'])
        x[str(a)] = sum(nSurvived == split['Y_cv'])/len(nSurvived)
    return x


# In[ ]:


TRIALS = 50
reg = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100]
regDict = {}
for a in reg:
    regDict[str(a)] = range(TRIALS)
predictions = pd.DataFrame(regDict, dtype=float)
predictions = predictions.apply(lambda x: testCVNeural(createSplitColinear(train), x, reg), axis=1)
predictions.describe()


# Based on this, the optimal regularization term is 10, so we will use this in the final model. Now, let's do the same thing for the SVM.

# In[ ]:


def testCVSVM(split, x, reg):
    layer_sizes = [len(split['X_train'][0]) * 3, len(split['X_train'][0])]
    for a in reg:
        machine = svm.SVC(kernel='rbf', C=a)
        machine.fit(split['X_train'], split['Y_train'])
        sSurvived = machine.predict(split['X_cv'])
        x[str(a)] = sum(sSurvived == split['Y_cv'])/len(sSurvived)
    return x


# In[ ]:


TRIALS = 50
reg = [1e-3, 3e-3, 1e-1, 3e-1, 1, 3, 10]
regDict = {}
for a in reg:
    regDict[str(a)] = range(TRIALS)
predictions = pd.DataFrame(regDict, dtype=float)
predictions = predictions.apply(lambda x: testCVSVM(createSplitColinear(train), x, reg), axis=1)
predictions.describe()


# Since 0.3 performed the best, we will use it.
# 
# ### Final Modeling

# In[ ]:


X_train = preprocessing.scale(numpy.transpose([train.isChild, train.Sex, train.Parch, train.Fare, train.emS, train.emC, train.Ticket, train.class1, train.class2, train.Age, train.SibSp, train.Cabin]))
Y_train = numpy.transpose(train.Survived)
X_test = preprocessing.scale(numpy.transpose([test.isChild, test.Sex, test.Parch, test.Fare, test.emS, test.emC, test.Ticket, test.class1, test.class2, test.Age, test.SibSp, test.Cabin]))
layer_sizes = [len(X_train[0]) * 3, len(X_train[0])]
neural =  MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(layer_sizes[0], layer_sizes[1]), random_state=1)    
forest = RandomForestClassifier(max_depth=5, random_state=0)
machine = svm.SVC(kernel='rbf', C=0.3)
neural.fit(X_train, Y_train)
forest.fit(X_train, Y_train)
machine.fit(X_train, Y_train)
nSurvived = neural.predict(X_test)
fSurvived = forest.predict(X_test)
sSurvived = machine.predict(X_test)
vSurvived = list(map(lambda x: int(sum(x) > 2), zip(nSurvived, fSurvived, sSurvived)))

pd.DataFrame({"PassengerId": test.PassengerId, "Survived": nSurvived}).to_csv("neural.csv", index=False)
pd.DataFrame({"PassengerId": test.PassengerId, "Survived": fSurvived}).to_csv("forest.csv", index=False)
pd.DataFrame({"PassengerId": test.PassengerId, "Survived": sSurvived}).to_csv("svm.csv", index=False)
pd.DataFrame({"PassengerId": test.PassengerId, "Survived": vSurvived}).to_csv("vote.csv", index=False)


# ## Conclusion and Final Thoughts
# 
# The overal success rates after uploading these to the kaggle website were as follows:
# 
# * forest: 0.78468
# * neural: 0.77990
# * svm: 0.79425
# * vote: 0.77990
# 
# I would like to thank anyone who took the time out to read this document. This was my first time trying a data science problem like this, and I would love to hear any feedback about how I can improve, and if it was a clear read through. Thank you!
