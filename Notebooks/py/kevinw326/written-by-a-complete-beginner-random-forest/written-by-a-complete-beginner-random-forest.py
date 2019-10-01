#!/usr/bin/env python
# coding: utf-8

# If someone does actually read this, please let me know in the comments if there's anything that I can improve on code (like more effcient ways to write the helpers etc.) or writing-wise, or with anything in regards to ML.

# **Setup for the File**

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import re
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

import os
print(os.listdir("../input"))


# **Getting the Training and Testing Data**

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **Functions for Encoding Categorical Data such as Male/Female, etc.**
# 
# Because some of the categories that are important are categorical, it is necessary to first encode them before running any functions on them. Furthermore, to further categorize the data, we can also split features such as age, which has a large span, into classes of their own, such as 0 to 10, 10 to 20, etc. We will just use these numbers, and following increments of 10, to encode age.  We can encode sex using 0 for male and 1 for female, and by encoding these values using simple functions, we can pass them through sklearn functions. However, we know that there are missing values in the training data, such as in age. Perhaps we can see whether we can just plug them into a most common group.

# In[ ]:


fig, ax = pyplot.subplots(1,figsize = (12,9))
sns.distplot(train['Age'].dropna(), bins = 16, ax = ax)
print("Missing Ages: ", len(train[train['Age'].isnull()]))


# We see that most of the passengers are around 15-30 years old. However, we see that there are 177 passengers with an unknown age. We can't just say that all of them are 15-30. So, even though it will affect the training set, we will just drop them for now. Later, maybe we can put them in to keep the data.

# In[ ]:


#drop the null Age values
#remove the null data for now
train_drop = train.copy()
train_drop = train_drop.dropna(subset = ['Age'])

#encoding the sex into 0s and 1s
def encodeSex(sex):
    if sex == "male": return 0
    #if female
    return 1

#encoding the ages of passengers into groups
def encodeAge(age):
    return int(age/5)

#encode the data in the DataFrame
train_drop.Sex = train_drop.Sex.apply(encodeSex)
train_drop.Age = train_drop.Age.apply(encodeAge)


# **Picking Features**
# 
# There are several factors that could affect the survival of a particular passenger. For example, features such as Pclass and Fare are representative of the passenger's wealth, and wealthier passengers could have been deemed important and were priorities to save. Sex and Age were important as women and children were prioritized as well. 

# In[ ]:


#The features that seem important to survival
features = ['Pclass','Sex', 'Age', 'Fare']


# Now that we have everything, we can put a Random Forest Classifier on our data. 

# In[ ]:


X = train_drop[features]
y = train_drop.Survived
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

#create a RandomForestClassifier
forest = RandomForestClassifier(max_leaf_nodes = 55)
forest.fit(train_X, train_y)
print("Feature importance: ", forest.feature_importances_)
print("Accuracy: ", forest.score(test_X, test_y))


# In testing, I found that ~50 for max_leaf_nodes results in, on average, the best accuracy, which seems to hover around 85%. Furthermore, using forest.feature_importances_, we see that age and fare seem to be the best indicators of survival. However, we did drop quite a few values. Perhaps, by keeping those values in and adding them to the most common group, something will change. But we can't just drop them all into the most common age group, as that would skew the data. Instead, we can try to predict the age of the passengers.
# 
# A pattern that one may see is in the titles given to the passengers, which could be a solid indicator of their average age. To test this theory, we can create a new feature for each passenger called Title, which will indicate what title the passenger has. We can see that titles such as Mrs. and  Mr. indicate that the passenger is married, and therefore may be in an older group, while Miss. and Master. mean that the passenger may be younger. Aside from these there are also other titles, such as Sir and Don. With these, we can see if there is a corrlation between age and title to predict the missing ages. To go along with this, we can use the other existing features to predict age as well. 

# In[ ]:


#Some of the code here is borrowed

def extractTitle(name):
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return ""
    
#Create a new column in the data called title
train_with_title = train.copy()
train_with_title = train_with_title.dropna(subset = ['Age'])
train_with_title['Title'] = train_with_title.Name.apply(extractTitle)

#plot the age against the title
fig, ax = pyplot.subplots(1,figsize = (18,6))
sns.scatterplot(x = "Title", y = "Age", data = train_with_title, ax = ax)


# Maybe there isn't much of a correlation. While it seems that Master generally indicates a young male, and Miss a young to middle-aged female, the other two significant titles, Mr and Mrs, have a wide span of ages. Nevertheless, we'll run a linear regression on this.

# In[ ]:


#encode the titles
def encodeTitle(title):
    if title == "Master": return 0
    if (title == "Miss" or title == "Mme" or
       title == "Ms" or title == "Mlle"): return 1
    if (title == "Mr" or title == "Mrs" or
       title == "Countess" or title == "Jonkheer"): return 2
    if (title == "Rev" or title == "Dr"): return 3
    if (title == "Don" or title == "Major" or
        title == "Lady" or title == "Sir"): return 4
    return 5

train_with_title.Title = train_with_title.Title.apply(encodeTitle)

#plot the age against the title
fig, ax = pyplot.subplots(1,figsize = (18,6))
sns.scatterplot(x = "Title", y = "Age", data = train_with_title, ax = ax)

test_features = ["Title"]

#We can test a decision tree on train_with_title
line = LinearRegression()

X = train_with_title[test_features]
y = train_with_title.Age

#training the data
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
line.fit(train_X, train_y)
print("Accuracy: ", line.score(test_X, test_y))


# It did better than I expected, but the prediction is still pretty bad. But, we can see an upward trend in age as the title number increases, so maybe not all is lost. Perhaps, by adding in some other features, we can increase the accuracy of our age predictor. 

# In[ ]:


test_features = ["Title", "SibSp", "Parch", "Pclass", "Fare"]

X = train_with_title[test_features]
y = train_with_title.Age

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
line.fit(train_X, train_y)
print("Accuracy: ", line.score(test_X, test_y))


# It does a little bit better, but it's still a pretty bad predictor. But, it's definitely better than just putting all the missings values into the same class, so we can add them in and see where it goes from there.

# In[ ]:


predicted_ages = train[train['Age'].isnull()]
predicted_ages['Title'] = predicted_ages.Name.apply(extractTitle)
predicted_ages['Title'] = predicted_ages.Title.apply(encodeTitle)
predictions = pd.Series(line.predict(predicted_ages[test_features]))

train_with_ages = train
#fill in the missing values
#There should be a cleaner and quicker way to write this
#because doing this is really really slow
inc = 0
for i in range(0, len(train['Age'])):
    if math.isnan(train['Age'][i]):
        train_with_ages['Age'][i] = predictions[inc]
        inc += 1


# Now that we've filled in the values, let's fit the classifier again. 

# In[ ]:


#encode the new df
#encode the data in the DataFrame
train_with_ages.Sex = train_with_ages.Sex.apply(encodeSex)
train_with_ages.Age = train_with_ages.Age.apply(encodeAge)

features = ['Pclass','Sex', 'Age', 'Fare']

X = train_with_ages[features]
y = train_with_ages.Survived

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

#fit the RandomForestClassifier
forest = RandomForestClassifier(max_leaf_nodes = 55)
forest.fit(train_X, train_y)
print("Feature importance: ", forest.feature_importances_)
print("Accuracy: ", forest.score(test_X, test_y))


# Now that we filled in the ages, we see that the accuracy actually decreased, likely due to some error from predicting the age with only a 40% accuracy rate. But, for now, I will accept it as is and submit. If anybody does happen to read this, please leave a comment with what I can improve or other ideas that I should look at. I'll probably come back to this and try to improve my accuracy periodically.

# In[ ]:


#making, training and submitting test
predicted_ages = test[test['Age'].isnull()]
predicted_ages['Title'] = predicted_ages.Name.apply(extractTitle)
predicted_ages['Title'] = predicted_ages.Title.apply(encodeTitle)
predictions = pd.Series(line.predict(predicted_ages[test_features]))

#fill in the missing values
#There should be a cleaner and quicker way to write this
inc = 0
for i in range(0, len(test['Age'])):
    if math.isnan(test['Age'][i]):
        test['Age'][i] = predictions[inc]
        inc += 1
        
#encode the new df
#encode the data in the DataFrame
test.Sex = test.Sex.apply(encodeSex)
test.Age = test.Age.apply(encodeAge)
test['Fare'] = test['Fare'].fillna(value = 13)

features = ['Pclass','Sex', 'Age', 'Fare']
X = test[features]

final_predict = forest.predict(X)
prediction = pd.DataFrame(test.PassengerId)
prediction['Survived'] = final_predict.astype('int')

prediction.to_csv('predict.csv',index = False)

