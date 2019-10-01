#!/usr/bin/env python
# coding: utf-8

# **Titanic Survival Predictions**
#  
#  
# **I hope you'll find value in this project of mine. Any upvotes or comments of support are highly appreciated :)**
# 
# I tried to make this kernal a fun kernal as well as an educational one. you'll see later on a funny incedent when trying to deal with missing age data, however what was important about displaying it in that sense was not just for entertainment. **it's also about how to deal with such incidents since they are bound to happen.**
# 
# Well let's get right to it shall we :)
# 
# We are here to predict survival rates of those unfortune lads on the RMS Titanic (Jack.. don't go jack..).
# (btw he was a fictional character... just saying...)
# 
# This project is divided into 4 parts:
# > 1. Importing Data
# > 2. Exploring the Data
# > 3. Feature Engineering and Dealing with Missing Data
# > 4. Fitting the data to several ML models.
# 
# Who will survive? if you watched the movie you'd probably guess.. women and children only! 
# 
# Well, let's see if that true. I'm sure there's more to this tragic event.
# 
# **First things First! let's import some libraries:**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#data analysis libraries 

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import os
print(os.listdir("../input"))

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.


# Okay, after importing important libraries and inputting our data. let's take a peak.

# In[ ]:


#import train and test CSV files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#take a look at the training data and it's shape
print(train.shape, test.shape)
train.describe(include="all")


# After import libraries, naturally the first step after that would be to check the nature of our data. it's always important to see the shape of the data before doing anything else. 
# 
# We have more rows for the training data than the test data, and it seems that we have an extra column for the training data (can you guess what it is?) 
# 
# It seems that we have a lot of NaNs to deal with as well. that's to be expected though.
# 
# let's see our columns:

# In[ ]:


print (train.columns)


# if you haven't figured out what's the extra column, it's actually 'Survived'. that's what we are trying to do here after all. to predict who survived and who didn't.
# 
# It doesn't seem that we have too many features to deal with. let's see the first few rows of the training data. maybe we can see something there:

# In[ ]:


train.head()


# it doesn't seem like we don't need passengerId since it won't help us with our predictions (no correlation between it and 'Survived')
# 
# let's go ahead and drop it (but will save it for later submisson of the results file):

# In[ ]:


#Save the 'Id' column
train_ID = train['PassengerId']
test_ID = test['PassengerId']

train = train.drop('PassengerId',axis=1)
test = test.drop('PassengerId',axis=1)


# Now let's see some correlations :)

# In[ ]:


corrmat = train.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);


# It seems that amongst the correlations in our heatmap (the numerical features), 'Pclass' seems to have the highest absolute value (0.34).
# 
# But wait, the categorical features (sex, cabin, ticket, embarked)?
# 
# **What about the 'Sex' feature ?**

# In[ ]:


sns.countplot(x='Sex', hue='Survived', data=train, palette='RdBu')
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Aha.. we can see that from all survivors, about 3/4 of them were women.
# 
# 
# **What about the 'Embarked' feature ?**

# In[ ]:


# What about the 'Embarked' feature ?
sns.countplot(x='Embarked', hue='Survived', data=train, palette='RdBu')
plt.xticks([0,1,2],['Southampton','Cherbourg ','Queenstown '])
# train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Aha.. another important insight, from the count plot we can  understand that the majority of the passengers were embarking from Southampton.
# 
# Now, let's check how many survived from each class (since it had the highest correlation with survived as we saw from the heatmap).

# In[ ]:


sns.barplot(train.Pclass ,train.Survived)
# train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# As expected, those of the first class had the greatest survival rate while those of the third where the most unfortunate :(
# 
# Remember that we said that it was mostly women and children?
# 
# well then, let's check the age feature and how to correlates with Survival.

# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# **Some conclusions from the graph:**
# 
# 1. if it looks strange to you that there are people with ages 0 who survived, that's because their age was missing.
# 
# 2. most people that didn't survive were between the ages of 15-25
# 
# 3. most people who survived we between the ages of 20-35
# 
# 4. babies of ages <5 had high survival rate.
# 
# (These insights are based on the graph. To get more accurate results, we could simply filter the data by age groups and plot a bar plot, or even add age groups as features to the dataframe. **we will do that later when we will try to fill missing data**)
# 
# okay..
# 
# Before we dive into feature engineering, let's join our training data and test data so that we won't get lost later and stay consistent with changes across the data

# In[ ]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Survived.values
all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['Survived'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# Now let's see how much of our data is missing:
# 

# In[ ]:


missingData = all_data.isnull().sum().sort_values(ascending=False)
percentageMissing = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)
totalMissing = pd.concat([missingData, percentageMissing], axis=1, keys=['Total','Percentage'])
totalMissing


# **Feature engineering & Dealing with missing data**
# 
# More than 3/4 of Cabin data is missing. hmmm.. 
# I want to understand why. could that be indicative of survival/non-survival?
# 
# well, let's see..

# In[ ]:


all_data["hasCabin"] = (all_data["Cabin"].notnull().astype('int'))
sns.barplot(x="hasCabin", y="Survived", data=all_data)
plt.show()
all_data[['hasCabin', 'Survived']].groupby(['hasCabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# all_data = all_data.dropna('Cabin',axis=1)
# all_data = all_data.dropna('Embarked', axis=0)


# As suspected, Having a Cabin, as a good predictor of survival.
# That's great. this could be an important feature in the endouver of predicting survival. 
# 
# let's drop the 'Cabin' feature, and stay with 'hasCabin'.
# 
# we can also drop the Ticket column since it doesn't have any special pattern that could aid us with predictions.

# In[ ]:


all_data = all_data.drop('Cabin',axis=1)
all_data = all_data.drop('Ticket',axis=1)


# replacing the 2 missing values in the Embarked feature with S
# since majority of people embarked in Southampton (S)
all_data = all_data.fillna({"Embarked": "S"})


all_data.head()


# As we saw from the missing data table, the second most missing data we are dealing with is the 'Age' feature. 
# 
# A creative way to deal with this would be to look at people's names and try to classify them to age groups based on their titles. 
# 
# let's try to extract all the titles (notice that each title ends with a '.') 

# In[ ]:


all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(all_data['Title'], all_data['Sex'])


# We can replace many titles with a more common name or classify them as 'other'.
# 
# 

# In[ ]:


#get cell value: all_data.loc[0].at['Title']
#set cell value: all_data.at[0,'Title'] = 'Mr'

for i,row in all_data.iterrows():
    x = all_data.loc[i].at['Title']
    if x in ['Capt','Col','Don' ,'Dr' ,'Major','Rev' ,'Sir']:
        all_data.at[i,'Title']= 'Mr'
    if x in ['Mlle','Ms' ,'Dona' ,'Lady']:
        all_data.at[i,'Title']= 'Miss'
    if x in ['Countess','Jonkheer','Mme']:
        all_data.at[i,'Title'] = 'other'
        
pd.crosstab(all_data['Title'], all_data['Sex'])


# LOL!!!!
# 
# it seems like we have a female Mr amongst our ship :P
# 
# let's try to figure out why...
# 
# if you look at the Dr title, you'll realize that we have 7 male doctors and 1 female doctor.
# 
# It seems that i assumed all doctors were female (sorry for the sexism, wasn't intentional i swear :) )
# 
# let's fix that shall we ;)

# In[ ]:


allFemales = all_data[all_data['Sex']=='female'] # select all females
ThatOneFemale = allFemales[all_data['Title']=='Mr'] # select all females with title Mr
ThatOneFemale


# In[ ]:


# extracted the index of ThatOneFemale to be 796
all_data.at[796,'Title']='Mrs'

pd.crosstab(all_data['Title'], all_data['Sex'])


# Alrighty then, moving on!
# 
# Now, let's convert the categorical titles to numeric.
# 

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "other": 5}
for i,row in all_data.iterrows():
    if all_data.loc[i].at['Title'] in title_mapping:
        all_data.at[i,'Title']= title_mapping[all_data.loc[i].at['Title']]
# all_data['Title']
all_data.head()


# Now let's fill in missing Age data based on the mean for each Title:

# In[ ]:


Mr_age = all_data[all_data['Title']==1].Age.mean()
Miss_age = all_data[all_data['Title']==2].Age.mean()
Mrs_age = all_data[all_data['Title']==3].Age.mean()
Master_age = all_data[all_data['Title']==4].Age.mean()
Other_age = all_data[all_data['Title']==5].Age.mean()
print(Mr_age, Miss_age, Mrs_age , Master_age, Other_age)

group_age_mapping = {1:Mr_age, 2: Miss_age, 3:Mrs_age, 4:Master_age, 5:Other_age}

for index,row in all_data.iterrows():
    if np.isnan(all_data.loc[index].at['Age']):
        all_data.at[index,'Age'] = group_age_mapping[all_data.loc[index].at['Title']]
        


# Now drop the Name column since we don't need it anymore.

# In[ ]:


all_data.drop('Name',axis=1,inplace=True)


# coverting the 'Sex' and 'Embarked' features to numeric:

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

for i,row in all_data.iterrows():
    if all_data.loc[i].at['Sex'] in sex_mapping:
        all_data.at[i,'Sex']= sex_mapping[all_data.loc[i].at['Sex']]
    if all_data.loc[i].at['Embarked'] in embarked_mapping:
        all_data.at[i,'Embarked']= embarked_mapping[all_data.loc[i].at['Embarked']]
all_data.head()


# Now for the Fare, we can complete the missing value, with the most frequent value. 
# 
# and dropping th survived column since we won't be needing that anymore (but will be used for modeling).

# In[ ]:


mode = all_data['Fare'].mode() # extract the mode
all_data['Fare'].fillna(mode[0], inplace=True) # fill NaNs with the mode

all_data.drop('Survived',axis=1, inplace=True) # drop survived column

missingData = all_data.isnull().sum().sort_values(ascending=False)
percentageMissing = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)
totalMissing = pd.concat([missingData, percentageMissing], axis=1, keys=['Total','Percentage'])
totalMissing


# Cool! no more missing values :)
# 
# let's take a look at our data and how it's coming along.
# 

# In[ ]:


all_data.head()


# Hmmm, everything seems in order. 
# 
# We just have this 'Fare' feature that looks like it has potential to guide us with our predictions. 
# 
# Let catogrize the 'Fare' feature as well. we'll do that by dividing it to 4 caregories.

# In[ ]:


FareBand = pd.qcut(all_data['Fare'], 4)
FareBand.unique()


# okay, let's adjust the 'Fare' feature:

# In[ ]:


# get cell value: all_data.loc[0].at['Title']
# set cell value: all_data.at[0,'Title'] = 'Mr'

for i,row in all_data.iterrows():
    currFare=all_data.loc[i].at['Fare']
    if (currFare > -0.001 and currFare <=7.896):
        all_data.at[i,'Fare'] = 1
    if (currFare > 7.896 and currFare <=14.454):
        all_data.at[i,'Fare'] = 2
    if (currFare > 140454 and currFare <=31.275):
        all_data.at[i,'Fare'] = 3
    if (currFare > 31.275 and currFare <=512.329):
        all_data.at[i,'Fare'] = 4
        
all_data.head(10)


# Awesome! now that our data is ready. we can go ahead to the fun part of this project :)
# 
# ***Modeling and Machine Learn***
# First things first, let's split our data to training data (to fit the model) and test data (to evaluate the model later on).

# In[ ]:


target = train['Survived']
trainData = all_data[0:ntrain]
testData = all_data[ntrain:]
target.shape, trainData.shape, testData.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(trainData, target, test_size=0.2, random_state=0)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# **Selecting the Best Model**
# 
# An accepted approch for selecting the best model is to try many differnt model and choose the model with the best results!

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(accuracy_score(y_pred, y_test), 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg = round(accuracy_score(y_pred, y_test), 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(accuracy_score(y_pred, y_test), 2)
print(acc_svc)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pred = decisiontree.predict(X_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test), 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) , 2)
print(acc_randomforest)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_test)
acc_gbk = round(accuracy_score(y_pred, y_test) , 2)
print(acc_gbk)


# In[ ]:


#predictions for submission
predictions = gbk.predict(testData)


# There are more classificationn algorithms that we can try. 
# 
# * like KNN or k-Nearest Neighbors
# * Perceptron
# * Stochastic Gradient Descent
# 
# and more...
# 
# however, it seems that Gradient Boosting Classifier gives us the best model. let's go ahead and submit the results ;)

# In[ ]:


output = pd.DataFrame({ 'PassengerId' : test_ID, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# **Thank you for joining me on this wonderful journey to Data Science :)**
# 
# **I hope you found value in this project of mine. Any upvotes or comments of support are highly appreciated :)**
