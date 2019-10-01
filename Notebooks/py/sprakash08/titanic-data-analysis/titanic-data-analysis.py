#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


#Get all the imports

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


get_ipython().magic(u'matplotlib inline')
sns.set_style('whitegrid')
sns.set_palette('viridis')


# In[ ]:


titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")


# In[ ]:


titanic_train.head(10)


# In[ ]:


titanic_test.head(10)


# From the titanic data, we can clear see that there are some missing data and unwanted cols which can be removed.

# In[ ]:


#Remove Columns
titanic_train.drop(columns=["PassengerId","Name","Ticket"],inplace=True)
test = titanic_test.drop(columns=["PassengerId","Name","Ticket"],inplace=False)


# In[ ]:


titanic_train.head(5)


# There are some nulls in the age column. We have to preprocess them as well.
# 
# The idea which i have followed is:
# 1. If Sex=Male and SibSp=1 and Parch > 0, take its mean which are not null
# 2. If Sex=Female and SibSp=1 and Parch > 0, take its mean which are not null
# 
# Do the same with SibSp=0

# In[ ]:


male_mar = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["SibSp"] == 1) & (titanic_train["Parch"] > 0)]
female_mar = titanic_train[(titanic_train["Sex"] == "female") & (titanic_train["SibSp"] == 1) & (titanic_train["Parch"] > 0)]
nan_child = male_mar = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["SibSp"] == 1) & (titanic_train["Parch"] == 0)]


# In[ ]:


male_mar_mean = male_mar["Age"].mean()
female_mar_mean = female_mar["Age"].mean()
nan_mean = nan_child["Age"].mean()

age_mean = (male_mar_mean + female_mar_mean + nan_mean) / 3

age_mean


# In[ ]:


type(titanic_train["Age"].iloc[0])


# In[ ]:


titanic_train.Age.fillna(age_mean,inplace=True)
test.Age.fillna(age_mean,inplace=True)
titanic_train.describe()


# Now, let us check how the age is distributed.

# In[ ]:


sns.distplot(titanic_train['Age'],bins=30,kde=False)


# How many number of male and female are present in the training set?

# In[ ]:


#Count of men
men = titanic_train[titanic_train["Sex"] == "male"]
#Count of female
women = titanic_train[titanic_train["Sex"] == "female"]

print("Men -->" + str(len(men)))
print("Women -->" + str(len(women)))


# In[ ]:


sns.countplot(x="Sex",data=titanic_train)


# 1. How are the passenger classes divided ?
# 
# Class-1 : Upper <br>
# Class-2: Middle <br>
# Class-3: Lower

# In[ ]:


ax = sns.countplot(x="Pclass",data=titanic_train,hue="Sex")


# In[ ]:


#How many non siblings are on the ship?
non_sibling = titanic_train[titanic_train["SibSp"] == 0]
sns.countplot(x='Sex',data=non_sibling)


# In[ ]:


#No of male having no siblings
non_sibling[non_sibling['Sex']=="male"]["Sex"].count()


# In[ ]:


#No of female having no siblings
non_sibling[non_sibling['Sex']=="female"]["Sex"].count()


# In[ ]:





# How many people survived and not ?

# In[ ]:


men_survived = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["Survived"] == 1)]
men_not_survived = titanic_train[(titanic_train["Sex"] == "male") & (titanic_train["Survived"] == 0)]
women_survived = titanic_train[(titanic_train["Sex"] == "female") & (titanic_train["Survived"] == 1)]
women_not_survived = titanic_train[(titanic_train["Sex"] == "female") & (titanic_train["Survived"] == 0)]
print("Survived Men --> " + str(len(men_survived)))
print("Survived Women --> " + str(len(women_survived)))
print("Not Survived Men --> " + str(len(men_not_survived)))
print("No Survived Women --> " + str(len(women_not_survived)))


# In[ ]:





# In[ ]:


sns.countplot(x="Survived",data=titanic_train,hue="Sex")


# In[ ]:


titanic_train["Embarked"].unique()


# # Making Predictions:
#     1. Applying Decision Trees

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


# In[ ]:


titanic_train["Embarked"].value_counts()

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")


# In[ ]:


dict = {"S":0,"C":1,"Q":2}

def convert_e_2_num(x):
    if x in dict:
        return int(dict[x])
    return 0


# In[ ]:


dictS = {"male":0,"female":1}

def convert_s_2_num(x):
    if x in dictS:
        return int(dictS[x])


# In[ ]:


titanic_train['Emb'] = titanic_train['Embarked'].apply(convert_e_2_num)
test["Emb"] = test['Embarked'].apply(convert_e_2_num)
titanic_train.head(10)


# In[ ]:


titanic_train["S"] = pd.to_numeric(titanic_train['Sex'].apply(convert_s_2_num))
test["S"] = pd.to_numeric(test['Sex'].apply(convert_s_2_num))


titanic_train.head(10)

titanic_train.info()



# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


#Taken from Github: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823 (Modified duplicate printing)
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    #heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    #heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


x = titanic_train.drop(columns=['Survived','Cabin','Sex','Embarked'],axis=1)
y = titanic_train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=0)


# In[ ]:


dtree.fit(x_train,y_train)


# In[ ]:


scores = cross_val_score(dtree,x_train,y_train,cv=10)


# In[ ]:


scores.mean()


# In[ ]:


test_1 = test.drop(columns=['Cabin','Sex','Embarked'],axis=1)
test_1.Fare.fillna(0.0,inplace=True)


# In[ ]:


pred = dtree.predict(x_test)


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


np = confusion_matrix(y_test,pred)


# In[ ]:


print_confusion_matrix(np,["Survived","Not Survived"],figsize=(8,5))


# In[ ]:


#some actual predictions
predictions = dtree.predict(test_1)


# In[ ]:


len(predictions)


# In[ ]:


#Not used
#Write to a csv
'''
count=0
preds = str(predictions,encoding="utf-8")
with open("../input/gender_submission.csv","w") as f:
    f.write("PassengerId", "Survived")
    while(count < len(predictions)):
        strObject = str(titanic_test["PassengerId"].iloc[count]) + "," + str(predictions[count])
        f.write(strObject)
        count+=1
    f.close()
'''


# In[ ]:


randf = RandomForestClassifier()


# In[ ]:


scoreR = cross_val_score(randf,x_train,y_train)
scoreR.mean()


# In[ ]:


randf.fit(x_train,y_train)


# In[ ]:


pred = randf.predict(x_test)


# In[ ]:


print(classification_report(y_test,pred))
cp = confusion_matrix(y_test,pred)


# In[ ]:


print_confusion_matrix(cp,["Survived","Not Survived"])


# In[ ]:





# In[ ]:





# ## More to follow - - - 
