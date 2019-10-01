#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


titanic_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


#Need to fill missing values for age so will explore that first
titanic_df['Age'].hist(bins=100, label="Train set")
test_df['Age'].hist(bins=100, label="Test set")
plt.legend()


# In[ ]:


#There doesn't appear to be any outliers so will replace all NaN values with median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)


# In[ ]:


#Define who is a child, female adult and male adult:
def child_female_male(passenger):
    Age, Sex = passenger
    if Age < 16:
        return 'child'
    else:
        return Sex


# In[ ]:


#Create new column to define if passenger was child/female/male
titanic_df['Type'] = titanic_df[['Age', 'Sex']].apply(child_female_male, axis = 1)
test_df['Type'] = test_df[['Age', 'Sex']].apply(child_female_male, axis = 1)


# In[ ]:


#plot this
sns.factorplot('Type',data=titanic_df, kind="count", palette='summer')


# In[ ]:


#Look at the amount of 'Type' of person in each class
sns.factorplot('Pclass', data=titanic_df, kind='count', hue='Type', x_order=(1,2,3), palette='winter')


# In[ ]:


#Take a look at the dispersion of age in the different classes in the training set
fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
fig.set(xlim=(0,titanic_df['Age'].max()))
fig.add_legend()


# In[ ]:


#Where passengers embarked from
sns.factorplot('Embarked', data=titanic_df, kind='count')


# In[ ]:


#Fill NaN value for "Embarked" with most common value
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")


# In[ ]:


#Replace the one NaN value in test set 'Fare'
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)


# In[ ]:


#Convert Fare to int value
titanic_df['Fare']= titanic_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)


# In[ ]:


#Map each object to an integer
titanic_df['Type'] = titanic_df['Type'].map({'male': 0, 'female': 1, 'child':2})
test_df['Type'] = test_df['Type'].map({'male':0, 'female':1, 'child':2})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_df['Embarked'] = test_df['Embarked'].map({'C':0, 'Q':1, 'S':2})


# In[ ]:


#Look who had family and who didn't
titanic_df['Family'] = titanic_df['Parch'] + titanic_df['SibSp']
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0


test_df['Family'] = test_df['Parch'] + test_df['SibSp']
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0


# In[ ]:


#Can view linear relationships between different features and Survived
#Younger passengers with family had a higher chance of surviving
sns.lmplot('Age', 'Survived', hue='Family', data=titanic_df)


# In[ ]:


#Those who paid a higher Fare also had a higher chance of surving
sns.lmplot('Fare', 'Survived', hue='Sex', data=titanic_df)


# In[ ]:


#Members of a small family (<5) had a higher chance of survival so merge train set and test set
titanic_all = pd.concat([titanic_df, test_df], ignore_index=True)


# In[ ]:


#Say a small family is less than 5 whilst a large family contains 5 or more
titanic_all['Small_fam'] = titanic_all['Parch'] + titanic_all['SibSp'] + 1
titanic_all['Small_fam'].loc[titanic_all['Small_fam'] >= 5] = 0
titanic_all['Small_fam'].loc[titanic_all['Small_fam'] < 5] = 1


# In[ ]:


#Split back into train set and test set
titanic_df = titanic_all[:891]


# In[ ]:


test_df = titanic_all[891:]
test_df = test_df.reset_index(drop=True)


# In[ ]:


#Drop the features we won't use in our models
titanic_df = titanic_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Survived'], axis=1)


# In[ ]:


X_train = titanic_df.drop(["Survived"],axis=1)
Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()


# In[ ]:


from sklearn.preprocessing import Imputer
X_train = Imputer().fit_transform(X_train)


# In[ ]:


from sklearn import cross_validation


# In[ ]:


Xcross_train, Xcross_test, ycross_train, ycross_test = cross_validation.train_test_split(
    X_train, Y_train, test_size=0.2, random_state=0)


# In[ ]:


#Try Logistic Regression
from sklearn import linear_model
logistic = linear_model.LogisticRegression()


# In[ ]:


clf = logistic.fit(Xcross_train, ycross_train)
clf.score(Xcross_test, ycross_test)


# In[ ]:


#Try Naive Bayes (GaussianNB as we have few features vs. size of training set)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf = gnb.fit(Xcross_train, ycross_train)
clf.score(Xcross_test, ycross_test)


# In[ ]:


#Try Support Vector Machines
from sklearn import svm
clf = svm.SVC()
clf = clf.fit(Xcross_train, ycross_train)
clf.score(Xcross_test, ycross_test)


# In[ ]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200,
    min_samples_split=4,
    min_samples_leaf=2)
clf = clf.fit(Xcross_train, ycross_train)
clf.score(Xcross_test, ycross_test)


# In[ ]:


#Highest score so far, let's look at the Classification Report
from sklearn.metrics import classification_report
from sklearn import metrics
y_true, y_pred = ycross_test, clf.predict(Xcross_test)
print(classification_report(y_true, y_pred))


# In[ ]:


y_pred = clf.predict(X_test).astype(int)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': y_pred})
submission.to_csv('titanic_submission.csv', index=False)


# In[ ]:


#See if ExtraTrees Classifier is an improvement on RandomForest
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=200, max_depth=None,min_samples_split=1, random_state=0)
clf = clf.fit(Xcross_train, ycross_train)
clf.score(Xcross_test, ycross_test)


# In[ ]:




