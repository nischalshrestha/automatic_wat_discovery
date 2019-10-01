#!/usr/bin/env python
# coding: utf-8

# * Titanic - Starting with Machine Learning

# # Titanic - Starting with Machine Learning

# This is my first experimentation with SKLearn and first time I use kaggle, so don't be hard on me ! :)

# ## Importing the data and fast review
# Start by making the necessary import, and loading the data

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# So here 'Survived' is our target.
# 
# We can see with have multiple type of data (categorical, numeric, string...).
# 
# Let's get more information :

# In[3]:


train.info()


# Age, Cabin have a lot of mising value, we will also have to deal with this later. Embarked has only 2 missing values, this would be easier.
# 
# Let's check the test dataframe :

# In[ ]:


test.info()


# Same as the train dataframe, we will need to deal with the 'Age' and 'Cabin' missing values.
# 
# We can also see that 'Fare' has one mising value.

# In[2]:


print('Mean survived : '+str(train['Survived'].mean()))


# ## Exploratory analysis and Data preparation - Feature by Feature

# ### Pclass and Sex

# In[4]:


sns.barplot('Pclass', 'Survived',data=train, order=[1,2,3])


# In[5]:


sns.barplot('Sex', 'Survived',data=train)


# In[ ]:


sns.barplot('Pclass', 'Survived', 'Sex', data=train)


# Nothing surprising here, let's just encode our features. 

# In[ ]:


train['Sex'] = train['Sex'].map({'male':0, 'female':1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})


# ### Family  features

# In[ ]:


sns.barplot('SibSp', 'Survived',data=train)


# In[ ]:


sns.barplot('Parch', 'Survived',data=train)


# Let's group this 2 features to form Family

# In[ ]:


train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']


# In[ ]:


sns.barplot('Family', 'Survived', data=train)


# We could group the value in 3 categories :  no relative (0), small (1-3) and large(>3).
# 
# When we look by class and sex , it's not that distinct so I decided to let it like this and let the model learn by himsef.

# In[ ]:


s = sns.FacetGrid(train, row='Pclass', col='Sex', aspect=3)
s.map(sns.barplot,'Family', 'Survived', ci=None)
s.add_legend()


# ### Embarked

# In[ ]:


train.groupby('Embarked')['Pclass'].count()


# In[ ]:


test.groupby('Embarked')['Pclass'].count()


# In[ ]:


sns.barplot('Embarked','Survived', data=train)


# We can clearly see that passager who embarked at Cherbourg have a better chance to survive.
# 
# But for Q, even if it looks like you have a better chance to survive...We don't have a lot of values, and  maybe this is specific to the train data, so in order to avoid overfiding, let's consider S and Q even.

# In[ ]:


def groupembarked(a):
    if a=='C':
        return 1
    return 0
train['EmbarkedAtC'] = train['Embarked'].apply(groupembarked) #I didn't fill the NaN value so we can't use map
test['EmbarkedAtC'] = test['Embarked'].apply(groupembarked)


# ### Cabin

# In[ ]:


train[train.Cabin.notnull()]['Cabin'][:10]


# We can see that data is  like Letter+N° , let's extract the letter and see what we have

# In[ ]:


def keepfirst(a):
    if isinstance(a, str):
        return a[0]
    return 'U'
train['Cabin'] = train['Cabin'].apply(keepfirst)
test['Cabin'] = test['Cabin'].apply(keepfirst)


# In[ ]:


sns.barplot('Cabin','Survived', data=train, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'])


# In[ ]:


train.groupby('Cabin')['Survived'].count()


# In[ ]:


test.groupby('Cabin')['Pclass'].count()


# With this graphic, it clear that knowing the cabin improve the chance of survival, I choose to make 3 group : Unknow 'U' + T (only 1 value... not sure this will change anything), B D E, and the rest together A C F G. This so completly arbitrary, and you may have another grouping ! 

# In[ ]:


def cabingroup(a):
    if a in 'CGAF':
        return 1
    elif a!='U':
        return 2
    else :
        return 0

train['CabinGroup'] = train['Cabin'].apply(cabingroup)
test['CabinGroup'] = test['Cabin'].apply(cabingroup)


# In[ ]:


sns.pointplot('CabinGroup','Survived', data=train)


# For the man,  It's clear that be a Master help you survive, but for the rest ? 

# ### Title And Name
# In this section, we extract the Title from the name. Then we group them into categories.

# In[ ]:


train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.groupby('Title')['Survived'].mean()


# In[ ]:


def replaceTitle(df):
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Dona'],'HighF')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Rev', 'Jonkheer', 'Don','Sir'], 'HighM')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df.loc[(df.Sex == 0)   & (df.Title == 'Dr'),'Title'] = 'Mr'
    df.loc[(df.Sex == 1) & (df.Title == 'Dr'),'Title'] = 'Mrs'
    
replaceTitle(train)
replaceTitle(test)
print('Title replace')


# In[ ]:


sns.barplot('Title','Survived','Pclass', data=train[train.Sex == 0], ci=None)


# For the man,  It's clear that be a Master help you survive, but for the rest ? 

# In[ ]:


sns.barplot('Title','Survived','Pclass', data=train[train.Sex == 1], ci=None)


# For the woman, most first class survive, so it doesn't look to change anything.

# In[ ]:


title_mapping = {"Mr": 0, "Miss": 0, "Mrs": 0, "Master": 1, "HighF": 0, 'HighM':0 }
train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)


# In[ ]:


sns.barplot('Title','Survived','Pclass', data=train[train.Sex == 0])


# ### Age

# In[ ]:


train[train.Age.notnull()].Survived.mean()


# In[ ]:


train[train.Age.notnull() == False].Survived.mean()


# 

# In[ ]:


combined = pd.concat([train, test])
age_mean = combined['Age'].mean()
age_std = combined['Age'].std()

def fill_missing_age(a):
    if np.isnan(a):
        return np.random.randint(age_mean-age_std, age_mean+age_std, size=1)
    return a
train['AgeFill']=train['Age'].apply(fill_missing_age)
test['AgeFill']=test['Age'].apply(fill_missing_age)


# In[ ]:


sns.regplot('Age', 'Survived', data=train, order=3)


# In[ ]:


s = sns.FacetGrid(train, row='Pclass', aspect=3)
s.map(sns.barplot,'Age', 'Survived', ci=None)
s.set(xlim=(1,30))
s.add_legend()


# 

# In[ ]:


def agegrouping(a):
    if a<2:
        return 1
    if a<12:
        return 2
    if a>60:
        return 3
    return 0
train['AgeGroup'] = train['AgeFill'].apply(agegrouping)
test['AgeGroup'] = test['AgeFill'].apply(agegrouping)


# ## Fare : Ticket not that useless

# In[ ]:


s = sns.FacetGrid(train,hue='Survived',aspect=3, size=4)
s.map(sns.kdeplot,'Fare',shade=True)
s.set(xlim=(0,200))
s.add_legend()


# Before we go further, what if we investigate a litlle more using the ticket :

# In[ ]:


#Ticket=train.groupby('Ticket').count()
combined = pd.concat([train.drop('Survived', axis=1), test])
ticket=combined[combined.duplicated(subset=['Ticket'], keep=False)].sort_values('Ticket')[['Name', 'Ticket','Fare', 'Family', 'Pclass', 'Cabin','Age', 'Embarked']]
ticket.head()


# The Fare is the fare of the ticket... one ticket can be multiple people.  We could even use this to correct the family feature ! (I will update the kernel with this later, to see if it improve our model)
# 
# Let's calcul the real fare by people.

# In[ ]:


combined = pd.concat([train.drop('Survived', axis=1), test])
ticket_count = combined[combined.duplicated(subset=['Ticket'], keep=False)].sort_values('Ticket')[['Name', 'Ticket']]
ticket_count = ticket_count.groupby('Ticket').count()[['Name']].reset_index()
ticket_count.columns = ['Ticket','Count']
ticket_count.head()


# In[ ]:


def calculFare(df, ticket_count):
    farecalcul = df['Fare'].copy()
    for row in df.iterrows():
        t = ticket_count[ticket_count.Ticket.str.match(row[1]['Ticket'])]
        if t.empty==False:
            farecalcul[row[0]]= row[1]['Fare']/t['Count'].values[0]
    return farecalcul
train['FareCalcul'] = calculFare(train, ticket_count)


# In[ ]:


s = sns.FacetGrid(train,hue='Survived',aspect=3, size=4)
s.map(sns.kdeplot,'FareCalcul',shade=True)
s.set(xlim=(0, 60))
s.add_legend()


# I bet the 3 picks we see correspond to the 3 classes :

# In[ ]:


s = sns.FacetGrid(train,hue='Survived', row='Pclass', aspect=3, size=4)
s.map(sns.kdeplot,'FareCalcul',shade=True)
s.set(xlim=(0, 60))
s.add_legend()


# Et voilà ! Almost a perfect match !  With this, I almost certain that fare is not going to help us ! We already have all the infos in the family feature ! 

# 

# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[ ]:


Keep = ['Pclass', 'Sex', 'Family', 'AgeGroup', 'EmbarkedAtC', 'CabinGroup', 'Title']
X=train[Keep]
y=train['Survived']


# In[ ]:


X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()
param = {'criterion': ['gini', 'entropy'],
         'max_depth': [3, 4, 5, 6, 10, 20, None],
         'max_features': ['sqrt','log2', None],
         'min_samples_leaf': [1, 2, 5, 0.05, 0.1, 0.2],
         'min_samples_split': [2, 0.05, 5, 0.1, 0.2, 0.3]}
GS = GridSearchCV(clf, param, scoring='accuracy', cv=5, n_jobs=5 )
GS.fit(X, y)
pred = GS.predict(X_test)
print(accuracy_score(y_test, pred))
print(GS.best_score_)
GS.best_estimator_


# In[ ]:


X_test=test[Keep]
DTC = DecisionTreeClassifier()
DTC.fit(X,y)
pred = DTC.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': pred })
output.to_csv('titanic-DecisionTree.csv', index = False)


# In[ ]:


X_test=test[Keep]
LR = LogisticRegression()
LR.fit(X,y)
pred = LR.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': pred })
output.to_csv('titanic-LogisticRegression.csv', index = False)


# In[ ]:


X_test=test[Keep]
svc = SVC()
svc.fit(X,y)
pred = svc.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': pred })
output.to_csv('titanic-SVC.csv', index = False)


# In[ ]:


X_test=test[Keep]
ensemble_voting = VotingClassifier(estimators=[('lg', LR), ('svm', svc), ('dc', DTC)], voting='hard')
ensemble_voting.fit(X, y)
pred = ensemble_voting.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
submission.to_csv('submission_ensemble_voting.csv', index=False)


# ### Final Though
# 
# So this is my first kernel, I really enjoy doing it, I learned a lot and tried to experiment as much as possible ! 
# I read a lot of kernel available on kaglle, so I'm happy to share this one.
# 
# The Decision Tree actually give a score of 0.803 on the leaderbord !
# 
# If anyone have any idea to imporve the model, i'm open to any sugestion !
