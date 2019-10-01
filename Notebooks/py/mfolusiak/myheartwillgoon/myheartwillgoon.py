#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')
combine = [train_df,test_df]

print(train_df.columns.values)


# In[ ]:


train_df.head()
train_df.tail()


# Learn more about data types, especially when these are mixed data types

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()
print('_'*40)


# Knowing the survival rate of 38% invesigate perciles .61 and .62

# In[ ]:


train_df.sort_values(by='Survived').describe(percentiles=[.61, .62, .63])


#   Parch distribution using `percentiles=[.75, .8]`
# 

# In[ ]:


train_df.sort_values(by='Parch').describe(percentiles=[.75, .80])


#   SibSp distribution `[.68, .69]`

# In[ ]:


train_df.sort_values(by='SibSp').describe(percentiles=[.68, .69])


#   Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

# In[ ]:


train_df.sort_values(by='Age').describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])


# In[ ]:


train_df.sort_values(by='Fare').describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])


# Are passengers' name unique?

# In[ ]:


train_df.Name.unique().size == train_df.Name.size


# Summary only on object columns - names are uniquem 577 males, 644 emabrked at Southampton

# In[ ]:


train_df.describe(include=['O'])


# Summary only on numerical columns

# In[ ]:


train_df.describe()


# Next do steps of:
# * correlating
# * completing
# * correcting the data - we decide to skip Name, Ticket, Cabin and PassangerId
# * create new features - family, title, banded age and proce instead of continuous
# * classify - add assumptions based on problem specification
# 
# Let's check some observations

# * 1st class passangers are more likely to survive

# In[ ]:


train_df[['Pclass','Survived']].groupby('Pclass', as_index=False).mean().sort_values(by='Survived', ascending=False)


# * Women are more likely to survive

# In[ ]:


train_df[['Sex','Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Survived', ascending=False)


# What about having children or siblings?

# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# From the above one can learn that:
# * infants were more likely to survive
# * old passangers survived
# * most passangers were 20-45 y.o.
# **Decisions.**
# - Consider Pclass for model training.

# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend();


# **Decisions.**
# - Consider Pclass for model training.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# Very interesting observations:
# - much lower survival rate of man, especially in the third class.
# - except passangers embarking from Q. Almost none
# **Decisions.**
# 
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# Those paying more had better survival chance. Again port of embarkation correlates with survival rate
# **Decisions.**
# 
# - Consider banding Fare feature.

# ## Wrangle data ##
# Now lets corect some features. We decided to drop Cabin and Ticket
# 

# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket','Cabin'],axis=1)
test_df = test_df.drop(['Ticket','Cabin'],axis=1)
combine = [train_df, test_df]
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# Before removing Names extract title
# 
# **Decision.**
# 
# - We decide to retain the new Title feature for model training.

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# Limit number of labels for features. Print mean survival rate for each title.
# From the data it looks that Miss and Mrs are scoring pretty well.

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# Drop PassangerId and Name

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)

train_df.head()


# Some data in Age column is missing, therefore we need to generate it. We will use the correlation between Age, Pclass and Gender for this.

# In[ ]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Lets prepare empty array to store guessed age data for different Pclass and Gender

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


for dataset in combine: # each of two datasets!
    for i in range(0,2):
        for j in range(0,3):
            guess_age_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            ageguess = guess_age_df.median()
            # convert to nearest 0.5
            guess_ages[i][j] = int(ageguess/0.5 +0.5) *0.5
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age' ] = guess_ages[i][j]
    dataset['Age'] = dataset['Age'].astype(int)
    
train_df.head()


# Create AgeBand and create correlation with Survive

# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'],10)
train_df[['AgeBand','Survived']].groupby('AgeBand',as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


for dataset in combine:
    dataset.loc[                     dataset['Age']<= 8 , 'Age'] = 0
    dataset.loc[(dataset['Age']> 8)&(dataset['Age']<=16), 'Age'] = 1
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=24), 'Age'] = 2
    dataset.loc[(dataset['Age']>24)&(dataset['Age']<=32), 'Age'] = 3
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=40), 'Age'] = 4
    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=48), 'Age'] = 5
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=56), 'Age'] = 6
    dataset.loc[(dataset['Age']>56)&(dataset['Age']<=64), 'Age'] = 7
    dataset.loc[(dataset['Age']>64)&(dataset['Age']<=72), 'Age'] = 8
    dataset.loc[(dataset['Age']>72)                     , 'Age'] = 9
train_df.head()


# Remove AgeBand

# In[ ]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[ ]:


train_df.groupby('Age').mean()

Create new feature to combine siblings and parents into one feature.
# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).count()


# Lets create IsAlone feature, maybe will have better trend?

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[ dataset['FamilySize'] == 1 , 'IsAlone' ] = 1

train_df[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()


# Let's drop Parch, SibSp, FamilySize

# In[ ]:


train_df = train_df.drop(['Parch','SibSp','FamilySize'],axis=1)
test_df  = test_df.drop (['Parch','SibSp','FamilySize'],axis=1)
combine = [train_df, test_df]


# We can create artificial class combining age and class

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset['Age'] * dataset['Pclass']

train_df.loc[:,['Age*Class','Age','Pclass']].head(10)


# '' Completing a categorical feature using the most frequent port of embarkation

# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# Converting categorical features to numeric

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)

train_df.head()

Deal with missing fare information for my test
# In[ ]:


test_df.Fare.fillna(test_df.Fare.dropna().median(), inplace=True)
test_df.head()


# We can now create FareBand

# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Convert the Fare feature to ordinal values based on the FareBand

# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[ ]:


test_df.head(10)


# #  Solve

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
acc_log = logreg.score(X_train,Y_train)
acc_log


# In[ ]:


corr_coef= pd.DataFrame(train_df.columns.delete(0),columns=['Feature'])
corr_coef['Correlation'] = pd.Series(logreg.coef_[0])
corr_coef.sort_values(by='Correlation',ascending=False)


# In[ ]:


# Support vector machines
svc = SVC()
svc.fit(X_train,Y_train)
acc_svc = svc.score(X_train,Y_train)
acc_svc


# In[ ]:


# k nearest neighbors - primitive lazy learning technique
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
acc_knn = knn.score(X_train,Y_train)
acc_knn


# In[ ]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = gaussian.score(X_train, Y_train)
acc_gaussian


# In[ ]:


# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
acc_perceptron = perceptron.score(X_train, Y_train)
acc_perceptron


# In[ ]:


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = sgd.score(X_train, Y_train)
acc_sgd


# In[ ]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = decision_tree.score(X_train, Y_train)
acc_decision_tree


# In[ ]:


#Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train,Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = random_forest.score(X_train,Y_train)
acc_random_forest


# In[ ]:


# Evaluate models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame.from_dict({'PassengerId': test_df['PassengerId'],'Survived' : Y_pred})
submission


# In[ ]:


# Submit
#submission.to_csv('../output/submission.csv',index=False)

