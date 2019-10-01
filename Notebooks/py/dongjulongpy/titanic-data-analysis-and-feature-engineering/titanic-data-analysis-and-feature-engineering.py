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


# importing algrebra and dataframe libraries 
import numpy as np
import pandas as np


# In[ ]:


# importing data analysis, graph libraries
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Reading train and test data and concat them.
# We are adding train and test data because a model can predict with same featues which we use train the model.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

titanic_df = train_df.append(test_df, ignore_index=True, sort=False)


# In[ ]:


titanic_df.head()


# # 1 - Step: Feature Engineering and Data Analysis
# ### We will analyze the data and try to generate some new features and clean some feature columns.

# In[ ]:


titanic_df.head()


# ## 1 - PassengerId:

# In[ ]:


# PassengerId is a irrelevant column with our dataset so will remove this column.

titanic_df.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


titanic_df.head(3)


# ## 2 - Pclass:

# In[ ]:


# checking whether it has missing values or not.
titanic_df['Pclass'].isnull().sum()


# In[ ]:


plt.figure(figsize=(12, 6)) # setting figure size.

sns.countplot(x='Pclass', data=titanic_df, hue='Survived')


# In[ ]:


# For Pclass column we will leave same for now maybe with other columns we can create some new features.


# ## 3 - Name:

# In[ ]:


titanic_df['Name'].isnull().sum()


# In[ ]:


titanic_df['Name'].head()


# In[ ]:


# For name column we will split the names and look the name titles.


# In[ ]:


def find_title(name):
    """
    This method takes a full name 
    and return the title of name
    
    """
    
    nameList = name.split()
    
    for i in nameList:
        if '.' in i:
            name = i[:-1]
    
    return name


# In[ ]:


# creating a new title column in titanic_df
titanic_df['Title'] = titanic_df['Name'].apply(find_title) 


# In[ ]:


titanic_df.head()


# In[ ]:


plt.figure(figsize=(16, 6)) # setting figure size.

sns.countplot(x='Title', data=titanic_df, hue='Survived')

plt.ylim(0, 500)


# In[ ]:


# In the graph we see that ['Mme', 'Ms', 'L', 'Lady', 'Sir', 'Mlle', 'Countess'] all survived.
# Also ['Don', 'Rev', 'Capt', 'Jonkheer'] all died.
# ['Mrs', 'Miss'] have similar ratio.
# So we will create 6 class for title:

# 1 - ['Mme', 'Ms', 'L', 'Lady', 'Sir', 'Mlle', 'Countess']
# 2 - ['Don', 'Rev', 'Capt', 'Jonkheer']
# 3 - ['Mrs', 'Miss']
# 4 - Master
# 5 - Mr
# 6 - Others


# In[ ]:


def title_class(title):
    
    if title in ['L', 'Lady', 'Sir', 'Countess', 'Mme', 'Mlle', 'Ms']:
        return 0
    elif title in ['Don', 'Rev', 'Capt', 'Jonkheer']:
        return 1
    elif title in ['Mrs', 'Miss']:
        return 2
    elif title in ['Master']:
        return 3
    elif title in ['Mr']:
        return 5
    else:
        return 6


# In[ ]:


titanic_df['Title'] = titanic_df['Title'].apply(title_class)


# In[ ]:


titanic_df.head()


# In[ ]:


# Now we don't need the name column anymore.
titanic_df.drop('Name', axis=1, inplace=True)


# ## 4 - Sex:

# In[ ]:


titanic_df['Sex'].isnull().sum()


# In[ ]:


def sex_column(sex):
    
    if sex == 'male':
        return 0
    else:
        return 1


# In[ ]:


titanic_df['Sex'] = titanic_df['Sex'].apply(sex_column)


# In[ ]:


titanic_df.head()


# ## 5 - Age:

# In[ ]:


titanic_df['Age'].isnull().sum()


# In[ ]:


# We have a lot of missing values. So first we have to fill these missing values.
# We can find relevant feature columns in age column and give some random age values for missing ages.


# In[ ]:


plt.figure(figsize=(16, 6)) # setting figure size.

sns.barplot(x='Pclass', y='Age', data=titanic_df)


# In[ ]:


titanic_df['Age'].describe()


# In[ ]:


titanic_df[titanic_df['Pclass'] == 1]['Age'].describe()


# In[ ]:


titanic_df[titanic_df['Pclass'] == 2]['Age'].describe()


# In[ ]:


titanic_df[titanic_df['Pclass'] == 3]['Age'].describe()


# In[ ]:


# Pclass is a option for predicting Age Column.


# In[ ]:


# We can scale the fare column and analyze relationship with age.


# In[ ]:


def fare_class(fare):
    return fare // 200


# In[ ]:


titanic_df['FareClass'] = titanic_df['Fare'].apply(fare_class)


# In[ ]:


titanic_df.head()


# In[ ]:


plt.figure(figsize=(16, 6)) # setting figure size.

sns.barplot(x='FareClass', y='Age', data=titanic_df)


# In[ ]:


# We can use both FareClass and Pclass to fill missing age values.
# FareClass 2 means Pclass 1 
# FareClass 1 means Pclass 2 
# FareClass 0 means Pclass 3 


# In[ ]:


import random

def fill_age(columns):
    
    age = columns[0]
    pclass = columns[1]
    fareclass = columns[2]
        
    if pd.isnull(age):
        pclass_mean = int(round(titanic_df[titanic_df['Pclass'] == pclass]['Age'].mean()))
        fareclass_mean = int(round(titanic_df[titanic_df['FareClass'] == fareclass]['Age'].mean()))

        pclass_std = int(round(titanic_df[titanic_df['Pclass'] == pclass]['Age'].std()))
        fareclass_std = int(round(titanic_df[titanic_df['FareClass'] == fareclass]['Age'].std()))

        age_max = int(round(((pclass_mean + fareclass_mean) + (pclass_std + fareclass_std))/2))
        age_min = int(round(((pclass_mean + fareclass_mean) - (pclass_std + fareclass_std))/2))

        random_age = random.randint(age_min, age_max)
        return random_age
    else:
        return age


# In[ ]:


titanic_df['Age'] = titanic_df[['Age', 'Pclass', 'FareClass']].apply(fill_age, axis=1)


# In[ ]:


titanic_df['Age'].isnull().sum()


# In[ ]:


titanic_df['Age'] = titanic_df['Age'].apply(int)


# In[ ]:


# We may create an age_class like 0-20, 21-40, 40-others


# In[ ]:


def age_class(age):
    
    if 0 <= age <= 20:
        return 0
    elif 20 < age <= 40:
        return 1
    elif 40 < age <= 60:
        return 2
    else:
        return 3


# In[ ]:


titanic_df['AgeClass'] = titanic_df['Age'].apply(age_class)


# In[ ]:


plt.figure(figsize=(16, 6)) # setting figure size.

sns.countplot(x='AgeClass', data=titanic_df, hue='Survived')


# In[ ]:


titanic_df.head()


# ## 6 - SibSp and Parch:

# In[ ]:


titanic_df['SibSp'].isnull().sum()


# In[ ]:


titanic_df['Parch'].isnull().sum()


# In[ ]:


# We can use these columns to create family size column.
# We can create a column which is passenger alone or not.


# In[ ]:


# We have to add 1 because we have to include passenger too.
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1


# In[ ]:


def alone(familysize):
    
    if familysize == 1:
        return 1
    else:
        return 0


# In[ ]:


titanic_df['Alone'] = titanic_df['FamilySize'].apply(alone)


# In[ ]:


titanic_df.head()


# In[ ]:


sizes = titanic_df['FamilySize'].unique() # unique familysize values.
sizes


# In[ ]:


# We can classify the familysize as small medium and big.


# In[ ]:


(max(sizes) - min(sizes)) / 3


# In[ ]:


def family_class(familysize):
    
    if familysize <= 3:
        return 0
    elif 3 < familysize <= 7:
        return 1
    else:
        return 2


# In[ ]:


titanic_df['FamilyClass'] = titanic_df['FamilySize'].apply(family_class)


# In[ ]:


titanic_df.head()


# ## 7 - Ticket:

# In[ ]:


titanic_df['Ticket'].isnull().sum()


# In[ ]:


tickets = titanic_df['Ticket'].unique()


# In[ ]:


# We can split ticket as digit and non digit tickets.
# We may use the first letter inside tickets.


# In[ ]:


tickets_int = list()
tickets_str = list()

for i in range(len(tickets)):
    try:
        tickets_int.append(int(tickets[i]))
    except:
        tickets_str.append(tickets[i])


# In[ ]:


def ticket_class(ticket):
    
    try:
        int(ticket)
        return 0
    except:
        return 1


# In[ ]:


titanic_df['TicketClass'] = titanic_df['Ticket'].apply(ticket_class)


# In[ ]:


plt.figure(figsize=(16, 6)) # setting figure size.

sns.countplot(x='TicketClass', data=titanic_df, hue='Survived')


# In[ ]:


titanic_df.drop('Ticket', axis=1, inplace=True)


# In[ ]:


titanic_df.head()


# ## 8 - Fare:

# In[ ]:


# We have already created fareclass just we can make them integer values
titanic_df['Fare'].isnull().sum()


# In[ ]:


titanic_df[titanic_df['Fare'].isnull() == True]


# In[ ]:


pclass_mean = int(round(titanic_df[titanic_df['Pclass'] == 3]['Fare'].mean()))
pclass_std = int(round(titanic_df[titanic_df['Pclass'] == 3]['Fare'].std()))

fare_min = pclass_mean - pclass_std
fare_max = pclass_mean + pclass_std

random_fare = random.randint(fare_min, fare_max)

titanic_df.loc[titanic_df['Fare'].isnull() == True, 'Fare'] = random_fare


# In[ ]:


titanic_df['Fare'].isnull().sum()


# In[ ]:


titanic_df['FareClass'].isnull().sum()


# In[ ]:


titanic_df.loc[titanic_df['FareClass'].isnull() == True, 'FareClass'] = random_fare // 200


# In[ ]:


titanic_df['FareClass'].isnull().sum()


# In[ ]:


titanic_df['FareClass'] = titanic_df['FareClass'].apply(int)


# In[ ]:


titanic_df.head()


# ## 9 - Cabin:

# In[ ]:


titanic_df['Cabin'].isnull().sum()


# In[ ]:


# There are too many missing cabin values that's why I will not use cabin column.


# In[ ]:


titanic_df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


titanic_df.head()


# ## 10 - Embarked:

# In[ ]:


titanic_df['Embarked'].isnull().sum()


# In[ ]:


titanic_df[titanic_df['Embarked'].isnull() == True]


# In[ ]:


# I will give random Embarked for these NaN values with no reason.


# In[ ]:


titanic_df.loc[titanic_df['Embarked'].isnull() == True, 'Embarked'] = 'S'


# In[ ]:


def embarked(embarked):
    
    embarked_dict = {'S' : 0, 'C' : 1, 'Q' : 2}
    return embarked_dict[embarked]


# In[ ]:


titanic_df['Embarked'] = titanic_df['Embarked'].apply(embarked)


# In[ ]:


titanic_df['Embarked'].isnull().sum()


# In[ ]:


titanic_df.head()


# In[ ]:


# Now, We have a featured data set which occured from train and test dataset.
# We will separete train and test dataset and then we will train our algorithms.


# In[ ]:


train_featured = titanic_df.iloc[:891]
test_featued = titanic_df.iloc[891:]

train_featured_copy = train_featured
test_featued_copy = test_featued


# # 2 - Step: Train Test Algorithm and Predict:

# In[ ]:


# First split our train data as train and test data to see accuract values.
# Sklearn has train_split for dividing dataset and shuffle it.
from sklearn.model_selection import train_test_split

train_df = train_featured_copy

X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# Some algorithms from sklearn to classification. 
# Actually, I didn't use many algorithms because they are almost all will give similar result.
# Because, the important thing is create a good featured dataset. If we have a good classifiable dataset
# mostly all algorithm will give similar result.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_predictions = logmodel.predict(X_test)
print(classification_report(y_test, log_predictions))


# In[ ]:


svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print(classification_report(y_test, svm_predictions))


# In[ ]:


rdm = RandomForestClassifier()
rdm.fit(X_train, y_train)
rdm_predictions = rdm.predict(X_test)
print(classification_report(y_test, rdm_predictions))


# In[ ]:


param_grid = {'C' : [1, 10, 100, 1000, 10000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))


# In[ ]:


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
print(classification_report(y_test, gbc_pred))


# In[ ]:


import itertools
import time

start = time.time()

y = train_df['Survived']
columns = list(train_df.columns)
columns.remove('Survived')

for k in range(0, len(columns)-8):
    
    features = list(itertools.combinations(columns, k))

    score_max = 0
    score_index = list()

    for i in range(len(features)):

        features_extra = list(features[i])
        features_extra.append('Survived')
        
        X = train_df.drop(features_extra, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        #logmodel = LogisticRegression()
        #logmodel.fit(X_train, y_train)
        #log_predictions = logmodel.predict(X_test)

        #svm_model = SVC()
        #svm_model.fit(X_train, y_train)
        #svm_predictions = svm_model.predict(X_test)

        #rdm = RandomForestClassifier()
        #rdm.fit(X_train, y_train)
        #rdm_predictions = rdm.predict(X_test)

        gbc = GradientBoostingClassifier()
        gbc.fit(X_train, y_train)
        gbc_predictions = gbc.predict(X_test)

        scores = dict()

        #scores[accuracy_score(y_test, log_predictions)] = "Logistic Regression: "
        #scores[accuracy_score(y_test, svm_predictions)] = "SVM: "
        #scores[accuracy_score(y_test, rdm_predictions)] = "Random Forest Classifier: "
        scores[accuracy_score(y_test, gbc_predictions)] = "GradientBoosting Classifier: "

        sorted_scores = sorted(scores, reverse=True)

        if score_max < max(sorted_scores):
            score_index = list()
            score_index.append(max(score_max, max(sorted_scores)))
            score_index.append(i)
            score_max = max(score_max, max(sorted_scores))

        #print("------------------------------------Test", i, '---------------------------------')    
        #print()

        #for j in sorted_scores:
        #    print(scores[j], j)

        #print()

    print("------------------------------------ Extra Feature", k, '---------------------------------')
    print("Extra Feature Count: ",  k, "\nMax Score:", score_index[0], "\nFeatue Index: ", score_index[1])
    print()
    
end = time.time()
time_comb = end - start
print("Time for Combinations of Features: ", time_comb)


# In[ ]:


# After I got some scores around 79-80 from algorithms. I thought, maybe I have some extra features 
# and I used bruce force to try exery combinations of extra features. 
# Because it takes long time I didn't use some algorithms above.
# The best score 0.8507462686567164 with 3 extra feature. Lets find these extra features.


# In[ ]:


features = list(itertools.combinations(columns, 5))
extra_features = list(features[1369])
extra_features.append('Survived')


# In[ ]:


# Now, to be sure I will remove these features and get some scores again from algorithms.


# In[ ]:


train_df = train_df

X = train_df.drop(extra_features, axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_predictions = logmodel.predict(X_test)
print(classification_report(y_test, log_predictions))


# In[ ]:


svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
print(classification_report(y_test, svm_predictions))


# In[ ]:


rdm = RandomForestClassifier()
rdm.fit(X_train, y_train)
rdm_predictions = rdm.predict(X_test)
print(classification_report(y_test, rdm_predictions))


# In[ ]:


param_grid = {'C' : [1, 10, 100, 1000, 10000], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test, grid_predictions))


# In[ ]:


gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
print(classification_report(y_test, gbc_pred))


# # Final Step:

# In[ ]:


train_df = train_df

X = train_df.drop(extra_features, axis=1)
y = train_df['Survived']


# In[ ]:


gbc.fit(X, y)


# In[ ]:


X_test = test_featued.drop(extra_features, axis=1)

gbc_predictions = gbc.predict(X_test)
gbc_predictions = pd.DataFrame(gbc_predictions, columns=['Survived'])


# In[ ]:


gbc_predictions['Survived'] = gbc_predictions['Survived'].apply(int)


# In[ ]:


gbc_predictions.set_index(test_df['PassengerId'], inplace=True)
gbc_predictions.to_csv('submission.csv')


# In[ ]:


data_df = pd.read_csv('submission.csv')

