#!/usr/bin/env python
# coding: utf-8

# ### Titanic: Machine Learning from Disaster
# - This Titanic dataset is a classic Machine Learning tutorial.
# - I will perform data cleaning, EDA, feature engineering and use classifcation models to predict survivors on the Titanic.
# - Analysis of this dataset was done during a bootcamp. Improved my original score from 0.65 before bootcamp to  0.79 during bootcamp.
# - Credit to [Geoffrey Wong](https://www.kaggle.com/csw4192) for helping me get over the damn hump that is 0.80 prediction score with his 'Survivor Confidence'.

# In[ ]:


# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print(train.info())
print('#'*30)
print(test.info())


# - For train and test set, majority of missing data are in 'Age' and 'Cabin'. 
# -  'Pclass' and 'Survived' are ordinal features. 'Age' and 'Fare' are continous features. 'Ticket' and 'Cabin' contains alphanumeric values. 'PassengerId' is an index.

# In[ ]:


# variable for test set 'PassengerId' needed for submission
passengerId = test['PassengerId']

# combine train and test set
titanic = train.append(test, ignore_index=True, sort=False )

# indexes for train and test set for modeling
train_idx = len(train)
test_idx = len(titanic) - len(test)


# #### EDA

# In[ ]:


# stats summary
train.describe()


# In[ ]:


# for correlation heatmap reassign female: 1 and male: 0
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})

# heatmap correlation to 'Survived'
corr = train.corr()
idx = corr.abs().sort_values(by='Survived', ascending=False).index
corr_idx = train.loc[:,idx]
train_corr = corr_idx.corr()
mask = np.zeros_like(train_corr)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(8,4))
sns.heatmap(train_corr, mask=mask, annot=True, cmap='seismic')


# - The highest numerical correlation is 'Fare,' one can assume that a passenger who bought a more expensive ticket such first class ticket can get to a life boat quicker then a passenger who bought a cheaper ticket, as first class was above deck. 
# - Categorical 'Pclass' has a negative correlation meaning that depending on which 'Pclass' a passenger is in, that passenger has a less chance of survival.
# - Logically 'Sex' is highest correlated as 'Women and Children first' code have priority evacuation to emergency lifeboats.
# - Surprisingly 'Age' correlation to survival is quite low.

# In[ ]:


plt.figure(figsize=(4,4))
train['Survived'].value_counts().plot.pie(autopct= '%1.1f%%', cmap='Pastel1')


# In[ ]:


train['Sex'] = train['Sex'].map({1:'female', 0:'male'})
titanic.groupby(['Pclass', 'Sex'])['Survived'].mean()


# - In each class, females had the highest percentage of survival.
# - 50% of female passengers in third class survived the lowest amount for females.
# - 36% of male passengers survived in first class the highest amount for males.
# - A total of 38.4% passengers survived from the dataset.

# In[ ]:


# countplot for 'Survived'
sns.catplot(x='Survived', hue='Sex', data=titanic, col='Pclass', kind='count', palette='seismic', height=4)


# In[ ]:


# swarmplot for 'Age'
sns.catplot(x='Survived', y='Age', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')


# In[ ]:


# swarmplot for 'Fare'
sns.catplot(x='Survived', y='Fare', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')


# In[ ]:


# swarmplot for 'SibSp'
sns.catplot(x='Survived', y='SibSp', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')


# In[ ]:


# swarmplot for 'Parch'
sns.catplot(x='Survived', y='Parch', hue='Sex', data=titanic, col='Pclass', kind='swarm', height=4, palette='seismic')


# - For passengers under 18 years old, it seems boys survived more than girls. 
# - For passengers over 18 years old, as expected females survived more than males because of "Women and Children first" code. 
# - In first and second class, passengers with family on board has a higher chance of surviving than without. Higher fares had more survivors than lower fares.
# - First class as expected has more survivors than other classes because first class is above deck and is in closer proximity to lifeboats. 
# - Females from third class survived more than males from first class.

# #### Data Preprocessing

# In[ ]:


# missing values
missing = titanic.isnull().sum().sort_values(ascending=False)
pct = (titanic.isnull().sum()/titanic.isnull().count()).sort_values(ascending=False)*100
total_missing = pd.concat([missing, pct], axis=1, keys=['total','percent'])
total_missing[total_missing['total']>0]


# - missing data from 'Survived' is for model prediction.

# In[ ]:


# 'Fare' NaN value
titanic[titanic['Fare'].isnull()]


# In[ ]:


# stats summary of 'Fare with 'Pclass and Embarked' groupby
titanic.groupby(['Pclass', 'Embarked'])['Fare'].describe()


# In[ ]:


# replace with median fare from 'Pclass' 3
titanic.iloc[1043,9] = 8.05


# In[ ]:


# 'Embarked' NaN value
titanic[titanic['Embarked'].isnull()]


# In[ ]:


# replace with 'C' as passengers' fare is closest to first class median price from 'Embarked' C
titanic.iloc[61,11] = 'C'
titanic.iloc[829, 11] = 'C'


# In[ ]:


# deeper look at 'Age' NaN values
#titanic[titanic['Age'].isnull()].sort_values(by='Name', ascending=True)


# - some passengers have $0.00 fare.
# - some passengers have same ticket.
# - some passengers have same surname but with different ticket.

# In[ ]:


# deeper look at $0.00 fare
titanic[titanic['Fare'] == 0]


# - Passengers with the ticket LINE are crew.
# - The rest all had a working relationship with the Titanic.

# In[ ]:


# median age for passengers with fare $0.00
work_median_age = titanic[(titanic['Fare'] == 0) & (titanic['Ticket'] != 'LINE')]['Age'].median()
work_median_age


# In[ ]:


# function to replace 'Age' NaN values for passengers with $0.00 fare
def workers_age(col):
    Age = col[0]
    Fare = col[1]
    if pd.isnull(Age):
        if Fare == 0:
            return work_median_age
    else:
        return Age


# In[ ]:


# apply function 
titanic['Age'] = titanic[['Age', 'Fare']].apply(workers_age, axis=1)


# In[ ]:


# to replace 'Age' NaN value for the remaining passengers, need to extract social class title from 'Name'
titanic['Title'] = titanic['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[ ]:


# new 'Title' column
titanic['Title'].value_counts()


# - A mix of English, French, royalty, and rare social titles aboard the Titanic

# In[ ]:


# title dictionary to combine similar and rare titles together
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }


# In[ ]:


# map title dictionary
titanic['Title'] = titanic['Title'].map(Title_Dictionary)


# In[ ]:


# groupby 'Pclass, Sex, and Title' to get age stats summary
titanic.groupby(['Pclass', 'Sex', 'Title'])['Age'].describe()


# - The title 'Masters' only represent boys under 15 years old.
# - Adults are age 15 and above.

# In[ ]:


# deeper look at under 18 passengers
u_18 = titanic[titanic['Age']<=18]
#u_18


# In[ ]:


# 'Master' median age
master_median_age = u_18[u_18['Title'] == 'Master']['Age'].median()


# In[ ]:


# function to replace 'Master' median age
def master_age(col):
    Age = col[0]
    Title = col[1]
    if pd.isnull(Age):
        if Title == 'Master':
            return master_median_age
    else:
        return Age


# In[ ]:


# apply 'master_age' function
titanic['Age'] = titanic[['Age', 'Title']].apply(master_age, axis=1)


# In[ ]:


# deeper look at under 18 female passengers
#u_18[u_18['Sex'] == 'female']


# - The title 'Miss' is to classify all single female passenger regardless of age. To create a group similar to 'Master' title, will classify girls under 15 as 'Missy' title.
# - By creating another 'Title' group, a more accurate age of passengers can be calculated to see if correlation has improved.
# - Filling in median age for female passengers should be more accurate based on title.
# 

# In[ ]:


# function to rename 'Miss' to 'Missy' to represent girls under 15
def girls_title(df):
    if df['Title'] == 'Miss':
        if df['Age'] < 15:
            return 'Missy'
        else:
            return df['Title']
    else:
        return df['Title']


# In[ ]:


# apply function 'girls_title'
titanic['Title'] = titanic[['Title', 'Age']].apply(girls_title, axis=1)

# groupby 'Pclass, Sex, and Title' updated stats summary for 'Age'
median_pclass_age = titanic.groupby(['Pclass', 'Sex', 'Title'])
#median_pclass_age['Age'].describe()


# In[ ]:


# lambda function to fill in remaining NaN values based on median age from 'median_pclass_age'
titanic['Age'] = median_pclass_age['Age'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


# fill in 'Cabin' NaN values with 'U' for unknown
titanic['Cabin'] = titanic['Cabin'].fillna('U')


# In[ ]:


# verify missing values, only 'Survived' should have missing values
titanic.isnull().sum().sort_values(ascending=False)


# #### Data Cleaning and Feature Engineering

# In[ ]:


# plots for continuous 'Age' and 'Fare'
fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.distplot(titanic['Age'], ax=axes[0])
sns.distplot(titanic['Fare'], ax=axes[1])


# - 'Age' is slightly positive skewed, will leave as is, however 'Fare' is positively skewed and not normal distribution. 
# - 'Fare' will be unskew and normalize if to use in prediction.
# - To unskew and normalize 'Fare', first find passengers that are traveling on same ticket, passengers that travel in a group their 'Fare' represents the total price paid for the ticket. 

# In[ ]:


# groupby ticket and count passengers traveling on same ticket
titanic['Same_Ticket'] = titanic.groupby('Ticket')['PassengerId'].transform('count')

# count the number of passengers traveling in a group
titanic[titanic['Same_Ticket'] >1]['Same_Ticket'].count()


# - There 596 passengers that are traveling in groups.
# - To find individual ticket price, divide 'Fare' by 'Same_Ticket'

# In[ ]:


# divide 'Fare' by 'Same_Ticket'
titanic['Fare'] = titanic['Fare'] / titanic['Same_Ticket']

# np.log1p 'Fare' to normalize
titanic['Fare_log1p'] = np.log1p(titanic['Fare'])

# updated distribution plots for fare
fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.distplot(titanic['Fare'], ax=axes[0])
sns.distplot(titanic['Fare_log1p'], ax=axes[1])


# In[ ]:


# new 'Family' to represent passengers traveling with family or not
titanic['Family'] = titanic['SibSp'] + titanic['Parch']

# new 'Family_size' to represent total number of family members, if equal 1, passenger is traveling alone or with non family group
titanic['Family_size'] = titanic['Parch'] + titanic['SibSp'] + 1
#titanic['Family_size'].value_counts()


# In[ ]:


# deeper look at passengers with no family
no_family = titanic[(titanic['SibSp'] == 0) & (titanic['Parch'] ==0)]

# groupby to count number of passengers with same ticket and no family members
no_family['Friends_group'] = no_family.groupby('Ticket')['PassengerId'].transform('count')

# add 'Family' and 'Friends_group' to get group size
no_family['Group_size'] = no_family['Family'] + no_family['Friends_group']


# In[ ]:


# update titanic dataset with 'no_family'
nf = no_family[['PassengerId', 'Group_size']]

# create 'Group_size' from 'Family_size'
titanic['Group_size'] = titanic['Family_size']

# update titanic with 'no_family' data
new_df = titanic[['PassengerId', 'Group_size']].set_index('PassengerId')
new_df.update(no_family.set_index('PassengerId'))
titanic['Group_size'] = new_df.values
titanic['Group_size'] = titanic['Group_size'].astype(int)


# In[ ]:


# clean 'Ticket' by extracting letters and converting digit only tickets to 'xxx'
tickets = titanic['Ticket'].apply(lambda t: t.split('.')[0].split()[0].replace('/','').replace('.',''))

# convert to list
tickets = tickets.tolist()


# In[ ]:


# function to convert digit only tickets to 'xxx'
def ticket_digits(t):
    v = []
    for i in t:
        if i.isnumeric():
            i == 'xxx'
            v.append(i)
        else:
            v.append(i)
    return v


# In[ ]:


# call 'ticket_digits' function
tickets = ticket_digits(tickets)

# assign to titanic dataset
titanic['Ticks'] = pd.DataFrame(tickets)

# number of clean tickets 
ticket_count = dict(titanic['Ticks'].value_counts())
titanic['Ticket_count'] = titanic['Ticks'].apply(lambda t: ticket_count[t])


# In[ ]:


# extract surnames from 'Name'
titanic['Surname'] = titanic['Name'].apply(lambda x: x.split(',')[0].strip())

# create 'SurnameId' to group same surname
titanic['SurnameId'] = titanic.groupby('Surname').ngroup().add(1)

# groupby 'Ticket' and 'Surname' to represent groups with same ticket or family
titanic['GroupId'] = titanic.groupby(['Ticket', 'Surname']).ngroup().add(1)


# - Some passengers incorrectly grouped, such as same family but bought different ticket.

# In[ ]:


# extract 'Cabin' letters to group
titanic['Cabin_group'] = titanic['Cabin'].apply(lambda x: x[0])


# #### Calculating group survival confidence 

# In[ ]:


# separate dataframe to calculate confidence
group_survival = titanic[['Pclass', 'Survived', 'Surname', 'SurnameId', 'Group_size', 'GroupId', 'Family_size', 'Ticket']]

# sum the number of survivors in a group
group_survival['group_survived'] = group_survival.groupby('GroupId')['Survived'].transform('sum')

# adjust the number of survivors in a group
group_survival['adj_survived'] = group_survival['group_survived'] - group_survival['Survived'].apply(lambda x: 1 if x == 1 else 0)

# sum the number of dead in a group
group_survival['group_dead'] = group_survival.groupby('GroupId')['Survived'].transform('count') - group_survival.groupby('GroupId')['Survived'].transform('sum')

# adjust the number of dead in a group
group_survival['adj_dead'] = group_survival['group_dead'] - group_survival['Survived'].apply(lambda x: 1 if x == 0 else 0)

# confidence of survival on single group of passengers
no_data = (group_survival['Group_size'] - group_survival['adj_survived'] - group_survival['adj_dead'])/(group_survival['Group_size'])

# calculate confidence
confidence = 1 - no_data
group_survival['confidence'] = confidence * ((1/group_survival['Group_size']) * (group_survival['adj_survived'] - group_survival['adj_dead']))

# assign back to titanic
titanic['confidence'] = group_survival['confidence']


# In[ ]:


# plot for 'Ticks'
plt.figure(figsize=(10,4))
sns.barplot(x= 'Ticks', y='Survived', data=titanic[titanic['Ticket_count']>10])
plt.axhline(y = np.mean(titanic.groupby('Ticks')['Survived'].mean()), linestyle='-.')


# In[ ]:


# plots for 'Family_size', 'Group_size', and 'Cabin_group'
fig, axes = plt.subplots(1,3, figsize=(16,4))
sns.barplot(x='Family_size', y='Survived', data=titanic, ax=axes[0])
axes[0].axhline(y=np.mean(titanic.groupby('Family_size')['Survived'].mean()), linestyle='-.')
sns.barplot(x='Group_size', y='Survived', data=titanic, ax=axes[1])
axes[1].axhline(y=np.mean(titanic.groupby('Group_size')['Survived'].mean()), linestyle='-.')
sns.barplot(x='Cabin_group', y='Survived', data=titanic, ax=axes[2])
axes[2].axhline(y=np.mean(titanic.groupby('Cabin_group')['Survived'].mean()), linestyle='-.')


# In[ ]:


# add column for 'Kid'
titanic['Kid'] = (titanic['Age'] < 15).astype(int)


# In[ ]:


# function to categorize 'Family_size'
def family_2_cat(df):
    if df <= 2:
        return 'single'
    elif (df > 2) & (df < 5):
        return 'small'
    elif df >= 5:
        return 'large'     


# In[ ]:


# apply function on 'Family_size'
titanic['Family_cat'] = titanic['Family_size'].apply(family_2_cat)


# In[ ]:


# bin 'Age' to range
pd.cut(titanic['Age'], 5).value_counts()


# In[ ]:


# function to categorize 'Age'
def age_2_cat(df):
    if df < 15:
        return 'kid'
    elif (df >= 15) & (df <= 32):
        return 'young adult'
    elif (df > 32) & (df <= 64):
        return 'adult'
    elif (df > 64):
        return 'senior'


# In[ ]:


# apply function 'age_2_cat'
titanic['Age_range'] = titanic['Age'].apply(age_2_cat)


# In[ ]:


# bin 'Fare' to a range
titanic['Fare_range'] = pd.qcut(titanic['Fare'],3, labels=False)


# In[ ]:


# select best and worst survival chance from 'Ticks'
titanic['PC'] = (titanic['Ticks'] == 'PC').astype(int)
titanic['CA'] = (titanic['Ticks'] == 'CA').astype(int)

# select best and worst survival chance from 'Cabin_group'
titanic['D'] = (titanic['Cabin_group'] == 'D').astype(int)
titanic['U'] = (titanic['Cabin_group'] == 'U').astype(int)


# #### Feature Selection and Modeling

# In[ ]:


# Feature correlation heatmap sorted by most correlated to "Survived"
corr = titanic.corr()
idx = corr.abs().sort_values(by='Survived', ascending=False).index
train_corr_idx = titanic.loc[:, idx]
train_corr = train_corr_idx.corr()
mask = np.zeros_like(train_corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(20,10))
sns.heatmap(train_corr, mask=mask, annot =True, cmap = 'seismic')


# In[ ]:


# select 
features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Title','Group_size', 'CA', 'PC','Kid','confidence', 'Fare_log1p']

titanic_full = titanic[features]

# map female to 0, male to 1
titanic_full['Sex'] = titanic_full['Sex'].map({'female': 0, 'male': 1})

# get dummy variables
titanic_feats = pd.get_dummies(titanic_full)


# In[ ]:


# assign to train and test set
df_train = titanic_feats[:train_idx]
df_test = titanic_feats[test_idx:]

# assign for train test split
X = df_train
y = train['Survived']
test_X = df_test


# In[ ]:


# import necessary modeling libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# scale continuous data
scaler = MinMaxScaler()

# fit, tranform on X and transform on test_X
X[['Fare_log1p','Group_size']] = scaler.fit_transform(X[['Fare_log1p', 'Group_size']])
test_X[['Fare_log1p', 'Group_size']] = scaler.transform(test_X[['Fare_log1p','Group_size']])


# In[ ]:


# train split test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 8)


# In[ ]:


# models
rfc = RandomForestClassifier()
svc = SVC()
knn = KNeighborsClassifier()
gboost = GradientBoostingClassifier()
logreg = LogisticRegressionCV()

models = [rfc, svc, knn, gboost, logreg]


# In[ ]:


for model in models:
    print('cross validation of: {0}'.format(model.__class__))
    score = cross_val_score(model, x_train, y_train, cv= 5, scoring = 'accuracy')
    print('cv score: {0}'.format(np.mean(score)))
    print('#'*50)


# In[ ]:


# RFC
rfc = RandomForestClassifier(oob_score=True)

# fit
rfc.fit(x_train, y_train)

# oob_score_
print(rfc.oob_score_)

# model score
print(rfc.score(x_train,y_train))

# prediction on x_test
y_pred = rfc.predict(x_test)

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# accuracy score
print('model accuracy: ',accuracy_score(y_test,y_pred))

# train error by RMSE
print('train error rmse: ',np.sqrt(mean_squared_error(y_train, rfc.predict(x_train))))


# In[ ]:


# features of importance plot
feats = pd.DataFrame()
feats['feats'] = x_train.columns
feats['importance'] = rfc.feature_importances_
feats.sort_values(by='importance', ascending=True, inplace=True)
feats.set_index('feats', inplace=True)


# In[ ]:


feats.plot(kind='barh')


# In[ ]:


rfc_submit = pd.DataFrame({'PassengerId': passengerId, 'Survived': rfc.predict(test_X)})
rfc_submit.to_csv('rfc_submit.csv', index=False)


# - got 0.75119 score
# - now to optimize model

# In[ ]:


# Optimize RFC parameters with GridSearchCV
model = RandomForestClassifier()

# parameters 
parameters = {
    "n_estimators": [50,100,200,300,400,500],
    "max_depth": [i for i in range(2,8)], 
    "min_samples_leaf": [i for i in range(2,8)],
    "max_leaf_nodes": [i for i in range(6,12)],
    "bootstrap": [True],
    'oob_score': [True],
    'max_features': [1,2,3]
}

# GridSearchCV (kaggle notebook reason will comment out)
#grid = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', n_jobs=-1, verbose=1, cv=5)


# In[ ]:


# fit x_train, y_train, for 
# grid.fit(x_train,y_train)


# In[ ]:


# print('best estimator: ', grid.best_estimator_)
# print('best params: ', grid.best_params_)


# In[ ]:


best_estimator = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features=3, max_leaf_nodes=7,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf='deprecated', n_estimators=50,
            n_jobs=None, oob_score=True, random_state=None, verbose=0,
            warm_start=False)

best_params = {'bootstrap': True, 'max_depth': 6, 'max_features': 3, 'max_leaf_nodes': 7, 'min_samples_leaf': 2, 'n_estimators': 50, 'oob_score': True}
    


# In[ ]:


rfc_grid = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features=3, max_leaf_nodes=7,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf='deprecated', n_estimators=50,
            n_jobs=None, oob_score=True, random_state=None, verbose=0,
            warm_start=False)


# In[ ]:


warnings.filterwarnings('ignore')
rfc_grid.fit(x_train, y_train)
print('oob score: ', rfc_grid.oob_score_)
print('accuracy score on x_test: ',accuracy_score(y_test, rfc_grid.predict(x_test)))


# In[ ]:


# optimized RFC prediction
grid_prediction = pd.DataFrame({'PassengerId': passengerId, 'Survived': rfc_grid.predict(test_X)})
grid_prediction.to_csv('prediction.csv', index=False)


# - Optimized RFC prediction: 0.80382
# - Thoughts to improve Kaggle score: better model optimization, feature engineering on passengers traveling as groups,  married couple only,  and parents with children only.
# - Thanks for taking a look!

# In[ ]:




