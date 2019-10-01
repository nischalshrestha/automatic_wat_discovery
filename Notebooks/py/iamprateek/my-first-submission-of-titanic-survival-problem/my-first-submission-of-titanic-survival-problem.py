#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
import re

#load the datasets in dataframes
input_dataset=pd.read_csv("../input/train.csv")
target_dataset=pd.read_csv("../input/test.csv")


# In[ ]:


input_dataset.head(30)


# In[ ]:


input_dataset.shape


# In[ ]:


input_dataset.describe()


# In[ ]:


survived_sex = input_dataset[input_dataset['Survived']==1]['Sex'].value_counts()
dead_sex = input_dataset[input_dataset['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))


# **Women are more likely to survive**

# In[ ]:


#correlate Fare with the survival.
figure = plt.figure(figsize=(15,8))
plt.hist([input_dataset[input_dataset['Survived']==1]['Fare'],input_dataset[input_dataset['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# **Passengers with cheaper ticket fares are more likely to die**

# In[ ]:


#combine the age, the fare and the survival on a single chart
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(input_dataset[input_dataset['Survived']==1]['Age'],input_dataset[input_dataset['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(input_dataset[input_dataset['Survived']==0]['Age'],input_dataset[input_dataset['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


# **the ticket fare correlates with the class**

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
input_dataset.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)


# In[ ]:


#how the embarkation site affects the survival
survived_embark = input_dataset[input_dataset['Survived']==1]['Embarked'].value_counts()
dead_embark = input_dataset[input_dataset['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=True, figsize=(15,8))


# **There seems to be no distinct correlation here.**

# In[ ]:


target_dataset.head()


# In[ ]:


target_dataset.Fare.fillna(7.925, inplace = True)


# In[ ]:


#define a print function that asserts whether or not a feature has been processed
def status(feature):
    print('Processing..',feature,': ok')


# In[ ]:


#merge the dataframes into one
def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    # reading test data
    test = pd.read_csv('../input/test.csv')
    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    # merging train data and test data for future feature engineering
    combined = train.append(test)
    # re-number the combined data set so there aren't duplicate indexes
    combined.reset_index(inplace=True)
    # reset_index() generates a new column that we don't want, so let's get rid of it
    combined.drop('index', inplace=True, axis=1)
    # the remaining columns need to be reindexed so we can access the first column at '0' instead of '1'
    #combined = combined.reindex(train.columns, axis=1)
    return combined


# In[ ]:


combined = get_combined_data()


# In[ ]:


combined.shape


# In[ ]:


# Replace missing values with "U0" in Cabin column
combined['Cabin'][combined.Cabin.isnull()] = 'U0'


# In[ ]:


#Extracting the passenger titles
def get_titles():
    global combined
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    # a map of more aggregated titles
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
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)


# In[ ]:


get_titles()


# In[ ]:


combined.head()


# In[ ]:


#processing the Age column
#Age variable is missing 177 values so we cannot simply remove it
grouped_train = combined.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()


# In[ ]:


grouped_median_train


# In[ ]:


grouped_median_test


# In[ ]:


#create a function that fills in the missing age in combined based on above different attributes
def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    def fillAges(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']
    
    combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    status('age')


# In[ ]:


process_age()


# In[ ]:


combined.info()


# In[ ]:


#create a function that drops the Name column 
#since we won't be using it anymore because we created a Title column
def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    status('names')


# In[ ]:


process_names()


# In[ ]:


combined.head()


# In[ ]:


combined.info()


# In[ ]:


#processing Fare column
def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)
    status('fare')


# In[ ]:


process_fares()


# In[ ]:


combined.info()


# In[ ]:


#processing Embarked
def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    status('embarked')


# In[ ]:


process_embarked()


# In[ ]:


combined.info()


# In[ ]:


#processing cabin column
def process_cabin():
    global combined
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined,cabin_dummies], axis=1)
    combined.drop('Cabin', axis=1, inplace=True)
    status('cabin')


# In[ ]:


process_cabin()


# In[ ]:


combined.info()


# In[ ]:


combined.head()


# In[ ]:


#processing sex column
def process_sex():
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    status('sex')


# In[ ]:


process_sex()


# In[ ]:


combined.info()


# In[ ]:


#processing Pclass
def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    status('pclass')


# In[ ]:


process_pclass()


# In[ ]:


combined.drop('PassengerId', inplace=True, axis=1)


# In[ ]:


combined.head()


# In[ ]:


def process_ticket():
    global combined
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    # Extracting dummy variables from tickets:
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)
    status('ticket')


# In[ ]:


process_ticket()


# In[ ]:


#Processing Family
def process_family():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)
    status('family')


# In[ ]:


process_family()


# In[ ]:


combined.info()


# In[ ]:


combined.shape


# In[ ]:


combined.head()


# In[ ]:


pd.isnull(combined).sum()


# In[ ]:


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# In[ ]:


def recover_train_test_target():
    global combined
    train0 = pd.read_csv('../input/train.csv')
    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]
    return train, test, targets


# In[ ]:


train, test, targets = recover_train_test_target()


# In[ ]:


pd.isnull(targets).sum()


# In[ ]:


train.Fare.loc[50:]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train,targets)
print(clf.feature_importances_)


# In[ ]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))


# **There is a great importance linked to Title_Mr, Age, Fare, and Sex
# There is also an important correlation with the Passenger_Id.**

# In[ ]:


#transform our train set and test set in a more compact datasets
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
train_reduced.shape


# In[ ]:


test_reduced = model.transform(test)
test_reduced.shape


# In[ ]:


#Hyperparameters tuning using Random Forest model
# turn run_gs to True if you want to run the gridsearch again.
run_gs = False
if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)
    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)
    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)


# In[ ]:


from sklearn.model_selection import cross_val_score
compute_score(model, train, targets, scoring='accuracy')


# In[ ]:


output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

