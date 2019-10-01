#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


# # I - Exploratory data analysis

# Import some useful libraries

# In[ ]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
pd.options.display.max_rows = 100


# Loading data

# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# Pandas allows us to statistically describe numerical features using the describe method.

# In[ ]:


data.describe()


# Age column has missing values. A solution is to replace the null values with the median age

# In[ ]:


data['Age'].fillna(data['Age'].median(), inplace=True)


# In[ ]:


data.describe()


# Draw some charts to understand more about the data

# In[ ]:


survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex, dead_sex])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(6, 4))


# The Sex variable seems to be a decisive feature. Women are more likely to survive
# 
# Let's now correlate the suvival with the age variable

# In[ ]:


figure = plt.figure(figsize=(6, 4))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# It seems that children age under 10 have high chance of survival

# Let's now focus on the Fare ticket

# In[ ]:


figure = plt.figure(figsize=(8, 4))
plt.hist([data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# Passengers with high fare ticket are likely to be survived

# Combine the age, the fare and the survival on a single chart

# In[ ]:


plt.figure(figsize=(8, 4))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'], data[data['Survived']==1]['Fare'], c='green', s=40)
ax.scatter(data[data['Survived']==0]['Age'], data[data['Survived']==0]['Fare'], c='red', s= 40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper righ', fontsize=15)


# Except for children, passengers with high fare ticket have high chance of survival

# The fare is correlated with the Pclass

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(6, 3), ax = ax)


# Let's now see how the embarkation site affects the survival

# In[ ]:


survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark, dead_embark])
df.index = ['Survived', 'Dead']
df.plot(kind='bar', stacked=True, figsize=(8,4))


# There seems to be no distinct correlation here

# # II - Feature Enginerring

# In[ ]:


# Function that asserts whether or not a feature has been processed
def status(feature):
    print('Processing %s :ok' %(feature))


# ## Loading data
# Load and combine train set and test set. Combined set will be tranning set for a model

# In[ ]:


def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')
    
    # extracting and then removing the targets from the traing data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    
    # Merging train data and test data for future engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined


# In[ ]:


combined = get_combined_data()


# In[ ]:


combined.shape


# In[ ]:


combined.head()


# ## Extracting the passenger titles

# In[ ]:


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


# ## Processing the ages
# 
# The are 177 values missing for Age. We need to fill the missing value

# In[ ]:


grouped = combined.groupby(['Sex', 'Pclass', 'Title'])
grouped.median()


# Look at the median age column and see how this value can be different based on the Sex, Pclass and Title put together.
# 
# For example:
# - If the passenger is female, from Pclass 1, and from royalty the median age is 39.
# - If the passenger is male, from Pclass 3, with a Mr title, the median age is 26.
# 
# Let's create a function that fills in the missing age in **combined** based on these different attributes.

# In[ ]:


def process_age():
    global combined
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')


# In[ ]:


process_age()


# In[ ]:


combined.info()


# There are still some missing value in Fare, Embarked, Cabin

# In[ ]:


def process_names():
    global combined
    
    combined.drop('Name', inplace=True, axis=1)
    title_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, title_dummies], axis=1)
    
    combined.drop('Title', axis=1, inplace=True)
    
    status('names')


# In[ ]:


process_names()


# ## Processing Fare

# In[ ]:


def process_fares():
    
    global combined
    
    combined.Fare.fillna(combined.Fare.mean(), inplace=True)
    
    status('fare')


# In[ ]:


process_fares()


# ## Processing Embarked
# Fill missing values with the most frequent Embarked value.

# In[ ]:


def process_embarked():
    global combined
    
    combined.Embarked.fillna('S', inplace=True)
    
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    
    status('embarked')


# In[ ]:


process_embarked()


# ## Processing Cabin
# Fill missing cabins with U (for Unknown)

# In[ ]:


def process_cabin():
    global combined
    
    combined.Cabin.fillna('U',inplace=True)
    
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    
    combined = pd.concat([combined, cabin_dummies], axis=1)
    
    combined.drop('Cabin', axis=1, inplace=True)
    
    status('cabin')


# In[ ]:


process_cabin()


# In[ ]:


combined.info()


# ## Processing Sex 

# In[ ]:


def process_sex():
    
    global combined
    
    combined['Sex'] = combined['Sex'].map({'male':1, 'female': 0})
    
    status('sex')


# In[ ]:


process_sex()


# ## Processing Pclass

# In[ ]:


def process_pclass():
    
    global combined
    
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    combined = pd.concat([combined, pclass_dummies], axis=1)
    
    combined.drop('Pclass', axis=1, inplace=True)
    
    status('pclass')


# In[ ]:


process_pclass()


# ## Processing Ticket

# In[ ]:


def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = list(map(lambda t : t.strip() , ticket))
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')


# In[ ]:


process_ticket()


# ## Processing Family

# In[ ]:


def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    
    status('family')


# In[ ]:


process_family()


# In[ ]:


combined.shape


# Now we have 68 features

# We need to scale all features

# In[ ]:


def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print('Features scaled sucessfully!')


# In[ ]:


scale_all_features()


# # III - Modeling

# We'll be using Random Forests. Random Froests has proven a great efficiency in Kaggle competitions.
# 
# For more details about why ensemble methods perform well, you can refer to these posts:
# - http://mlwave.com/kaggle-ensembling-guide/
# - http://www.overkillanalytics.net/more-is-always-better-the-power-of-simple-ensembles/
# 
# Steps:
# 1. Break the combined dataset to train set and test set
# 2. Use the train set to build a predictive model
# 3. Evaluate the model using the train set
# 4. Test the model using the test set

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score


# We use 5-fold cross validation with the Accuracy metric

# In[ ]:


def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# Now we need to separate training set and test set from the combined set

# In[ ]:


def recover_train_test_target():
    train0 = pd.read_csv('../input/train.csv')
    
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    return train,test,targets


# In[ ]:


train,test,targets = recover_train_test_target()


# ## Feature selection

# We select features from 68 features:
# 
# - This decreases redundancy among the data
# - This speeds up the training process
# - This reduces overfitting
# 
# Tree-based estimators can be used to compute feature importances

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)


# In[ ]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_


# In[ ]:


features.sort(['importance'], ascending=False)


# Now we transform the train set and test set in a more compact datasets.

# In[ ]:


model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape


# In[ ]:


test_new = model.transform(test)
test_new.shape


# ## Hyperparameters tuning
# 
# As mentioned in the beginning of the Modeling part, we will be using a Random Forest model.
# 
# Random Forest are quite handy. They do however come with some parameters to tweak in order to get an optimal model for the prediction task.
# 
# To learn more about Random Forests, you can refer to this link: https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

# In[ ]:


forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {'max_depth' : [4,5,6,7,8],
                  'n_estimators':[200,210,240,250],
                  'criterion':['gini', 'entropy']}
cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# Now we generate solution for sumission

# In[ ]:


output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('titanic_submission.csv',index=False)


# In[ ]:




