#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---
get_ipython().magic(u'matplotlib inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
pd.options.display.max_rows = 100


# In[ ]:


def status(feature):
    print('Processing', feature, ':OK')


# In[ ]:


def get_combined_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    targets = train.Survived # extracting and removing the targets from training data
    train.drop(['Survived'], 1, inplace=True)
    
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    return combined


# In[ ]:


combined = get_combined_data()


# In[ ]:


combined.shape


# **Extracing the passenger titles**

# In[ ]:


def get_titles():
    global combined
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {
        'Capt':    'Officer',
        'Col':     'Officer',
        'Major':   'Officer',
        'Jonkheer':'Royalty',
        'Don':     'Royalty',
        'Sir':     'Royalty',
        'Dr':      'Officer',
        'Rev':     'Officer',
        'the Countess':'Royalty',
        'Dona':    'Royalty',
        'Mme':'Mrs',
        'Mlle':'Miss',
        'Ms':'Mrs',
        'Mr':'Mr',
        'Mrs':'Mrs',
        'Miss':'Miss',
        'Master':'Master',
        'Lady':'Royalty'
    }
    combined['Title'] = combined.Title.map(Title_Dictionary)
    combined.drop('Name', 1, inplace=True)


# In[ ]:


get_titles()
combined.head(5)


# **Processing the ages**

# In[ ]:


grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()


# In[ ]:


def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
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
    combined.Age = combined.apply(lambda r: fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    status('age')


# In[ ]:


process_age()
combined.info()


# In[ ]:


def process_names():
    global combined
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    combined.drop('Title', axis=1, inplace=True)
    status('Name')


# In[ ]:


process_names()
combined.head(5)


# In[ ]:


def process_fare():
    global combined
    combined.Fare.fillna(combined.Fare.mean(), inplace=True)
    status('fare')


# In[ ]:


process_fare()


# In[ ]:


def process_embarked():
    global combined
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    status('Embarked')


# In[ ]:


process_embarked()


# In[ ]:


def process_cabin():
    global combined
    combined.Cabin.fillna('U', inplace=True)
    # mapping each 
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    # dummy encoding
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    combined.drop('Cabin', axis=1, inplace=True)
    status('Cabin')


# In[ ]:


process_cabin()
combined.head(5)


# In[ ]:


def process_sex():
    global combined
    combined['Sex'] = combined['Sex'].map({'male':0, 'female':1})
    status('sex')


# In[ ]:


process_sex()


# In[ ]:


def process_pclass():
    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix='Pclass')
    combined = pd.concat([combined, pclass_dummies], axis=1)
    combined.drop('Pclass', axis=1, inplace=True)
    status('pclass')


# In[ ]:


process_pclass()


# In[ ]:


def process_ticket():
    global combined
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = map(lambda t : t.strip(), ticket)
        # print(type(ticket))
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'
    # extracing dummy variables from tickets
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)
    status('Ticket')


# In[ ]:


process_ticket()
combined.head(5)


# In[ ]:


def process_family():
    global combined
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['BigFamily'] = combined['FamilySize'].map(lambda s : 1 if s > 4 else 0)
    status('family')


# In[ ]:


process_family()
combined.shape
combined.head(5)


# In[ ]:


def scale_all_features():
    global combined
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x : x/x.max(), axis=0)
    print('Features scaled successfully!')


# In[ ]:


scale_all_features()
combined.head(5)


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score


# In[ ]:


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)


# In[ ]:


def recover_train_test_target():
    global combined
    train0 = pd.read_csv('../input/train.csv')
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    return train, test, targets


# In[ ]:


train, test, targets = recover_train_test_target()


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)


# In[ ]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort(['importance'], ascending=False)


# In[ ]:


# do feature selection
model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape


# In[ ]:


test_new = model.transform(test)
test_new.shape


# In[ ]:


forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
    'max_depth' : [4,5,6,7,8],
    'n_estimators' : [200, 300, 400],
    'criterion' : ['gini', 'entropy']
}
cross_validation = StratifiedKFold(targets, n_folds=5)
grid_search = GridSearchCV(forest, param_grid=parameter_grid, cv=cross_validation)
grid_search.fit(train_new, targets)

print('Best score : {}'.format(grid_search.best_score_))
print('Best parameters : {}'.format(grid_search.best_params_))


# In[ ]:


pipeline = grid_search
output = pipeline.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId', 'Survived']].to_csv('output.csv', index=False)

