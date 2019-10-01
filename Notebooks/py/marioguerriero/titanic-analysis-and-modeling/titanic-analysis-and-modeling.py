#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries and dataset and defining utility functions

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import re, math

from sklearn import svm, tree
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

def turn_dummy(df, prop):
    dummies = pd.get_dummies(df[prop], prefix=prop)
    df.drop(prop, axis=1, inplace=True)
    return pd.concat([df, dummies], axis=1)


# In[11]:


# Read datasets
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
test_passenger_id = test_df['PassengerId']
train_df.head()


# In[12]:


test_df['Survived'] = np.nan
dataset = train_df.append(test_df)
dataset = dataset.drop(['PassengerId', 'Name', 'Parch', 'Ticket', 'Cabin'], axis=1)

for col in dataset.columns.tolist():
    if dataset[col].dtype == 'object':
        dataset = turn_dummy(dataset, col)


# In[3]:


def show_cor(df, x, y):
    print(df[[x, y]].groupby([x], as_index=False).mean())
    
def make_bins(df, feature, n_bins):
    """
    In place creation of bins
    """
    bin_label = feature + '_Bin'
    df[bin_label] = pd.cut(df[feature], n_bins)
    label = LabelEncoder()
    df[bin_label] = label.fit_transform(df[bin_label])
    
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
        
def plot_cor(x, y, labels=None, title=None, xlabel=None, ylabel=None):
    _,ax = plt.subplots()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    rect = ax.bar(x, y, align='center', tick_label=labels)
    ax.set_title(title)
    
def parameters_grid_search(classifier, params, x, y, cv=10):
    """
    Grid Search to find best parameters for a certain classifier whose
    performances are evaluated using cross-validation
    """
    gs = GridSearchCV(classifier(), params, cv=cv)
    gs.fit(x, y)    
    return (gs.cv_results_['mean_test_score'].mean(), gs.best_estimator_, gs.best_params_)


# # Exploratory Analysis
# Let's examine the correlation between each feature and the output label and let's see if we can extract some additional information from the existing data.

# ## Check for missing values
# Number of missing attributes per class:

# In[4]:


train_df.isnull().sum()


# ## PClass
# Not much to do here. Just check percentage of survived people per class.

# In[5]:


tmp = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

_,ax = plt.subplots()
rect = ax.bar(tmp['Pclass'], tmp['Survived'], align='center', tick_label=[1,2,3])
ax.set_title('% of survival per class')


# ## Gender
# Turn gender information into a numeric value. And see its correlation with the output label.

# In[8]:


train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female':1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female':1})


# In[9]:


# Plot correlation
tmp = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
plot_cor(tmp['Sex'], tmp['Survived'], ['Male', 'Female'], '% of survival per gender')


# ## Age
# Deal with missing values as there are 177 of them.

# In[10]:


train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)


# Divide age values into bin a check correlations between bins and the output label.

# In[11]:


# Build age ranges
n_bins = 5
make_bins(train_df, 'Age', n_bins)
make_bins(test_df, 'Age', n_bins)
tmp = train_df[['Age_Bin', 'Survived']].groupby(['Age_Bin'], as_index=False).mean()
plot_cor(tmp['Age_Bin'], tmp['Survived'], title='% of survival per age bin', xlabel='Age range')


# ## Name/Title
# Do some feature engineering on the name, trying to extract nobiliar titles. People with higher nobiliar titles are usually richer and richer people had more chances of survival as seen from the survival percentage per ticket class.

# In[12]:


def title_engineering(df):
    df['Title'] = df['Name'].map(lambda x: re.compile(', ([a-zA-Z]*).').findall(x)[0])
    df['Title'] = df['Title'].apply(lambda x: 'Woman' if x in ['Mrs', 'Ms','Mlle', 'Mme', 'Miss'] else x)
    df['Title'] = df['Title'].apply(lambda x: 'Man' if x == 'Mr' else x)
    df['Title'] = df['Title'].apply(lambda x: 'Noble_Man' if x in ['Dr', 'Master', 'Rev', 'Jonkheer', 'Capt', 'Don', 'Major', 'Col', 'Sir'] else x)
    df['Title'] = df['Title'].apply(lambda x: 'Noble_Lady' if x in ['Dona', 'Lady', 'the Countess', 'the'] else x)
    
title_engineering(train_df)
title_engineering(test_df)

# Display correlation between titles and survival changes
tmp = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
plot_cor(tmp['Title'], tmp['Survived'], title='% of survival per nobiliar title', xlabel='Nobiliar Title Group')

# Now factorize nobiliar title for future numerical analysis
train_df['Title'] = pd.factorize(train_df['Title'])[0]
test_df['Title'] = pd.factorize(test_df['Title'])[0]


# ## Cabin
# Cabin position may have been a key factor for the survival of certain people. However the cabin feature has a lot of missing values wich may affect our model performances. Since 78% of the total cabin values are missing both in the training set and in the test set, it makes no sense in further analyzing it for now so I will just skip on this.

# ## Family
# People with bigger families may have had less chances to survive. Let's explore this possibility.

# In[13]:


def family_engineering(df):
    # Compute family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # Check if the person was alone
    df['Is_Alone'] = (df['FamilySize'] == 1).astype(int)
    n_bins = 3
    make_bins(df, 'FamilySize', n_bins)
    
def family_engineering1(df):
    # Check for chances of survival of entire families
    df['LastName'] = df['Name'].apply(lambda x: str.split(x, ",")[0])
    DEFAULT_SURVIVAL_VALUE = 0.5
    df['FamilySurvival'] = DEFAULT_SURVIVAL_VALUE

    for grp, grp_df in df[['Survived','Name', 'LastName', 'Fare', 'Ticket', 'PassengerId',
                               'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['LastName', 'Fare']):
        if (len(grp_df) != 1):
            # A Family group is found.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df.loc[df['PassengerId'] == passID, 'FamilySurvival'] = 1
                elif (smin==0.0):
                    df.loc[df['PassengerId'] == passID, 'FamilySurvival'] = 0

family_engineering(train_df)
family_engineering1(train_df)

family_engineering(test_df)

# Display correlation between titles and survival changes
tmp = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
plot_cor(tmp['FamilySize'], tmp['Survived'], title='% of survival per family size', xlabel='Family Size')

# Display correlation between family size and survival chances
tmp = train_df[['FamilySize_Bin', 'Survived']].groupby(['FamilySize_Bin'], as_index=False).mean()
plot_cor(tmp['FamilySize_Bin'], tmp['Survived'], title='% of survival per family size group', xlabel='Family Size Group')

# Display correlation between lone people and survival chances
tmp = train_df[['Is_Alone', 'Survived']].groupby(['Is_Alone'], as_index=False).mean()
plot_cor(tmp['Is_Alone'], tmp['Survived'], title='% of survival of lone people')

# Display correlation between family survival and survival chances
tmp = train_df[['FamilySurvival', 'Survived']].groupby(['FamilySurvival'], as_index=False).mean()
plot_cor(tmp['FamilySurvival'], tmp['Survived'], 
         labels=['Not Survived', 'Unknown', 'Survived'], xlabel='Family\'s fate', title='% of survival if entire family survived')


# The most meaningful correlation I can see here is the one regairding plain family size groups. In fact, smaller families are more likely to survive.

# # Modeling
# 
# ## Missing values

# In[31]:


def compute_missing_values(dataset):
    tmp_df = dataset.drop(['Survived'], axis=1)
    total = tmp_df.isnull().sum().sort_values(ascending=False)
    percent = (tmp_df.isnull().sum()/tmp_df.isnull().count() * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percentage'])
    return missing_data[missing_data['Percentage'] > 0]

compute_missing_values(dataset)


# In[33]:


dataset['Age'].fillna(0, inplace=True)
dataset['Fare'].fillna(0, inplace=True)
compute_missing_values(dataset)


# In[34]:


train_df = dataset[:len(train_df)].copy()
test_df = dataset[-len(test_df):].copy()


# ## Data Preparation
# Let's prepare data to send in order to make it possible to work on them. I decided to use only the following features: `['Pclass', 'Sex', 'Age_Bin', 'Title', 'FamilySize_Bin']`

# In[35]:


# Data for the training stage
X_train = train_df.drop(['Survived'], axis=1).as_matrix()#[['Pclass', 'Sex', 'Age_Bin', 'Title', 'FamilySize_Bin']].values
y = train_df[['Survived']].values.ravel()

# Data for the test stage
X_test = test_df.drop(['Survived'], axis=1).as_matrix()#[['Pclass', 'Sex', 'Age_Bin', 'Title', 'FamilySize_Bin']].values


# ## Stacking Classifier

# In[43]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier

import numpy as np

clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = RandomForestClassifier(n_estimators=20, random_state=1)
clf3 = GaussianNB()
xgboost = XGBClassifier()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, xgboost], 
                          meta_classifier=lr)

print('10-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_train, y, 
                                              cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# In[44]:


sclf.fit(X_train, y)


# # Produce output
# Here I will just use the stacking classifier I just built.

# In[17]:


# Predict values
prediction = sclf.predict(X_test)

# Build output dataframe
out_df = pd.DataFrame({
    'PassengerId': test_passenger_id,
    'Survived': prediction.astype(int)
})

# Write to CSV
out_df.to_csv('titanic-result.csv', index=False)


# 
