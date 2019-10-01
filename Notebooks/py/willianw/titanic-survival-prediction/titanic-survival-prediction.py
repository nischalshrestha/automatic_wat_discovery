#!/usr/bin/env python
# coding: utf-8

# ### To-do's
# * Cross validation mimic test set
# * Feature Engineering -> is mother? is father? has mother? has father? couple?
# * Remove outliers

# In[ ]:


from sklearn.model_selection import cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import f1_score, make_scorer, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Imputer
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np
import pandas as pd
import glob
import re


# In[ ]:


import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


ts = pd.read_csv('../input/train.csv')
vd = pd.read_csv('../input/test.csv')
vd['Survived'] = np.nan
df = pd.concat([ts, vd], sort=False)


# In[ ]:


df.describe(include=['int64', 'float64']).T


# In[ ]:


df.describe(include=['object', 'category']).T


# # Preprocessing

# In[ ]:


df['Train'] = df['Survived'].notna() * 1


# ### Sex

# In[ ]:


df['Sex'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1).astype('int')
sns.countplot('Sex', hue='Train', data=df);


# ### Age (without regression)

# In[ ]:


fig, axs = plt.subplots(1, 3, figsize=(20,5))
sns.distplot(df[df['Age'].notna()]['Age'], ax=axs[0])
axs[1].hist(df[df['Age'].notna()]['Age'], cumulative=True)
sns.violinplot('Survived', 'Age', data=df, inner='quartile', ax=axs[2])


# In[ ]:


def age_treatment(age):
    if age == age:
        if age < 2:
            return 0
        elif age < 10:
            return 1
        elif age < 25:
            return 2
        elif age < 40:
            return 3
        elif age < 60:
            return 4
        else:
            return 5
    return np.nan
df['Age_categorical'] = df['Age'].apply(age_treatment)
df['Age_isna'] = (df['Age'].isna()) * 1


# In[ ]:


df['Cabin'] = df['Cabin'].fillna('').str.replace(r'[^A-Z]', '').apply(lambda x: str(x)[0] if len(str(x)) > 0 else '').astype('category')
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip()).astype('category')
df['Family_size'] = df['Parch'] + df['SibSp']+1
df['IsAlone'] = (df['Family_size'] == 1) * 1
df['Name'] = df['Name'].str.replace(r",.*", "").astype('category')
fam = df.groupby(['Name', 'Family_size'], as_index=False, group_keys=False).agg({'Fare': sum, 'Survived': np.mean})
fam['per_capita'] = fam['Fare']/fam['Family_size']
df = pd.merge(df, fam, left_on=['Name', 'Family_size'], right_on=['Name', 'Family_size'], suffixes=['_ind', '_fam'])


# ### Title

# In[ ]:


f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='row', figsize=(25,16))
sns.boxplot('Title', 'Age_categorical', data=df, ax=ax1)
sns.boxplot('Title', 'Fare_ind', data=df, ax=ax2)
sns.boxplot('Title', 'SibSp', data=df, ax=ax3)
sns.boxplot('Title', 'Parch', data=df, ax=ax4)
ax1.set_title("Grouping Titles");


# In[ ]:


print("Let's apply the titles transformation!")
titles = {
    'Capt': 'Major',
    'Col':'Major',
    'Don': 'Rev',
    'Dona': 'Mrs',
    'Dr': 'Rev',
    'Jonkheer': 'Mrs',
    'Lady': 'Mrs',
    'Mlle': 'Miss',
    'Mme': 'Miss',
    'Ms': 'Mrs',
    'Sir': 'Major',
    'the Countess': 'Mrs',
}


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(25,5))
sns.countplot('Title', data=df, hue='Train', ax = axs[0])
axs[0].set_title("Then")
df['Title'] = df['Title'].apply(lambda x: titles[x] if x in titles.keys() else x).astype('category')
sns.countplot('Title', data=df, hue='Train', ax = axs[1])
axs[1].set_title("Now")
plt.show();


# ### Embarked

# In[ ]:


df['Embarked'] = df['Embarked'].fillna('S').astype('category')
fig, axs = plt.subplots(1, 3, figsize=(25, 4)) 
sns.barplot('Embarked', 'Survived_ind', data=df, ax=axs[0])
axs[0].set_title('Survival Rate by Embarked Spot')
sns.violinplot('Embarked', 'Age', data=df, ax=axs[1])
axs[1].set_title('Age by Embarked Spot')
sns.countplot('Embarked', hue='Title', data=df, ax=axs[2])
plt.show()


# ### Fare

# In[ ]:


df['Fare_ind'] = df['Fare_ind'].fillna(df['Fare_ind'].dropna().median()).apply(lambda x: np.log(x) if x > 0 else 0)
fig, axs = plt.subplots(1, 2, figsize=(25,5))

axs[0].hist(
    [df[(df['Survived_ind']==1)&(df['Train']==1)]['Fare_ind'], df[(df['Survived_ind']==0)&(df['Train']==1)]['Fare_ind']],
    label=["Survived", "Died"], alpha=1, stacked=True)
axs[0].set_title("Train")
axs[0].legend()

axs[1].hist(
    [df[(df['Train']==0)]['Fare_ind']],
    label=["Test"], alpha=1, stacked=True)
axs[1].set_title("Test")
axs[1].legend()
axs[1].set_ylim(top=axs[0].get_ylim()[1])
plt.show();


# ### Family Identification

# In[ ]:


df.sort_values(by='Name').head(10)


# In[ ]:


df.drop(['Train', 'Ticket', 'Name', 'Fare_fam', 'per_capita'], axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x=df['Cabin'], y=df['Survived_ind'], hue=df['Cabin'], capsize=0.2)
plt.title("Survivance Rate vs. Cabin")
plt.show();


# ### Age

# In[ ]:


age_pat = pd.get_dummies(df.drop(['Survived_ind'], axis=1))
has_age = age_pat[age_pat['Age'].notna()]

params = {
        'learning_rate': np.logspace(-3, 0, 10),
        'gamma': np.linspace(0.005, 5.0, 10),
        'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
        'min_child_weight': range(1, 10, 2),
        'reg_lambda': np.linspace(0.1, 1.0, 10),
        'scale_pos_weight': [df[df['Survived_ind']==0]['Survived_ind'].shape[0]/df[df['Survived_ind']==1]['Survived_ind'].shape[0]]
    }

age_regression = True       # Should we create a regression for Age? Else we impute with median
if age_regression:
    age = xgb.XGBClassifier(verbose=True)
    rscv = RandomizedSearchCV(age, params, n_jobs=5, cv=3, n_iter=20, scoring='r2', verbose=True)
    rscv.fit(has_age.drop(['Age'], axis=1), has_age['Age'])
    
else:
    age = df.groupby(by=['Sex', 'Title'], as_index=False, group_keys=False)['Age'].median()
    df = df.merge(age, left_on=['Sex', 'Title'], right_on=['Sex', 'Title'], suffixes=['', '_age'])
    df.loc[df['Age_age'].isna(), 'Age_age'] = df.loc[df['Age'].notna(), 'Age'].median()
    df.loc[df['Age'].isna(), 'Age'] = df.loc[df['Age'].isna(), 'Age_age']
    df.drop(['Age_age'], axis=1, inplace=True)
    display(df.head())


# In[ ]:


if age_regression:
    print("Mean  ", cross_val_score(DummyRegressor("mean"), has_age.drop(['Age'], axis=1), has_age['Age'], scoring='r2'))
    print("XGB   ", cross_val_score(rscv.best_estimator_, has_age.drop(['Age'], axis=1), has_age['Age'], scoring='r2'))
    df.loc[df['Age'].isna(), 'Age'] = rscv.best_estimator_.predict(age_pat[age_pat['Age'].isna()].drop(['Age'], axis=1))
    display(pd.DataFrame(rscv.cv_results_).sort_values(by='mean_test_score', ascending=False)).head(5)


# In[ ]:


if age_regression:
    n, score_ts, score_vd = learning_curve(rscv.best_estimator_, age_pat[age_pat['Age'].notna()].drop(['Age'], axis=1), age_pat[age_pat['Age'].notna()]['Age'], cv=5, scoring=make_scorer(r2_score));
    plt.figure(figsize=(20, 5))
    plt.plot(n, score_ts.mean(axis=1), label='Training')
    plt.plot(n, score_vd.mean(axis=1), label='Validation')
    plt.legend()
    plt.show();


# In[ ]:


ages = {0: 'baby', 1:'kid', 2:'teen', 3:'adult', 4:'adult+', 5:'senior'}
df['Age_categorical'] = df['Age_categorical'].map(ages).astype('category')


# # Analysis

# ## Correlation

# In[ ]:


df = pd.get_dummies(df)
corr = pd.DataFrame(df.corr().abs())
plt.figure(figsize=(25, 8))
sns.heatmap(corr.abs());


# ## Is using Survived_fam a good idea?
# * Maybe using Survived_fam is a bad idea :(
# * Maybe not :)

# In[ ]:


if False:
    df.drop('Survived_fam', axis=1, inplace=True)
else:
    df['Survived_fam'].fillna(df['Survived_fam'].mean(), inplace=True)


# ## What about babies?

# In[ ]:


df.loc[df['Age']<1, 'Survived_ind'].value_counts()


# # Training

# In[ ]:


target = df['Survived_ind'].isna()
ts = df[~target]
vd = df[target]


# In[ ]:


from sklearn.utils import shuffle
X, y = shuffle(ts.drop('Survived_ind', axis=1), ts['Survived_ind'])


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


def model_name(model):
    return str(model).split('(')[0]
model_search = {}
models = [xgb.XGBClassifier(), GaussianNB(), BernoulliNB(), MultinomialNB(), LogisticRegression(), Lasso(), GaussianProcessClassifier(),
         KNeighborsClassifier(), RandomForestClassifier(), MLPClassifier()]
for model in models:
    results = cross_val_score(model, X, y, cv=5)
    model_search[model_name(model)] = results
    print(model_name(model), model_search[model_name(model)], "\t {:.2f}({:.2f})".format(results.mean(), results.std()))


# In[ ]:


search_xgb = False
if search_xgb:
    clf = xgb.XGBClassifier()
    rscv = RandomizedSearchCV(clf, params, n_jobs=-1, cv=5, n_iter=50, scoring='accuracy', verbose=True, return_train_score=False)
    rscv.fit(X, y);

    results = pd.DataFrame(rscv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    results = results[['mean_test_score','std_test_score']]
    model = rscv.best_estimator_
    results.head()
else:
    model = LogisticRegression()
    model.fit(X, y)


# In[ ]:


cross_val_score(model, X, y, cv=10)


# In[ ]:


vd['Survived'] = model.predict(vd.drop(['Survived_ind'], axis=1)).astype(int)


# In[ ]:


vd[['PassengerId','Survived']].to_csv('output.csv', index=False)


# In[ ]:


vd[vd['PassengerId'] == 1284]['Survived']


# In[ ]:




