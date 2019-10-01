#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# "Titanic: Machine Learning from Disaster" is [Kaggle competition](https://www.kaggle.com/c/titanic),
# which is starting my adventure with machine learning.
# The goal is to predict if a passenger survived the sinking of the Titanic or not.
# 
# 
# Competition contain dataset presented in table:
# 
# Variable|Definition|Key
# :-------|:--------:|---
# Survival|Survival|0 = No<br>1 = Yes
# Pclass|Ticket class|1 = 1st<br>2 = 2nd<br>3 = 3rd
# Sex|Sex|
# Age|Age in years|
# SibSp|# of siblings / spouses<br>aboard the Titanic|
# Parch|# of parents / children<br>aboard the Titanic|
# Ticket|Ticket number|
# Fare|Passenger fare|
# Cabin|Cabin number|
# Embarked|Port of Embarkation|C = Cherbourg<br>Q = Queenstown<br>S = Southampton
# 
# 
# At the beggining, we need import libraries which I used later:

# In[2]:


# data manipulation 
import numpy as np
import pandas as pd

# data visualization 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
get_ipython().magic(u'matplotlib inline')

# preprocesing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

# algorithms
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb

# metrics
from sklearn.metrics import accuracy_score, classification_report

# optimization
from functools import partial
from hyperopt import hp, fmin, tpe

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Next I defined required functions to analyze, train and predict model:

# In[2]:


# Look for missing values in DataFrame
def missing_values(df):
    for column in df.columns:
        null_rows = df[column].isnull()
        if null_rows.any() == True:
            print('%s: %d nulls' % (column, null_rows.sum()))

def survived_plot(feature):
    all = df_train.groupby(('Survived', feature)).size().unstack()
    all.index = ['Survived', 'Dead']
    all.plot(kind='bar', stacked=True,
             title = 'Who survived?', figsize = (12, 6))

def good_feats(df):
    feats_from_df = set(df.select_dtypes([np.int, np.float]).columns.values)
    bad_feats = {'PassengerId', 'Survived', 'SibSp', 'Parch'}
    return list(feats_from_df - bad_feats)

def factorize(df, *columns):
    for column in columns:
        df[column + '_cat'] = pd.factorize(df[column])[0]

def plot_learning_curve(model, title, X, y, ylim=None, cv = None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    if ylim is not None:plt.ylim(*ylim)

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")
    return plt

def title_feat(*dfs):
    for df in dfs:
        title = df['Name'].apply(lambda x:
                                       x.split(',')[1].split('.')[0].strip())
        # title mapping
        title_map = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master': 2, 'Rev': 5,
                     'Dr':6, 'Col':8, 'Mlle':2, 'Major':8, 'Ms':1,
                     'the Countess':3, 'Don':1, 'Capt':8, 'Lady':4, 'Mme':4,
                     'Sir':4, 'Jonkheer':7, 'Dona':3}
        df['Title_cat'] = title.apply(lambda x: title_map[x])
    
def age_feat(*dfs):
    for df in dfs:
        age_title = df.groupby(['Title_cat'])['Age'].median().to_dict()
        df['Age'] = df.apply(lambda row: age_title[row['Title_cat']]
                                         if pd.isnull(row['Age'])
                                         else row['Age'], axis=1)
        df['Age_norm'] = pd.cut(df['Age'],[0,5,18,35,60,80])
        factorize(df, 'Age_norm')
    
def family_feat(*dfs):
    for df in dfs:
        df['Family'] = df['SibSp'] + df['Parch'] + 1
        
def model_train_predict(model, X, y, success_metric=accuracy_score):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return success_metric(y_test, y_pred)


# In[3]:


# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_test['Survived'] = np.nan


# # Collect some important information

# In[4]:


# simple info about train dataset
df_train.info()


# In[5]:


# get five first rows
df_train.head()


# In[6]:


# get five random rows
df_train.sample(5)


# In[7]:


# statistic information about training set
df_train.describe()


# In[8]:


# columns names
df_train.columns


# In[9]:


# columns correlation
df_train.corr()


# In[10]:


# visualization correlation
plt.rcParams['figure.figsize']=(15,7)
sns.heatmap(df_train.corr().abs(), annot=True, linewidths=.5, cmap="Blues");


# Two important things: 
#     - correlation between two features (should be removed from model)
#     - correlation between feature and target variable (should be added to model)
#     
# In this case we have correlated two features: Parch and SibSp, so we can deduce, that marrieds had children.
# 
# Next correlation, which is worth paying attention is correlation between target variable and feature Fare. I think, that it is main point to study. If someone was payed to much for, a ticket, that had better chance of survival.

# # How Titanic sank?
# 
# ![Sinking of the RMS Titanic animation](images/Sinking_of_the_RMS_Titanic_animation.gif)
# 
# https://en.wikipedia.org/wiki/April_1912#/media/File:Sinking_of_the_RMS_Titanic_animation.gif
# 
# # Where was the lifeboats?
# 
# ![Titanic_Boat_Deck_plan_with_lifeboats.png](images/Titanic_Boat_Deck_plan_with_lifeboats.png)
# 
# https://en.wikipedia.org/wiki/Lifeboats_of_the_RMS_Titanic#/media/File:Titanic_Boat_Deck_plan_with_lifeboats.png
# 
# I think it's very important questions, because depending on where someone had a cabin, they had a different chance of survival. Some cabins were sunk at the very beginning. Other cabins were far to the lifeboats.
# 
# # Visualization
# I used groupby() method to group survivors with feature, but I found another method in someone repository, so I compared them:

# In[11]:


# curiosity about function
def func():
    together = df_train.groupby(['Sex', 'Survived']).size()
    
def func2():
    survived = df_train[df_train.Survived == 1]['Sex'].value_counts()
    dead = df_train[df_train.Survived == 0]['Sex'].value_counts()

print('Calculation time for groupby():')
get_ipython().magic(u'time for i in range(10000): func()')
print('Calculation time for another method:')
get_ipython().magic(u'time for i in range(10000): func2()')


# As you can see, function with groupby() is three times faster than alternative. Why? The second function browses dataset twice, but groupby() in first function browse it only once.

# In[12]:


# who survived? per sex
survived_plot('Sex')


# In[13]:


# who survived? per pclass
survived_plot('Pclass')


# In[14]:


# survival possibility dependent on age
sns.set(style="darkgrid")

sns.FacetGrid(df_train, hue = 'Survived', size = 2.5,
              aspect = 5, palette = 'Blues') \
  .map(sns.kdeplot, 'Age', shade=True) \
  .set(xlim = (0, df_train['Age'].max())) \
  .add_legend();


# In[15]:


# how survival possibility depend on class 
sns.set(rc = {'figure.figsize':(11.7,8.27)})
sns.barplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = df_train);


# In[16]:


# dependency between Pclass and Sex - who had change of survived?
sns.pointplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = df_train);


# # Basic model
# 
# First I choose features from dataset. I search it with select_dtypes() method. I choose numerical variable 
# because classification model works only on numerical variables. Frequently objects cointains categorial variable, but we can't use it in this form - we must change it to numerical. We could do this for example by use factorization function.
# 
# ```feats_from_dataset = df.select_dtypes([np.int, np.float]).columns.values```
# 
# 
# Variable 'Survived' is our target so we mustn't put it to models's features. 'PassengerId' is id of passenger 
# on titanic. It isn't necessary for us.
# 
# ```bad_feats = ['PassengerId', 'Survived']```
# 
# ```good_feats = [ feats for feats in feats_from_dataset if feats not in bad_feats ]```

# In[17]:


df_train = df_train[pd.notnull(df_train['Embarked'])]

factorize(df_train, 'Sex', 'Embarked')
factorize(df_test, 'Sex', 'Embarked')
df_train.head(5)


# In[18]:


# we have two new columns with factorized variables: 'Sex_cat', 'Embarked_cat'
good_feats(df_train)


# In[19]:


# basic model
X = df_train[good_feats(df_train)].values
y = df_train['Survived']

model = DummyClassifier()
score = model_train_predict(model, X, y)
print("Score: %.2f" % score)


# In[20]:


plt = plot_learning_curve(model, "Learning Curves (Dummy Classifier)", X, y, ylim=(0.5, 1.0), cv=10, n_jobs=4)
plt.show()


# Red line shows result for train dataset. This line must be better than green line, because we learn on train dataset.
# However, we want that green line also have hight score.

# # Feature engineering

# In[21]:


title_feat(df_train, df_test)
age_feat(df_train, df_test)
family_feat(df_train, df_test)


# # Advanced model

# In[22]:


X = df_train[good_feats(df_train)].values
y = df_train['Survived']

models = [
    LogisticRegression(),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10),
    ExtraTreesClassifier(max_depth=20)
]

for model in models:
    print(str(model) + ": ")
    get_ipython().magic(u'time score = model_train_predict(model, X, y)')
    print(str(score) + "\n")
    plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.0), n_jobs=4)
    plt.show()


# In[23]:


for model in models:
    print(str(model) + ": ")
    get_ipython().magic(u"time cross_validate(model, X, y, scoring='accuracy', cv=3)")
    print(str(score) + "\n")
    plt = plot_learning_curve(model, "Learning Curves", X, y, ylim=(0.5, 1.0), cv=3, n_jobs=4)
    plt.show()


# Hmmmm... Logistic Regresion is the best. The rest have overfitting.

# In[24]:


# Feature importances
# graphs show feature importances

models = [
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10),
    ExtraTreesClassifier(max_depth=20)
]

for model in models:
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title('Feature importances: ' + str(model).split('(')[0])
    plt.bar(range(X.shape[1]), model.feature_importances_[indices],
           color = 'g', align = 'center')
    plt.xticks(range(X.shape[1]), [ good_feats(df_train)[x] for x in indices])
    plt.xticks(rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


# # xgboost

# In[25]:


model = xgb.XGBClassifier()
model.fit(X, y)

y_pred = model.predict(X)
score = accuracy_score(y, y_pred)
print("Score: %.2f" % score)


# In[26]:


X = df_train[good_feats(df_train)].values
y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def compute(params):
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    #print("Score: %.2f" % score)
    #print(params)
    return (1 - score)

space = {
        'max_depth':  hp.choice('max_depth', range(4,6)),
        'min_child_weight': hp.uniform('min_child_weight', 0, 10),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05)
    }

best = fmin(compute, space, algo=tpe.suggest, max_evals=250)
print(best)


# In[27]:


X_train = df_train[good_feats(df_train)].values
y_train = df_train['Survived']
X_test = df_test[good_feats(df_test)].values

model = xgb.XGBClassifier(**best)
model.fit(X_train, y_train)


# In[28]:


print(classification_report(y, 
                            y_pred, 
                            target_names=['Not Survived', 'Survived']))

