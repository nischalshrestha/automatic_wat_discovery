#!/usr/bin/env python
# coding: utf-8

# # Start
# Import libraries & load data<br>
# 
# The target is to predict Survived so train and test data should merge as one.

# In[487]:


from pprint import pprint

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


# In[488]:


train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')
df = train_df.append(test_df)


# # Recognize data

# In[489]:


df.head()


# In[490]:


df.info()


# In[491]:


df.describe()


# In[492]:


df.corr()


# # Should know which columns miss data
# There are Age, Cabin, Embarked and Fare columns have null data.

# In[493]:


null_series = df.isnull().any()
null_series = null_series[null_series == True]
null_series.index


# # Analyze by groupby data

# In[494]:


def extract_mean_survived_groupby_column(df, row_name, column_name):
    return df[[row_name, column_name]].groupby([row_name], as_index=True).mean()         .sort_values(by=column_name, ascending=False)


# In[495]:


extract_mean_survived_groupby_column(df, 'Pclass', 'Survived')


# In[496]:


extract_mean_survived_groupby_column(df, 'Sex', 'Survived')


# In[497]:


extract_mean_survived_groupby_column(df, 'SibSp', 'Survived')


# In[498]:


extract_mean_survived_groupby_column(df, 'Parch', 'Survived')


# # Analyze by visualizing data

# In[499]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[500]:


g = sns.FacetGrid(df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
g.map(plt.hist, 'Age', alpha=.5, bins=20)
g.add_legend()


# In[501]:


grid = sns.FacetGrid(df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[502]:


grid = sns.FacetGrid(df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[503]:


grid = sns.FacetGrid(df, row='Pclass', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[504]:


grid = sns.FacetGrid(df, row='Pclass', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Embarked', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# # Drop unimportant data
# 
# Cabin misses data too much. Ticket hasn't enough reference to relate with Survived.

# In[505]:


df = df.drop(['Ticket', 'Cabin'], axis=1)


# # Creating new feature extracting from existing

# In[506]:


df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[507]:


pd.crosstab(df['Title'], df['Sex'])


# In[508]:


df.Title.value_counts()


# In[509]:


title_series = df.Title.value_counts()
title_series = title_series >= 61
title_majority_list = title_series[title_series == True].index.tolist()
title_minority_list = title_series[title_series == False].index.tolist()


# In[510]:


title_majority_df = df[df.Title.isin(title_majority_list)]
title_minority_df = df[df.Title.isin(title_minority_list)]


# In[511]:


fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.swarmplot(x='Survived', y="Age", hue='Title', data=title_majority_df, size=10, ax=ax, palette='muted')


# In[512]:


fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.swarmplot(x='Survived', y='Age', hue='Title', data=title_minority_df, size=10, ax=ax, palette=sns.color_palette("Set1", n_colors=len(title_minority_list), desat=1))


# In[513]:


title_dummy = pd.get_dummies(title_majority_df.Title)
tmp_df = df.drop(['Title'], axis=1)
tmp_df = pd.concat([tmp_df, title_dummy], axis=1)
tmp_df.corr()


# In[514]:


'''
Please refer here:
Mr. on wiki: https://en.wikipedia.org/wiki/Mr.
Miss on wiki: https://en.wikipedia.org/wiki/Miss
Ms on wiki: https://en.wikipedia.org/wiki/Ms.
'''
title_map = {
    'Capt': 'Mr',
    'Col': 'Mr',
    'Don': 'Mr',
    'Dona': 'Mrs',
    'Dr': 'Mr',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Major': 'Mr',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Ms': 'Miss',
    'Rev': 'Mr',
    'Sir': 'Mr',
    'Countess': 'Mrs'}
df.Title = df.Title.replace(title_map)
df.Title.value_counts()


# In[515]:


extract_mean_survived_groupby_column(df, 'Title', 'Survived')


# In[516]:


df = df.drop(columns=['Name'], axis=1)


# In[517]:


title_dummies = pd.get_dummies(df.Title, prefix='title')
df = pd.concat([df, title_dummies], axis=1)
df = df.drop(columns=['Title'], axis=1)


# # Completion
# 
# * Age
# * Fare
# * Embarked

# ## Completion Age

# In[518]:


grid = sns.FacetGrid(df, row='Embarked', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[519]:


grid = sns.FacetGrid(df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[520]:


pclass_dummies = pd.get_dummies(df.Pclass)
embarked_dummies = pd.get_dummies(df.Embarked)
sex_dummies = pd.get_dummies(df.Sex)
tmp_df = pd.concat([df, pclass_dummies, embarked_dummies, sex_dummies], axis=1)
tmp_df.corr()


# In[521]:


grid = sns.FacetGrid(df[df.Embarked == 'S'], row='Pclass', col='Sex', 
                     size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[522]:


# data of Embarked C is too few
grid = sns.FacetGrid(df[df.Embarked == 'C'], row='Pclass', col='Sex', 
                     size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[523]:


# data of Embarked Q is too few
grid = sns.FacetGrid(df[df.Embarked == 'Q'], row='Pclass', col='Sex', 
                     size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[524]:


for embark_index in df.Embarked.value_counts().index:
    the_embark_series = df.Embarked == embark_index
    for sex_index in df.Sex.value_counts().index:
        the_sex_series = df.Sex == sex_index
        median = df[the_embark_series & the_sex_series].Age.median()
        df.loc[the_embark_series & the_sex_series & df.Age.isnull(), 'Age'] = median
df.Age = df.Age.astype(int)


# In[525]:


age_intervals = (
    (-1, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (15, 18),
    (18, 21),
    (21, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, 60),
    (60, 65),
    (65, 80)
)
def apply_age_intervals(age):
    for index, age_interval in enumerate(age_intervals):
        left, right = age_interval
        if left < age <= right:
            return index
df['age_stage'] = df.Age.apply(apply_age_intervals).astype(int)


# In[526]:


df[['age_stage', 'Survived']].groupby(['age_stage'], as_index=False).mean()     .sort_values(by='age_stage', ascending=True)


# In[527]:


age_stage_dummies = pd.get_dummies(df.age_stage, prefix='age_stage')
df = pd.concat([df, age_stage_dummies], axis=1)
df = df.drop(columns=['Age', 'age_stage'], axis=1)


# ## Completion Fare

# In[528]:


for embark_index in df.Embarked.value_counts().index:
    the_embark_series = df.Embarked == embark_index
    for pclass_index in df.Pclass.value_counts().index:
        the_pclass_series = df.Pclass == pclass_index
        median = df[the_embark_series & the_pclass_series].Fare.median()
        df.loc[the_embark_series & the_pclass_series & df.Fare.isnull(), 'Fare'] = median


# ## Completion Embarked

# In[529]:


most_embarked = df.Embarked.value_counts().index[0]
df.Embarked.fillna(most_embarked, inplace=True)


# In[530]:


embarked_dummies = pd.get_dummies(df.Embarked, prefix='embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df = df.drop(columns=['Embarked'], axis=1)


# ## Create feature

# In[531]:


is_alone_series = (df.Parch == 0) & (df.SibSp == 0)
df['is_alone'] = is_alone_series.astype(int)


# In[532]:


sex_dummies = pd.get_dummies(df.Sex, prefix='sex')
df = pd.concat([df, sex_dummies], axis=1)
df = df.drop(columns=['Sex'], axis=1)


# # Model, predict and solve

# In[555]:


from sklearn.model_selection import cross_val_score


# In[546]:


train_df = df[df.Survived.notnull()]
test_df = df[df.Survived.isnull()]
train_df.Survived = train_df.Survived.astype(int)
test_df = test_df.drop(columns=['Survived'], axis=1)


# In[564]:


scores = []
the_range = list(range(10, 40, 5))
for n_estimators in the_range:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    #score = clf.score(train_df.drop(columns=['Survived'], axis=1), train_df.Survived)
    cv_scores = cross_val_score(clf,
                                train_df.drop(columns=['Survived'], axis=1), 
                                train_df.Survived,
                                cv=20)
    scores.append(cv_scores.mean())
    
plt.plot(the_range, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()

pprint(dict(zip(the_range, scores)))


# In[562]:


scores = []
the_range = list(range(15, 31))
for n_estimators in the_range:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    #score = clf.score(train_df.drop(columns=['Survived'], axis=1), train_df.Survived)
    cv_scores = cross_val_score(clf,
                                train_df.drop(columns=['Survived'], axis=1), 
                                train_df.Survived,
                                cv=10)
    scores.append(cv_scores.mean())
    
plt.plot(the_range, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()

pprint(dict(zip(the_range, scores)))


# In[548]:


test_df.info()


# In[565]:


clf = RandomForestClassifier()
clf = clf.fit(train_df.drop(columns=['Survived'], axis=1), 
              train_df.Survived)


# In[566]:


predict_result = clf.predict(test_df)


# In[577]:


test_df = pd.read_csv('../input/test.csv')
test_df['Survived'] = predict_result.astype(int)


# In[586]:


test_df[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# In[ ]:




