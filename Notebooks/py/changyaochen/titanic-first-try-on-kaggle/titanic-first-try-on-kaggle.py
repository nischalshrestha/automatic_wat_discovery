#!/usr/bin/env python
# coding: utf-8

# ## Data inspection
# Let's get started! First of all, let me take a look at the structure of the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bokeh.plotting as bkp  # for nice plotting
import bokeh.charts as bkc  # for nice plotting
import bokeh.models as bkm  # for nice plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()


# - *Survived*
# 
# Keep in mind that our target (predicted value) is whether a passenger will survive. Therefore, the first question is that, how many people have survived?

# In[ ]:


df['Survived'].value_counts(dropna = False)


# This is a simple enough math so that I don't the computer to do it for me: out of the 891 total passengers shown in this dataset, 342 survived, and the survival rate is 0.384.  It also reveals that there is no missing entry in this column.  
# 
# Note that, there are multiple features for each passenger, and part (or all) of them can be used to build the model. Therefore, we might want to take a closer look at each of them. 
# 
# - *Pclass*
# 
# First, let's take a look how many passengers survived according to Pclass:

# In[ ]:


# Pclass 
survived_pclass = df.groupby('Pclass')['Survived'].value_counts().unstack()
survived_pclass['Rate'] = survived_pclass[1]/(survived_pclass[1] + survived_pclass[0])
survived_pclass


# How does it look graphically?

# In[ ]:


bkp.output_notebook()
bar1 = bkc.Bar(df, values = 'Survived', label = 'Pclass', agg = 'count',
            tools='pan,box_zoom,reset,resize,save,hover', 
               stack=bkc.attributes.cat(columns='Survived', sort=False), 
            legend='top_left', plot_width=600, plot_height=300)
hover = bar1.select(dict(type = bkm.HoverTool))
hover.tooltips = dict([("Num", "@height{int}")])
bar1.yaxis.axis_label = 'Number of passengers'
bkp.show(bar1)


# Apparently, passengers from higher classes are more likely to survive, both in terms of number and percentage. Furthermore, the survival rates for Pclass 1 and 3 are all quite different from 0.5, hence, 'quite pure'. In another word, during prediction, if I see a passenger is from Pclass 1(3), I would likely to bet he/she will (not) survive. We should take this into account into our future models.

# - *Name*
# 
# The next column in the data is 'Name'. What information that we can possibly get from them? Humm.. how about the titles? 

# In[ ]:


# Name
import re
title = df['Name'].map(lambda x: re.split('[,.]', x)[1].strip())
df['Title'] = title
survived_title = df['Survived'].groupby(df['Title']).value_counts().unstack()
survived_title.fillna(0, inplace=True)
survived_title['Rate'] = survived_title[1]/survived_title.sum(axis=1)
survived_title.sort_values(by='Rate', ascending=False, inplace=True)
survived_title


# - *Gender*
# 
# Next let's take a look at the survivors' age distribution, grouped by gender.

# In[ ]:


# gender (or sex)
survived_sex = df.groupby('Sex')['Survived'].value_counts().unstack()
survived_sex['Rate'] = survived_sex[1]/(survived_sex.sum(axis=1))
survived_sex


# Apparently, ladies are much more likely to survive, than the gentlemen. 

# - *Age*
# 
# Next up, let's take a look at how the age will affect one's chance to survive.. Note that there are many missing values in Age entries.

# In[ ]:


# age histogram of survivors
survived_age = df[['Survived', 'Age', 'Sex']].copy()
survived_age['Survived'] = survived_age['Survived'].astype(int)
print('Total number of NAs in Age: {}'.format(survived_age['Age'].isnull().sum()))
survived_age.dropna(inplace=True)
hist1 = bkc.Histogram(survived_age, values = 'Age', color = 'Sex', bins = 50,
                     plot_width=600, plot_height=300)
bkp.show(hist1)


# - *SibSp and Parch* 
# 
# How the family size will affect the survival?

# In[ ]:


# SibSp and Parch
survived_sibsp = df['Survived'].groupby(df['SibSp']).value_counts().unstack()
survived_sibsp.fillna(0, inplace=True)
survived_sibsp['Rate'] = survived_sibsp[1]/survived_sibsp.sum(axis=1)
survived_sibsp.sort_values(by='Rate', ascending=False, inplace=True)
print(survived_sibsp)
print('Total number of NAs in SibSp: {}'.format(df['SibSp'].isnull().sum()))

# Parch
survived_parch = df['Survived'].groupby(df['Parch']).value_counts().unstack()
survived_parch.fillna(0, inplace=True)
survived_parch['Rate'] = survived_parch[1]/survived_parch.sum(axis=1)
survived_parch.sort_values(by='Rate', ascending=False, inplace=True)
print('\n', survived_parch)
print('Total number of NAs in Parch: {}'.format(df['Parch'].isnull().sum()))

# family size
df['Family Size'] = df['SibSp'] + df['Parch']
survived_family = df['Survived'].groupby(df['Family Size']).value_counts().unstack()
survived_family.fillna(0, inplace=True)
survived_family['Rate'] = survived_family[1]/survived_family.sum(axis=1)
survived_family.sort_values(by='Rate', ascending=False, inplace=True)
print('\n', survived_family)


# - *Fare*
# 
# We also notice that there is a large range in ticket fare, thus we are interested to see how much you can pay to survive... for the illustrative purpose, I will through the age into the mix...

# In[ ]:


# Fare
p = bkc.Scatter(df, x = 'Fare', y = 'Age', color = 'Survived',
                plot_width = 700, plot_height = 500, legend = 'top_right')
bkp.show(p)


# - *Cabin*
# 
# How about cabin? Naturally, one would this feature would have a great impact on the final survival rate. However, there are many missing values in the entries. Furthermore, the structure of the cabin number is in the form of letter + number. In order to get a better sense, I will need to strip these two values.

# In[ ]:


# cabin
print('Total number of non-NAs in Cabin: {}'.format(df['Cabin'].notnull().sum()))
print('Total number of NAs in Cabin: {}'.format(df['Cabin'].isnull().sum()))
cabin = df[['Survived', 'Cabin']].copy()
cabin.dropna(inplace=True)
def find_num(x):
    result = re.search('([0-9]+)', x)
    if result:
        return result.group()
    else:
        return '0'
cabin['Header'] = cabin['Cabin'].map(lambda x: re.findall('[A-Z]', x)[0])
cabin['Number'] = cabin['Cabin'].map(find_num)
survived_cabin_h = cabin['Survived'].groupby(cabin['Header']).value_counts().unstack()
survived_cabin_h.fillna(0, inplace=True)
survived_cabin_h['Rate'] = survived_cabin_h[1]/survived_cabin_h.sum(axis=1)
survived_cabin_h.sort_values(by='Rate', inplace=True, ascending=False)
print(survived_cabin_h)


# - *Embarked*
# 
# How about the feature Embarked? Maybe the passengers are allocated to different parts of the ship, which affect their final survival chance?

# In[ ]:


# Embarked
survived_embarked = df['Survived'].groupby(df['Embarked']).value_counts().unstack()
survived_embarked.fillna(0, inplace=True)
survived_embarked['Rate'] = survived_embarked[1]/survived_embarked.sum(axis=1)
survived_embarked.sort_values(by='Rate', ascending=False, inplace=True)
print(survived_embarked)


# ## Data wrangling
# We note that there are many missing values in the features Age (177), Cabin (687), and Embarked (2). We need to take care of them before we can build a model. Additionally, since we are going to use sklearn, we need to convert the categorical features to numbers. 

# In[ ]:


# how many na values for each column?
print(df.isnull().sum())
df_clean = df.copy()


# - *Age*
# 
# There are 177 missing entries in Age, too many that I can't afford to throw those rows away. Therefore, the question becomes, how to fill these missing values intelligently?
# 
# One method will be to fill all the 177 missing values with same number, for example, 0, or the mean age of the rest of the dataset.

# In[ ]:


age_mean = df[df['Age'].notnull()]['Age'].mean()
df_clean.loc[df['Age'].isnull(), 'Age'] = age_mean


# Can we do it more cleverly? For example, if we look at the row with a missing age, we could look at how many siblings or parents / children this particular passenger has. Then if (a big if) we can find the ages of his/her family, then we might be a clever way to guess the age of that passenger. However, this sounds like a solution would take many lines to present, hence I will stick with the average age. 
# 
# With the further modeling in mind, I can further bin the ages into just few ranges:

# In[ ]:


bins = [0, 20, 40, 60, 80, 100]
df_clean['Age range'] = pd.cut(df_clean['Age'], bins, labels=False)


# - *Cabin*
# 
# The next feature with many missing values is Cabin. From the entries that with Cabin values, it seems that the "header" of the Cabin can be a good indicator of the survival chance. For example, if the Cabin value starts with B, C, D, E, F, I would guess a higher chance of survival. How is the survival results looks for the passengers without a Cabin record?

# In[ ]:


df[df['Cabin'].isnull()]['Survived'].value_counts()


# What I can do, is to fill all the missing Cabin values with a distinct header, say, 'X'. Then I will carry this information for later modeling step. 
# 

# In[ ]:


df_clean.loc[df_clean['Cabin'].isnull(), 'Cabin'] = 'X000'
df_clean['Cabin_h'] = df_clean['Cabin'].map(lambda x: x[0])


# - *Embarked*
# 
# There are only 2 rows with missing Embarked values, therefore, I will fill them with the most frequent value, S. 

# In[ ]:


df_clean.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'


# Let's do a final check to make sure there is no missing value anywhere. Looking good!

# In[ ]:


df_clean.isnull().sum()


# Before we proceed, let's take a close look at those categorical features, and ask the question: are there too many different categories? If so, shall we group some of them?

# In[ ]:


df_clean.dtypes


# In[ ]:


df_clean['Cabin_h'].value_counts()
df_clean['Title'].value_counts()


# It seems, there are only three dominant categories in the 'Title' feature (a derived feature), therefore, I will group all but the three title.

# In[ ]:


# group titles
def group_title(x):
    if x not in ['Mr', 'Miss', 'Mrs']:
        return 'Other'
    else:
        return x
df_clean['Title'] = df_clean['Title'].map(group_title)


# ## Model building
# The goal of the model is to predict whether a passenger will survive. Let's first establish some baseline for the prediction. 
# The first crudest, and meaningless one: everyone will survive (or die), what's the accuracy?
# Well, the ratio between survived passengers and total passengers is 342/(342+549) = 0.383. Therefore if the model says everyone dies, the accuracy will be 1 - 0.383 = **0.617**. This is **baseline 1**.
# 
# We also know that, the survival percentage of female passengers is higher than that of male passengers. So what if guess: all female will survive (or equivalently, all male will die)? Let's take a look at what's the ratio:

# In[ ]:


df['Survived'].groupby([df['Sex'], df['Survived']]).count().unstack()


# The survival rate for female is 233/(233+81) = 0.742, and the death rate for male is 468/(468+109) = 0.811. Let's say, in the prediction, the chance to encounter a male or a female is 50-50, then the accuracy for this model is 0.742*0.5 + 0.811*0.5 = **0.777**. This is **baseline 2**.

# The accuracy of any realistic model, should at least beat these baselines. Let's start!
# Let me first partition the original data into training set and test set.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# make a copy, for no reason...
df_clean_run = df_clean.copy()
# do the OneHot encoding
# the added derived features are 'Age range', 'Family Size', 'Cabin_h', 'Title'
df_clean_run = pd.get_dummies(df_clean_run, columns=['Sex', 'Cabin_h', 'Pclass', 'Embarked', 'Title'])
# initilize the classifier
clf = RandomForestClassifier(n_estimators=1000, max_depth=5)
# split the training set, for x-validation
train, test = train_test_split(df_clean_run, test_size = 0.2)
features = train.columns.tolist()
remove_list = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 
               'Age range', 'SibSp', 'Parch']
for item in remove_list:
    features.remove(item)
print(features, '\n')
clf.fit(train[features], train['Survived'])
importances = [(f, i) for f, i in zip(features, clf.feature_importances_)]
importances.sort(key = lambda x: x[1], reverse=True)
#for f, i in importances:
#    print('Importance: {:>10}:{:4.3f}'.format(f, i))
print('\nTraining Accurancy: {:<30}'.format(clf.score(train[features], train['Survived'])))
print('Test Accurancy: {:<30}'.format(clf.score(test[features], test['Survived'])))


# ## Medel prediction
# 
# Now I am ready to take on the test dataset. There are three preprocessing steps I need perform on the testing dataset (order matters):
# 
# 1. Fill the missing values
# 2. Add the derived features
# 3. OneHot encoding
# 
# After the OneHot encoding, I also need to check whether there is any difference between the test dataset features, and the training dataset features.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df_clean = test_df.copy()
# preprocessing
# fill missing values
test_df_clean.isnull().sum()
age_mean = test_df[test_df['Age'].notnull()]['Age'].mean()
test_df_clean.loc[test_df['Age'].isnull(), 'Age'] = age_mean
test_df_clean.loc[test_df_clean['Cabin'].isnull(), 'Cabin'] = 'X000'
test_df_clean['Cabin_h'] = test_df_clean['Cabin'].map(lambda x: x[0])
test_df_clean.loc[test_df_clean['Fare'].isnull(), 'Fare'] = test_df[test_df['Fare'].notnull()]['Fare'].mean()


# In[ ]:


# 2. Add derived features
# the added derived features are 'Age range', 'Family Size', 'Cabin_h', 'Title'
test_df_clean['Age range'] = pd.cut(test_df_clean['Age'], bins, labels=False)
test_df_clean['Family Size'] = test_df_clean['SibSp'] + test_df_clean['Parch']
test_df_clean['Cabin_h'] = test_df_clean['Cabin'].map(lambda x: x[0])
test_df_clean['Title'] = test_df_clean['Name'].map(lambda x: re.split('[,.]', x)[1].strip())
test_df_clean['Title'] = test_df_clean['Title'].map(group_title)


# In[ ]:


# OneHot encoding
test_df_clean = pd.get_dummies(test_df_clean, columns=['Sex', 'Cabin_h', 'Pclass', 'Embarked', 'Title'])


# In[ ]:


# check for possible missing features
for fe in features:
    if fe not in test_df_clean.columns.tolist():
        test_df_clean[fe] = 0.0


# In[ ]:


# predict!
output = clf.predict(test_df_clean[features])


# In[ ]:


df_submit = test_df_clean[['PassengerId']].copy()
df_submit['Survived'] = pd.Series(output)
df_submit.to_csv('output.csv', index=False)


# In[ ]:




