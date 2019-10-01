#!/usr/bin/env python
# coding: utf-8

# # My Take on the Beginner Titanic Kernel
# Welcome to my first kernel! I will be exploring the [Titanic data set](https://www.kaggle.com/c/titanic) as a way to practice and learn data science techniques. Many of the ideas I use here will be shamelessly stolen from other kernels. I will cite any code or plots that are lifted directly from someone else.
# 
# ### Table of Contents
# 1. [Introduction](#introduction-what-we-expect-to-see)
# 2. [Import the data and first observations](#import-the-data-and-first-observations)
# 3. [Super Simple Model](#super-simple-model)
# 4. [Improved Model](#improved-model)
# 5. [Predictions](#predictions)

# ## Introduction: What we expect to see
# The Titanic data set contains records of passengers aboard the ship and whether they survived or died when the ship sank.
# 
# Here are the columns in the training set from the [data dictionary](https://www.kaggle.com/c/titanic/data)
# 
# | Variable | Definition                                 | Key                                            |
# |:---------|:-------------------------------------------|:-----------------------------------------------|
# | survival | Survival                                   | 0 = No, 1 = Yes                                |
# | pclass   | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
# | sex      | Sex                                        |                                                |
# | Age      | Age in years                               |                                                |
# | sibsp    | # of siblings / spouses aboard the Titanic |                                                |
# | parch    | # of parents / children aboard the Titanic |                                                |
# | ticket   | Ticket number                              |                                                |
# | fare     | Passenger fare                             |                                                |
# | cabin    | Cabin number                               |                                                |
# | embarked | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |
# 
# 
# I expect from prior knowledge of this disaster that women, children, and the upper class will have better survival rates. We could make a baseline model using only those features before attempting to engineer others.

# ## Import the data and first observations

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read train and test data into pandas dataframes
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Take a look at the training data
train.sample(5)


# In[ ]:


train.describe()


# **Observations**
# * We want to use the gender of the passengers in our model. The values in the column `Sex` are encoded as strings: `female` and `male`. We will have to convert this column to a numerical encoding to use it for modeling.
# * `Age` has a lot of `NaN` values (its count in the training set is 714, compared to 891 rows). Since we expect we will want to use this, we need to either drop all the rows with `NaN` values (which would throw away 177 rows, kind of a lot) or come up with a way to impute the missing data.

# ## Super simple model
# We'll make a basic data set, with the most obvious features: `Age`, `Sex`, and `Pclass`. We will have to clean `Age` and extract `Sex` into a usable form. 
# 
# Once we construct the training and test data with these columns, we will use the Gaussian Naïve Bayes model to make predictions and get a score. Later, when we make more sophisticated features and use other models, we can use this score as a baseline.

# ### Encoding Categorical Sex Column
# To use the `Sex` column, we must convert it from a categorical encoding (which has as its values strings `female` and `male`) to a one-hot encoding (with two columns, `Sex_female` and `Sex_male`, taking values 0 and 1). The way this is often done is with the pandas function `get_dummies`. However, this function can have unwanted side effects on test data. If the training data and the test data do not have the same set of values, then the resulting one-hot encoded training and test data will not have the same number of columns. This causes a problem with downstream models, which require training and test data columns to be identical. So, to use `get_dummies`, we must resolve any problems by hand. At best, this is a fiddly, manual process; at worst this can introduce data leakage of the test set into the training set. Both of these can and should be avoided!
# 
# There are encoders in scikit-learn to take care of this, by using the training set to create the encoded columns and fitting the test set onto those same columns. But as far as I know, they don't handle string categories well (or at all). There is a CategoricalEncoder that is present in the source repository, but has not been released yet. As of now, we can use the `OneHotEncoder` from the `category_encoders` library to do the same function.
# 
# I will not do what I have seen in other tutorials, which is to simply map values in the `Sex` column to 1 if `male` and 0 if `female`. For one reason, that approach does not work on categorical variables in general; it presumes we know all the values beforehand. And for another, gender isn't a binary, and I as a matter of princliple will never use a single binary column for it.

# In[ ]:


import category_encoders as ce

ohe = ce.one_hot.OneHotEncoder(cols=['Sex'], handle_unknown='ignore', use_cat_names=True)
train_basic = ohe.fit_transform(train[['Pclass', 'Age', 'Sex']])

# If this were our actual model for submission, we would transform the test data as well
# test_basic = ohe.transform(test[['Pclass', 'Age', 'Sex']])

train_basic.head()


# ### Impute Missing Age Data: Basic Approach
# Now that we have one-hot encoded `Sex`, we need to impute missing data for `Age`. There are a lot of different ways we could do this, but for this simple model we take the simplest one: find the mean of all values of `Age`, and use that mean to fill in the missing values.
# 
# Note that we only take the mean over the training data, and use that mean to fill missing values in both training and test data. Always remember to avoid data leakage!

# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
train_basic = imp.fit_transform(train_basic)
# test_basic = imp.transform(test_basic)


# ### Training and Validation Data
# We need to have a way to score our models without using the testing data. For the purposes of model exploration, we split the training data into a smaller training set and a validation set.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(train_basic, train['Survived'], random_state=43210)


# ### Fit a Model and Make Predictions
# It is time now to predict the survival from passenger records in the test set. 
# 
# There are many models we could choose. But, again, this model should stick to basics. We will go through other models later when we have a more interesting data set. So, for now, we will choose Gaussian Naïve Bayes, simply to have some baseline for later comparisons.

# In[ ]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_validation, y_validation)


# We got 80% accuracy for our simple model. That isn't terrible! But let's see what else we can do to make it better.

# ## Improved Model
# How can we improve on this model? 
# * Of the features we included in the simple model, `Pclass` and `Sex` seem to be pretty finished. There isn't much else we can do to improve them. We could improve `Age` by finding a better way to impute the missing values.
# * There are lots of other columns in the data set that we simply ignored when making our basic model. We might be able to pull something out of those.
# 
# In looking at these other columns, we should compare the values we see to `Survival`. We want to see whether some difference in the column's values correlates to a difference in `Survival` values. If we find that, then hopefully we have found a meaningful signal our model can use to improve its accuracy.

# ### Cabin
# Since we expect that passenger class predicts survival, might cabin numbers also be predictive? Let's take a look.

# In[ ]:


train['Cabin'].describe()


# In[ ]:


train['Cabin'].unique()


# #### Cabin Observations
# * A ton of unique values. Which we would expect. Most cabins would only hold a few people.
# * Also a lot of `nan` values.
# * The cabin numbers begin with a letter, which identifies the deck. The lower the letter, the higher the class of the passenger.
# * Given that we see more lower letters and fewer higher letters (few `E`s and almost no `F`s, for instance) we infer that it is more likely that higher-class passengers had their cabins recorded.
# * There is one `T` in there, which is weird. There was no deck `T` on Titanic. We'll check on that passenger.
# * Several entries have more than one cabin. I assume those are families that booked a block of cabins together, and each person had all the cabins on their records. We'll check that.
# * A couple records are anomalous in that they have a letter with no number, but then a different letter+number combo. For instance: "`F G73`" and "`F E69`". I don't know what to think about those.
# 
# Let's look at some of the outlier entries.

# In[ ]:


train[(train['Cabin'] == 'T') | (train['Cabin'] == 'B51 B53 B55') | (train['Cabin'] == 'F E69') | (train['Cabin'] == 'F G73')]


# #### Cabin Observations, Continued
# Don't know what to think about those anamalous entries.
# * My hypothesis about multiple cabin numbers being for families is wrong. There are two passengers with cabin entry `B51 B53 B55`. 
#     * I thought it would have been three, since there are three cabins.
#     * They aren't in the same family. They have different names. One of them is traveling alone, the other with a parent or child.
#     * They didn't even embark from the same port.
# * Other anomalous cabins are similarly perplexing. I can't find a pattern in what I'm seeing.
# 
# I guess I'll just ignore these strange entries and proceed as if they aren't anomalous at all. 

# What we'll do with the cabin is create a numeric feature:
# * We extract one deck letter from the cabin value.
# * We map that letter onto a numeric value (A->7, B->6, etc.)
# * For any missing values, we impute the mean for passengers of the same class.

# In[ ]:


import re

singleLetterRe = re.compile(r"^[A-Z]$") # This will clean the weird 'T' value
cabinRe = re.compile(r"^([A-Z] )?([A-Z])\d+.*$")
decks = dict(zip('ABCDEFG', range(7, 0, -1)))

# First, make the numeric deck column for train and test, preserving nan
for df in (train, test):
    df['Deck'] = (df['Cabin'].replace(singleLetterRe, np.nan)
                  .replace(cabinRe, '\\2')
                  .map(decks, na_action='ignore'))

# Next, fill in missing deck values
# We group decks by pclass and take the mean, then fill all the missing values
# with the mean for their plass
deckmeans = train[['Pclass', 'Deck']].groupby('Pclass')['Deck'].mean()
for pclass in 1,2,3:
    train.loc[(train['Pclass'] == pclass) & (train['Deck'].isna()), 'Deck'] = deckmeans[pclass]
    test.loc[(test['Pclass'] == pclass) & (test['Deck'].isna()), 'Deck'] = deckmeans[pclass]
print(train.groupby('Deck')['Deck'].count())

plt.figure(figsize=(14,6))
sns.barplot(x="Deck", y="Survived", data=train, ax=plt.gca());


# ### Cabin: present or not?
# Another signal we can interpret from the `Cabin` column is whether a cabin was recorded for a passenger or not. 

# In[ ]:


for df in train, test:
    df['cabin_was_recorded'] = ~df['Cabin'].isna()
sns.barplot(x='cabin_was_recorded', y='Survived', data=train);


# This looks like it separates the values pretty well. We can include it and see if it will be helpful.

# ### Title
# We can't really use the passenger's names directly. There is a ton of variation, most of which is noise. Or so I assume; maybe there is a signal lurking in there that I can't see.
# 
# One thing that is somewhat regular in the name is the title. Every passenger's name has some kind of title, like `Mr.` or `Mrs.`. Some are only used for younger passengers, like `Master.` or `Miss.` We can extract this title into a column.
# 
# While it may or may not be useful on its own, the title can give us a better way to impute missing age values.

# In[ ]:


# This process of splitting gets us the word immediately before a '.'
for df in train,test:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train.groupby('Title')['Title'].count()


# In[ ]:


plt.figure(figsize=(18,6));
sns.barplot(x='Title', y='Survived', data=train, ax=plt.gca());


# Okay, so that's all the titles. Doesn't look like we can use it in a model super directly. But how to they break down by age? Let's look at the age distributions for the four most common titles.

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(18,6))
bins = range(0, 70, 5)
for title, ax in zip(('Master', 'Miss', 'Mr', 'Mrs'), axes.flatten()):
    sns.distplot(train.loc[train['Title']==title, 'Age'].dropna(), bins=bins, kde=False, ax=ax, label=title)
    ax.legend()


# This looks to me like you can guess pretty well the age of the passengers by looking at their titles. So that's how we will impute the age values. We'll group by Title, take the mean of the ages, and fill missing values for Age based on their Title's mean. (If any are still missing after that, we can try the same procedure with Sex.)

# In[ ]:


titleagemeans = train[['Title', 'Age']].groupby('Title')['Age'].mean()
for title in train['Title'].unique():
    if titleagemeans[title] == np.nan:
        # If, say, one of the rare titles is missing all age values,
        # its mean will still be nan.
        # Skip it for now. We can check later if there are 
        # still nan values for Age to fill in
        continue
    train.loc[(train['Title'] == title) & (train['Age'].isna()), 'Age'] = titleagemeans[title]
    test.loc[(test['Title'] == title) & (test['Age'].isna()), 'Age'] = titleagemeans[title]    


# In[ ]:


# Now sweep up any values left over
imp = SimpleImputer(strategy='mean')
train['Age'] = imp.fit_transform(train['Age'].values.reshape(-1, 1))
test['Age'] = imp.transform(test['Age'].values.reshape(-1, 1))


# In[ ]:


np.any(train['Age'].isna())


# ### Age Bins
# Now that we have filled in the missing Age values, let's discretize the data. We can convert these data into bins; let's say we want 5 bins, which would make them of size (80-0)/5=16. 
# 
# We'll create this column by hand.

# In[ ]:


binsize = 16
for df in train, test:
    df['Age band'] = (df['Age'] - df['Age'].mod(binsize)).div(binsize).astype(int)
train['Age band'].value_counts().to_frame()


# ### Traveling with Family or Alone
# Next we try to pull something out of `SibSp` and `Parch`. A reminder of the column definitions:
# *  **sibsp** # of siblings / spouses aboard the Titanic
# * **parch** # of parents / children aboard the Titanic
# 
# In particular, I'm going to look at whether a passenger was traveling alone. Does that have any predictive value we can pull out?

# In[ ]:


for df in train, test:
    df['Alone'] = 0
    df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0), 'Alone'] = 1


# In[ ]:


sns.barplot(x='Alone', y='Survived', hue='Sex', data=train);


# It looks as if being alone gives a small improvement to the survival chance of women, but decreases the survival chance of men.
# 
# What about different ages? Can we see an effect on whether being alone impacts survival based on age?

# In[ ]:


g = sns.FacetGrid(train, col='Alone', row='Sex', margin_titles=True, size=5)
g.map(sns.barplot, 'Age band', 'Survived');


# I can see two useful features we could engineer out of here. 
# * We could make a binary column for "Alone_male". This seems pretty correlated to a low survival rate. (Not depicted in the plots, however, is the 80-year-old man traveling alone who did survive.)
# * The survival rate appears linear for women who are not traveling alone. We can make a column for this, with the values for women not alone being equal to their age band, and the value for everyone else set to -1.

# In[ ]:


for df in train, test:
    df['Alone_male'] = 0
    df.loc[(df['Alone'] == 1) & (df['Sex'] == 'male'), 'Alone_male'] = 1
    
    df['Accompanied_female_age_band'] = -1
    accompanied_females = (df['Alone'] == 0) & (df['Sex'] == 'female')
    df.loc[accompanied_females, 'Accompanied_female_age_band'] = df.loc[accompanied_females, 'Age band']

fig, axes = plt.subplots(1, 2, figsize=(14,6))
sns.barplot(x='Alone_male', y='Survived', data=train, ax=axes[0]);
sns.barplot(x='Accompanied_female_age_band', y='Survived', data=train, ax=axes[1]);


# ### Fare
# I don't really know anything about the Fare column. I don't see how it could give us more information than Pclass. But let's check it out.

# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(train['Fare'], ax=plt.gca());


# #### Observations
# * Almost everyone paid very little for their tickets.
# * A few paid more, some a lot more.
# 
# Who paid over £500 for their tickets?

# In[ ]:


train[train['Fare'] > 500]


# Since I assume fares are correlated with Pclass, let's examine those together.

# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18,6))

for i in range(3):
    sns.distplot(train.loc[train['Pclass'] == i+1, 'Fare'], ax=axes[i])
    axes[i].set_title('Fares in Pclass {}'.format(i+1));


# I can't make anything out of this. And given that I don't expect the fare a passenger paid would have any correlation to their survival, I'm not going to include it.

# ## Predictions
# Let's return to our Gaussian Naïve Bayes model. Are we doing better than we did before?
# 
# ### Features
# The features we are going to keep:
# * Pclass
# * Deck
# * cabin_was_recorded
# * Age band
# * Alone
# * Alone_male
# * Accompanied_female_age_band
# * Sex_male
# * Sex_female
# 
# The features we are going to drop:
# * PassengerId
# * Name
# * Age
# * SibSp
# * Parch
# * Ticket
# * Fare
# * Cabin
# * Embarked
# * Title

# In[ ]:


drop_cols = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title']
test_ids = test['PassengerId']
survived = train['Survived']
train = train.drop(drop_cols + ['Survived'], axis=1)
test = test.drop(drop_cols, axis=1)


# Lastly, don't forget to encode Sex as a categorical feature

# In[ ]:


ohe = ce.one_hot.OneHotEncoder(cols=['Sex'], handle_unknown='ignore', use_cat_names=True)
train = ohe.fit_transform(train)
test = ohe.transform(test)


# In[ ]:


train.head()


# In[ ]:


X_train, X_validation, y_train, y_validation = train_test_split(train, survived, random_state=43210)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_validation, y_validation)


# Ok, our score got worse. That's a bit disheartening.

# In[ ]:


X_train, X_validation, y_train, y_validation = train_test_split(train[['Pclass', 'Age band', 'Sex_male', 'Sex_female']], survived, random_state=43210)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb.score(X_validation, y_validation)


# Well, at least we can recover our 80% score by keeping only the few features we had for our simple model. Still, it doesn't feel great to have done all the work of putting together those features and seeing it make things worse.

# ### XGBoost

# In[ ]:


import xgboost as xg
from sklearn.model_selection import cross_val_score

xgb = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
result=cross_val_score(xgb, train, survived, cv=5, scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())


# In[ ]:


xgb.fit(train, survived)
predictions = xgb.predict(test)


# In[ ]:


results = pd.DataFrame()
results['PassengerId'] = test_ids
results['Survived'] = predictions
results.head()


# In[ ]:


results.to_csv('results.csv', index=False)


# In[ ]:




