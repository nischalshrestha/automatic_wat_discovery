#!/usr/bin/env python
# coding: utf-8

# # Sinking of the Titanic - From Data to Insights
# 
# ## Goal
# 
# My goal is to understand the factors that made people survive the sinking of the Titanic. 
# 
# ## Strategy
# 
# This notebook's structure is as follows:
# 
# 1. Check how the data looks like.
# 1. Gain some initial insights out of the available variables.
# 1. Check if there are missing values and if so, impute them.
# 1. Engineer some variables that may bring more insigths or descriminative power for modelling.
# 1. Conclude on what made people survive the sinking of the Titanic.
# 1. Define next steps.
# 1. Indicate external references, if applicable.
# 
# ## How The Data Looks Like?

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz


# In[ ]:


sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
sns.set_palette(sns.color_palette("husl", 10))
get_ipython().magic(u'matplotlib inline')

# not good practice; remove when scipy warnings don't show anymore
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# The data schema of `train` and `test` match, with the exception that train has an extra variable, the dependent variable. Thus it's possible to concatenate `train` and `test` into just one data frame. This will facilitate the pre-processing of the data and be able to get better descriptive statistics.

# In[ ]:


titanic = pd.concat([train, test], axis=0, sort=False).reset_index()
titanic.info()


# To get situational awareness I recommend reading the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) and [The Sinking Of The Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic).
# 
# #### Some Observations
# 
# - Some variables are continuous and others are discrete or categorical.
# - There is missing data in variables `Cabin`, `Age`, `Embarked` and `Fare`.
# - `PassengerId` is a unique sequential numeric identifier and seems not important.
# - `Survived` is the dependent variable and for 1/3 of the passengers I don't know if they survived or not.
# - `Pclass` is strongly correlated with `Fare` as a proxy for economic status.
# - `Name` maybe is useful to extract the passenger's title and the family surname.
# - `Age` can be re-written as age bins.
# - `SibSp` indicates the number of siblings and spouse abord the Titanic.
# - `Parch` indicates the number of parents and children aboard the Titanic.
# - `Ticket` indicates ticket number. Doesn't seem useful, unless I can derive a new feature out of it.
# - `Cabin` has many missing points but maybe I can get something (like deck) out of that.
# - `Embarked` indicates in which port the passengers embarked.

# ## What Can We Learn From Each Variable?
# 
# ### Sex, Pclass and Fare
# 
# #### Does Survival Relates With Gender or Social Class?

# In[ ]:


f, (g1, g2) = plt.subplots(1, 2, figsize=(15,5))
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train, ax=g1).set_title(
    "Average Survival per Gender and Social Class");
sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=train, ax=g2).set_title(
    "Average Survival per Gender and Social Class");


# In our sample women were more likely to survive than men. Even more if they were rich.
# 
# Men in 2nd class were as screwed as the ones in 3rd class. However that didn't happen to women.

# #### How Survival Relates With Social Class?

# In[ ]:


f, (g1, g2, g3) = plt.subplots(1, 3, figsize=(15,5))
sns.distplot(titanic.Fare.dropna(), ax=g1, hist=False, color='g').set_title(
    "How Fare is Distributed");
sns.boxplot(x='Fare', y='Pclass', data=titanic, orient='h', ax=g2).set_title(
    "Boxplot of Fare per Social Class");
sns.boxplot(x='Fare', y='Survived', data=titanic, orient='h', ax=g3).set_title(
    "Boxplot of Fare per Survival Status");
titanic.groupby('Pclass')['Fare'].describe()


# This matches my expectations but let me check one of extreme data points on the right side of the boxplot.

# In[ ]:


titanic.query("Fare > 300")


# All 4 passengers travelled on the same ticket and had 4 cabins, looking at their surnames, number of parents/children and ages I'd say passengers 680 and 1235 were a mother and son while the other 2 passengers may be their servants.

# In[ ]:


f, (g1, g2) = plt.subplots(1, 2, figsize=(15,5))
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=train, ax=g1).set_title(
    "Average Survival per Social Class and Gender");
sns.countplot(x='Pclass', hue='Survived', data=train, ax=g2).set_title(
    "Nr of Survivals/Deaths per Social Class");


# Women and men travelling in 1st class had more chances of surviving than passengers travelling in other classes.

# #### Take Home Message(s)
# 
# Social class and gender played an important role in surviving the sinking of the Titanic.
# 
# #### TODO
# 
# 1. Impute the missing value in variable `Fare`.
# 
# ### Age

# In[ ]:


titanic[['Age']].describe().T


# In[ ]:


f, (g1, g2, g3) = plt.subplots(1, 3, figsize=(15, 5))
sns.distplot(
    titanic['Age'].dropna(), bins=int(titanic['Age'].max()), 
    color='g', ax=g1).set_title("Age Distribution");

sns.distplot(
    titanic.query("Sex == 'female'")['Age'].dropna(), bins=int(titanic['Age'].max()),
    color='red', hist=False, ax=g2, label='Women').set_title(
    "Age Distribution for Women and for Men");
sns.distplot(
    titanic.query("Sex == 'male'")['Age'].dropna(), bins=int(titanic['Age'].max()),
    color='b', hist=False, ax=g2, label='Men');

sns.distplot(
    titanic.query("Survived == 0")['Age'].dropna(), bins=int(titanic['Age'].max()),
    hist=False, label="Died", color='black', ax=g3).set_title(
    "Age Distribution per Survival Status");
sns.distplot(
    titanic.query("Survived == 1")['Age'].dropna(), bins=int(titanic['Age'].max()),
    hist=False, label="Survived", color='g', ax=g3);


# #### Observations
# 
# 1. The passengers were mostly young people, 75% of them had less than 39 years old.
# 1. There were more children with 4 years old or less than children between 5 and 12.
# 1. There were more girls than boys (age $\in [0, 15]$).)
# 1. Children with less than 10 years old were favoured compared with other age groups.

# In[ ]:


f, (g1, g2, g3) = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(titanic['Age'], orient='h', ax=g1).set_title(
    "Boxplot of Age");
sns.boxplot(x='Age', y='Embarked', data=titanic, orient='h', ax=g2).set_title(
    "Boxplot of Age per Port");
sns.boxplot(x='Age', y='Pclass', data=titanic, orient='h', ax=g3).set_title(
    "Boxplot of Age per Social Class");


# #### Take Home Message(s)
# 
# 1. First class passengers are older than in 2nd and 3rd class probably because they had more time to make fortune or were travelling for leisure while some passengers in 3rd class were migrating to US.
# 1. Perhaps there is a relation between passengers that embarked in Queenstown and passengers travelling in 3rd class.
# 
# #### TODO
# 1. Impute the missing values in variable `Age`.
# 1. Build a new variable that breaks age into bins like babies, children, teenagers, young adults, adults and seniors.
# 
# ### Cabin
# 
# The data seems too sparse to be imputed but maybe I can extract the deck and see if I can do something out of that.

# In[ ]:


cabin_not_null = titanic[~titanic.Cabin.isnull()].copy()
cabin_not_null['deck'] = cabin_not_null['Cabin'].str[0]
f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(
    x='deck', y='Survived', data=cabin_not_null, order=sorted(cabin_not_null.deck.unique()),
    ax=g1).set_title("Average Survival per Deck");
sns.countplot(
    x="deck", hue="Pclass", data=cabin_not_null, palette="Greens_d", 
    order=sorted(cabin_not_null.deck.unique()), ax=g2).set_title(
    "Nr of Passengers per Deck and Social Class");


# In[ ]:


counts = train['Survived'].value_counts()
print("Overall Survival Rate:\t\t {}".format(counts[1] / counts.sum()))
print("Cabin-Not-Null Survival Rate:\t {}".format(
    cabin_not_null.Survived.sum() / cabin_not_null.shape[0]))


# Survival rates seem almost uniform among the different decks.
# 
# Seems that the passengers that have information about their cabin have a higher chance of surviving (0.46) than the overall average (0.38).
# 
# Seems that decks A, B, and C were  1st class only.
# 
# #### Take Home Message(s)
# 
# In our sample having information about the passengers' cabins is more common among 1st class than in 2nd and 3rd class. It seems having some relation with survival rate.
# 
# #### TODO
# 1. Build a variable that indicates if cabin information is available or not.
# 
# ### Embarked

# In[ ]:


titanic.groupby('Embarked')['Fare'].describe()


# In[ ]:


titanic[titanic.Embarked.isnull()]


# I don't know where those two passengers embarked but their fare was 80 and fares are distributed differently among the 3 different embarkement points. Thus it's likely that the passengers embarked in Cherbourg.

# In[ ]:


f, ((g1, g2), (g3, g4)) = plt.subplots(2, 2, figsize=(12,12))
sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=titanic, ax=g1).set_title(
    "Average Survival per Port and Social Class");
sns.countplot(x='Embarked', hue='Survived', data=titanic, ax=g2).set_title(
    "Nr of Passengers per Port and Survival Status");
sns.barplot(x='Embarked', y='Fare', hue='Survived', data=titanic, ax=g3).set_title(
    "Average Fare per Port and Survival Status");
sns.barplot(x='Embarked', y='Fare', hue='Pclass', data=titanic, ax=g4).set_title(
    "Average Fare per Port and Social Class");


# #### Take Home Message(s)
# 
# - For our sample, port of embarkment is related with survival rate because it's a proxy of `Pclass` and `Fare`.
# - Curiousl in Queenstown passengers in 2nd class had a better survival rate than the ones in 1st class probably due to the fact the sample size is small.
# - Many rich passengers embarked in Cherbourg and this port had a better survival rate.
# 
# #### TODO
# 1. Impute the two missing values in `Embarked` to be Cherbourg.
# 
# ### Name

# In[ ]:


titanic.Name.head()


# I can extract title and surname out of the names.

# In[ ]:


titles = titanic.Name.apply(lambda s: s.split(',')[1].split('.')[0])
titles.value_counts()


# #### TODO
# 1.  Build new features `title` and `surname` out of `Name`.
# 
# ### Parch and SibSp
# 
# I can compute the size of the family based on these two variables.

# In[ ]:


df = titanic[['Survived']].copy()
df['family_size'] = titanic['Parch'] + titanic['SibSp'] + 1

f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x="family_size", y="Survived", data=df, ax=g1).set_title(
    "Average Survival per Family Size");
sns.countplot(x="family_size", data=df, ax=g2, palette='Greens_d').set_title(
    "Nr of Passengers For Each Group In Family Size");


# #### Take Home Message(s)
# In our sample small families had higher chances of surviving.
# 
# #### TODO
# 1. Build a feature that indicates family size.
# 1. Build a feature that indicates if it's a mother/father of a baby or child - use `Ticket`, `Age`, `family_size` and `Parch`.
# 
# ### Ticket

# In[ ]:


print(titanic.Ticket.count())
print(titanic.Ticket.nunique())
titanic.Ticket.unique()[:50]


# Tickets don't seem unique and sometimes they've prefixes, can any of these be useful?
# 
# #### TODO
# 
# 1. Use ticket number to identify people travelling together.
# 1. Extract prefixes and see how they relate with `Survived`.

# ## Can We Build New Variables that Bring Us Extra Information?
# 
# Implement the ideas that I indicated previously in the TODOs.
# 
# ### Impute Missing Data
# 
# #### Embarked

# In[ ]:


feats = titanic.copy()
feats[feats.Embarked.isnull()]


# Looking at how fares are distributed among the 3 ports and thinking that these two passengers have a fare of 80 it's likely that they embarked in Cherbourg.

# In[ ]:


feats['Embarked'].fillna('C', inplace=True)
feats['Embarked'].isnull().sum()


# #### Fare

# In[ ]:


feats[feats.Fare.isnull()]


# I chose to  impute this null with the mean of the passengers that embarked in Southampton and travelled in 3rd class.

# In[ ]:


desc = feats.query("Embarked == 'S' and Pclass == 3")['Fare'].describe()
feats['Fare'].fillna(desc['mean'], inplace=True)
feats['Fare'].isnull().sum()


# #### Age
# 
# Here my choice is to do linear regression to impute the missing data in variable `Age`.

# In[ ]:


predictors = ['Fare', 'Parch', 'Pclass', 'SibSp', 'Age']
age_train = feats.loc[~feats.Age.isnull(), predictors]
age_test = feats.loc[feats.Age.isnull(), predictors]

lm = LinearRegression()
lm.fit(age_train.drop('Age', axis=1), age_train['Age'], )
predicted_age = lm.predict(age_test.drop('Age', axis=1))
feats.loc[feats.Age.isnull(), 'Age'] = predicted_age
feats.info()


# In[ ]:


sns.distplot(titanic.Age.dropna(), hist=False, color='r', label='Before Imputation');
sns.distplot(feats.Age, hist=False, color='g', label='After Imputation').set_title(
    "KDEs of Age Distributed Before and After Imputation");


# The distributions match except on their peak. Seems that the majority of the imputations happened in the range [$20, 35]$. Perhaps it's worth checking if predicting age is being well modelled but for now I'll move forwards with these results.

# ### Encode Existing  Categorical Variables Into Numeric Variables
# 
# #### Sex

# In[ ]:


sex = pd.get_dummies(feats['Sex'])
sex.head()


# #### Embarked

# In[ ]:


dic = {'C': 'cherbourg', 'Q': 'queenstown', 'S': 'southampton'}
embarked = pd.get_dummies(feats['Embarked'])
embarked.columns = [dic[i] for i in embarked.columns if i in dic.keys()]
embarked.head()


# ### Create New Variables
# 
# ### Break Age Into Bins

# In[ ]:


bins = [-1, 2, 6, 12, 16, 40, 60, 100]
group_names = ['baby', 'small_child', 'child', 'teenager', 'young_adult', 'adult', 'senior']
categories = pd.cut(feats['Age'], bins, labels=group_names)
feats['age_bins'] = categories


# In[ ]:


f, (g1, g2) = plt.subplots(2, 1, figsize=(10, 12))
sns.countplot(
    x='age_bins', hue='Survived', data=feats, order=group_names, 
    palette="Greens_d", ax=g1).set_title(
    "Nr of Survivals and Deaths per Age Group");
sns.barplot(
    x='age_bins', y='Survived', hue='Sex', data=feats, order=group_names, 
    palette="Greens_d", ax=g2).set_title(
    "Average Survival per Age Group and Gender");
g2.legend(loc='upper left');


# #### Take Home Message(s)
# 
# - Being a baby or a small child increased the chance of surviving.
# - In our sample, women had more chances to survive then men, excepting if they were children (6-12 years old) but this is a small group and thus its likely that the survival mean is not a solid statistic in this case.

# In[ ]:


age = pd.get_dummies(feats['age_bins'])
age.head()


# ### Cabin Info Is Available

# In[ ]:


cabin_not_null = pd.Series([0] * feats.shape[0]).rename('cabin_not_null')
cabin_not_null[~feats.Cabin.isnull()] = 1
cabin_not_null.head()


# ### Title and Surname

# In[ ]:


surname = pd.DataFrame(
    feats.Name.apply(lambda s: s.split(',')[0].split('.')[0]).rename('surname'))
feats['surname'] = surname
surname.head()


# In[ ]:


feats['title'] = pd.DataFrame(
    feats.Name.apply(lambda s: s.split(',')[1].split('.')[0].strip(' ')).rename('title'))
changes = {
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
    'Sir': 'Noble', 'Lady': 'Noble', 'the Countess': 'Noble', 'Jonkheer': 'Noble', 'Don': 'Noble', 'Dona': 'Noble',
    'Major': 'Militar', 'Capt': 'Militar', 'Col': 'Militar'}
feats['title'].replace(changes, inplace=True)
print(feats.title.value_counts())

title = pd.get_dummies(feats['title'])
title.columns = [i.lower() for i in title.columns]
title.head()


# In[ ]:


f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(x='title', hue='Survived', data=feats, palette="Greens_d", ax=g1).set_title(
    'Nr of Survivals and Deaths per Title');
sns.barplot(x='title', y='Survived', data=feats, palette="Greens_d", ax=g2).set_title(
    "Average Survival per Title ");


# #### Take Home Message(s)
# 
# The groups that had a better chance of surviving were women, noble people and well-off people (Masters).

# ### Family Size/Type

# In[ ]:


feats['family_size'] = feats['Parch'] + feats['SibSp'] + 1
bins = [0, 1, 4, 11]
feats['family_type'] = pd.cut(
    feats['family_size'], bins, labels=['single', 'small_family', 'large_family'])
feats['family_size'].head()


# In[ ]:


f, (g1, g2) = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(x='family_type', y='Survived', data=feats, ax=g1).set_title(
    "Average Survival per Family Type");
sns.countplot(x='family_type', hue='Survived', data=feats, palette="Greens_d", ax=g2).set_title(
    'Nr of Survivals and Deaths per Family Type');


# In[ ]:


family_type = pd.get_dummies(feats['family_type'])
family_type.head()


# ### Is She A Young Mother / Is He A Young Father

# Find who are the passengers that share the the same ticket.

# In[ ]:


tickets = feats.query("family_size > 1")['Ticket'].copy(
    ).str.replace('.', '').rename('ticket').to_frame()
split = tickets.ticket.str.split(' ')
tickets['ticket_nr'] = split.apply(lambda s: s.pop())
def get_element(s):
    '''Get the element of a list.'''
    try:
        return s[0]
    except Exception as e: 
        return None
tickets['ticket_prefix'] = split.apply(lambda s: get_element(s))
tickets[['ticket', 'ticket_prefix', 'ticket_nr']].head()


# For each group of shared tickets, check if there is a baby and if so, identify the parents.

# In[ ]:


pars = []
for t in tickets.ticket_nr.unique():
    dat = feats.iloc[tickets[tickets.ticket_nr == t].index.tolist()] 
    if dat.shape[0] == 1 or not any(dat.Age <= 6):
        continue  # skips if there is only 1 passenger per ticket number or if there is no baby
    family = pd.concat([dat.query("Age <= 6"), dat.query("Parch > 0 and Age > 15")])
    pars.append(family)
parents = pd.concat(pars).query("Parch > 0 and Age > 15")
parents['parents'] = 1
parents[['surname', 'Age', 'Parch', 'Pclass', 'Sex', 
         'SibSp', 'Survived', 'age_bins', 'family_size']].head()
feats = feats.join(parents['parents'])
feats['parents'].fillna(0, inplace=True)


# In[ ]:


feats['parents'].sum()


# Unfortunately this approach only finds a limited number of parents (57).

# ### Ticket Is Unique and Prefixes

# In[ ]:


tickets.ticket_prefix.value_counts().rename('nr_tickets').to_frame()


# Checked the survival rate of the people with prefixes and I didn't get anything that seemed useful - also the ammount of tickets with prefix is quite low.
# 
# Later I'll use the ticket number to identify groups of people travelling together.

# ## What Makes A Survivor?

# In[ ]:


n = train.shape[0]
dfs = [
    feats.iloc[:n], sex[:n], embarked[:n], cabin_not_null[:n], 
    age[:n], title[:n], family_type[:n]]
new_feats = pd.concat(dfs, axis=1, ignore_index=False).drop('index', axis=1)
new_feats.columns = map(str.lower, new_feats.columns.tolist())


# In[ ]:


new_feats.columns.unique()


# In[ ]:


predictors = ['fare', 'pclass', 'male', 'cabin_not_null', 
              'master', 'small_family', 'age',
             ]
x = new_feats[predictors]
y = new_feats['survived']
x.head()


# In[ ]:


dt = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=5)
dt = dt.fit(x, y)


# In[ ]:


pd.Series(dict(zip(x.columns, dt.feature_importances_))).rename(
    'feat_importance').sort_values(ascending=True).plot(
    kind='barh', title='Feature Importance Ranking', color='g');


# In[ ]:


scores = cross_val_score(dt, x, y, cv=5, scoring='accuracy', n_jobs=-1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# A white-box model like the one above would yield a decent performance.

# In[ ]:


dot_data = tree.export_graphviz(
    dt, out_file=None, feature_names=predictors,
    filled=True, rounded=True, special_characters=True)
graphviz.Source(dot_data)


# ## Conclusion
# 
# - Being a woman was an important factor to survive the sinking of the Titanic, even more if she was rich.
# - Being a rich male and travelling solo or with their spouse increased the chance of survival compared with other males.
# - Knowing the passengers' cabin seems having some influence.
# - Smaller familiies had a better chance of surviving than larger families.
# - Younger people had better chances as well.
# 
# ## Next Steps
# 
# As you noticed this notebook was  aimed exclusively at understanding the data and obtaining some insights. My next steps are:
# 
# 1. Build a better model to estimate the missing values of Age.
# 1. Digging more into Cabin, Ticket, SibSp, Parch and Embarked to do extra feature engineering.
# 1. Transforming Fare into a better feature.
# 1. Modelling who survived the sinking of the Titanic.
# 
# ## External References
# 
# - [Titanic](https://en.wikipedia.org/wiki/RMS_Titanic)
# - [Sinking of the RMS Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_RMS_Titanic)
# - This Kaggle Kernel is innovative and well written: [Divide and Conquer 0.82297](https://www.kaggle.io/svf/1518354/600ec57f850ab2c2f347caea5465ab87/__results__.html#)
