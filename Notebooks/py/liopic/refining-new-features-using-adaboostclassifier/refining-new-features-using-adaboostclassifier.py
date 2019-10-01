#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# * [1. Introduction](#1.-Introduction)
# * [2. Quick exploration of the given data](#2.-Quick-exploration-of-the-given-data)
# 	* [2.1 Loading the data](#2.1-Loading-the-data)
# 	* [2.2 Data basic description](#2.2-Data-basic-description)
# * [3. Features completion](#3.-Features-completion)
# 	* [3.1 Age](#3.1-Age)
# 	* [3.2 Cabin](#3.2-Cabin)
# 		* [3.2.1 Cabin from name?](#3.2.1-Cabin-from-name?)
# 		* [3.2.2 Cabin from Ticket?](#3.2.2-Cabin-from-Ticket?)
# 	* [3.3 Embarked](#3.3-Embarked)
# 	* [3.4 Fare](#3.4-Fare)
# * [4. Feature detailed analysis and engineering](#4.-Feature-detailed-analysis-and-engineering)
# 	* [4.1 Quantitative features](#4.1-Quantitative-features)
# 		* [4.1.1 Age](#4.1.1-Age)
# 		* [4.1.2 Fare](#4.1.2-Fare)
# 		* [4.1.3 SibSp (Sibblings + Spouses) and Parch (Parents + children)](#4.1.3-SibSp-%28Sibblings-+-Spouses%29-and-Parch-%28Parents-+-children%29)
# 	* [4.2 Categorical features](#4.2-Categorical-features)
# 		* [4.2.1 Sex](#4.2.1-Sex)
# 		* [4.2.2 Embarked](#4.2.2-Embarked)
# 		* [4.2.3 Pclass](#4.2.3-Pclass)
# 	* [4.3 Feature engineering with non-classifiable](#4.3-Feature-engineering-with-non-classifiable)
# 		* [4.3.1 Cabin_letter](#4.3.1-Cabin_letter)
# 		* [4.3.2 Ticket](#4.3.2-Ticket)
# 			* [4.3.2.1 Sharing the Ticket](#4.3.2.1-Sharing-the-Ticket)
# 			* [4.3.2.2 Head of family/friends from Ticket](#4.3.2.2-Head-of-family/friends-from-Ticket)
# 			* [4.3.2.3 Type of Ticket](#4.3.2.3-Type-of-Ticket)
# 		* [4.3.3 Name](#4.3.3-Name)
# 			* [4.3.3.1 Language from Name](#4.3.3.1-Language-from-Name)
# 			* [4.3.3.2 Common surname](#4.3.3.2-Common-surname)
# 			* [4.3.3.3 Title from Name](#4.3.3.3-Title-from-Name)
# 	* [4.4 Exploring which features are really useful](#4.4-Exploring-which-features-are-really-useful)
# * [5 Training a ML model](#5-Training-a-ML-model)
# * [6 Prediction](#6-Prediction)
# 

# # 1. Introduction

# In this notebook we'll work with [the Titanic competition on Kaggle](https://www.kaggle.com/c/titanic).
# 
# First we will explore the given data and complete missing data. Then we will create new features based on the data provided, which is called __feature engineering__. Finally we will predict test cases using __Random Forest__.

# # 2. Quick exploration of the given data

# Before starting we need to load the data and explore it a bit to understand its meaning.

# ## 2.1 Loading the data

# In[1]:


# import the usual libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# We will load both train and test data, and concat them to work on both at the same time. Just notice that the test data has the _Survived_ feature missing.

# In[2]:


train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')
df = pd.concat([train_df, test_df])

df.sample(10)


# You can refer to [its data dictionary](https://www.kaggle.com/c/titanic/data) to know more about these features.

# ## 2.2 Data basic description

# Using pandas __.describe()__ method we can see general statistics for each feature.

# In[3]:


# Have a look at other numerical features
df.describe()


# Joining train and test sets there are 1309 people. There are 891 people in the train set (the ones with Survived data), but only 38% survived. The average age is 29.8 years. The fare has a wide range of prices, from free to 512. Around 38% travelled without parents nor children. The most common Pclass is 3rd. Almost half of the people travelled without sibblings nor spouse.

# In[4]:


# Basic statistics for non-numerical cases
df.describe(include=['O'])


# There is only information about 295 people with cabin. Most people embarked at "S", which is Southampton, UK. And there are more male than female. Finally, some people shared tickets, being 11 the most remarkable case.

# # 3. Features completion

# Let's first see which features need completion.

# In[5]:


df.isnull().sum()


# There are 263 passengers with missing Age and a lot without Cabin. Also we have 2 missing cases in Embarked and 1 in Fare.

# ## 3.1 Age

# Firstly there are some ages that, according with the data dictionary, are actually estimated: those that come in the form xx.5. So let's mark them, and also the ones we will be estimating too.

# In[6]:


is_estimated_or_null = lambda x: pd.isnull(x) or (x>1 and divmod(x, 1)[1] == 0.5)
df['estimated_age'] = df.Age.apply(lambda age: 1 if is_estimated_or_null(age) else 0)


# In order to estimate the missing ages we could guess that young people were not labeled as 1st class. The age can depend on the sex, and perhaps even on the embarkation port.

# In[7]:


# Let's verify the guess grouping
age_grouped = df[['Pclass','Sex','Embarked','Age']].groupby(['Pclass','Sex','Embarked']).median()
age_grouped


# Given that the guess looks quite correct, let's complete the missing Age cases just depending on passenger's class.

# In[8]:


real_age = lambda row: row.Age if not pd.isnull(row.Age) else age_grouped.loc[row.Pclass].loc[row.Sex].loc[row.Embarked].Age
df['Age'] = df[['Pclass','Sex','Embarked','Age']].apply(real_age, axis=1)


# ## 3.2 Cabin 

# There are a lot of missing data in this feature. We could start grouping cabins by its initial letter, which is its vessel's section or deck.

# In[9]:


df['cabin_letter'] = df.Cabin.apply(lambda c: c[0] if not pd.isnull(c) else 'N') # N=none

df.sample(5)


# In[10]:


# Grouping by cabin letter should show us some insights...
survival_ratio = df[['cabin_letter','Pclass','Survived']].groupby(['cabin_letter']).mean()
people_count = df[['cabin_letter','Name']].groupby(['cabin_letter']).count().rename(columns={'Name': 'passenger_count'})

pd.concat([survival_ratio,people_count], axis=1)


# It is clear that different cabins have different survival expectation, due to the situation in the ship. Apparently letters A, B, C, and T are related to 1st class; same for G, related to 3rd class. Other letters have people in different classes.
# 
# As the missing data comes from people in all classes, we can't assign easily neither cabin nor cabin letter.

# ### 3.2.1 Cabin from name?

# If you have a look at the names provided, they are in the form "Surname, Title. Name", and in the cases of a wife "Man_surname, Man_title. Man_name (Woman_name Woman_surname)". We could use this information to find people with same family names and assign the same cabin.

# In[11]:


df['surname'] = df.Name.apply(lambda n: n.split(',')[0])
df.sample(10)


# In[12]:


#Group by surname and class, in order to find people that could be a family
surnames = df[['surname','Cabin','Pclass','Name']].groupby(['surname','Pclass']).count()
surnames.head()


# In[13]:


# Find cases with more people than assigned cabin
missing = surnames[(surnames.Cabin>0) & (surnames.Cabin<surnames.Name)] # Notice the element-wise binary logical operator '&'
missing.rename(columns={'Name': 'passenger_count'})


# We found just 7 cases of (probably) families with missing cabins. Let's have a look at the Brown...

# In[14]:


df[df.surname=='Brown']


# In this case we can't assign any cabin to the passengers 671, 685 and 1067 (the ones without cabin) because they are actually an independent family, as they have the same ticket number and moreover Parch and SibSp numbers made us to suspect so.
# 
# Let's try with the Hoyt...

# In[15]:


df[df.surname=='Hoyt']


# In this case too we can't assign any cabin as the third passenger is clearly not related with the couple.
# 
# We have seen 2 cases but didn't found any useful way to get the cabin from the surname, a pity.

# ### 3.2.2 Cabin from Ticket?

# We could find people without cabin set, but with the ticket id shared with other people who have cabin assigned.

# In[16]:


tickets_grouped = df[['Ticket','Cabin','Name']].groupby('Ticket').count()

# Filter: With at least a Cabin, with at least 2 people, and more people than cabins
candidate_tickets = tickets_grouped[(tickets_grouped['Cabin']>=1) & (tickets_grouped['Name']>=2) & (tickets_grouped['Cabin']<tickets_grouped['Name'])]
candidate_tickets


# Nice, we got some candidates! Let's verify with one of them.

# In[17]:


df[df.Ticket=='113781']


# Great! We can complete some Cabins!

# In[18]:


shared_tickets = candidate_tickets.index.tolist()

find_cabin_given_ticket = lambda ticket: df[(df.Ticket==ticket) & (pd.notnull(df.Cabin))].Cabin.values[0]
def assign_cabin(row):
    if pd.isnull(row.Cabin) and row.Ticket in shared_tickets: 
        return find_cabin_given_ticket(row.Ticket) 
    return row.Cabin

df['Cabin'] = df[['Cabin', 'Ticket']].apply(assign_cabin, axis=1)
df['cabin_letter'] = df['Cabin'].apply(lambda c: c[0] if not pd.isnull(c) else 'N') # N=none

df[df.Ticket=='113781']


# In[19]:


df.Cabin.isnull().sum()


# We started with 1014 passengers without cabin, but at least we completed 16 direct cases.

# ## 3.3 Embarked

# There were 3 ports of embarkation, coded as: C = Cherbourg (France), Q = Queenstown (UK), S = Southampton (UK). Let's see how many passengers embarked in each port.

# In[20]:


df[['Embarked', 'Survived', 'Name', 'Pclass']].groupby('Embarked').agg(
    {'Name': ['count'], 'Pclass': ['mean'], 'Survived': ['mean']})


# Most people embarked at Southampton (914) but we can notice that people that embarked in the continent (C) have more chances of surviving, perhaps due to a higher Pclass.
# 
# An easy way to fill in missing values is using "S", as it's by far the most common case: this is called imputation. As there is just 2 missing cases, this solution will be enough.

# In[21]:


df['Embarked'].fillna('S', inplace=True)


# ## 3.4 Fare

# As there is only one missing value, so let's explore its case directly:

# In[22]:


df[pd.isnull(df.Fare)]


# We can assign a fare given the average fare of similar cases.

# In[23]:


estimated_fare = df[(df.Embarked=='S') & (df.Pclass==3) & (df.Sex=='male')].Fare.mean()
df['Fare'].fillna(estimated_fare, inplace=True)


# # 4. Feature detailed analysis and engineering

# Some quantitative features could be used directly as input of the ML model (expecting there will be some correlation). Other features (categorical and non-classificable ones) will need further process to make them useful.
# 
# * **Quantitative**, that is, numbers that are easy to work with
#     * Continuous: Age, Fare
#     * Discrete: SibSp, Parch
# 
# 
# * **Categorical**, that represent categories, and will need some processing
#     * Nominal: Embarked, Sex
#     * Ordered: Pclass
#     
#  
# * **Non classifiable**, that will need some feature engineering to make them useful
#     * Strings: Cabin, Ticket, Name

# ## 4.1 Quantitative features

# ### 4.1.1 Age

# Let's explore now the relation between survived ratio and age, grouping by decades.

# In[24]:


grouped_ages = df[['Age','Survived']].groupby(by=lambda index: int(df.loc[index]['Age']/10)).mean()
grouped_ages.plot(x='Age', y='Survived')


# In[25]:


#Why the line goes up in 80 years? An outlier?
df[df['Age']>=80]


# In most cases, the older the person the less probability of survival. So it's obvious this feature will be helpful for our ML model.
# 
# However there is a clear outliner, that we will remove from our dataframe to avoid learning from *a bad case*.

# In[26]:


df = df[df['Age']<80]


# We could use age feature directly, or group by decades; as we don't know which one could be better, let's give both to the ML model.
# 
# We will save interesting features' names in a variable called _useful_.

# In[27]:


df['decade'] = df['Age'].apply(lambda age: int(age/10))

# We will save useful features (column names) for later.
useful = ['Age', 'decade']


# ### 4.1.2 Fare

# The first intuition is to think that cheapest tickets will be related with more deaths, but let's plot it.

# In[28]:


# Grouping by 100s
fare_grouped = df[['Fare', 'Survived']].groupby(by=lambda i: int(df.loc[i]['Fare']/100)).mean()
fare_grouped.plot(x='Fare', y='Survived')


# It is clear that cheaper tickets will have less chance of survival. We can use this as input too.

# In[29]:


useful.append('Fare')


# ### 4.1.3 SibSp (Sibblings + Spouses) and Parch (Parents + children)

# Will having family of the similar age (sibblings + spouse) help? Let's find out!

# In[30]:


sibblings_grouped = df[['SibSp', 'Survived']].groupby('SibSp').mean()
sibblings_grouped.plot()


# Being alone or having a big family seems a problem. Let's see the case with parents and children value.

# In[31]:


generations_grouped = df[['Parch', 'Survived']].groupby('Parch').mean()
generations_grouped.plot()


# In this case, again being alone or being in a big family is a problem for your survival, but the correlation is not that clear.
# 
# We could reframe this information with new features: being alone and the size of the family.

# In[32]:


df['family_size'] = df['SibSp'] + df['Parch'] + 1

# Let's see if there is a clear limit
df[['family_size', 'Survived']].groupby('family_size').mean().plot()


# Let's create new features for family sizes.

# In[33]:


df['small_family'] = df['family_size'].apply(lambda size: 1 if size<=4 else 0)
df['big_family'] = df['family_size'].apply(lambda size: 1 if size>=7 else 0)
df['no_family'] = df['family_size'].apply(lambda s: 1 if s==1 else 0)

useful.extend(['SibSp', 'Parch', 'family_size', 'small_family', 'big_family', 'no_family'])


# ## 4.2 Categorical features

# ### 4.2.1 Sex

# Let's see which sex has more chances of surviving.

# In[34]:


survived_sex = df[df['Survived']==1]['Sex'].value_counts()
survived_sex.name='Survived'
dead_sex = df[df['Survived']==0]['Sex'].value_counts()
dead_sex.name='Dead'

table = pd.DataFrame([survived_sex,dead_sex])

table.T.plot(kind='bar', stacked=True, color='gr')


# It is clear that most females survived, while most men died. Let's encode it as values.

# In[35]:


df['male'] = df['Sex'].map({'male': 1, 'female': 0})

useful.append('male')


# ### 4.2.2 Embarked

# As we have seen while completing the missing values of this feature, there is some relation with the survival chances.
# 
# Let's see it again, but splitting by Pclass.

# In[36]:


embarked_grouped = df[['Embarked', 'Pclass', 'Survived']].groupby(['Embarked','Pclass']).mean()
embarked_grouped.plot(kind='barh')


# It's clear than there is differences, for instance with the second class of people embarked in Q. We need to encode this feature in multiple columns (one-hot encoding), that can be done with pandas get_dummies().

# In[37]:


df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='embarked')], axis=1)

useful.extend(['embarked_{}'.format(x) for x in ['C', 'S', 'Q']])

#Let's see how these multiple columns look like
df.sample(5)


# ### 4.2.3 Pclass

# We have seen several times by now that there is a direct correlation between the class and the chance of survival. As this feature is already a number, we do not need to apply any process.

# In[38]:


useful.append('Pclass')


# ## 4.3 Feature engineering with non-classifiable 

# While doing feature completion we have already done some feature engineering that need further process: Cabin_letter (deck) and surname. Let's try out more ideas!

# ### 4.3.1 Cabin_letter

# We just need to encode it in multiple columns.

# In[39]:


df = pd.concat([df, pd.get_dummies(df['cabin_letter'], prefix='deck')], axis=1)

letters = df['cabin_letter'].unique()
useful.extend(['deck_{}'.format(x) for x in letters])


# ### 4.3.2 Ticket

# #### 4.3.2.1 Sharing the Ticket

# From previous exploration we noticed that a ticket can be shared among several passengers. Let's explore it a bit more.

# In[40]:


ticket_count = df[['Ticket', 'Name']].groupby('Ticket').count().rename(columns={'Name':'count'}).sort_values(by='count', ascending=False)
ticket_count.head()


# Several people have shared a ticket in some cases, which means that there were groups of friends or colleages that, despite they don't have family relation, we should consider them as a group. Let's verify if with ticket "CA- 2343" and  "1601".

# In[41]:


df[df['Ticket']=='CA. 2343']


# In[42]:


df[df['Ticket']=='1601']


# So our guess can be true. Ticket "CA. 2343" owners are from the same family. And ticket '1601' owners are people with similar names' origin, but no family indicators (both Parch and SibSp are zeros).
# 
# Wait a moment! If the ticket is shared, we should divide the Fare among the people who share the ticket. Let's add this new feature too.

# In[43]:


df['ticket_owners'] = df['Ticket'].apply(lambda x: ticket_count.loc[x])
df['shared_fare'] = df['Fare'] / df['ticket_owners']

df['alone'] = df[['ticket_owners','no_family']].apply(lambda row: 1 if row.ticket_owners==1 and row.no_family==1 else 0 , axis=1)

useful.extend(['ticket_owners', 'shared_fare', 'alone'])


# #### 4.3.2.2 Head of family/friends from Ticket

# If the people sharing the ticket are family or friends, we could guess that the oldest of family's (the head of family) age can be relevant to the rest of the family/friends. Younger head of family could've helped better the rest of the group, for instance.

# In[44]:


df['ticket_owners'].describe()


# The 50% percentile shows that more than half of passengers traveled without sharing ticket, which is not helping our guess. Let's follow with the calculation anyway.

# In[45]:


older_age = df[['Ticket', 'Age']].groupby('Ticket').max()
df['older_relative_age'] = df['Ticket'].apply(lambda ticket: older_age.loc[ticket])

useful.extend(['older_relative_age'])


# #### 4.3.2.3 Type of Ticket

# It looks like there are different type of tickets: some with letters and numbers, other with just numbers, etc.

# In[46]:


import re


def ticket_type(t):
    if re.match('^\d+$', t):
        return 'len' + str(len(t))
    else:
        return re.sub('[^A-Z]', '', t)


df['ticket_type'] = df['Ticket'].apply(ticket_type)

df[['ticket_type', 'Survived']].groupby(
    'ticket_type').agg({'Survived': ['mean', 'std','count']}).sort_values(('Survived','count'), ascending=False)


# It seems that standard deviation is quite wide in most cases, except in "A" type. We could group clear cases with most occurrences but also a short deviation.

# In[47]:


def useful_ticket_type(ticket_type):
    useful_types = ['A', 'SOTONOQ', 'WC']
    if ticket_type in useful_types:
        return ticket_type
    else:
        return 'other'


df['useful_ticket_type'] = df['ticket_type'].apply(useful_ticket_type)

df = pd.concat(
    [df, pd.get_dummies(df['useful_ticket_type'], prefix='ticket_type')], axis=1)

letters = df['useful_ticket_type'].unique()
useful.extend(['ticket_type_{}'.format(x) for x in letters])

df.sample(10)


# ### 4.3.3 Name

# A trivial feature could be the length of the name. Longer names tend to be from the royalty.

# In[48]:


df['name_length'] = df.Name.apply(len)
df[['name_length', 'Survived']].groupby('name_length').mean().plot()


# There is a subtle correlation up to 35 chars, then a chaos up to 58 chars, and then everybody with really long names survived. Let's group this in 3 cases.

# In[49]:


df['name_length_short'] = df['name_length'].apply(lambda s: 1 if s <= 35 else 0)
df['name_length_mid'] = df['name_length'].apply(lambda s: 1 if 35 < s <=58 else 0)
df['name_length_long'] = df['name_length'].apply(lambda s: 1 if s > 58 else 0)

useful.extend(['name_length', 'name_length_short', 'name_length_mid', 'name_length_long'])


# #### 4.3.3.1 Language from Name

# As Name field is text, we could try to get the language using _langid_ library. Perhaps people who spoke other languages except English had more problems to understand emergency directions.

# In[50]:


import langid

df['lang'] = df['Name'].apply(lambda n: langid.classify(n)[0])
df[['Name','lang']].sample(10)


# Despite _langid_ did not a perfect job with just a few words to work with, we can get an idea about the language that each passenger could have used.
# 
# Let's explore the most common languages.

# In[51]:


lang_count = df[['lang','Name']].groupby('lang').count().rename(columns={'Name':'count'})
lang_class = df[['lang','Pclass']].groupby('lang').mean()
lang_survived = df[['lang','Survived']].groupby('lang').mean()
pd.concat([lang_count, lang_class, lang_survived], axis=1).sort_values(by='count', ascending=False).head(15)


# English(en) and German(de) are the most common cases, with a survived ratio around 38%. Then we have languages that come from latin: French(fr), Spanish(es) and Italian(it) with a better ratio, around 40~50%, surprisingly. Moreover, looking at the mean Pclass, both groups have similar social status.
# 
# Funny enough, there are several cases labelled as Estonian(et), which could be an effect of _langid_ with short strings, but with a really bad survived ratio.
# 
# We may want to make groups, just to help the ML model. We will add some African and Asian languages as groups too, expecting .

# In[52]:


language_groups = {
    'uk': ('cy', 'en'),
    'germanic': ('da', 'de', 'nl'),
    'latin': ('es', 'fr', 'it', 'la', 'pt', 'br', 'ro'),
    'african': ('af', 'rw', 'xh'),
    'asian': ('id', 'tl', 'tr')
}
language_map = { y:x for x in language_groups for y in language_groups[x]}    

df['lang_group'] = df['lang'].apply(lambda l: language_map[l] if l in language_map else 'other')
survived_avg_per_group = df[['lang_group','Survived']].groupby('lang_group').mean()
survived_std_per_group = df[['lang_group','Survived']].groupby('lang_group').std().rename(columns={'Survived':'std'})
pd.concat([survived_avg_per_group, survived_std_per_group], axis=1)


# The standard deviation is too wide, but let's add it to the useful features' list anyway.

# In[53]:


df = pd.concat([df, pd.get_dummies(df['lang_group'], prefix='lang_group')], axis=1)

langs = df['lang_group'].unique()
useful.extend(['lang_group_{}'.format(x) for x in langs])


# #### 4.3.3.2 Common surname

# Let's guess that a common surname will not help the person, while a rare surname it's an indicator for a special person.

# In[54]:


surnames = df[['surname', 'Name']].groupby('surname').count().rename(columns={'Name':'count'})
df['surname_count'] = df['surname'].apply(lambda x: surnames.loc[x])

useful.append('surname_count')


# #### 4.3.3.3 Title from Name

# Given that the Name has the format "Surname, Title. Name", let's extract the titles and group them.

# In[55]:


df['title'] = df['Name'].apply(lambda n: n.split(',')[1].split('.')[0].strip())

df[['title', 'Survived']].groupby('title').agg({'Survived': [
    'mean', 'std', 'count']}).sort_values(('Survived', 'count'), ascending=False)


# Let's group these titles!

# In[56]:


title_groups = {
    "Capt": "sacrifies",
    "Col": "army",
    "Rev": "sacrifies",
    "Major": "army",
    "Mr" : "Mr",
    "Master": "Master",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Ms": "Mrs",
    "Mlle": "Miss"
}

df['title_group'] = df['title'].apply(lambda t: title_groups[t] if t in title_groups else 'other')

df = pd.concat([df, pd.get_dummies(df['title_group'], prefix='title_group')], axis=1)

t_g = df['title_group'].unique()
useful.extend(['title_group_{}'.format(x) for x in t_g])


# ## 4.4 Exploring which features are really useful

# We have created missing values and new features freely. But in order to see which features could be more relevant to our model, we can explore their direct correlation with Survived. We can even train a Random Forest to get some insights too.

# In[57]:


df.corr()['Survived'].sort_values()


# The correlation show us that being titled as Mr. is the biggest problem for a passenger (correlated -0.55). The second feature in importance is male sex (-0.54). Then title Mrs, Pclass, title Miss, etc.
# 
# Let's do a quick training with a Random Forest Classifier to compare feature importance. Firsly we should split out the train set from our complete dataframe.

# In[58]:


train = df[df['Survived'].notnull()]
train_X = train[useful]
train_y = train['Survived']


# We will run a quick try with Random Forest just to get an estimation of feature importance. This is useful to know, when further feature engineering is necesary, which cases are candidates to further analysis and engineering.

# In[59]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf = clf.fit(train_X, train_y)


# In[60]:


importances = pd.DataFrame(clf.feature_importances_, index=train_X.columns, columns=['importance']).sort_values(by='importance')
importances.plot.barh(figsize=(16,8), legend=None, title='Feature importance')


# Some of our synthetic features are preforming well, and the same happens with Fare, Age and Pclass. We can explore the relations among top features, in order to see which ones can be candidates for refinement.

# In[61]:


top_features = importances.tail(8).index.tolist()
top_features.append('Survived')
top_correlations = df[top_features].corr()

sns.heatmap(top_correlations, annot=True)


# Clearly male and title_group_Mr are strongly correlated (0.87), so we could remove the second one. Same happens with shared_fare and Fare, and with Age and older_relative_age.
# 
# Let's choose only the most relevant features.

# In[62]:


train_X.shape


# Given 890 passengers with 48 features in the train set...

# In[63]:


useful = importances.tail(33).index.tolist()
useful.remove('title_group_Mr')
useful.remove('Fare')
useful.remove('older_relative_age')
train_X=train[useful]
train_X.shape


# We end with 30 useful features.

# # 5 Training a ML model

# Now let's do a extend search for the best hyperparams that we will use in Random Forest. With GridSearchCV() we can try out different hyperparams for our model so automatically it will find the best combination. This will take a bit.

# In[64]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

# Set to True or False to search all combinations or use previous results
search_best_hyperparameters = False

if search_best_hyperparameters:
    parameter_grid = {
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'learning_rate': [0.1, 0.2, 0.5, 1, 1.2],
        'random_state': [1]
    }
    model = AdaBoostClassifier()
    gs = GridSearchCV(
        model,
        scoring='accuracy',
        param_grid=parameter_grid,
        cv=4,
        n_jobs=-1)
    gs.fit(train_X, train_y)
    params = gs.best_params_
    print(params)
else:
    params = {
        'learning_rate': 0.1,
        'n_estimators': 500,
        'random_state': 1
    }


# In[65]:


# Use the params to get a score with the training set
clf = AdaBoostClassifier(**params)
clf = clf.fit(train_X, train_y)
clf.score(train_X, train_y)


# # 6 Prediction

# Finally, let's use the trained classifier to get predictions for the test dataset.

# In[66]:


test = df[df['Survived'].isnull()]
test_X = test[useful]
test_y = clf.predict(test_X)


# In[67]:


submit = pd.DataFrame(test_y.astype(int), index=test_X.index, columns=['Survived'])
submit.head()


# In[68]:


submit.to_csv('submission.csv')


# This submission scored quite well in the competition. As a next step, a deep analysis on outliers could be done.
# 
# **Thank you** for reading!
