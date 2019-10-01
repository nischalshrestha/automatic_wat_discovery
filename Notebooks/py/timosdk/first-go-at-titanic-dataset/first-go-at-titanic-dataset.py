#!/usr/bin/env python
# coding: utf-8

# # Analysis of the Titanic data set with Python
# 
# After completing the excellent intro to machine learning course on Udacity (https://classroom.udacity.com/courses/ud120) I decided to test the teachings on some introductory datasets. This Titanic dataset contains a lot of interesting data - but can get quite morbid as we are trying to predict who lives and who dies. 
# 
# The following feature engineering borrows heavily from many of the other great kernels on this data set and I try to reference througout whenever I borrow something directly - but I also try to advance and go beyond the work in these other kernels. 
# 
# # Python Setup
# 
# **Load the libraries and the data**
# 
# Load the libraries that we need and the data that we need.
# 
# I smash together the training and the test dataset into one big frame so I only have to do the transformations and feature engineering once. This also gets around the issue of creating training and test set specific means and averages - when really both datasets should be transformed in the same way.

# In[ ]:


import numpy as np 
import pandas as pd 
import re
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import svm, neighbors, naive_bayes
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation

# plotting setup
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format='retina'")
plt.rcParams['figure.figsize'] = [14.0, 6.0]

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# training data
data_train = pd.read_csv("../input/train.csv")

# test data
data_test = pd.read_csv("../input/test.csv")

# combine test and train into one dataframe - can always split again by ['Survived']
data_full = pd.concat([data_train, data_test])
print('Full dataset columns with null values: \n', data_full.isnull().sum())

# show us some representative data.
data_full.sample(5)


# # Analyze the data and features
# 
# Data and feature analysis will be performed on the trainig data set when we need to see patterns in survival since survival data is missing in the test set. For general range analysis of the data I use the full dataset.
# 
# ### Quick analysis of data
# 
# To see what we are up against here are the probablities of surviving in general as well as split out for men and women. Based on this it seems prudent that the first engineered feature should hold information about gender.

# In[ ]:


print("Chance of surviving in training set:", data_train['Survived'].sum()/len(data_train['Survived']))
print("Chance of surviving if you are a male:",data_train.loc[data_train['Sex'] == 'male']['Survived'].sum()/data_train['Sex'].value_counts()['male'])
print("Chance of surviving if you are a female:",data_train.loc[data_train['Sex'] == 'female']['Survived'].sum()/data_train['Sex'].value_counts()['female'])

data_full['IsFemale'] = 1
data_full.loc[data_full['Sex'] == 'male', 'IsFemale'] = 0


# ### Name length
# 
# Crazy as it sounds name length seems to correlate with survival. This is partially due to women being listed as Mrs. Husband with their names in parentheses. And we know that women had better chances of survival.

# In[ ]:


data_full["NameLength"] = data_full["Name"].apply(lambda x: len(x))

fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(data_full.loc[(data_train['Survived'] == 0), 'NameLength'], color='gray',shade=True,label='dead', bw=3)
ax=sns.kdeplot(data_full.loc[(data_train['Survived'] == 1), 'NameLength'], color='g',shade=True,label='alive', bw=3)
plt.title('Name Length Distribution - Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Name Length", fontsize = 15)

print("Longest Name:", data_full.loc[data_full['NameLength']==data_full['NameLength'].max()].Name.values[0])


# ### Titles
# 
# Some people have used titles to get information about social status. There is some information encoded in the title for sure. Probably not much for Mr - the only information there is that you are a male and maybe lower social status. For females there is Miss and Mrs - this encodes marital status so this offers information above just gender. Then there is the long tail of other titles. The ones that occur only twice or less it seems pointless to try and train on those titles. However, it does look like Master, Dr, and Rev have interesting survival distributions so these titles are probably valuable for creating predictions. 

# In[ ]:


test_titles = data_train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print(test_titles.value_counts())

print("-"*10)

# The most common ones are tied to gender and age and the less common ones are so uncommon that they make 
# little sense to encode - could encode as common vs fancy - but likely the fancy ones are also in high
# class and or expensive ticket.

title = 'Mr'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Miss'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Mrs'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Master'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Dr'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Rev'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

print("-"*10)



# We will encode the titles in the full dataset and just keep the 6 most frequent tiltles - the rest will be named fancy.  Titles will be used for imputing age later.

# In[ ]:


# get title from name
data_full['Titles'] = data_full['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

# fix these simple ones
data_full['Titles'] = data_full['Titles'].replace('Mlle', 'Miss')
data_full['Titles'] = data_full['Titles'].replace('Ms', 'Miss')
data_full['Titles'] = data_full['Titles'].replace('Mme', 'Mrs')

# get a list of the 6 most frequent titles
mostfrequenttitles = data_full['Titles'].value_counts().nlargest(6).keys()

# if your title is not in the top 6 you are a fancy person
data_full.loc[(data_full['Titles'].isin(mostfrequenttitles)==False), 'Titles'] = "Fancy"

# here is the value counts for the full dataset
print ("Title frequencies\n", data_full['Titles'].value_counts())


# In[ ]:


# create dummies from titles
dummies = pd.get_dummies(data_full['Titles'])
data_full = pd.concat([data_full, dummies], axis = 1)

data_full.sample(5)


# ### Age
# 
# From the age vs survival kde plot below it is looks like having age bins that are around 15 years in size would adequately divide the various ages into bins that have different chances of survival. 

# In[ ]:


fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 0) & (data_train['Age'].isnull() == False), 'Age'], color='gray',shade=True,label='dead', bw=3)
ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 1) & (data_train['Age'].isnull() == False), 'Age'], color='g',shade=True, label='survived', bw=3)
ax=sns.kdeplot(data_full.loc[(data_train['Age'].isnull() == False), 'Age'], color='b',shade=False, label='full dataset', bw=3)
plt.title('Age Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Age", fontsize = 15)


# People have been creating bins to avoid having age as a continuous variable and instead have it as a discrete variable. 
# 
# However, people have been using pd.cut for this and doing the cuts on the respective datasets. This would be fine if the max and min ages were the same for the test and trainig sets - but they are not. As can be seen below the cut creates different boundaries for these two datasets.
# 
# For my implementation I force a min and a max by inserting these at the end before I start pd.cutting. I also use labels so we can get these age bins as int values in one go. I set max age to 120 and use a 8 total bins to create 15 year bins.
# 
# Interestingly it looks like the main effect of age is in combination with gender - since young males are the only ones that appear to do better than the general average of their gender. 

# In[ ]:


# For demonstration purposes only - cutting on the separated set gives rise to different and nonsensical cutoffs:
pd.cut(data_train['Age'], bins = 5)
#[(0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] < (48.168, 64.084] < (64.084, 80.0]]
pd.cut(data_test['Age'], bins = 5)
#[(0.0942, 15.336] < (15.336, 30.502] < (30.502, 45.668] < (45.668, 60.834] < (60.834, 76.0]]

# Now we will create age bins for the full dataset
ages = data_full['Age']
ages = ages.append(pd.Series([0, 80]))
bins = pd.cut(ages, bins = 8, labels = [1, 2, 3, 4, 5, 6, 7, 8])

data_full['AgeBin'] = bins[:-2].astype(float)

fig = plt.figure(figsize=(15,8))
sns.pointplot(x='AgeBin', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True)


# From all this it looks like it would be good to create a dummy for kids - since agebin does not really provide a whole lot of differentiation except in the agebin=1 category

# In[ ]:


data_full['IsKid'] = 0
data_full.loc[data_full['AgeBin'] == 1, 'IsKid'] = 1


# ** Dealing with missing values in the age feature **
# 
# Age is the feature that has the most missing values after the cabin feature. In this case I want to fill the agebin with the mode of the agebin based on the title of the person. This especially seems to make sense for Master/Mr and Mrs/Miss where the title allows us to extract information about age based instead of just picking at random.

# In[ ]:


# iterate through titles and fill in missing values
titles = data_full['Titles'].value_counts().keys()
for title in titles:
    age_mode = data_full.loc[data_full['Titles']==title, 'AgeBin'].mode().values[0]
    data_full.loc[(data_full['Titles']==title) & (data_full['AgeBin'].isnull()), 'AgeBin'] = age_mode

# now convert agebin to int
data_full['AgeBin'] = data_full['AgeBin'].astype(int)

print('Full dataset columns with null values: \n', data_full.isnull().sum())


# ### Family
# 
# The creation of family size variable based on sibsp & parch has been discussed elsewhere. I will use the same methods and also create an is alone feature.

# In[ ]:


# Family size
data_full['FamilySize'] = data_full['SibSp'] + data_full['Parch'] + 1

# Is this person alone on the ship
data_full['IsAlone'] = 0
data_full.loc[data_full['FamilySize'] == 1, 'IsAlone'] = 1


# ### Fare
# 
# First thing is to fill in the missing value for Mr. Storey. Will do this based on median fare for people travelling on the same class as Mr. Storey (3rd class).

# In[ ]:


# the guy is on 3rd class so lets use the median fare of 3rd class to fill this value
data_full.loc[data_full['Fare'].isnull(), 'Fare'] = data_full.loc[data_full['Pclass'] == 3, 'Fare'].median()

data_full.loc[data_full['Ticket']=='3701']


# Again we will create fare bins based on the data. Interestingly there is a huge number of unique fares for this dataset - it seems like prices were more fluid back in the day!

# In[ ]:


print("Number of unique fares:", data_full['Fare'].nunique())

plt.figure(figsize=[15,6])

plt.subplot(121)
plt.boxplot(x=data_full['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(122)
plt.hist(x = data_full['Fare'], color = ['g'], bins = 8)
plt.title('Fare Histogram')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers - log scale')
plt.yscale('log', nonposy='clip')


# In[ ]:


# add fare bins to the full dataset
data_full['FareBin'] = pd.qcut(data_full['Fare'], 6, labels = [1, 2, 3, 4, 5, 6]).astype(int)

fig = plt.figure(figsize=(15,8))
sns.pointplot(x='FareBin', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True)


# ### Tickets
# 
# We will look at tickets and the survival of people booked on the same tickets. There is a grave danger of over fitting here - if we look at dead men and alive females - these are the most common cases - we can easily overfit on the data. So instead we will concentrate on the less common cases and see how these play out: dead women and kids and alive males. 
# 
# This analysis is heavily borrowed from: https://www.kaggle.com/francksylla/titanic-machine-learning-from-disaster/code
# 
# The graphs below visualize what happens here:
# * Dead female on ticket - then all the men on ticket dies also
# * Man survives - then all females survive
# * Kid dies - both men and women on ticket die

# In[ ]:


# create table with counts of people per ticket
ticket_table = pd.DataFrame(data_full["Ticket"].value_counts())
ticket_table.rename(columns={'Ticket': 'People_on_ticket'}, inplace = True)

ticket_table['Dead_female_on_ticket'] = data_full.Ticket[(data_full.AgeBin > 1) & (data_full.Survived < 1) & (data_full.IsFemale)].value_counts()
ticket_table['Dead_female_on_ticket'].fillna(0, inplace=True)
ticket_table.loc[ticket_table['Dead_female_on_ticket'] > 0, 'Dead_female_on_ticket'] = 1
ticket_table['Dead_female_on_ticket'] = ticket_table['Dead_female_on_ticket'].astype(int)

ticket_table['Dead_kid_on_ticket'] = data_full.Ticket[(data_full.AgeBin == 1) & (data_full.Survived < 1)].value_counts()
ticket_table['Dead_kid_on_ticket'].fillna(0, inplace=True)
ticket_table.loc[ticket_table['Dead_kid_on_ticket'] > 0, 'Dead_kid_on_ticket'] = 1
ticket_table['Dead_kid_on_ticket'] = ticket_table['Dead_kid_on_ticket'].astype(int)

ticket_table['Alive_male_on_ticket'] = data_full.Ticket[(data_full.AgeBin > 1) & (data_full.Survived > 0) & (data_full.IsFemale == False)].value_counts()
ticket_table['Alive_male_on_ticket'].fillna(0, inplace=True)
ticket_table.loc[ticket_table['Alive_male_on_ticket'] > 0, 'Alive_male_on_ticket'] = 1
ticket_table['Alive_male_on_ticket'] = ticket_table['Alive_male_on_ticket'].astype(int)

# unique identifiers for tickets with more than 2 people
ticket_table["Ticket_id"]= pd.Categorical(ticket_table.index).codes
ticket_table.loc[ticket_table["People_on_ticket"] < 3, 'Ticket_id' ] = -1

# merge with the data_full
data_full = pd.merge(data_full, ticket_table, left_on="Ticket",right_index=True,how='left', sort=False)


# In[ ]:


fig, (maxis1, maxis2, maxis3) = plt.subplots(1, 3,figsize=(15,6))
sns.pointplot(x='Dead_female_on_ticket', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis1)
sns.pointplot(x='Alive_male_on_ticket', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis2)
sns.pointplot(x='Dead_kid_on_ticket', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis3)


# ### Last names
# Similar to tickets - last names can be used to group people. For this one we want to make sure that we only count dead or alive if this person has family size > 1.
# 
# Visualize: This is similar to tickets - but since it is possible for two Andersons and Smiths to not be related the correlations here are not as strong as they were for people on the same tickets.

# In[ ]:


data_full['Lastname'] = data_full["Name"].apply(lambda x: x.split(',')[0].lower())

lastname_table = pd.DataFrame(data_full["Lastname"].value_counts())
lastname_table.rename(columns={'Lastname': 'People_w_lastname'}, inplace = True)

lastname_table['Dead_mom_w_lastname'] = data_full.Lastname[(data_full.AgeBin > 1) & (data_full.Survived < 1) & (data_full.FamilySize > 1) & (data_full.IsFemale)].value_counts()
lastname_table['Dead_mom_w_lastname'].fillna(0, inplace=True)
lastname_table.loc[lastname_table['Dead_mom_w_lastname'] > 0, 'Dead_mom_w_lastname'] = 1
lastname_table['Dead_mom_w_lastname'] = lastname_table['Dead_mom_w_lastname'].astype(int)

lastname_table['Dead_kid_w_lastname'] = data_full.Lastname[(data_full.AgeBin == 1) & (data_full.Survived < 1) & (data_full.FamilySize > 1)].value_counts()
lastname_table['Dead_kid_w_lastname'].fillna(0, inplace=True)
lastname_table.loc[lastname_table['Dead_kid_w_lastname'] > 0, 'Dead_kid_w_lastname'] = 1
lastname_table['Dead_kid_w_lastname'] = lastname_table['Dead_kid_w_lastname'].astype(int)

lastname_table['Alive_dad_w_lastname'] = data_full.Lastname[(data_full.AgeBin > 1) & (data_full.Survived > 0) & (data_full.IsFemale==False) & (data_full.FamilySize > 1)].value_counts()
lastname_table['Alive_dad_w_lastname'].fillna(0, inplace=True)
lastname_table.loc[lastname_table['Alive_dad_w_lastname'] > 0, 'Alive_dad_w_lastname'] = 1
lastname_table['Alive_dad_w_lastname'] = lastname_table['Alive_dad_w_lastname'].astype(int)

# unique identifiers for lastname with more than 2 people
lastname_table["Lastname_id"]= pd.Categorical(lastname_table.index).codes
lastname_table.loc[lastname_table["People_w_lastname"] < 3, 'Lastname_id' ] = -1

# merge with the data_full table
data_full = pd.merge(data_full, lastname_table, left_on="Lastname",right_index=True,how='left', sort=False)


# In[ ]:


fig, (maxis1, maxis2, maxis3) = plt.subplots(1, 3,figsize=(15,6))
sns.pointplot(x='Dead_mom_w_lastname', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis1)
sns.pointplot(x='Alive_dad_w_lastname', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis2)
sns.pointplot(x='Dead_kid_w_lastname', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis3)


# ### Embarkment
# 
# This visualization was borrowed from https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy. However, in that kernel the author did not specify the hue_order and was suppressing warnings so the middle plot endded up with male and female colors reversed. With the hue_color specified it is clear that men do not do well regardless of their port of origin. However, the patterns are different based on port so there is some interesting information here!

# In[ ]:


e = sns.FacetGrid(data_train, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep', order = [1, 2, 3], hue_order = ['male', 'female'])
e.add_legend()

# fill embared with mode
data_full['Embarked'] = data_full['Embarked'].fillna(data_full['Embarked'].mode().values[0])

# add dummies for the embarked as there is no linear relationship here
dummies = pd.get_dummies(data_full['Embarked'], prefix = 'embrk')
data_full = pd.concat([data_full, dummies], axis = 1)


# ### Decks and cabins
# 
# The Cabin feature seems like it may have some interesting information - like maybe people on certain decks could get to the lifeboats faster or there may be some social class information encoded in the Cabin feature. Cabin consist of a letter followed by a number - on in the case of the $$500 ticket consist of 3 letter/number separated by space. The letter is presumably the deck and the number the cabin number on that deck. Also, there is a large proportion of people where we have no cabin information.
# 
# Interestingly it does not look like the deck significanlty affects survival - but the survival rate of people with an assigned cabin certainly is better than the average - so we will just encode a 'has cabin' feature.

# In[ ]:


data_full['Deck'] = data_full['Cabin'].str[:1]

# setup has cabin
data_full['HasCabin'] = 1
data_full.loc[data_full['Cabin'].isnull(), 'HasCabin'] = 0

fig = plt.figure(figsize=(15,8))
sns.pointplot(x='Deck', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])

# transform decks to integers
data_full['Deck'] = pd.Categorical(data_full['Deck'].fillna('N')).codes


# ### Pclass
# 
# We will just make dummies from this. There is relationship between class and social status but usually 1st class is not just one better than 2nd class - they are usually in totally different stratospheres - so I do not want to force a linear relationship on this data.

# In[ ]:


# create dummies from Pclass

dummies = pd.get_dummies(data_full['Pclass'], prefix = 'class')
data_full = pd.concat([data_full, dummies], axis = 1)
data_full.sample(10)


# # Prepare data for machine learning
# 
# ### Get the NumPy arrays
# 
# Now we are ready to extract the np arrays from the data frames. First we need to select what deatures to use for the modeling and then we are going to grab features from train and test data and setup our labels.
# 
# I end up dropping most of the lastname features - I found these to lead to overfitting and not be as informative as wether people were travelling on the same ticket. 

# In[ ]:


# get rid of superfluous columns
final_full_fram = data_full.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 
                                  'Embarked', 'Titles', 'Cabin',
                                  'Fare', 'Age', 'Ticket_id', 'Lastname',
                                  'Lastname_id', 'People_w_lastname', 
                                  'Alive_dad_w_lastname', 
                                  'Rev', 'People_on_ticket', 'Fancy',
                                  'Dr', 
                                 ], axis = 1)


# Split the data back to test and train
data_train1 = final_full_fram[final_full_fram['Survived'].isnull() == False]
print('Train columns (', data_train1.shape[1], ') with null values:\n', data_train1.isnull().sum())

print(" -"*20)

data_test1 = final_full_fram[final_full_fram['Survived'].isnull() == True]
print('Train columns (', data_test1.shape[1], ') with null values:\n', data_test1.isnull().sum())


# In[ ]:


# make sure we only have numerical values in our dataframe
#data_test1.info()

#check ranges
data_test1.describe()


# In[ ]:


features = data_train1.columns
features = features.drop(['PassengerId', 'Survived'])
print ("These are the features we will use for modeling:\n", features)

# create the np arrays that we need for training and testing
np_train_features = data_train1.as_matrix(columns = features)
print ("training features shape", np_train_features.shape)

np_train_labels = data_train1['Survived']
print ("training labels shape", np_train_labels.shape)

np_test_features = data_test1.as_matrix(columns = features)
print ("testing features shape", np_test_features.shape)


# ### Feature scaling
# 
# It is always good to do feature scaling. This is not super necessary in this case since most of these are dummies or have low ranges (we binned the fares and the ages) 

# In[ ]:


# fit scaler on full dataset
scaler = StandardScaler()
scaler.fit(np.concatenate((np_train_features, np_test_features), axis=0))

print(scaler.mean_)

np_train_features = scaler.transform(np_train_features)
np_test_features  = scaler.transform(np_test_features) 


# ### Check on feature importances
# 
# For this I am getting the p-values for a general f_classifier using SelectKBest. The list of features that we drop earlier is based on p-values above 0.05 and also the check of feature importances done for the random forest regressor.

# In[ ]:


selector = SelectKBest(f_classif, k=len(features))
selector.fit(data_train1[features], np_train_labels)

scores = selector.pvalues_
indices = np.argsort(scores)[::1]
print("Features p-values :")
for f in range(len(scores)):
    print("%.3e %s" % (scores[indices[f]],features[indices[f]]))


# # Learning using cross validation and grid search
# 
# I use GridSearchCV to optimize hyperparameters. I use the default 3 fold cross validation (the CV part of GridSearchCV) since on the training data set (891 samples) that will lead to around 300 samples per fold - so it will train on 600 samples and test on 300. 
# 
# +---test---+----------+----------+
# 
# +----------+---test---+----------+
# 
# +----------+----------+---test---+
# 
# ## Support Vector Machine
# 
# First algorithm is support vector machine. 
# 

# In[ ]:


svc = svm.SVC()
parameters = {'kernel': ['linear', 'rbf'],
              'C':[1, 2, 4, 8]}

clf_svm = model_selection.GridSearchCV(svc, parameters, n_jobs = 2)
clf_svm.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_svm.best_score_, clf_svm.best_params_))


# ## Random Forest
# 
# Next we will try a random forest classifier. This one will generate slightly different scores on consecutive runs due to the random part of the random forest. 
# 
# Even the optimal hyperparameters are sightly different on consecutive runs.

# In[ ]:


rf_regr = RandomForestClassifier()
parameters = {"min_samples_split" :[4]
            ,"n_estimators" : [50, 100]
            ,"criterion": ('gini','entropy')
             }

clf_rf = model_selection.GridSearchCV(rf_regr, parameters, n_jobs = 2)
clf_rf.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_rf.best_score_, clf_rf.best_params_))


# ## eXtreme Gradient Boosting
# 
# XGBoost is a fast and gives quite good results. http://xgboost.readthedocs.io/en/latest/model.html
# 

# In[ ]:


xgb = XGBClassifier()
parameters = {'learning_rate': [0.05, 0.1, .25, 0.5], 
              'max_depth': [1,2,4,8], 
              'n_estimators': [50, 100]
             }

clf_xgb = model_selection.GridSearchCV(xgb, parameters, n_jobs = 2)
clf_xgb.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_xgb.best_score_, clf_xgb.best_params_))


# ## Naive Bayes
# 
# For some reason this one gives real poor results.

# In[ ]:


nb = naive_bayes.GaussianNB()
parameters = {'priors': [None]}

clf_nb = model_selection.GridSearchCV(nb, parameters)
clf_nb.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_nb.best_score_, clf_nb.best_params_))


# ## K-Nearest Neighbors
# 
# This is a fast algorithm especially on a small dataset like this. 

# In[ ]:


knn = neighbors.KNeighborsClassifier()
parameters = {'n_neighbors': [1,2,3,4,5,6,7],
              'weights': ['uniform', 'distance'], 
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
             }

clf_knn = model_selection.GridSearchCV(knn, parameters, n_jobs = 2)
clf_knn.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_knn.best_score_, clf_knn.best_params_))


# ## Gradient Boosting
# Another algorithm

# In[ ]:


gbk = GradientBoostingClassifier()

parameters = {
            'learning_rate': [0.05, 0.1],
            'n_estimators': [50, 100], 
            'max_depth': [2,3,4,5]   
             }

clf_gbk = model_selection.GridSearchCV(gbk, parameters, n_jobs = 2)
clf_gbk.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_gbk.best_score_, clf_gbk.best_params_))


# ## Another random forest
# 
# This time we will run with class weights - we know that 1502 out of 2224 died - so survived weight should be 0.325 and dead weight should be 0.675.
# 

# In[ ]:


rfc = RandomForestClassifier(n_estimators=50, min_samples_split=4, class_weight={0:0.675,1:0.325})

# for this one we will use kfold validation since we already have hyper parameters specified
kf = cross_validation.KFold(np_train_labels.shape[0], n_folds=3, random_state=42)

scores = cross_validation.cross_val_score(rfc, np_train_features, np_train_labels, cv=kf)
print("Accuracy on 3-fold XV: %0.3f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))

rfc.fit(np_train_features, np_train_labels)
score = rfc.score(np_train_features, np_train_labels)
print("Accuracy on full set: %0.3f" % (score*100))

print(" *"*15)
print("Feature importances in this model:")

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features)):
    print("%0.2f%% %s" % (importances[indices[f]]*100, features[indices[f]]))



# ## Note on Accuracy
# 
# Whoooo those are some nice accuracies! 98% accuracy from XGB! We nailed it. 
# 
# But not so fast - since we have coded in dead and alive males, females, and kids we have coded the labels into the features! Thus the within sample accuracy on crossvalidation is nolonger going to be reflective of out of sample accuracy - the test data will not perform as well since these features will only help people that are on tickets are also present in the training set. 
# 
# By definition the training set people are all in the training set so if we really wanted to calculate accuracy for the algorithms we would need to partition the training set _before_ feature engineering and make sure that the partition of tickets and lastnames matches the partition seen in the train/test set. This could be done with an elaborate pipeline if someone is up for it.
# 
# Also -  we are most likely in overfitting territory here...

# # Submission
# 
# In this example it looked like our support vector machine was doing the best job on the cross validated data so that is what we will use for the first submission.

# In[ ]:


data_test1.loc[:, 'Survived-SVM'] = clf_svm.best_estimator_.predict(np_test_features)
print("SVM predicted number of survivors:", data_test1['Survived-SVM'].sum())

data_test1.loc[:, 'Survived-RF'] = clf_rf.best_estimator_.predict(np_test_features)
print("RF predicted number of survivors:", data_test1['Survived-RF'].sum())

data_test1.loc[:, 'Survived-RFC'] = rfc.predict(np_test_features)
print("RFC predicted number of survivors:", data_test1['Survived-RFC'].sum())

data_test1.loc[:, 'Survived-XGB'] = clf_xgb.best_estimator_.predict(np_test_features)
print("XGB predicted number of survivors:", data_test1['Survived-XGB'].sum())

# use the RFC data
data_test1.loc[:, 'Survived'] = data_test1['Survived-RFC'].astype(int)

submit = data_test1[['PassengerId', 'Survived']]
submit.to_csv("submit.csv", index=False)


# ## Future work
# 
# We have some nice models here and some nice results - it may be possible to do some voting on these results to get the best prediction.

# In[ ]:


data_test1[['Survived-SVM', 'Survived-RF', 'Survived-RFC', 'Survived-XGB', 'Survived']]


# In[ ]:




