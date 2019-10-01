#!/usr/bin/env python
# coding: utf-8

# # Introduction
# I have created this kernel for beginners. However I am also a beginner, so it has been a way of learning for me. I would really appreciate any expert comments. Many features are defined (most based on other kernels I studied), but not all of them are used. I tried to keep explanations concise.
# 
# Some kernels I studied: 
# - [Erik Bruin's kernel](https://www.kaggle.com/erikbruin/titanic-2nd-degree-families-and-majority-voting)
# - [Konstantin's kernel](https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)
# - [Manav Sehgal's kernel](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

# In[ ]:


# Settings
show_graphs = True
add_interactions = False
model_tuning = False
feature_selection = False 


# In[ ]:


# The features with 1 will be used as predictors directly in the models.  

used_features = {
    'PassengerId': 0,
    'Pclass': 1,
    'Name': 0,
    'LastName': 0,
    'Title': 0,                 
    'Sex': 1,
    'Sex-female x Pclass-1-2': 1,
    'Sex-male x Pclass-3': 0,
    'SibSp': 0, 
    'Parch': 0,
    'FamilySize': 0,
    'FamilySizeBin': 0,
    'IsAloneF': 0,
    'Age': 0,
    'AgeBin': 1,
    'IsChild': 0,
    'IsChild x Pclass-1-2': 1,
    'Cabin': 0, 
    'HasCabin': 0,
    'CabinType': 0,
    'Embarked': 0,
    'Ticket': 0,
    'TicketSize': 1,
    'TicketSizeBin': 0,
    'IsAloneT': 1,
    'Fare': 0,
    'FareOrig': 0,
    'FareBin': 1,
    'NameFareSize': 0,
    'Group': 0,
    'GroupSurvived': 1,
    'GroupSize': 0
}


# In[ ]:


import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

# Utils
import os
import scipy
from itertools import compress
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA


# In[ ]:


# Functions
def gridgraph(df, x, col=None, row=None, hue=None, fun=sns.distplot, **fun_kwargs):
    grid = sns.FacetGrid(df, col=col, row=row, hue=hue, size=4)
    grid = grid.map(fun, x, **fun_kwargs)
    grid.add_legend()
    if fun_kwargs['kde']:
        for ax in grid.axes.flat:
            drawmedian(ax)
    
def catgroup(df, catcol, target, fun="mean"):
    print(df.groupby(catcol, as_index=False)[target].agg(fun))
    
def drawmedian(ax):
    for line in ax.get_lines():
        x, y = line.get_data()
        cdf = scipy.integrate.cumtrapz(y, x, initial=0)
        middle = np.abs(cdf-0.5).argmin()

        median_x = x[middle]
        median_y = y[middle]

        ax.vlines(median_x, 0, median_y)


# # Read data
# Here we read the input files, and concatenate them to have the full data in one dataframe. This will make data cleaning tasks easier. If we need to use the target, we will filter to the training part.

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_valid = pd.read_csv('../input/test.csv')
yt = df_train['Survived']

# Create unified data
df_train['Data'] = 'T'
df_valid['Data'] = 'V'
df_full = pd.concat([df_train, df_valid], sort=False, ignore_index=True) 
mask_train = df_full['Data'] == 'T' 
mask_valid = df_full['Data'] == 'V' 


# # Data cleaning and EDA
# Here I will look at each feature individually, and do the following:
# - Analyze: Check stats, correlate to target, visualize, etc.
# - Correct: Fix data errors, outliers, check if values are reasonable, handle missing values. 
# - Derive: Do feature engineering. 
# - Convert: Use correct datatypes, data format (e. g. dummies), do necessary transformations (e. g. scaling).

# ### Overall stats
# Here we look at overall statistics as a starting point.

# In[ ]:


# Check datatypes, missing values
df_full.info()


# In[ ]:


# Check stats of the columns.
df_full.describe(include='all')


# We have seen that
# - Age, Fare, Embarked, and Cabin have missing values.
# - There are 0 values in Fare, which is strange.

# ### PassengerId
# Not really useful as predictor, because it is just an ID.

# ### Pclass
# We check the average rate of survival in each class.

# In[ ]:


catgroup(df_full.loc[mask_train], 'Pclass', 'Survived')


# In[ ]:


if show_graphs:
    sns.barplot(x='Pclass', y='Survived', order=[1,2,3], data=df_full[mask_train], palette='colorblind')


# We see that the survival rate is much higher if class value is lower. This feature can be useful in the model. 

# ### Name
# Name contains many information. We can extract titles and last names, and form groups based on them. So we will use Name to derive new features.

# ### LastName (derived from Name)
# We might be able to find families based on last name, provided we can differentiate between families with the same name. We will go deeper into this later. 

# In[ ]:


df_full['LastName'] = df_full['Name'].str.extract('^([^,]+),', expand=False)


# ### Title (derived from Name)
# Title can tell us about sex (Mr./Mrs.), age (Master is child, Miss is young), class (Countess, Lady, etc. probably have 1st class ticket), so it can be useful.
# 
# There is some noise in this feature, in the form of rare titles. We group these first.

# In[ ]:


df_full['Title'] = df_full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Change rare titles to more common categories
dict_replace = {
    "Capt": "Officer",
    "Col": "Officer",
    "Countess": "Noble",
    "Don": "Noble",
    "Dona": "Noble",
    "Dr": "Noble",
    "Jonkheer": "Noble",
    "Lady": "Noble",
    "Major": "Officer",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    "Rev": "Noble",
    "Sir": "Noble",
}

df_full['Title'] = df_full['Title'].replace(dict_replace)


# Title may not be that useful as predictor, because We already have the information it would give in columns Age, Sex, and Pclass.

# ### Sex
# Sex is probably the most useful feature, see below.

# In[ ]:


catgroup(df_full.loc[mask_train], 'Sex', 'Survived')


# We convert this to integers.

# In[ ]:


df_full['Sex'] = df_full['Sex'].map({'male': 0, 'female': 1})


# We see that females have a lot more chance to survive, so Sex is very predictive.

# ### Sex-female x Pclass-1-2 (derived from Sex and Pclass)
# We can check the rate of survival in groups formed by Pclass and Sex categories.

# In[ ]:


if show_graphs:
    gridgraph(df_full[mask_train], 'Survived', col='Pclass', row='Sex', bins=2, kde=False)


# We see that in 1st and 2nd class, Sex is much more predictive, so we create an interaction feature.

# In[ ]:


mask_female = df_full['Sex'] == 1
mask_class12 = df_full['Pclass'].isin([1, 2])
df_full['Sex-female x Pclass-1-2'] = (mask_female & mask_class12).astype(int)


# Check how predictive it is.

# In[ ]:


catgroup(df_full.loc[mask_train], 'Sex-female x Pclass-1-2', 'Survived')


# It predicts female survival very accurately in 1st and 2nd class.

# ### Sex-male x Pclass-3 (derived from Sex and Pclass)
# In 3rd class, Sex is also much more predictive, so we create another interaction feature.

# In[ ]:


mask_male = df_full['Sex'] == 0
mask_class3 = df_full['Pclass'] == 3
df_full['Sex-male x Pclass-3'] = (mask_male & mask_class3).astype(int)


# In[ ]:


catgroup(df_full.loc[mask_train], 'Sex-male x Pclass-3', 'Survived')


# This feature predicts male perishing very accurately in 3rd class.

# ### SibSp and Parch 
# These features are not that useful in their initial form, but we can use them to derive other features.

# ### FamilySize (derived from SibSp and Parch)
# We can check how predictive this feature is.

# In[ ]:


df_full['FamilySize'] = df_full['SibSp'] + df_full['Parch'] + 1


# In[ ]:


catgroup(df_full[mask_train], 'FamilySize', 'Survived')


# In[ ]:


if show_graphs:
    sns.barplot(x='FamilySize', y='Survived', data=df_full[mask_train], palette='colorblind')


# It seems that having a family size of 2-4 is best. We can create bins that reflect this more.

# ### FamilySizeBin (derived from FamilySize)
# We will use three categories here.

# In[ ]:


# Create categories for FamilySize
df_full['FamilySizeBin'] = pd.cut(df_full['FamilySize'], [0, 1, 4, 20], labels=["alone", "normal", "big"])
catgroup(df_full.loc[mask_train], 'FamilySizeBin', 'Survived')


# We encode this feature as integers.

# In[ ]:


df_full['FamilySizeBin'] = df_full['FamilySizeBin'].map({'alone': 0, 'normal': 1, 'big': 2})


# ###  IsAloneF (derived from FamilySizeBin)
# We will create a separate variable for being alone, it may help in some models. 

# In[ ]:


df_full['IsAloneF'] = (df_full['FamilySizeBin'] == 0).astype(int)
catgroup(df_full.loc[mask_train], 'IsAloneF', 'Survived')


# ### Age
# There are a lot of missing values here, so first we check who has missing age. It is important to get ages right. My experience is that it can add a lot to performance, but it is easy to overfit.

# In[ ]:


# Check who has missing age.
mask_noage = df_full['Age'].isnull()
df_noage = df_full.loc[mask_noage]

df_noage.groupby(['Title', 'Pclass'], as_index=False)['Name'].count()


# Mostly men from 3rd class have missing age. Children with missing age are also almost exclusively from 3rd class.
# 
# Now let us check the Age distributions in each Pclass value, for survived and perished passengers.

# In[ ]:


# Check Age distributions.
if show_graphs:
    bins = np.linspace(0, 100, 20)
    gridgraph(df_full.loc[mask_train], 'Age', col='Pclass', hue='Survived', kde=False, bins=bins)


# We see that children have a better survival rate in 1st and 2nd class. In 3rd class, there is not much difference. This means that it is not so important to impute accurate ages for 3rd class children. If we still wanted to do that, then for female children, the title "Miss" and Parch > 0 could be a good age predictor, because most young passengers are with parents, and while Parch > 0 can also indicate a child for an older passenger, their title is more likely Mrs, not Miss.
# 
# We expect that the median age is different in each Pclass, and for each Sex. Title is also a good proxy for Age, and it contains also sex information (Master is male child, Mr is male adult, Miss is young female, Mrs is older female).
# 
# Therefore we could use the median ages of (Pclass, Title) groups, however my experience is that using only Title medians gives better performance.

# Fixing missing values: 
# 
# - For the rest of theall of the passengers, we will group by Pclass and Title, and impute with the group median age.

# In[ ]:


# Check Age distributions.
if show_graphs:
    bins = np.linspace(0, 100, 20)
    gridgraph(df_full, 'Age', row='Title', kde=True, bins=bins)


# In[ ]:


# Impute Age
df_medians = df_full.groupby('Title')['Age'].median()
for idx, median in df_medians.iteritems():
    mask_group = df_full['Title'] == idx
    df_full.loc[mask_group & mask_noage, 'Age'] = median


# ### AgeBin (derived from Age)
# We create categorical feature from Age here.

# In[ ]:


df_full['AgeBin'] = pd.qcut(df_full['Age'], 4, labels=False).astype(int)
if show_graphs:
    sns.barplot(x='AgeBin', y='Survived', data=df_full[mask_train], palette='colorblind')


# ### IsChild (derived from Age and Pclass)
# This feature represents the fact that children have better survival rates.

# In[ ]:


df_full['IsChild'] = (df_full['Age'] < 16).astype(int)
catgroup(df_full.loc[mask_train], 'IsChild', 'Survived')


# ### IsChild x Pclass-1-2 (derived from IsChild and Pclass)
# We have seen that children have better chances only in 1st and 2nd classes. Thus we create an interaction feature representing this.

# In[ ]:


mask_class12 = df_full['Pclass'].isin([1, 2])
df_full['IsChild x Pclass-1-2'] = df_full['IsChild'] * mask_class12.astype(int)
catgroup(df_full.loc[mask_train], 'IsChild x Pclass-1-2', 'Survived')


# ### Cabin
# Cabin information was found mostly for 1st class passengers only, so most of it is missing. We can still check whether having cabin information or the type of cabin gives any advantage.

# ### HasCabin
# We look at the effect of having cabin information. It is expected to correlate with being in 1st class.

# In[ ]:


df_full['HasCabin'] = df_full['Cabin'].notnull().astype(int)
mask_class1 = (df_full['Pclass'] == 1).astype(int)

print("Correlation with 1st class: ", df_full['HasCabin'].corr(mask_class1))
catgroup(df_full.loc[mask_train], 'HasCabin', 'Survived')


# The group that has cabin information truly correlates with 1st class.  

# ### CabinType
# We check if the type of cabin adds any advantage.

# In[ ]:


df_full['CabinType'] = df_full['Cabin'].str[0]
catgroup(df_full.loc[mask_train], 'CabinType', 'Survived')


# In the group that has cabin info, there is no predictive power in having any cabin type. (There is only one passenger with type T, so that does not count.)

# ### Embarked
# We can check for any correlation between survival chance and town of boarding. It is hard to imagine any strong causality in the background though, so we do not expect too much useful result.
# 
# First, we fill the missing values.

# In[ ]:


# See who is missing Embarked
mask_noembarked = df_full['Embarked'].isnull()
df_full[mask_noembarked]


# Their Ticket number does not unambiguously identify Embarked, but some internet search reveals that it is Southampton, which is also the most frequent value.

# In[ ]:


# Impute missing Embarked with most frequent value ('S')
df_full['Embarked'].fillna(df_full['Embarked'].mode()[0], inplace=True)
if show_graphs:
    sns.barplot(x='Embarked', y='Survived', data=df_full[mask_train], palette='colorblind')


# We see that passengers from Cherbourg have the highest chance to survive, but it could be just noise.

# ### Ticket
# We can observe that sometimes multiple people have the same ticket number. This suggests that they are a group traveling together. We can use this information to create a feature similar to FamilySize.

# ### TicketSize (derived from Ticket)
# This feature also captures the dependence of survival chance on the size of the traveling group, just like FamilySize. 

# In[ ]:


# Check survival rate in function of group size.
df_ticket = df_full.loc[mask_train].groupby('Ticket', as_index=False)['Survived', 'Name'].agg({'Survived': 'mean', 'Name': 'count'})
df_ticket = df_ticket.groupby('Name', as_index=False)['Survived'].mean()
df_ticket = df_ticket.sort_values(by='Survived')
df_ticket


# We see that similarly to FamilySize, groups of 2-4 seem to be ideal. 

# In[ ]:


df_ticket = df_full.groupby('Ticket')['Name'].count()
df_full['TicketSize'] = df_full['Ticket'].map(df_ticket)


# Check the correlation between TicketSize and FamilySize.

# In[ ]:


print('Correlation: ', df_full[['TicketSize', 'FamilySize']].corr().values[0, 1])
(df_full['TicketSize'] - df_full['FamilySize']).hist(bins=20)


# TicketSize and FamilySize are strongly correlated.

# ### TicketSizeBin (derived from TicketSize)
# We will assign Ticket based group sizes into categories, like we did with FamilySize.

# In[ ]:


df_full['TicketSizeBin'] = pd.cut(df_full['TicketSize'], [0, 1, 4, 20], labels=["alone", "normal", "big"])
catgroup(df_full.loc[mask_train], 'TicketSizeBin', 'Survived')


# We encode this feature as integers.

# In[ ]:


df_full['TicketSizeBin'] = df_full['TicketSizeBin'].map({'alone': 0, 'normal': 1, 'big': 2})


# ###  IsAloneT (derived from TicketSizeBin)
# We will create the "traveling alone" indicator from the TicketSizeBin feature too.

# In[ ]:


df_full['IsAloneT'] = (df_full['TicketSizeBin'] == 0).astype(int)
catgroup(df_full.loc[mask_train], 'IsAloneT', 'Survived')


# It seems that IsAloneT is slightly better predictor than IsAloneF (though the difference can be just noise).

# ### Fare
# We can observe in the data that Fare is given for the ticket, not for the individual passenger. Therefore we divide fare values with TicketSize. (We could also try FamilySize though.)

# In[ ]:


fare_scaler = 'TicketSize'
df_full['Fare'] = df_full['Fare'] / df_full[fare_scaler]


# We have seen that there are passengers with zero fare. They are either members of the Titanic "guarantee group", or their tickets were bought by their company, or they worked on the Philadelphia, that was canceled due to the coal strike (LINE tickets). 
# 
# We could think that
# - these are data errors, and we do not expect similar records in real data, therefore we have to fix them using e. g. some group medians. 
# - these are special kind of passengers, and similar ones can appear in real data too, so we do not touch them. 
# 
# Here we go with the second assumption.

# In[ ]:


mask_zerofare = df_full['Fare'] == 0
df_full.loc[mask_zerofare]


# These are all adult (aged 20-50) males embarked at Southampton and traveling alone. We could assign a median fare for them based on this group, if we wanted. 

# There is also a 1st class passenger with a fare of 5 (Carlsson, Mr. Frans Olof), which is an outlier. (He bought this for the St Louis, which was canceled because of the coal strike, so his company bought him a 1st class ticket for Titanic.) 
# 
# We could fix this too, but we will not. (In fact, there could be many more such deviations in the data, it is not our goal to look for these here...)

# Now, we impute missing values with the median fare. Only one value is missing, so it does not matter too much in this case.

# In[ ]:


df_full['Fare'].fillna(df_full['Fare'].median(), inplace=True)


# We would like to keep the original fare values too, in case we need them.

# In[ ]:


df_full['FareOrig'] = df_full['Fare'] * df_full[fare_scaler] 


# ### FareBin (derived from Fare)
# Here we create a feature that is a categorized version of Fare.

# In[ ]:


df_full['FareBin'] = pd.qcut(df_full['Fare'], 4, labels=False).astype(int)


# ### NameFareSize (derived from LastName and FareOrig)
# This feature is a third way of grouping. We can observe that fares are sometimes the same for people even if they are not on the same ticket. So we can make groups based on this.

# In[ ]:


# Check survival rate in function of group size.
df_familygroup = df_full.loc[mask_train].groupby(['LastName', 'FareOrig'], as_index=False)['Survived', 'Name'].agg({'Survived': 'mean', 'Name': 'count'})
df_familygroup = df_familygroup.groupby('Name', as_index=False)['Survived'].mean()
df_familygroup = df_familygroup.sort_values(by='Survived')
df_familygroup


# We see that this feature is similar to FamilySize and TicketSize, groups of 2-4 seem to be ideal. 

# In[ ]:


df_familygroup = df_full.groupby(['LastName', 'FareOrig'])['Name'].count()
df_full['NameFareSize'] = df_full[['LastName', 'FareOrig']].apply(lambda row: df_familygroup[(row[0], row[1])], axis=1)


# Check the correlation between NameFareSize, TicketSize and FamilySize.

# In[ ]:


df_corr = df_full[['NameFareSize', 'TicketSize', 'FamilySize']].corr()
print("Correlation Familysize - NameFaresize = ", df_corr.loc['FamilySize', 'NameFareSize'])
print("Correlation Ticketsize - NameFaresize = ", df_corr.loc['TicketSize', 'NameFareSize'])
(df_full['NameFareSize'] - df_full['FamilySize']).hist(bins=20)


# We see that NameFareSize correlates more with FamilySize, so we can use just FamilySize instead. This also suggests that using FamilySize in the model and for scaling Fares could be better.

# ### GroupSurvived (derived from LastName, FareOrig, and Ticket)
# Not only the size of groups can be important, but also whether any group members survived. The assumption here is that a passenger has better chance to survive if somebody survived in their group.
# 
# We will construct the groups here from both LastName, Fare, and Ticket information. First we use LastName and Fare.

# In[ ]:


df_full['Group'] = ''
df_groups = df_full.groupby(['LastName', 'FareOrig'])

for group, df_group in df_groups:    
    for idx, row in df_group.iterrows():
        group_members = df_group.drop(idx)['PassengerId'].tolist()
        df_full.at[idx, 'Group'] = group_members


# Second, we construct groups based on Ticket.

# In[ ]:


df_groups = df_full.groupby('Ticket')

for group, df_group in df_groups:    
    for idx, row in df_group.iterrows():
        group_members = df_group.drop(idx)['PassengerId'].tolist()
        df_full.at[idx, 'Group'].extend(group_members)
df_full['Group'] = df_full['Group'].map(set)


# Last, we merge the two kind of groups for each passenger.

# In[ ]:


def group_survived(group):
    mask_group = df_full['PassengerId'].isin(group)
    s = df_full.loc[mask_group, 'Survived'].max()
    return s if pd.notnull(s) else 0.5 

df_full['GroupSurvived'] = df_full['Group'].apply(group_survived)


# We could construct another GroupSize type feature based on these groups, and check its performance, but we will skip this for now.

# ### GroupSize (derived from Group)
# This is the fourth type of traveling group size feature.

# In[ ]:


df_full['GroupSize'] = df_full['Group'].str.len() + 1


# Check how it performs.

# In[ ]:


# Check survival rate in function of group size.
df_fullgroup = df_full.loc[mask_train].groupby('GroupSize', as_index=False)['Survived'].mean()
df_fullgroup = df_fullgroup.sort_values(by='Survived')
df_fullgroup


# Being in a 3-4 sized group is the best here. We can check how similar it is to previous group size features.

# In[ ]:


df_corr = df_full[['NameFareSize', 'TicketSize', 'FamilySize', 'GroupSize']].corr()
print("Correlation Familysize - GroupSize = ", df_corr.loc['FamilySize', 'GroupSize'])
print("Correlation Ticketsize - Groupsize = ", df_corr.loc['TicketSize', 'GroupSize'])
print("Correlation NameFareSize - Groupsize = ", df_corr.loc['NameFareSize', 'GroupSize'])
(df_full['GroupSize'] - df_full['TicketSize']).hist(bins=20)


# We see that GroupSize correlates more with TicketSize, so we can use just TicketSize instead.

# ### Used features
# We are done with feature engineering, it is time to drop features, which we will not use for modeling.

# In[ ]:


list_drop_features = [name for name, include in used_features.items() if not include]

df_full.drop(columns=list_drop_features, inplace=True)


# Check the correlation between remaining features.

# In[ ]:


df_full.loc[mask_train].corr()
#sns.heatmap(df_full[mask_train], annot=True)


# # Feature transformations

# ### Scaling
# Here we apply scaling on the feature set. It might not be necessary for all of them, but it does not hurt.

# In[ ]:


base_columns = ['Survived', 'Data']
data_columns = [col for col in df_full.columns if col not in base_columns]

scaler = StandardScaler()
df_full.loc[mask_train, data_columns] = scaler.fit_transform(df_full.loc[mask_train, data_columns])
df_full.loc[mask_valid, data_columns] = scaler.transform(df_full.loc[mask_valid, data_columns])


# We separate the full dataset to training and validation parts.

# In[ ]:


Xt = df_full.loc[mask_train].drop(columns=base_columns)
Xv = df_full.loc[mask_valid].drop(columns=base_columns)


# # Feature selection
# Here is a check that shows feature importances using LASSO.

# In[ ]:


# Logistic Regression LASSO
if feature_selection:
    threshold_pct = 0.1
    lasso = LogisticRegression(penalty='l1', C=10, random_state=0, solver='saga', max_iter=200)
    lasso.fit(Xt, yt)
    print("Lasso accuracy on training data: ", lasso.score(Xt, yt))
    
    # Select features
    coefs = np.absolute(lasso.coef_.flatten())
    plt.hist(coefs, bins=20)
    mask_features = coefs > (np.max(coefs) * threshold_pct)
    new_columns = Xt.columns[mask_features]
    df_features = pd.DataFrame({'Features': new_columns, 'Strength': coefs[mask_features]}).sort_values(by='Strength', ascending=False)
    print(df_features)
    
    # Drop features with low importance
    Xt = Xt[new_columns]
    Xv = Xv[new_columns]


# # Modeling

# ### Model evaluation
# Here we evaluate some simple models using cross validation.

# In[ ]:


# Models
models = {}
models['Logistic Regression'] = LogisticRegression(penalty='l2', C=1.0, random_state=0, solver='saga', max_iter=300)
models['SVC_rbf'] = SVC(probability=True, kernel='rbf', gamma='scale', random_state=0)
models['SVC_lin'] = SVC(probability=True, kernel='linear', random_state=0)
models['KNN'] = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski', metric_params=None, 
                                     n_jobs=1, n_neighbors=10, p=3, weights='uniform')
dtree_params = {'criterion': 'entropy', 'max_depth': 5, 'max_features': None, 'min_impurity_decrease': 0.0, 
              'min_samples_leaf': 0.01, 'min_samples_split': 0.01, 'min_weight_fraction_leaf': 0.0, 'splitter': 'best'}
models['Decision Tree'] = DecisionTreeClassifier(**dtree_params)
models['Random Forest'] = RandomForestClassifier(criterion='entropy', n_estimators=200, oob_score=True)
xgb_params = {'subsample': 0.5, 'reg_lambda': 5, 'reg_alpha': 0, 'n_estimators': 200, 'min_child_weight': 0, 'max_depth': 6, 
              'max_delta_step': 1, 'learning_rate': 1.0, 'gamma': 2, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5}
models['XGBoost'] = XGBClassifier(objective='binary:logistic', **xgb_params)


# In[ ]:


# Run CV using randomized folds (these can overlap).
scoring = ['accuracy']  # We can give multiple metrics here 
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

for mname, model in models.items():
    result = cross_validate(model, Xt, yt, scoring=scoring, cv=cv, return_train_score=False)
    print("CV results for model {}: mean {:2.4f}, std {:2.4f}".format(mname, np.mean(result['test_accuracy']), np.std(result['test_accuracy'])))


# ### Model tuning
# Here we can do hyperparameter optimization.

# In[ ]:


if model_tuning:
    param_grid_XGBoost = {
        'colsample_bytree': [0.1, 0.5, 1],
        'colsample_bylevel': [0.1, 0.5, 1],
        'subsample': [0.1, 0.5, 1], 
        'learning_rate': [0.05, 0.1, 0.3, 1.0],
        'max_depth': [0, 3, 6, 10], 
        'reg_alpha': [0, 0.1, 1, 5],
        'reg_lambda': [0, 0.1, 1, 5],
        'gamma': [0, 1, 2, 5], 
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [0, 1, 2, 5],
        'max_delta_step': [0, 1, 2, 5],
    }

    param_grid_DTC = {
        'criterion': ['gini', 'entropy'], 
        'splitter': ['best', 'random'], 
        'max_depth': [None, 3, 5, 7, 10], 
        'min_samples_split': [2, 0.01, 0.05, 0.1],
        'min_samples_leaf': [1, 0.01, 0.05, 0.1],
        'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
        'max_features': [None, 'auto'], 
        'min_impurity_decrease': [0.0, 0.2, 0.4, 0.7],
    }
    #tune_model = RandomizedSearchCV(models['XGBoost'], param_distributions=param_grid_XGBoost, scoring='roc_auc', cv=cv, n_iter=1000)
    tune_model = GridSearchCV(models['Decision Tree'], param_grid=param_grid_DTC, scoring='roc_auc', cv=cv)
    tune_model.fit(Xt, yt)
    print('Best parameters:\n', tune_model.best_params_)


# # Prediction
# Now we use the validation data and create the predictions, and the submission files. 
# 
# NOTE: If we predicted based on Sex only, we could reach 74% accuracy. So we expect much better results from more complex models.

# In[ ]:


list_submit = models.keys()
dict_submissions = {}

# Loop over models
for mname in list_submit:
    model = models[mname]
    model.fit(Xt, yt)

    # Check train data score
    ytp = model.predict(Xt)
    acc = model.score(Xt, yt)
    #print("Accuracy of model ", mname, " on train data: ", acc)

    # Generate validation data score
    yvp = model.predict(Xv)
    dict_submissions[mname] = yvp
    submission = pd.DataFrame({"PassengerId": df_valid["PassengerId"], "Survived": yvp})
    submission.to_csv('submission_{}.csv'.format(mname), index=False)


# Performances:
# - KNN: 0.8181
# - Decision Tree: 0.8086
# - SVC: 0.8086
# - Logistic Regression: 0.7894 
