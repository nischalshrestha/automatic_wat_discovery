#!/usr/bin/env python
# coding: utf-8

# # Exploring and Processing Data

# In[ ]:


# imports
import pandas as pd
import numpy as np
import os


# ## Import Data

# In[ ]:


# read the data with all default parameters
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')


# In[ ]:


# get the type
type(train_df)


# ## Basic Structure

# In[ ]:


# use .info() to get brief information about the dataframe 
train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


test_df['Survived'] = -888 # Adding Survived with a default value


# In[ ]:


df = pd.concat((train_df, test_df),axis=0)


# In[ ]:


df.info()


# In[ ]:


# use .head() to get top 5 rows
df.head()


# In[ ]:


# use .tail() to get last 5 rows
df.tail()


# In[ ]:


# use .head(n) to get top-n rows
df.head(10)


# In[ ]:


# column selection using dot
df.Name


# In[ ]:


# selection using column name as string
df['Name']


# In[ ]:


# selecting multiple columns using a list of column name strings
df[['Name','Age']]


# In[ ]:


# indexing : use loc for label based indexing 
# all columns
df.loc[5:10,]


# In[ ]:


# selecting column range
df.loc[5:10, 'Age' : 'Pclass']


# In[ ]:


# selecting discrete columns
df.loc[5:10, ['Survived', 'Fare','Embarked']]


# In[ ]:


# indexing : use iloc for position based indexing 
df.iloc[5:10, 3:8]


# In[ ]:


# filter rows based on the condition 
male_passengers = df.loc[df.Sex == 'male',:]
print('Number of male passengers : {0}'.format(len(male_passengers)))


# In[ ]:


# use & or | operators to build complex logic
male_passengers_first_class = df.loc[((df.Sex == 'male') & (df.Pclass == 1)),:]
print('Number of male passengers in first class: {0}'.format(len(male_passengers_first_class)))


# ## Summary Statistics

# In[ ]:


# use .describe() to get statistics for all numeric columns
df.describe()


# In[ ]:


# numerical feature
# centrality measures
print('Mean fare : {0}'.format(df.Fare.mean())) # mean
print('Median fare : {0}'.format(df.Fare.median())) # median


# In[ ]:


# dispersion measures
print('Min fare : {0}'.format(df.Fare.min())) # minimum
print('Max fare : {0}'.format(df.Fare.max())) # maximum
print('Fare range : {0}'.format(df.Fare.max()  - df.Fare.min())) # range
print('25 percentile : {0}'.format(df.Fare.quantile(.25))) # 25 percentile
print('50 percentile : {0}'.format(df.Fare.quantile(.5))) # 50 percentile
print('75 percentile : {0}'.format(df.Fare.quantile(.75))) # 75 percentile
print('Variance fare : {0}'.format(df.Fare.var())) # variance
print('Standard deviation fare : {0}'.format(df.Fare.std())) # standard deviation


# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


# box-whisker plot
df.Fare.plot(kind='box')


# In[ ]:


# use .describe(include='all') to get statistics for all  columns including non-numeric ones
df.describe(include='all')


# In[ ]:


# categorical column : Counts
df.Sex.value_counts()


# In[ ]:


# categorical column : Proprotions
df.Sex.value_counts(normalize=True)


# In[ ]:


# apply on other columns
df[df.Survived != -888].Survived.value_counts() 


# In[ ]:


# count : Passenger class
df.Pclass.value_counts() 


# In[ ]:


# visualize counts
df.Pclass.value_counts().plot(kind='bar')


# In[ ]:


# title : to set title, color : to set color,  rot : to rotate labels 
df.Pclass.value_counts().plot(kind='bar',rot = 0, title='Class wise passenger count', color='c');


# ## Distributions

# In[ ]:


# use hist to create histogram
df.Age.plot(kind='hist', title='histogram for Age', color='c');


# In[ ]:


# use bins to add or remove bins
df.Age.plot(kind='hist', title='histogram for Age', color='c', bins=20);


# In[ ]:


# use kde for density plot
df.Age.plot(kind='kde', title='Density plot for Age', color='c');


# In[ ]:


# histogram for fare
df.Fare.plot(kind='hist', title='histogram for Fare', color='c', bins=20);


# In[ ]:


print('skewness for age : {0:.2f}'.format(df.Age.skew()))
print('skewness for fare : {0:.2f}'.format(df.Fare.skew()))


# In[ ]:


# use scatter plot for bi-variate distribution
df.plot.scatter(x='Age', y='Fare', color='c', title='scatter plot : Age vs Fare');


# In[ ]:


# use alpha to set the transparency
df.plot.scatter(x='Age', y='Fare', color='c', title='scatter plot : Age vs Fare', alpha=0.1);


# In[ ]:


df.plot.scatter(x='Pclass', y='Fare', color='c', title='Scatter plot : Passenger class vs Fare', alpha=0.15);


# ## Grouping and Aggregations

# In[ ]:


# group by 
df.groupby('Sex').Age.median()


# In[ ]:


# group by 
df.groupby(['Pclass']).Fare.median()


# In[ ]:


df.groupby(['Pclass']).Age.median()


# In[ ]:


df.groupby(['Pclass'])['Fare','Age'].median()


# In[ ]:


df.groupby(['Pclass']).agg({'Fare' : 'mean', 'Age' : 'median'})


# In[ ]:


# more complicated aggregations 
aggregations = {
    'Fare': { # work on the "Fare" column
        'mean_Fare': 'mean',  # get the mean fare
        'median_Fare': 'median', # get median fare
        'max_Fare': max,
        'min_Fare': np.min
    },
    'Age': {     # work on the "Age" column
        'median_Age': 'median',   # Find the max, call the result "max_date"
        'min_Age': min,
        'max_Age': max,
        'range_Age': lambda x: max(x) - min(x)  # Calculate the age range per group
    }
}


# In[ ]:


df.groupby(['Pclass']).agg(aggregations)


# In[ ]:


df.groupby(['Pclass', 'Embarked']).Fare.median()


# ## Crosstabs

# In[ ]:


# crosstab on Sex and Pclass
pd.crosstab(df.Sex, df.Pclass)


# In[ ]:


pd.crosstab(df.Sex, df.Pclass).plot(kind='bar');


# ## Pivots

# In[ ]:


# pivot table
df.pivot_table(index='Sex',columns = 'Pclass',values='Age', aggfunc='mean')


# In[ ]:


df.groupby(['Sex','Pclass']).Age.mean()


# In[ ]:


df.groupby(['Sex','Pclass']).Age.mean().unstack()


#   

# ## Data Munging : Working with missing values

# In[ ]:


# use .info() to detect missing values (if any)
df.info()


# ### Feature : Embarked

# In[ ]:


# extract rows with Embarked as Null
df[df.Embarked.isnull()]


# In[ ]:


# how many people embarked at different points
df.Embarked.value_counts()


# In[ ]:


# which embarked point has higher survival count
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Embarked)


# In[ ]:


# impute the missing values with 'S'
# df.loc[df.Embarked.isnull(), 'Embarked'] = 'S'
# df.Embarked.fillna('S', inplace=True)


# In[ ]:


# Option 2 : explore the fare of each class for each embarkment point
df.groupby(['Pclass', 'Embarked']).Fare.median()


# In[ ]:


# replace the missing values with 'C'
df.Embarked.fillna('C', inplace=True)


# In[ ]:


# check if any null value remaining
df[df.Embarked.isnull()]


# In[ ]:


# check info again
df.info()


# ### Feature : Fare

# In[ ]:


df[df.Fare.isnull()]


# In[ ]:


median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'),'Fare'].median()
print(median_fare)


# In[ ]:


df.Fare.fillna(median_fare, inplace=True)


# In[ ]:


# check info again
df.info()


# ### Feature : Age

# In[ ]:


# set maximum number of rows to be displayed
pd.options.display.max_rows = 15


# In[ ]:


# return null rows
df[df.Age.isnull()]


# #### option 1 : replace all missing age with mean value

# In[ ]:


df.Age.plot(kind='hist', bins=20, color='c');


# In[ ]:


# get mean
df.Age.mean()


# issue : due to few high values of 70's and 80's pushing the overall mean
# 
# 

# In[ ]:


# replace the missing values
# df.Age.fillna(df.Age.mean(), inplace=True)


# #### option 2 : replace with median age of gender

# In[ ]:


# median values
df.groupby('Sex').Age.median()


# In[ ]:


# visualize using boxplot
df[df.Age.notnull()].boxplot('Age','Sex');


# In[ ]:


# replace : 
# age_sex_median = df.groupby('Sex').Age.transform('median')
# df.Age.fillna(age_sex_median, inplace=True)


# #### option 3 : replace with median age of Pclass

# In[ ]:


df[df.Age.notnull()].boxplot('Age','Pclass');


# In[ ]:


# replace : 
# pclass_age_median = df.groupby('Pclass').Age.transform('median')
# df.Age.fillna(pclass_age_median , inplace=True)


# #### option 4 : replace with median age of title

# In[ ]:


df.Name


# In[ ]:


# Function to extract the title from the name 
def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title


# In[ ]:


# use map function to apply the function on each Name value row i
df.Name.map(lambda x : GetTitle(x)) # alternatively you can use : df.Name.map(GetTitle)


# In[ ]:


df.Name.map(lambda x : GetTitle(x)).unique()


# In[ ]:


# Function to extract the title from the name 
def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]



# In[ ]:


# create Title feature
df['Title'] =  df.Name.map(lambda x : GetTitle(x))


# In[ ]:


# head 
df.head()


# In[ ]:


# Box plot of Age with title
df[df.Age.notnull()].boxplot('Age','Title');


# In[ ]:


# replace missing values
title_age_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median , inplace=True)


# In[ ]:


# check info again
df.info()


# ## Working with outliers

# ### Age

# In[ ]:


# use histogram to get understand the distribution
df.Age.plot(kind='hist', bins=20, color='c');


# In[ ]:


df.loc[df.Age > 70]


# ### Fare

# In[ ]:


# histogram for fare
df.Fare.plot(kind='hist', title='histogram for Fare', bins=20, color='c');


# In[ ]:


# box plot to indentify outliers 
df.Fare.plot(kind='box');


# In[ ]:


# look into the outliers
df.loc[df.Fare == df.Fare.max()]


# In[ ]:


# Try some transformations to reduce the skewness
LogFare = np.log(df.Fare + 1.0) # Adding 1 to accomodate zero fares : log(0) is not defined


# In[ ]:


# Histogram of LogFare
LogFare.plot(kind='hist', color='c', bins=20);


# In[ ]:


# binning
pd.qcut(df.Fare, 4)


# In[ ]:


pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']) # discretization


# In[ ]:


pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']).value_counts().plot(kind='bar', color='c', rot=0);


# In[ ]:


# create fare bin feature
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high'])


# ## Feature Engineering

# ### Feature : Age State ( Adult or Child )

# In[ ]:


# AgeState based on Age
df['AgeState'] = np.where(df['Age'] >= 18, 'Adult','Child')


# In[ ]:


# AgeState Counts
df['AgeState'].value_counts()


# In[ ]:


# crosstab
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].AgeState)


# ### Feature : FamilySize

# In[ ]:


# Family : Adding Parents with Siblings
df['FamilySize'] = df.Parch + df.SibSp + 1 # 1 for self


# In[ ]:


# explore the family feature
df['FamilySize'].plot(kind='hist', color='c');


# In[ ]:


# further explore this family with max family members
df.loc[df.FamilySize == df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]


# In[ ]:


pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].FamilySize)


# ### Feature : IsMother

# In[ ]:


# a lady aged more thana 18 who has Parch >0 and is married (not Miss)
df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age > 18) & (df.Title != 'Miss')), 1, 0)


# In[ ]:


# Crosstab with IsMother
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].IsMother)


# ### Deck

# In[ ]:


# explore Cabin values
df.Cabin


# In[ ]:


# use unique to get unique values for Cabin feature
df.Cabin.unique()


# In[ ]:


# look at the Cabin = T
df.loc[df.Cabin == 'T']


# In[ ]:


# set the value to NaN
df.loc[df.Cabin == 'T', 'Cabin'] = np.NaN


# In[ ]:


# look at the unique values of Cabin again
df.Cabin.unique()


# In[ ]:


# extract first character of Cabin string to the deck
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck'] = df['Cabin'].map(lambda x : get_deck(x))


# In[ ]:


# check counts
df.Deck.value_counts()


# In[ ]:


# use crosstab to look into survived feature cabin wise
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Deck)


# In[ ]:


# info command 
df.info()


# ### Categorical Feature Encoding

# In[ ]:


# sex
df['IsMale'] = np.where(df.Sex == 'male', 1, 0)


# In[ ]:


# columns Deck, Pclass, Title, AgeState
df = pd.get_dummies(df,columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])


# In[ ]:


print(df.info())


# ### Drop and Reorder Columns

# In[ ]:


# drop columns
df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1, inplace=True)


# In[ ]:


# reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]


# In[ ]:


# check info again
df.info()


# ### Advanced visualization using MatPlotlib

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


plt.hist(df.Age)


# In[ ]:


plt.hist(df.Age, bins=20, color='c')
plt.show()


# In[ ]:


plt.hist(df.Age, bins=20, color='c')
plt.title('Histogram : Age')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()


# In[ ]:


f , ax = plt.subplots()
ax.hist(df.Age, bins=20, color='c')
ax.set_title('Histogram : Age')
ax.set_xlabel('Bins')
ax.set_ylabel('Counts')
plt.show()


# In[ ]:


# Add subplots
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(14,3))

ax1.hist(df.Fare, bins=20, color='c')
ax1.set_title('Histogram : Fare')
ax1.set_xlabel('Bins')
ax1.set_ylabel('Counts')

ax2.hist(df.Age, bins=20, color='tomato')
ax2.set_title('Histogram : Age')
ax2.set_xlabel('Bins')
ax2.set_ylabel('Counts')

plt.show()


# In[ ]:


# Adding subplots
f , ax_arr = plt.subplots(3 , 2 , figsize=(14,7))

# Plot 1
ax_arr[0,0].hist(df.Fare, bins=20, color='c')
ax_arr[0,0].set_title('Histogram : Fare')
ax_arr[0,0].set_xlabel('Bins')
ax_arr[0,0].set_ylabel('Counts')

# Plot 2
ax_arr[0,1].hist(df.Age, bins=20, color='c')
ax_arr[0,1].set_title('Histogram : Age')
ax_arr[0,1].set_xlabel('Bins')
ax_arr[0,1].set_ylabel('Counts')

# Plot 3
ax_arr[1,0].boxplot(df.Fare.values)
ax_arr[1,0].set_title('Boxplot : Age')
ax_arr[1,0].set_xlabel('Fare')
ax_arr[1,0].set_ylabel('Fare')

# Plot 4
ax_arr[1,1].boxplot(df.Age.values)
ax_arr[1,1].set_title('Boxplot : Age')
ax_arr[1,1].set_xlabel('Age')
ax_arr[1,1].set_ylabel('Age')

# Plot 5
ax_arr[2,0].scatter(df.Age, df.Fare, color='c', alpha=0.15)
ax_arr[2,0].set_title('Scatter Plot : Age vs Fare')
ax_arr[2,0].set_xlabel('Age')
ax_arr[2,0].set_ylabel('Fare')

ax_arr[2,1].axis('off')
plt.tight_layout()


plt.show()


# #### Extra Visualization Material : For Your Practice

# In[ ]:


# family size 
family_survived = pd.crosstab(df[df.Survived != -888].FamilySize, df[df.Survived != -888].Survived)
print(family_survived)


# In[ ]:


# impact of family size on survival rate
family_survived =  df[df.Survived != -888].groupby(['FamilySize','Survived']).size().unstack()
print(family_survived)


# In[ ]:


family_survived.columns = ['Not Survived', 'Survived']


# In[ ]:


# Mix and Match
f, ax = plt.subplots(figsize=(10,3))
ax.set_title('Impact of family size on survival rate')
family_survived.plot(kind='bar', stacked=True, color=['tomato','c'], ax=ax, rot=0)
plt.legend(bbox_to_anchor=(1.3,1.0))
plt.show()


# In[ ]:


family_survived.sum(axis = 1)


# In[ ]:


scaled_family_survived = family_survived.div(family_survived.sum(axis=1), axis=0)


# In[ ]:


scaled_family_survived.columns = ['Not Survived', 'Survived']


# In[ ]:


# Mix and Match
f, ax = plt.subplots(figsize=(10,3))
ax.set_title('Impact of family size on survival rate')
scaled_family_survived.plot(kind='bar', stacked=True, color=['tomato','c'], ax=ax, rot=0)
plt.legend(bbox_to_anchor=(1.3,1.0))
plt.show()


# In[ ]:


df.info()


# In[ ]:


def read_data():
    # set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    # read the data with all default parameters
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')
    test_df['Survived'] = -888
    df = pd.concat((train_df, test_df), axis=0)
    return df

def process_data(df):
    # using the method chaining concept
    return (df
         # create title attribute - then add this 
         .assign(Title = lambda x: x.Name.map(get_title))
         # working missing values - start with this
         .pipe(fill_missing_values)
         # create fare bin feature
         .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low','low','high','very_high']))
         # create age state
         .assign(AgeState = lambda x : np.where(x.Age >= 18, 'Adult','Child'))
         .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)
         .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))
          # create deck feature
         .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin)) 
         .assign(Deck = lambda x : x.Cabin.map(get_deck))
         # feature encoding 
         .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1,0))
         .pipe(pd.get_dummies, columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])
         # add code to drop unnecessary columns
         .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1)
         )

def get_title(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def fill_missing_values(df):
    # embarked
    df.Embarked.fillna('C', inplace=True)
    # fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    # age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median , inplace=True)
    return df


# In[ ]:


train_df = process_data(train_df)


# In[ ]:


test_df = process_data(test_df)


# In[ ]:


test_df.info()


# ### Data Preparation

# In[ ]:


X = train_df.loc[:,'Age':].values.astype('float')
y = train_df['Survived'].ravel()


# In[ ]:


print(X.shape, y.shape)


# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[ ]:


# average survival in train and test
print('mean survival in train : {0:.3f}'.format(np.mean(y_train)))
print('mean survival in test : {0:.3f}'.format(np.mean(y_test)))


# ### Baseline Model

# In[ ]:


import sklearn

# import function
from sklearn.dummy import DummyClassifier


# In[ ]:


# create model
model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)


# In[ ]:


# train model
model_dummy.fit(X_train, y_train)


# In[ ]:


print('score for baseline model : {0:.2f}'.format(model_dummy.score(X_test, y_test)))


# In[ ]:


# peformance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


# In[ ]:


# accuracy score
print('accuracy for baseline model : {0:.2f}'.format(accuracy_score(y_test, model_dummy.predict(X_test))))


# In[ ]:


# confusion matrix
print('confusion matrix for baseline model: \n {0}'.format(confusion_matrix(y_test, model_dummy.predict(X_test))))


# In[ ]:


# precision and recall scores
print('precision for baseline model : {0:.2f}'.format(precision_score(y_test, model_dummy.predict(X_test))))
print('recall for baseline model : {0:.2f}'.format(recall_score(y_test, model_dummy.predict(X_test))))


# ### Logistic Regression Model

# In[ ]:


# import function
from sklearn.linear_model import LogisticRegression


# In[ ]:


# create model
model_lr_1 = LogisticRegression(random_state=0)


# In[ ]:


# train model
model_lr_1.fit(X_train,y_train)


# In[ ]:


# evaluate model
print('score for logistic regression - version 1 : {0:.2f}'.format(model_lr_1.score(X_test, y_test)))


# In[ ]:


# performance metrics
# accuracy
print('accuracy for logistic regression - version 1 : {0:.2f}'.format(accuracy_score(y_test, model_lr_1.predict(X_test))))
# confusion matrix
print('confusion matrix for logistic regression - version 1: \n {0}'.format(confusion_matrix(y_test, model_lr_1.predict(X_test))))
# precision 
print('precision for logistic regression - version 1 : {0:.2f}'.format(precision_score(y_test, model_lr_1.predict(X_test))))
# precision 
print('recall for logistic regression - version 1 : {0:.2f}'.format(recall_score(y_test, model_lr_1.predict(X_test))))


# In[ ]:


# model coefficients
model_lr_1.coef_


# ### Part 2 

# ### Hyperparameter Optimization

# In[ ]:


# base model 
model_lr = LogisticRegression(random_state=0)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# In[ ]:


print('best score : {0:.2f}'.format(clf.best_score_))


# In[ ]:


# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test, y_test)))


# ### Feature Normalization and Standardization

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# #### Feature Normalization

# In[ ]:


# feature normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)


# In[ ]:


X_train_scaled[:,0].min(),X_train_scaled[:,0].max()


# In[ ]:


# normalize test data
X_test_scaled = scaler.transform(X_test)


# #### Feature Standardization

# In[ ]:


# feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #### Create model after standardization

# In[ ]:


# base model 
model_lr = LogisticRegression(random_state=0)
parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
clf.fit(X_train_scaled, y_train)


# In[ ]:


clf.best_score_


# In[ ]:


# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test_scaled, y_test)))


# ### Random Forest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_rf_1 = RandomForestClassifier(random_state=0)
model_rf_1.fit(X_train_scaled, y_train)


# In[ ]:


# evaluate model
print('score for random forest - version 1 : {0:.2f}'.format(model_rf_1.score(X_test_scaled, y_test)))


# ### HyperParameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'n_estimators':[10, 100, 200], 
              'min_samples_leaf':[1, 5,10,50],
              'max_features' : ('auto','sqrt','log2'),
               }
rf = RandomForestClassifier(random_state=0, oob_score=True)
clf = GridSearchCV(rf, parameters)


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


clf.best_estimator_


# In[ ]:


# best score
print('best score for random forest : {0:.2f}'.format(clf.best_score_))


# ### Confusion Metrics , Precision and Recall

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


from sklearn import metrics


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


pred = model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
metrics.auc(fpr, tpr)


# In[ ]:


# Predict on Final Test data


# In[ ]:


test_X = test_df.as_matrix().astype('float')


# In[ ]:


test_X = scaler.transform(test_X)


# In[ ]:


predictions = model.predict_proba(test_X)


# In[ ]:


print(predictions.shape)


# In[ ]:




