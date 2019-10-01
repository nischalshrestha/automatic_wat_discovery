#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #seaborn visualisation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Acquire training and testing data using panda

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# 1. First we must understand what our data looks like and get its overview
# 2. Secondly, we inspect if any of the features have missing values.

# In[ ]:


# We will find total number of data and data types of different features.

train_data.info()

#We can see from the data above, that Age and Cabin features have a lot of missing values.


# We can get more numerical information about our data using describe function in pandas.

# In[ ]:


train_data.describe(include = 'all')


# # 1. Exploratory Data Analysis: 
# ### Analyse, Explore and Identify pattern in data

# In[ ]:


# First, we plot a graph to get an idea of the number of people who survived and who died
train_data.Survived.value_counts().plot(kind='bar')
import matplotlib.pyplot as plt
plt.title('Death (0) vs Survival (1)')
print("It is clear that fewer people were able to survive and more than 500 people died!")


# Let us start by understanding correlations between Age and our solution goal (Survived).
# 
# A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
# 
# Note that x-axis in historgram visualizations represents the count of samples or passengers.
# 

# In[ ]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)
fig = plt.figure(figsize=(25, 7))

sns.violinplot(x='Sex', y='Age', 
               hue='Survived', data=train_data, 
               split=True,
               palette={0: "r", 1: "g"}
              );


# 
# **Observations.**
# 
# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.
# 
# **Decisions.**
# 
# This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
# 
# - We should consider Age in our model training.
# - Complete the Age feature for null values
# - We should band age groups

# Let us consider correlating Embarked, Sex and Pclass

# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# **Observations.**
# 
# - Female passengers had much better survival rate than males.
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. 
# - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. 
# 
# **Decisions.**
# 
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

# Lets see how Class affected the Survival rates of passengers.

# In[ ]:


train_data.Pclass.value_counts().plot(kind='barh')
plt.title('Class Distribution')
plt.xlabel('No of Passengers')
plt.ylabel('Class of Ticket')


# In[ ]:


grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# **Observations.**
# 
# - Pclass=3 had most passengers, however most did not survive. 
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived. 
# - Most passengers in Pclass=1 survived.
# - Pclass varies in terms of Age distribution of passengers.
# 
# **Decisions.**
# 
# - Consider Pclass for model training.

# We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).

# In[ ]:


grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=train_data, split=True, palette={0: "r", 1: "g"});


# **Observations.**
# 
# - Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
# - Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
# 
# **Decisions.**
# 
# - Consider banding Fare feature.

# # 2. Feature Engineering: 
#  
# In the previous part, we flirted with the data and spotted some interesting correlations.
# 
# In this part, we'll see how to process and transform these variables in such a way the data becomes manageable by a machine learning algorithm.
# 
# We'll also create, or "engineer" additional features that will be useful in building the model.
# 
#  Basically: 
#   1. Filling missing values (to Age, Embarked, Fare)
#   2. Coverting categorical variables to integers (Age and Embarked)
#   3. Creating new feature extracting from existing or combine existing features to create a new feature
# 

# ### But first, let's define a print function that asserts whether or not a feature has been processed.

# In[ ]:


def status(feature):
    print ('Processing', feature, ':ok')


# One trick when starting a machine learning problem is to append the training set to the test set together.
# 
# We'll engineer new features using the train set to prevent information leakage. Then we'll add these variables to the test set.

# In[ ]:


def get_combined_data():
    
    # extracting and then removing the targets from the training data 
    targets = train_data.Survived
    train_data.drop(['Survived'], 1, inplace=True)
    
    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = train_data.append(test_data)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)
    
    return combined


# In[ ]:


combined = get_combined_data()
print (combined.shape)


# > # 1. Extracting the passenger titles

# When looking at the passenger names one could wonder how to process them to extract a useful information.
# 
# If you look closely at these first examples:
# 
# Braund, Mr. Owen Harris
# Heikkinen, Miss. Laina
# Oliva y Ocana, Dona. Fermina
# Peter, Master. Michael J
# You will notice that each name has a title in it ! This can be a simple Miss. or Mrs. but it can be sometimes something more sophisticated like Master, Sir or Dona. In that case, we might introduce an additional information about the social status by simply parsing the name and extracting the title and converting to a binary variable.
# 
# Let's first see what the different titles are in the train set

# In[ ]:


titles = set()
for name in train_data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

print(titles)


# In[ ]:


Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

def get_titles():
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated title
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
    status('Title')
    return combined


# In[ ]:


combined = get_titles()


# In[ ]:


combined[combined['Title'].isnull()]


# There is indeed a NaN value in the line 1305. In fact the corresponding name is Oliva y Ocana, Dona. Fermina.
# 
# This title was not encoutered in the train dataset.

# ># 2. Processing the ages

# We have seen in the first part that the Age variable was missing 177 values. This is a large number ( ~ 13% of the dataset). Simply replacing them with the mean or the median age might not be the best solution since the age may differ by groups and categories of passengers.
# 
# To understand why, let's group our dataset by sex, Title and passenger class and for each subset compute the median age.
# 
# To avoid data leakage from the test set, we fill in missing ages in the train using the train set and we fill in ages in the test set using values calculated from the train set as well.

# Number of missing ages in train set

# In[ ]:


print(combined.iloc[:891].Age.isnull().sum())


# Number of missing ages in test set
# 
# 

# In[ ]:


print(combined.iloc[891:].Age.isnull().sum())


# In[ ]:


grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]


# In[ ]:


grouped_median_train.head()


# This dataframe will help us impute missing age values based on different criteria.
# 
# Look at the median age column and see how this value can be different based on the Sex, Pclass and Title put together.
# 
# For example:
# 
# If the passenger is female, from Pclass 1, and from royalty the median age is 40.5.
# If the passenger is male, from Pclass 3, with a Mr title, the median age is 26.
# Let's create a function that fills in the missing age in combined based on these different attributes.

# In[ ]:


def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])
    ) 
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    combined.loc[ combined['Age'] <= 16, 'Age'] = 0
    combined.loc[(combined['Age'] > 16) & (combined['Age'] <= 32), 'Age'] = 1
    combined.loc[(combined['Age'] > 32) & (combined['Age'] <= 48), 'Age'] = 2
    combined.loc[(combined['Age'] > 48) & (combined['Age'] <= 64), 'Age'] = 3
    combined.loc[ combined['Age'] > 64, 'Age'] = 4
    combined['Age'] = combined['Age'].astype(int)
    status('age')
    return combined


# In[ ]:


combined = process_age()


# Perfect. The missing ages have been replaced.
# 
# However, we notice a missing value in Fare, two missing values in Embarked and a lot of missing values in Cabin. We'll come back to these variables later.
# 

# # 3. Processing names

# In[ ]:


def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)
    
    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)
    
    status('names')
    return combined


# This function drops the Name column since we won't be using it anymore because we created a Title column.
# 
# Then we encode the title values using a dummy encoding.

# In[ ]:


combined = process_names()


# In[ ]:


combined.head()


# ># 4. Processing Fare

# Let's imputed the missing fare value by the average fare computed on the train set (only one value is missing).

# In[ ]:


def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    status('fare')
    return combined


# In[ ]:


combined = process_fares()


# ># 5. Processing Embarked

# In[ ]:


def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    freq_port = train_data.Embarked.dropna().mode()[0]
    combined.Embarked.fillna(freq_port, inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return combined
freq_port = train_data.Embarked.dropna().mode()[0]


# In[ ]:


combined = process_embarked()


# In[ ]:


combined.head()


# ># 6. Processing Cabin

# In[ ]:


train_cabin, test_cabin = set(), set()

for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')
        
for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')


# In[ ]:


print(train_cabin)


# In[ ]:


print(test_cabin)


# In[ ]:


def process_cabin():
    global combined    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')    
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return combined


# This function replaces NaN values with U (for Unknow). It then maps each Cabin value to the first letter. Then it encodes the cabin values using dummy encoding again.

# In[ ]:


combined = process_cabin()


# In[ ]:


combined.head()


# > # 7. Processing Sex

# In[ ]:


def process_sex():
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})
    status('Sex')
    return combined


# This function maps the string values male and female to 1 and 0 respectively.

# In[ ]:


combined = process_sex()


# ># 8. Processing Pclass

# In[ ]:


def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies],axis=1)
    
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('Pclass')
    return combined


# This function encodes the values of Pclass (1,2,3) using a dummy encoding.
# 
# 

# In[ ]:


combined = process_pclass()


# ># 9. Processing Ticket

# Let's first see how the different ticket prefixes we have in our dataset

# In[ ]:


combined.Ticket.head()


# In[ ]:


def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'


# In[ ]:


tickets = set()
for t in combined['Ticket']:
    tickets.add(cleanTicket(t))


# In[ ]:


print(len(tickets))


# In[ ]:


def process_ticket():
    
    global combined

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('Ticket')
    return combined


# In[ ]:


combined = process_ticket()


# ># 9. Processing Family
# This part includes creating new variables based on the size of the family (the size is by the way, another variable we create).
# 
# This creation of new variables is done under a realistic assumption: Large families are grouped together, hence they are more likely to get rescued than people traveling alone.

# In[ ]:


def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    combined.drop('Parch', inplace=True, axis=1)
    combined.drop('SibSp', inplace=True, axis=1)
    status('family')
    return combined


# This function introduces 4 new features:
# 
# FamilySize : the total number of relatives including the passenger (him/her)self.
# Sigleton : a boolean variable that describes families of size = 1
# SmallFamily : a boolean variable that describes families of 2 <= size <= 4
# LargeFamily : a boolean variable that describes families of 5 < size

# In[ ]:


combined = process_family()


# In[ ]:


print(combined.shape)


# We end up with a total of 67 features.

# In[ ]:


#We encode Sex as 0 or 1 for male or female.
#from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#train_data.Sex = labelencoder.fit_transform(train_data.Sex)
#test_data.Sex = labelencoder.fit_transform(test_data.Sex)
#print('Label Encoder for Sex, Done!')


# ## **We are almost done with our data preprocessing. Now we are ready to train a model and predict the required solution.**

# # 3. Machine Learning Models: Training and Evaluation
# Machine learning has around 60+ predictive modelling algorithms to choose from. So we must understand the type of problem at hand and solution requirement to narrow down to  few models which we can evaluate.
# 
# Here want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). As such it is a Supervised Machine Learning problem dealing with classification (Survived or not) problem.
# 
# We will be using four classification algorithms which are -
# 1. Logistic Regression
# 2. Support Vector Machine
# 3. K Nearest Neighbors and 
# 4. Random Forest
# 5. XGBoost

# Back to our problem, we now have to:
# 
# Break the combined dataset in train set and test set.
# Use the train set to build a predictive model.
# Evaluate the model using the train set.
# Test the model using the test set and generate and output file for the submission.
# Keep in mind that we'll have to reiterate on 2. and 3. until an acceptable evaluation score is achieved.
# 
# Let's start by importing the useful libraries.

# In[ ]:


from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest, SelectFromModel

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


#  We'll define a small scoring function recover the train set and the test set from the combined dataset.

# In[ ]:


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

def recover_train_test_target():
    global combined
    
    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values
    train = combined.iloc[:891]
    test = combined.iloc[891:]
    
    return train, test, targets


# In[ ]:


train, test, targets = recover_train_test_target()


# ### Feature Selection
# We've come up to more than 30 features so far. This number is quite large.
# 
# When feature engineering is done, we usually tend to decrease the dimensionality by selecting the "right" number of features that capture the essential.
# 
# In fact, feature selection comes with many benefits:
# 
# It decreases redundancy among the data
# It speeds up the training process
# It reduces overfitting
# Tree-based estimators can be used to compute feature importances, which in turn can be used to discard irrelevant features.

# In[ ]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)


# Let's have a look at the importance of each feature.

# In[ ]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)


# In[ ]:


features.plot(kind='barh', figsize=(25, 25))


# As you may notice, there is a great importance linked to Title_Mr, Age, Fare, and Sex.
# Let's now transform our train set and test set in a more compact datasets.

# In[ ]:


model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)


# In[ ]:


test_reduced = model.transform(test)
print(test_reduced.shape)


# Now we're down to a lot less features.
# 
# We'll see if we'll use the reduced or the full version of the train set.

# # Data Correlation Analysis
# We will visualize the correlation of features that we found above using heatmap. In heat map, we will use absolute values of the correlation, this make variables which have close to 0 correlation appear dark, and everything which is correlated (or anti-correlated) is bright. The shades of the color gives the relative strength of correlation.
# 
# ##### Note: np.abs() : gives absolute value element wise

# In[ ]:


corr = train.corr()
sns.heatmap(np.abs(corr), xticklabels = corr.columns, yticklabels = corr.columns)
print('Heatmap of the correlation values for the ')

corr_test = train_reduced.corr()
sns.heatmap(np.abs(corr_test), xticklabels = corr_test.columns, yticklabels = corr_test.columns)
print('Heatmap of the correlation values (reduced)')


# ## Let's try different base models

# In[ ]:


logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()
svc = SVC()
knn = KNeighborsClassifier()

models = [logreg, logreg_cv, rf, gboost, svc, knn]


# In[ ]:


for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')


# In[ ]:


for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')


# We will be using a Random Forest model. It may not be the best model for this task but we'll tune it. This work can be applied to different models.
# 
# Random Forest are quite handy. They do however come with some parameters to tweak in order to get an optimal model for the prediction task.

# >### Hyper-parameter Tuning for dealing with Overfitting and Underfitting

# In[ ]:


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {
             'max_depth' : [4, 6, 8],
             'n_estimators': [10, 50,70],
             'max_features': ['sqrt', 'auto', 'log2'],
             'min_samples_split': [0.001,0.003,0.01],
             'min_samples_leaf': [1, 3, 10],
             'bootstrap': [True,False],
             }
forest = RandomForestClassifier()
cross_validation = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(estimator = forest,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = cross_validation,
                           verbose = 1
                          )
grid_search = grid_search.fit(train_reduced, targets)
params = grid_search.best_params_

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))



# ###### Now we use the result of Best params as hyperparameters to train our final machine learning "model"  

# In[ ]:


model = RandomForestClassifier(**params)

model.fit(train_reduced, targets)
print(compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy'))



# ### So we obtain a 82+% classification accuracy with our machine learning model. We will submit our prediction to check how good we did in our test data set. We will use the predictions made from Random Forest y_pred.

# In[ ]:


y_pred = model.predict(test_reduced).astype(int)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanic.csv', index=False)

