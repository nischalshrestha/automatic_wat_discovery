#!/usr/bin/env python
# coding: utf-8

# ## Tackling the Titanic
# Rebecca Vislay Wade

# #### Overview
# This notebook describes a solution to the Titanic problem implemented in Python 3. There are many like it but this one is mine :) Thanks for checking it out!
# 
# 1. Data Load & Inspection  
# 2. Feature Engineering & Missing Value Imputation  
#     A. New variable, 'Deck'  
#     B. New variables describing family size  
#     C. New variables from 'Name'  
#     D. New variables from 'Ticket'  
#     E. Indicator variables for imputed values  
#     F. Imputing missing values of 'Fare' & 'Embarked'  
#     G. Using 'Title' to impute missing 'Age' values  
#     H. Classification tree model to impute 'Deck'  
# 3. Model Construction  
#     A. Final data prep  
#     B. Model 1: L1-Penalized Logistic Regression (0.77511 public leaderboard score)  
#     C. Model 2: ElasticNet Logistic Regression (0.77950)  
#     D. Model 3: Random Forest (*best performing model* 0.79425)  
#     E. Model 4: Gradient Boosted Tree (0.77033)  
#     F. Model 5: Gradient Boosted Tree with Reduced Predictor Set (0.75598)  
# 4. Model Comparison  
# 
# **Some of the model parameter grid searches take a long time to run.**

# ### Data Load & Inspection

# First, we import set a working directory and import the train and test csv file. I'm using the index_col argument in the read_csv function to set the index equal to 'PassengerId'.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# read individual CSVs into pandas DataFrames
train = pd.read_csv('../input/train.csv', index_col = 'PassengerId')
test = pd.read_csv('../input/test.csv', index_col = 'PassengerId')


# Here, I add a column of NaNs as placeholders for the 'Survived' variable in the test dataset and then combine it with the train data into a single dataframe, 'data'.

# In[ ]:


# add "Survived" column to test & combine into a single DataFrame
import numpy as np

test['Survived'] = np.nan
data = train.append(test)


# Take a look at the first few rows...

# In[ ]:


data.head()


# Now we'll look at a frequency table for the target variable, 'Survived'...

# In[ ]:


# look at counts for target variable 'survived'
pd.crosstab(index = data['Survived'], columns = 'Count')


# ...and the number of missing values for each variable.

# In[ ]:


data.isnull().sum()


# ### Feature Engineering & Missing Value Imputation

# Age, Cabin, Embarked, and Fare are all missing one or more values in the dataset that will have to be imputed. We will take care of these a little later. First, let's create some new variables.

# ##### Create new variable 'Deck'
# The Titanic struck the fateful iceberg at 11:40pm at night. Presumably, most passengers would have been turning in for the night. The Cabin varible contains information about where on the ship passengers most likely were when it had the collision and began to sink. Such information could be predictive of survival. 
# 
# For the passengers that have one indicated, the Cabin variable consists of a letter and a number. The letter corresponds to a deck on the ship and the number to the passenger's room(s) on that deck.

# In[ ]:


data.Cabin.head(10)


# Let's extract the letter character and make it a new variable, 'Deck'.

# In[ ]:


# Define new variable 'Deck' from first character of the string in 'Cabin'
data['Deck'] = data['Cabin'].astype(str).str[0]

# replace n with NaNs
data['Deck'] = data.Deck.replace('n', np.nan)

# check again
data.Deck.head(10)


# ##### Create new variables related to family size
# Another factor that could have impacted survival is whether or not passengers were traveling with family members or alone. The dataset contains two variables pertaining to family size: 'Parch', the number of parents and/or children the passenger was traveling with, and ' SibSp', the number of siblings and/or spouses. Together, these two variables add to give us the size of the passenger's family.

# In[ ]:


# Adding 1 (the passenger) + Parch (# of parents & children traveling with) + SibSp (# of siblings & spouses traveling with)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# check new variable
data.FamilySize.head(10)


# Let's look at FamilySize in more detail. Here's a contingency table looking at FamilySize and Survived.

# In[ ]:


# crosstab of FamilySize
pd.crosstab(data['FamilySize'], data['Survived'])


# It looks like traveling with 1-3 family members (FamilySize = 2-4) could have positively affected the probability of survival. Here's a way of visualizing that. Since Survived is coded (0,1), a barplot of Survived versus FamilySize gives us the sample proportions of of surivivors in each group. 

# In[ ]:


# make a dataframe of just FamilySize and Survived
famsizes = data[['FamilySize','Survived']]

sns.set_context('talk')
sns.barplot(x = 'FamilySize', y = 'Survived', data = famsizes, color = 'green')


# FamilySizes of 2-4 are associated with a greater than 50% chance of survival per the sample. Let's create some variables based on these observations.

# In[ ]:


# new variables (fun with list comprehensions!)
data['Alone'] = [1 if familysize == 1 else 0 for familysize in data['FamilySize']]
data['LargeFamily'] = [1 if familysize >= 5 else 0 for familysize in data['FamilySize']]
data['SmallFamily'] = [1 if familysize >= 2 and familysize < 5 else 0 for familysize in data['FamilySize']]


# Crosstabs to check for correct totals.

# In[ ]:


# crosstab of FamilySize
pd.crosstab(data['FamilySize'], 'Count')


# In[ ]:


# crosstabs of new variables
pd.crosstab(data['Alone'], 'Count')


# In[ ]:


pd.crosstab(data['LargeFamily'], 'Count') # 22 + 25 + 16 + 8 + 11 = 82


# In[ ]:


pd.crosstab(data['SmallFamily'], 'Count') # 235 + 159 + 43 = 437


# ##### What's in a name? A whole lot of information!
# The 'Name' variable is rich with potential information that could be predictive of survival.

# In[ ]:


data.Name.head()


# We begin by chopping up the Name variable to give us Title, LastName, MaidenName (in parentheses), Nickname (between double quotes), and FirstName.

# In[ ]:


# Split off last name into new column
data['LastName'] = data['Name'].str.split(',').str.get(0)
data['LastName'] = data['LastName'].str.strip() # strip whitespace from ends

# Split off Title into new column
data['Title'] = data['Name'].str.split('.').str.get(0)
data['Title'] = data['Title'].str.split(',').str.get(1)
data['Title'] = data['Title'].str.strip()

# Excise parenthetical full maiden names
data['MaidenName'] = data['Name'].str.split('(').str.get(1)
data['MaidenName'] = data['MaidenName'].str.split(')').str.get(0)

# Excise nicknames
data['Nickname'] = data['Name'].str.split('"').str.get(1)
data['Nickname'] = data['Nickname'].str.strip()

# Get first name
data['FirstName'] = data['Name'].str.split('.').str.get(1)
data['FirstName'] = data['FirstName'].str.split(' ').str.get(1)


# Married women are listed under their husband's full name with their actual first names now in MaidenName. Let's replace their husbands' names with their actual first names.

# In[ ]:


# get maiden first names
data['MaidenFirstName'] = data['MaidenName'].str.split(' ').str.get(0)
# Replace FirstName with MaidenFirstName except with the NaNs filled with 
# FirstName and strip
data['FirstName'] = data['MaidenFirstName'].fillna(data['FirstName'])
data['FirstName'] = data['FirstName'].str.strip()
# drop MaidenFirstName
data = data.drop(['MaidenFirstName'], axis = 1)


# Let's grab the wives' maiden last names, too.

# In[ ]:


# Get MaidenLastName from MaidenName
data['MaidenLastName'] = data['MaidenName'].str.rsplit(' ', expand = True, n=1)[1]
# replace 'None' with NaN and strip MaidenLastName
data['MaidenLastName'] = data.MaidenLastName.replace('None', np.nan)
data['MaidenLastName'] = data['MaidenLastName'].str.strip()
# Drop MaidenName
data = data.drop('MaidenName', axis =1)


# ##### Extracting & Combining Ticket Prefixes
# Another potentially information-rich variable is 'Ticket'. These are alphanumeric ticket codes that may indicate things such as class, fare, embarkment, and originating ticket agent. 

# In[ ]:


data.Ticket.head()


# Families or groups traveling together may have had similar prefix codes, for example. Let's extract those prefixes...

# In[ ]:


# Let's define a ticket prefix as all the letters coming before the first 
# space. Some have '.' or '/' in the letters so we first have to remove those.
data['Ticket'] = data.Ticket.str.replace('.','')
data['Ticket'] = data.Ticket.str.replace('/','')

data.Ticket.head()


# In[ ]:


# Now split at the spaces
data['TicketPrefix'] = data.Ticket.str.split().str.get(0)

# list comprehension to replace numeric values of TicketPrefix with 'None'    
data['TicketPrefix'] = ['None' if prefix.isnumeric() == True else prefix for prefix in data.TicketPrefix]

# take a look again at the cleaned up TicketPrefix Variable
data['TicketPrefix'].tail(25)


# Here is the number of passengers in each of the ticket prefix groups...

# In[ ]:


# counts of TicketPrefix
data['TicketPrefix'].value_counts()


# Some of these could be duplicate groups. For example, 'SCPARIS' and 'SCParis' are most likely the same prefix so let's combine them.

# In[ ]:


# Replace 'SCParis' with 'SCPARIS'
data.TicketPrefix = data.TicketPrefix.str.replace('SCParis', 'SCPARIS')


# Also, let's put all the prefixes with only one member into a single category called 'Unique'.

# In[ ]:


# combine prefix categories with single members into new cateogry, 'Unique'.
prefixes = data['TicketPrefix'].value_counts()
uniquePrefixes = list(prefixes[prefixes == 1].index)
data.TicketPrefix = ['Unique' if prefix in uniquePrefixes else prefix for prefix in data.TicketPrefix]

# look at value counts again
data['TicketPrefix'].value_counts()


# Excellent! Now we'll make dummy variables for each TicketPrefix category...

# In[ ]:


# make dummies for TicketPrefix
prefixDummies = pd.get_dummies(data['TicketPrefix'], prefix = 'TicketPrefix')
data = pd.concat([data, prefixDummies], axis = 1)


# ##### Make indicators for imputed values
# Sometimes the fact that a varible is missing in the dataset is predictive. Below, indicator variables are made with the suffix '_M' that take the value 1 if it was imputed and 0 if not.

# In[ ]:


# create flag variables for each column with missing values where 1 means value was imputed and 0 means value was not imputed
for column in data.columns:
    if data[column].isnull().sum() != 0:
        data[column + '_M'] = data[column].isnull()
        data[column + '_M'] = data[column + '_M'].astype('int64').replace('True', 1)
        data[column + '_M'] = data[column + '_M'].astype('int64').replace('False', 0)

# Rename the 'Survived_M' variable as Test since it identifies the members of the test set
data.rename(columns = {'Survived_M':'Test'}, inplace = True)

data.columns


# Let's drop the 'Cabin' and 'Cabin_M' variable for now since we'll focus on 'Deck' instead.

# In[ ]:


data = data.drop(['Cabin', 'Cabin_M'], axis = 1)


# ##### Impute missing values of Embarked and Fare
# Two values of Embarked and one value of Fare are missing in the dataset

# In[ ]:


# check number of missing values for each variable again
data.isnull().sum()


# Embarked is a categorical variable that can take one of three values S (Southampton), Q (Queenstown), or C (Cherbourg). 

# In[ ]:


# Look at Embarked
pd.crosstab(index = data['Embarked'], columns = 'Count') # 914 'S'


# Let's replace the two missing values of Embarked with the most common value, S.

# In[ ]:


# replace those with the most common value ('S')
data.Embarked = data.Embarked.astype('str').replace('nan', 'S')

# Check again
pd.crosstab(index = data['Embarked'], columns = 'Count') # 916 'S'


# Now we make some dummy variables.

# In[ ]:


# Make dummies for the Embarked variable
embarkedDummies = pd.get_dummies(data['Embarked'], prefix = 'Embarked')
data = pd.concat([data, embarkedDummies], axis = 1)


# Fare values vary greatly depending on the accomodation class (1st, 2nd, or 3rd) encoded by the variable 'Pclass'. First we find the value of Pclass for the passenger with the missing Fare value.

# In[ ]:


# Find the Pclass value for the passenger missing an entry for Fare
data['Pclass'][data['Fare'].isnull() == True] # Pclass = 3


# The passenger was a 3rd class ticket holder. Now we set the missing Fare value equal to the mean Fare value among 3rd class passengers.

# In[ ]:


# Set missing Fare value equal to the average for 3rd Class
data['Fare'] = data['Fare'].fillna(data.groupby('Pclass').Fare.mean()[3])


# While we're at it, let's change the 'Sex' variable so that male/female is encoded 0/1.

# In[ ]:


# replace female/male with 1/0
data['Sex'] = data['Sex'].replace('female', 1)
data['Sex'] =data['Sex'].replace('male', 0)


# ##### Exploring the 'Title' variable & imputing missing Age values
# Age could be a very important variable in predicting survival considering the 'women and children first' rule that seems to have been followed on the Titanic. As such, we want to impute the missing values of Age with a greater degree of precision than we'd have if we just used the mean or median of the entire sample.
# 
# Luckily, we have the 'Title' variable.

# In[ ]:


pd.crosstab(data['Title'], 'Count')


# Let's start by cleaning up the categories a little. First, some of the titles are non-English versions of English ones. Let's begin by changing them to their English versions.

# In[ ]:


# replace non-English, non-honorific titles with English versions
data.Title = data.Title.str.replace('Don', 'Mr')
data.Title = data.Title.str.replace('Dona', 'Mrs')
data.Title = data.Title.str.replace('Mme', 'Mrs')
data.Title = data.Title.str.replace('Ms', 'Mrs')
data.Title = data.Title.str.replace('Mra', 'Mrs')
data.Title = data.Title.str.replace('Mlle', 'Miss')

pd.crosstab(data['Title'], 'Count')


# Some titles, like Dr and Rev, we'll create specific indicator variables for.

# In[ ]:


data['Title_Dr'] = [1 if title in ['Dr'] else 0 for title in data['Title']]

data['Title_Rev'] = [1 if title in ['Rev'] else 0 for title in data['Title']]


# Others, like military or noble titles, we'll group together and create indicators for.

# In[ ]:


militaryTitles = ['Capt', 'Col', 'Major']
data['MilitaryTitle'] = [1 if title in militaryTitles else 0 for title in data['Title']]

nobleTitles = ['Jonkheer', 'Lady', 'Sir', 'the Countess']
data['NobleTitle'] = [1 if title in nobleTitles else 0 for title in data['Title']]


# Aside from those special titles, we have four categories: Master, Mr, Miss, and Mrs. For those with learned (Dr, Rev), military (Capt, Col, Major), or noble (Jonkheer, Lady, Sir, the Countess), we want to put them into one of the four major title categories. We'll start with the masculine titles. Military titles as well as 'Sir' most likely refer to men ('Mr') but 'Jonkheer' is a Dutch honorific that could be applied to a younger male (or, 'Master'). Let's chack the age of our Jonkheer...

# In[ ]:


data.loc[data['Title'].isin(['Jonkheer'])]['Age']    # 38


# Since our Jonkheer is 38 years old, let's include him in the list of titles we change to 'Mr'.

# In[ ]:


# replace male special titles with Mr.
male = dict.fromkeys(['Dr','Rev', 'Capt', 'Col', 'Major', 'Jonkheer', 'Sir'], 'Mr')
data['Title'] = data.Title.replace(male)


# Let's check the ages of the passengers with the female honorifics in Title - 'Lady' and 'the Countess'.

# In[ ]:


# check ages of the Lady and the Countess
data.loc[data['Title'].isin(['Lady'])]['Age']   # 48


# In[ ]:


data.loc[data['Title'].isin(['the Countess'])]['Age']    # 33


# It's reasonable to put both of these in the 'Mrs' category.

# In[ ]:


# replace Lady and the Countess with Mrs in Title column
female = dict.fromkeys(['Lady', 'the Countess'], 'Mrs')
data['Title'] = data.Title.replace(female)


# Remember that ultimately we'd like to use the Title information to impute missing values of Age. Here's a boxplot showing the distribution of ages in the different Title groups.

# In[ ]:


sns.set_context('talk')
sns.set_style('darkgrid')

# boxplot of Age by Title
sns.boxplot(x='Title', y='Age', data=data)


# The male title groups appear to be distinct, possibly separable distributions. This is even more evident from the histograms.

# In[ ]:


# two histogram plots, one for males + one for females
gents = ['Master', 'Mr']
colGents = ['blue', 'green']
for i in range(len(gents)):
    a_gents = sns.distplot(data[data['Title'] == gents[i]].Age.dropna(), 
                                label = gents[i],
                                color = colGents[i])
a_gents.legend()


# The situation is different for the female titles.

# In[ ]:


ladies = ['Miss', 'Mrs']
colLadies = ['orange', 'red']
for i in range(len(ladies)):
    a_ladies = sns.distplot(data[data['Title'] == ladies[i]].Age.dropna(),
                            label = ladies[i],
                            color = colLadies[i])
a_ladies.legend()


# Even though the female title distributions overlap considerably, using the median Age of the passenger's Title group to impute missing values seems reasonable.

# In[ ]:


# Masters
masters = pd.DataFrame(data[data.Title == 'Master'].Age, columns = ['Age'])
masters.Age = masters.fillna(masters.median())
data = data.combine_first(masters)

# Miss's
misses = pd.DataFrame(data[data.Title == 'Miss'].Age, columns = ['Age'])
misses.Age = misses.fillna(misses.median())
data = data.combine_first(misses)

# Mr's
misters = pd.DataFrame(data[data.Title == 'Mr'].Age, columns = ['Age'])
misters.Age = misters.fillna(misters.median())
data = data.combine_first(misters)

# Mrs's
missuses = pd.DataFrame(data[data.Title == 'Mrs'].Age, columns = ['Age'])
missuses.Age = missuses.fillna(missuses.median())
data = data.combine_first(missuses)

# check missing values
data.Age.isnull().sum()


# Let's check that Age-Title boxplot again...

# In[ ]:


sns.boxplot(x='Title', y='Age', data=data)


# Now we make some dummy variables for Title and add them to the dataframe.

# In[ ]:


# get dummies
titledummies = pd.get_dummies(data['Title'], prefix = 'Title')
# add to dataset
data = pd.concat([data, titledummies], axis=1)


# ##### Decision tree-based imputation of missing 'Deck' values
# The 'Deck' variable is missing values for ~1/3 of the total dataset. Let's have a look at a frequency table for the Deck values we *do* have.

# In[ ]:


# crosstab of Deck
pd.crosstab(data['Deck'], columns = 'Count')


# The T Deck looks like an outlier (and it was in a sense). T deck cabin belonged to Mr. Stephen Blackwell, a first class passenger.

# In[ ]:


data[data.Deck == 'T'].Name


# In[ ]:


data[data.Deck == 'T'].Pclass


# For the sake of modeling, let's add Mr. Blackwell to the exclusively 1st Class 'A' Deck.

# In[ ]:


# crosstab of Pclass and Deck
pd.crosstab(data['Deck'], data['Pclass'])


# In[ ]:


data['Deck'] = data.Deck.str.replace('T', 'A')


# The majority of 3rd class passengers would have been in steerage with no cabin designation. Let's look at the distribution of Deck among 3rd class passengers.

# In[ ]:


# fill nan's with 'missing'
data.Deck = data.Deck.fillna('missing')
# crosstab
pd.crosstab(data[data['Pclass'] == 3]['Deck'], 'Count', margins = True)


# Let's assume the 693 3rd class passengers missing a Deck were in steerage. We'll create a new category of 'Deck' called 'S' for steerage. All 3rd class passengers with missing Deck values will be assigned to the 'S' class. 

# In[ ]:


data.loc[data['Pclass'] == 3, 'Deck'] = data.loc[data['Pclass'] == 3, 'Deck'].str.replace('missing', 'S')

# crosstab of Pclass and Deck
pd.crosstab(data['Deck'], data['Pclass'], margins = True)


# That leaves us with 67 1st Class and 254 2nd Class passengers to find Deck vlaues for. I thought it would be fun to impute these using a decision tree model. First, we create train and test datasets then isolate the inputs and target for the model.

# In[ ]:


# isolate train and test subsets for Deck imputation model
train_deck = data[data['Deck'] != 'missing'].drop(['Deck_M', 'Embarked', 'Fare_M',
                 'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
                 'Survived', 'Test', 'Ticket', 'TicketPrefix', 'Title'], axis = 1)

# drop the ticket prefix indicators for now
train_deck = train_deck[['Age', 'Age_M', 'Alone', 'Deck', 'Embarked_C', 'Embarked_M', 'Embarked_Q', 'Embarked_S', 
                         'FamilySize', 'Fare', 'LargeFamily', 'MaidenLastName_M', 'MilitaryTitle', 'Nickname_M', 
                         'NobleTitle', 'Parch', 'Pclass', 'Sex', 'SibSp', 'SmallFamily', 'Title_Dr', 'Title_Master', 
                         'Title_Mr', 'Title_Mrs', 'Title_Miss']]

# since we only have 1st and 2nd class passengers to predict Deck for, we'll only use 1st and 2nd class passengers to build 
# the model
train_deck = train_deck[train_deck.Pclass != 3] 

missing_deck = data[data['Deck'] == 'missing'].drop(['Deck_M', 'Embarked', 'Fare_M',
                 'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
                 'Survived', 'Test', 'Ticket', 'TicketPrefix', 'Title'], axis = 1)

missing_deck = missing_deck[['Age', 'Age_M', 'Alone', 'Deck', 'Embarked_C', 'Embarked_M', 'Embarked_Q', 'Embarked_S', 
                             'FamilySize', 'Fare', 'LargeFamily', 'MaidenLastName_M', 'MilitaryTitle', 'Nickname_M', 
                             'NobleTitle', 'Parch', 'Pclass', 'Sex', 'SibSp', 'SmallFamily', 'Title_Dr', 'Title_Master', 
                             'Title_Mr', 'Title_Mrs', 'Title_Miss']]

# separate into inputs and outputs
X_train_deck = train_deck.drop('Deck', axis = 1)
X_missing_deck = missing_deck.drop('Deck', axis = 1)
y_train_deck = train_deck['Deck']

# feature names
X_names = X_train_deck.columns


# Here's a correlation heatmap for the predictors in the Deck training set.

# In[ ]:


# take a look at correlations among predictors for Deck
corrDeckX = X_train_deck.corr()

# plot the heatmap
fig, ax = plt.subplots()
fig.set_size_inches(14, 10)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.heatmap(corrDeckX, 
            xticklabels=corrDeckX.columns,
            yticklabels=corrDeckX.columns,
            cmap = 'PiYG')


# To tune the parameters of the tree model, we'll use a grid search. First, we construct the grid and the random forest classifier object.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV

# Set up a dict with values to test for each parameter/argument in the model object
deck_grid = {'max_depth'         : np.arange(1,30),
             'min_samples_split' : np.arange(2,20),
             'min_samples_leaf'  : np.arange(1,20)}

# Construct random forest object
tree_deck = DecisionTreeClassifier(random_state = 538, 
                                   max_features = 'sqrt',
                                   presort = True)
treeGrid = GridSearchCV(tree_deck, deck_grid)


# In[ ]:


# fit trees
treeGridFit = treeGrid.fit(X_train_deck,y_train_deck)


# In[ ]:


treeGrid.best_score_


# Here are the parameters for the model with the highest accuracy on the training set.

# In[ ]:


treeGrid.best_params_


# Now we construct a random forest classifier object with them.

# In[ ]:


# Build the best tree model
treeDeckBest = DecisionTreeClassifier(random_state = 538, 
                                      max_features = 'sqrt',
                                      presort = True,
                                      max_depth = 14,
                                      min_samples_split = 8,
                                      min_samples_leaf = 1)

treeDeckBestFit = treeDeckBest.fit(X_train_deck, y_train_deck)


# We can estimate the performance of the model out-of-sample using cross-validation. 

# In[ ]:


# calculate test accuracy estimate for best model (use default 3-fold CV)
cv_error_tree = np.mean(cross_val_score(treeDeckBest, X_train_deck, y_train_deck,
                                      scoring = 'accuracy'))
print('Est. Test Accuracy: ', cv_error_tree)


# Even though this model doesn't seem too great (accuracy < 0.5), we replace the missing values of Deck with predictions from the model and add them back to the original dataset.

# In[ ]:


# Predict the missing values of Deck
missing_deck['Deck'] = treeDeckBest.predict(X_missing_deck)

# replace 'missing' entries in data 'Deck' column with NaNs again
data.Deck = data.Deck.replace('missing', np.nan)

# use combine_first to replace missing values in data with imputed values now
# in missing_deck
data = data.combine_first(missing_deck)


# Here's the contingency table for Deck and Pclass again. We can see that, even though the model had a low accuracy, it distributed the missing values among the 1st and 2nd classes.

# In[ ]:


# crosstab of Pclass and Deck
pd.crosstab(data['Deck'], data['Pclass'], margins = True)


# Taking a look again at the missing values of each variable, we can see that only values for Nickname and MaidenName are missing. (Those missing 'Survived' are the test observations.)

# In[ ]:


data.isnull().sum()


# Finally, we make some dummy variables for the Deck variable and add them to the dataset.

# In[ ]:


# Make dummies for the imputed Deck variable
deckDummies = pd.get_dummies(data['Deck'], prefix = 'Deck')
data = pd.concat([data, deckDummies], axis = 1)


# In[ ]:


data.columns


# ## Model Construction

# ### Final preparations for model building
# We have some cleaning up and organizing to do before we build models of Survived. First, we'll separate out the test and train datasets and drop some of the variables we don't need.

# In[ ]:


# Separate out into train and test sets; drop text variables
test_model = data[data['Survived'].isnull()].drop(['Deck', 'Embarked', 'Fare_M',
            'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
            'Test', 'Ticket', 'TicketPrefix', 'Title', 'Deck_M'], axis = 1)
train_model = data[data['Survived'].notnull()].drop(['Deck', 'Embarked', 'Fare_M',
            'FirstName', 'LastName', 'MaidenLastName', 'Name', 'Nickname',
            'Test', 'Ticket', 'TicketPrefix', 'Title', 'Deck_M'], axis = 1)


# Here's the correlation heatmap for the predictors in the training set.

# In[ ]:


# take a look at correlations among the final set of predictors and the target, Survived
corrModelX = train_model.corr()

fig, ax = plt.subplots()
sns.set_context('talk')
sns.set_style('darkgrid')
fig.set_size_inches(15,10)

# plot the heatmap
sns.heatmap(corrModelX, 
            xticklabels=corrModelX.columns,
            yticklabels=corrModelX.columns,
            cmap = 'PiYG')


# Split into input and output dataframes and define a variable to hold the names of the features.

# In[ ]:


# separate into inputs and outputs
X_train = train_model.drop(['Survived'], axis = 1)
y_train = train_model['Survived']
X_test = test_model.drop(['Survived'], axis = 1)

# Feature names
features = X_train.columns


# We'll construct a few models so here we initialize a dataframe to hold the results. Along with the model name, the dataframe will contain 'TrainAcc' (the accuracy of the model on the training set) and 'TestAccCVEst' (the cross-validation estimate of the out-of-sample accuracy).

# In[ ]:


# Create a dataframe to hold model results
model_results = pd.DataFrame(columns = ['Model', 'TrainAcc', 'TestAccCVEst'])


# Finally, we set up some cross-validation schemes using the KFold function from sklearn.model_selection. I've got one for 5-fold and one for 10-fold ross-validation.

# In[ ]:


# set up 5-fold and 10-fold cross-validation schemes for test error estimation
from sklearn.model_selection import KFold
cv_5fold = KFold(n_splits = 5, shuffle = True, random_state = 237) 

cv_10fold = KFold(n_splits = 10, shuffle = True, random_state = 237)


# ### Model 1: L1-Penalized Logistic Regression

# L1-regularization can be used for variable selection as it shrinks the coefficients of some variables to 0. I try two different functions from the sklearn.linear_model package to do regularized logistic regression: LogisticRegression and SGDClassifier. For the first - LogisticRegression - I'll build an L1-penalized model. For the second, I'll construct and ElasticNet model that combines L1 and L2 (ridge) regularization.

# In[ ]:


# First, LogisticRegression:
from sklearn.linear_model import LogisticRegression
lr1 = LogisticRegression(penalty = 'l1',
                         random_state = 555,
                         solver = 'liblinear')

# fit the model
lr1_fit = lr1.fit(X_train, y_train)

# make predictions on the training set
lr1_preds = lr1.predict(X_train)

# confusion matrix for training set
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_train, lr1_preds)


# Here's a classification report...

# In[ ]:


# classification report
print(classification_report(y_train, lr1_preds,
                            target_names = ['Died', 'Survived']))


# For this model, 'precision' can be thought of as the ability of the model to pick out the true survivors and not label passengers as survivors when they in fact perished. 0.81 means that 81% of the passengers that the model labeled as survivors were actually survivors.
# 
# 'Recall' in this case would be the ability of the model to pick out all the survivors. 0.76 in this case means that the model was able to correctly identify 76% of the survivors. 
# 
# Let's explicitly calculate the in-sample training accuracy...

# In[ ]:


# training accuracy
lr1_trainAcc = lr1.score(X_train, y_train) # 0.8395
lr1_trainAcc


# ...and make an estimate of the out-of-sample test accuracy using 5-fold cross-validation.

# In[ ]:


# calculate test accuracy estimate
lr1_testErrEst = np.mean(cross_val_score(lr1, X_train, y_train,
                                         scoring = 'accuracy', 
                                         cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', lr1_testErrEst)    # 0.8283


# As mentioned earlier, L1 regularization shrinks some of the coefficients to exactly zero and in this way does variable selection. Here's a list of the predictors and their coefficients.

# In[ ]:


# construct a coefficent table
lr1_coefs = [coef for coef in lr1.coef_[0]]
featuresList = list(features)
lr1_coefs = pd.DataFrame(list(zip(featuresList, lr1_coefs)), columns = ['Feature', 'Coef'])
lr1_coefs


# Let's get a list of the non-zero coefficients.

# In[ ]:


lr1_nonzeroCoef = lr1_coefs[lr1_coefs.Coef != 0]
lr1_nonzeroCoef


# This model appears to put the strongest emphasis on Sex, Title_Master, ticket prefixes STONO and SWPP, and Deck_E.
# 
# Let's add the results to the model_results dataframe that we'll use to compare models.

# In[ ]:


# add metrics to model_results dataframe
lr1_results = pd.DataFrame([['l1LogReg', lr1_trainAcc, lr1_testErrEst]],
                           columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(lr1_results)
model_results


# These models are built for Kaggle's Titanic competition so the below code makes predictions on the test set and outputs the results to a csv file for submission.

# In[ ]:


# make predictions on test set
lr1_test = pd.DataFrame(lr1.predict(X_test).astype(int), 
                        columns = ['Survived'],
                        index = test.index)

# write to csv
lr1_test.to_csv('L1lr_test.csv')


# ### Model 2: ElasticNet
# sklearn.linear_model's SGDClassifier function can be used to produce a number of different linear models (logistic regression, SVMs), all with training by stochastic gradient descent. ElasticNet models are linear models that combine L1 and L2 regularization. The ratio between them is a tunable parameter as is 'alpha', the regularization parameter that controls the total amount of penalty applied to the linear model. 
# 
# Here I use a grid search to find optimal parameters for both alpha and the L1-to-L2 ratio.

# In[ ]:


from sklearn.linear_model import SGDClassifier

# Set up a dict with values to test for each parameter/argument in the model object
en1_grid = {'alpha'    : [0.005, 0.01, 0.015], 
            'l1_ratio' : np.arange(0, 1, 0.05)}

# SGD Classifier object; log loss makes this logistic regression
en1 = SGDClassifier(loss = 'log',
                    penalty = 'elasticnet',
                    random_state = 237,
                    learning_rate = 'optimal',
                    max_iter = 500)

# set up the grid search
en1_GridSearch = GridSearchCV(en1, en1_grid)

# fit trees
en1_fit = en1_GridSearch.fit(X_train,y_train)


# Let's look at the score and the parameters for the 'best' model.

# In[ ]:


en1_GridSearch.best_score_


# In[ ]:


en1_GridSearch.best_params_


# Construct and fit the 'best' elasticnet model.

# In[ ]:


# best model
en1_best = SGDClassifier(alpha = 0.005,
                         l1_ratio = 0.15,
                         loss = 'log',
                         penalty = 'elasticnet',
                         random_state = 237,
                         learning_rate = 'optimal',
                         max_iter = 500)

en1BestFit = en1_best.fit(X_train, y_train)


# Make predictions on the training set and construct a confusion matrix...

# In[ ]:


# make prediction on the training set
en1_preds = en1_best.predict(X_train)

# confusion matrix for training set
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_train, en1_preds)


# ...and a classification report...

# In[ ]:


# classification report
print(classification_report(y_train, en1_preds,
                            target_names = ['Died', 'Survived']))


# This model appears to do a slightly better job at picking out the true survivors. Let's get the training and estimated test accuracy to include in model_results...

# In[ ]:


# training accuracy
en1_trainAcc = en1_best.score(X_train, y_train)

# calculate test accuracy estimate for best model using 5-fold CV
en1_testErrEst = np.mean(cross_val_score(en1_best, X_train, y_train,
                                        scoring = 'accuracy',
                                        cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', en1_testErrEst)


# In[ ]:


# add metrics to model_results dataframe
en1_results = pd.DataFrame([['ElasticNet', en1_trainAcc, en1_testErrEst]],
                           columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(en1_results)
model_results


# Finally, make predictions and write to csv...

# In[ ]:


# make predictions on test set
en1_test = pd.DataFrame(en1_best.predict(X_test).astype(int), 
                       columns = ['Survived'],
                       index = test.index)

# write to csv
en1_test.to_csv('en1_test.csv')


# ### Model 3: Random Forest
# The random forest model is a favorite for the Titanic problem for several reasons. Serveral of the predictors in the dataset are correlated with each other. Such multicollinearity can cause problems for linear models, like Models 1 and 2. Random forests can handle highly correlated predictors since they are an ensemble of smaller tree models that use only a few predictors each.
# 
# Several parameters can be adjusted inan attempt to optimize a random forest model. Here we create a grid of parameter values to test and fit models using them.

# In[ ]:


# Set up a dict with values to test for each parameter/argument in the model object
rf1_grid = {'n_estimators'      : [20, 50, 100],
            'max_depth'         : np.arange(1,5),
            'min_samples_split' : np.arange(6,20,2),
            'min_samples_leaf'  : np.arange(3,15,3),
            'max_leaf_nodes'    : [5, 10, 15]}

# Construct random forest object
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(random_state = 555)

# set up the grid search
rf1_GridSearch = GridSearchCV(rf1, rf1_grid)

# fit trees
rf1_fit = rf1_GridSearch.fit(X_train,y_train)


# The best score in the grid search...

# In[ ]:


rf1_GridSearch.best_score_


# The parameters of the model with that score...

# In[ ]:


rf1_GridSearch.best_params_


# Construct the best model and fit it to the training set...

# In[ ]:


# Build the best RF model with hyperparameters determined above
from sklearn.ensemble import RandomForestClassifier
rf1_best = RandomForestClassifier(max_depth = 3,
                                  max_leaf_nodes = 15,
                                  min_samples_leaf = 3,
                                  min_samples_split = 18,
                                  n_estimators = 20)

r1fBestFit = rf1_best.fit(X_train, y_train)

# make prediction on the training set
rf1_preds = rf1_best.predict(X_train)

# confusion matrix for training set
confusion_matrix(y_train, rf1_preds)


# Classification report...

# In[ ]:


# classification report
print(classification_report(y_train, rf1_preds,
                            target_names = ['Died', 'Survived']))


# Get the in-sample and estimated out-of-sample accuracies...

# In[ ]:


# training accuracy
rf1_trainAcc = rf1_best.score(X_train, y_train)

# calculate test accuracy estimate for best model using 5-fold CV
rf1_testErrEst = np.mean(cross_val_score(rf1_best, X_train, y_train,
                                        scoring = 'accuracy',
                                        cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', rf1_testErrEst)


# Add to model_results...

# In[ ]:


# add metrics to model_results dataframe
rf1_results = pd.DataFrame([['RandomForest', rf1_trainAcc, rf1_testErrEst]],
                           columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(rf1_results)

model_results


# Make predictions on the test set and output csv...

# In[ ]:


# make predictions on test set
rf1_test = pd.DataFrame(rf1_best.predict(X_test).astype(int), 
                       columns = ['Survived'],
                       index = test.index)

# write to csv
rf1_test.to_csv('rf1_test.csv')


# Random forest models have the added benefit of providing variable importance information. We can make a bar graph of variable importances.

# In[ ]:


# look at feature importances
rf1_importances = rf1_best.feature_importances_

headers_rf1 = ["Variable", "Importance"]
values_rf1 = pd.DataFrame(sorted(zip(X_train.columns, rf1_importances), key=lambda x: x[1] * -1), columns = headers_rf1)

# horizontal bar plot of importances
fig, ax = plt.subplots()
fig.set_size_inches(12, 13)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.barplot(x = 'Importance', y = 'Variable', data = values_rf1, orient = 'h', color = 'green')


# ### Model 4: Gradient Boosted Tree Classifier
# Gradient boosting is a way to improve the predictive ability of an ensemble model through recursive addition and weighting of the weak learners averaged to construct it. In this case, we recursively construct small tree models using only a few randomly-chosen features. Each tree is added to the ensemble after being weighted based on it's individual accuracy. More accurate weak learners are weighted more and therefore contribute more to final model.
# 
# As with the random forest ensemble model, there are several parameters that require tuning in order to find the settings that produce the 'best' version of the classifier.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
# Set up a dict with values to test for each parameter/argument in the model
# object
gbc_grid = {'n_estimators'        : np.arange(20,100,20),
            'learning_rate'       : [0.05, 0.1, 0.15],
            'max_features'        : np.arange(1,6),
            'max_depth'           : np.arange(1,4),
            'min_samples_split'   : np.arange(10,20,5),
            'min_samples_leaf'    : np.arange(3,21,7),
            'subsample'           : [0.3, 0.5, 0.7],
            'max_leaf_nodes'      : np.arange(5,20,5)}


# Construct random forest object
gbc = GradientBoostingClassifier(random_state = 555)

# set up the grid search
gbc_GridSearch = GridSearchCV(gbc, gbc_grid)

# fit learners
gbc_fit = gbc_GridSearch.fit(X_train,y_train)


# Best score...

# In[ ]:


gbc_GridSearch.best_score_


# Parameters for the 'best' model...

# In[ ]:


gbc_GridSearch.best_params_


# Fit the best model and produce a confusion matrix

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbc_best = GradientBoostingClassifier(n_estimators = 40,
                                      learning_rate = 0.15,
                                      max_depth = 3,
                                      max_features = 5,
                                      min_samples_leaf = 3,
                                      min_samples_split = 10,
                                      max_leaf_nodes = 10,
                                      subsample = 0.5,
                                      random_state = 555)

gbcBestFit = gbc_best.fit(X_train, y_train)

# make prediction on the training set
gbc_preds = gbc_best.predict(X_train)

# confusion matrix for training set
confusion_matrix(y_train, gbc_preds)


# Classification report...

# In[ ]:


# classification report
print(classification_report(y_train, gbc_preds,
                            target_names = ['Died', 'Survived']))


# Training and estimated test accuracy...

# In[ ]:


# training accuracy
gbc_trainAcc = gbc_best.score(X_train, y_train)

# calculate test accuracy estimate for best model using 5-fold CV
gbc_testErrEst = np.mean(cross_val_score(gbc_best, X_train, y_train,
                                         scoring = 'accuracy',
                                         cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', gbc_testErrEst)


# Add metrics to model_results...

# In[ ]:


# add metrics to model_results dataframe
gbc_results = pd.DataFrame([['GradientBoostedTree', gbc_trainAcc, 
                             gbc_testErrEst]],
                             columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(gbc_results)

model_results


# Make predictions on the test set and write them to csv...

# In[ ]:


# make predictions on test set
gbc_test = pd.DataFrame(gbc_best.predict(X_test).astype(int), 
                        columns = ['Survived'],
                        index = test.index)

# write to csv
gbc_test.to_csv('gbc_test.csv')


# As with the random forest model, a gradient boosted classifier can provide us with information about variable importance. Here's a graph of those importances.

# In[ ]:


# look at feature importances
gbc_importances = gbc_best.feature_importances_

headers_gbc = ["Variable", "Importance"]
values_gbc = pd.DataFrame(sorted(zip(X_train.columns, gbc_importances), key=lambda x: x[1] * -1), columns = headers_gbc)

# horizontal bar plot of importances
fig, ax = plt.subplots()
fig.set_size_inches(12, 13)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.barplot(x = 'Importance', y = 'Variable', data = values_gbc, orient = 'h',color = 'green')


# It is interesting to note the differences between the variable importances from the random forest model and the gradient boosted tree. The random forest seems to put more emphasis on a passenger's sex as a predictor of survival while the gradient boosted tree sees age and fare/class as most important.

# ### Model 5: Gradient Boosted Tree with Reduced Parameter Set
# The models built so far seem to have a problem with *overfitting*. That is, they learn to predict survivor status on the training set *so well* that they have a difficult time with other data. One potential issue is that there are predictors in the models that aren't associated with survival status at all. These predictors only contribute noise. In a sense, an overfitting model 'learns' the noise and that can diminish the model's performance on data other than that with which it was trained.
# 
# To explore this possibility, I've constructed a second gradient boosted tree model on a subset of the predictors used to build the previous four models. The subset consists of the top 25 most important variables as determined from the previous gradient boosted model.

# In[ ]:


# let's take the top 25 predictors
reducedFeatures = values_gbc.Variable.head(25)

# Extract only those features from the test and train sets
test_model_red = test_model[reducedFeatures]
train_model_red = train_model[reducedFeatures]
train_model_red.loc[:,'Survived'] = train_model['Survived']


# Let's look at the correlations between this new predictor set and Survived...

# In[ ]:


# Correlations in the new training set
corrRedModelX = train_model_red.corr()

fig, ax = plt.subplots()
sns.set_context('talk')
sns.set_style('darkgrid')
fig.set_size_inches(15,10)

# plot the heatmap
sns.heatmap(corrRedModelX, 
            xticklabels=corrRedModelX.columns,
            yticklabels=corrRedModelX.columns,
            cmap = 'PiYG')


# Separate into inputs and outputs...

# In[ ]:


# separate into inputs and outputs
X_train_red = train_model_red.drop(['Survived'], axis = 1)
y_train_red = train_model_red['Survived']
X_test_red = test_model_red


# Do our grid search for good parameters...

# In[ ]:


# Set up a dict with values to test for each parameter/argument in the model
# object
gbcRed_grid = {'n_estimators'        : [50, 75, 100],
               'learning_rate'       : [0.05, 0.1, 0.15],
               'max_features'        : np.arange(1,4),
               'max_depth'           : np.arange(1,4),
               'min_samples_split'   : np.arange(8,20,4),
               'min_samples_leaf'    : [5, 7, 9],
               'subsample'           : [0.3, 0.5, 0.7],
               'max_leaf_nodes'      : [10, 15, 20]}

# Construct random forest object
gbcRed = GradientBoostingClassifier(random_state = 555)

# set up the grid search
gbcRed_GridSearch = GridSearchCV(gbcRed, gbcRed_grid)

# fit trees
gbcRed_fit = gbcRed_GridSearch.fit(X_train_red,y_train_red)


# Best score from the grid search...

# In[ ]:


gbcRed_GridSearch.best_score_


# Parameters of the 'best' model...

# In[ ]:


gbcRed_GridSearch.best_params_


# Construct the 'best' model, fit it, make predictions on the training set, and produce a confusion matrix..

# In[ ]:


# Build the best model
gbcRed_best = GradientBoostingClassifier(n_estimators = 80,
                                         learning_rate = 0.15,
                                         max_depth = 3,
                                         max_features = 4,
                                         min_samples_leaf = 3,
                                         min_samples_split = 15,
                                         max_leaf_nodes = 10,
                                         subsample = 0.7,
                                         random_state = 555)

gbcRedBestFit = gbcRed_best.fit(X_train_red, y_train_red)

# make prediction on the training set
gbcRed_preds = gbcRed_best.predict(X_train_red)

# confusion matrix for training set
confusion_matrix(y_train_red, gbcRed_preds)


# Classification report...

# In[ ]:


# classification report
print(classification_report(y_train_red, gbcRed_preds,
                            target_names = ['Died', 'Survived']))


# This model appears to be the best preforming in terms of precision and recall on the training set. 
# 
# Here's the training and estimated test accuracy...

# In[ ]:


# training accuracy
gbcRed_trainAcc = gbcRed_best.score(X_train_red, y_train_red)

# calculate test accuracy estimate for best model using 1-fold CV
gbcRed_testErrEst = np.mean(cross_val_score(gbcRed_best, X_train_red, y_train_red,
                                         scoring = 'accuracy',
                                         cv = cv_5fold))
print('Est. Test Accuracy of Best Model: ', gbcRed_testErrEst)


# Add metrics to model_results...

# In[ ]:


# add metrics to model_results dataframe
gbcRed_results = pd.DataFrame([['GBC_ReducedX', gbcRed_trainAcc, 
                                 gbcRed_testErrEst]],
                              columns = ['Model', 'TrainAcc', 'TestAccCVEst'])
model_results = model_results.append(gbcRed_results)

model_results


# Make predictions on the test set and write to csv...

# In[ ]:


# make predictions on test set
gbcRed_test = pd.DataFrame(gbcRed_best.predict(X_test_red).astype(int), 
                        columns = ['Survived'],
                        index = test.index)

# write to csv
gbcRed_test.to_csv('gbcRed_test.csv')


# The variable importances from the boosted model on the reduced dataset...

# In[ ]:


# look at feature importances
gbcRed_importances = gbcRed_best.feature_importances_

headers_gbcRed = ["Variable", "Importance"]
values_gbcRed = pd.DataFrame(sorted(zip(X_train_red.columns, gbcRed_importances), key=lambda x: x[1] * -1), 
                             columns = headers_gbcRed)

# horizontal bar plot of importances
fig, ax = plt.subplots()
fig.set_size_inches(12, 13)
sns.set_context('talk')
sns.set_style('darkgrid')
sns.barplot(x = 'Importance', y = 'Variable', data = values_gbcRed, orient = 'h', color = 'green')


# ### Final Model Comparison
# Conveniently, we've saved the in- and estimated out-of-sample accuracy metrics for each of the five models in the dataframe, model_results. We'll add a column for the public leaderboard results.

# In[ ]:


# Add a column with Kaggle public leaderboard results
model_results['PublicLeaderboard'] = [0.77511, 0.77950, 0.79425, 0.77033, 0.75598]


# In[ ]:


model_results

