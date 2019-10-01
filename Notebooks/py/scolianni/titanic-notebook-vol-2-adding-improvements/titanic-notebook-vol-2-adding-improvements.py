#!/usr/bin/env python
# coding: utf-8

# ## This is an improvement on my first Kaggle notebook
# My goal here is to do some simple improvements on my previous notebook.  
# 
# ### Possible Improvements
# 
# 1.  Instead of default values for NaN's, clean missing values with column averages.
#     - Done
# 2.  Remove columns that offer little or no information.  Eg. PassengerId.
#     - Done
# 3.  Add additional features (eg. related to family size, etc.).
#     - Done
# 4.  Assess the accuracy by implementing k-fold cross validation.
#     - Done
# 5.  Implement Gird Search to tune hyper parameters
#     - Done
# 6.  Learn how to make an evaluation for how valuable each feature is.  Remove unimportant features.
#     - Done
# 7.  Add in other common classifiers to test
# 8.  Add in some boosting method.

# ## Read in the data

# In[ ]:


# The first order of business, as always, is to read in the mother fucking data

import pandas as pd
dfTrain = pd.read_csv('../input/train.csv')
dfTest = pd.read_csv('../input/test.csv')

dfTrain.head()


# ## Look at the format of the data
# 
# We need to know which columns contain NaN's.

# In[ ]:


dfTrain.info()


# We can see that there are 891 rows, but the Age, Cabin, and Embarked columns have fewer.  This means that these columns have NaN's.

# ## Analyze features
# 
# We now need to do two things:
# 
# 1.  Calculate the average probability of death and survival.
# 2.  Calculate the conditional probability of death and survival given each feature.
# 
# This will allow us to tell which features contain useful information.

# ### Average chance of death and survival

# In[ ]:


dfTrain['Survived'].value_counts(normalize=True)


# In[ ]:


import seaborn as sns
sns.countplot(dfTrain['Survived'])


# We can see that the average probability of death is ~62%, and the average probability of survival is ~38%.

# ### PassengerId
# 
# The PassengerId feature is unique to each passenger.  As such, it logically should have no predictive power.  therefore, this attribute will be removed from the dataframes before testing.

# ### Pclass
# 
# Pclass refers to the class of the ticket.  For some reason, I am suspecting that the attribute of having money is going to make one statistically better at coping with icebergs.  Why could this be?

# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Pclass']).mean()


# In[ ]:


sns.countplot(dfTrain['Pclass'])


# In[ ]:


sns.countplot(dfTrain['Pclass'], hue=dfTrain['Survived'])


# We can see that class is an important indicator for survival.  The average person has a 38% chance or survival, but if you're in first or second class, your chances are significantly higher.  If you are in third class, your chances are significantly lower.
# 
# This could be due to bias when filling the life boats.  It could also be that the lower class passengers were in the lower decks, and thus were the last to get to the life boats / were the worst and first affected by the disaster.

# ### What's in a Name?
# 
# The Name feature, in theory, would be unique like PassengerId.  There is some interesting information that could be unpacked here, however.  A person's title (Mr., Mrs., Fr., etc.) might contain information about the probability of survival.  Additionally, the length of a person's name might similarly contain information about survival, as snooty important people are likely to have longer names.  In a similar light, I define a new attribute called "Name_Complexity" which equals the number of distinct names in the original Name.  This also is based on the theory that snooty people usually have more names.

# In[ ]:


dfTrain['Name_Len'] = dfTrain['Name'].apply(lambda x: len(x))
pd.qcut(dfTrain['Name_Len'], 5).value_counts()


# In[ ]:


dfTrain['Survived'].groupby(pd.qcut(dfTrain['Name_Len'], 5)).mean()


# In[ ]:


dfTrain['Name_Title'] = dfTrain['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
dfTrain['Name_Title'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Name_Title']).mean()


# In[ ]:


dfTrain['Name_Complexity'] = dfTrain['Name'].apply(lambda x: len(x.split()))
dfTrain['Name_Complexity'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Name_Complexity']).mean()


# ### Sex
# 
# Women will likely have a higher survival rate due to chivalry.

# In[ ]:


dfTrain['Sex'].value_counts(normalize=True)


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Sex']).mean()


# ### Age
# 
# I'm guessing the young will survive better.  Or maybe not, as survival is biased towards the upper class, and young people probably couldn't afford a first class ticket.

# In[ ]:


pd.qcut(dfTrain['Age'], 5).value_counts()


# In[ ]:


dfTrain['Survived'].groupby(pd.qcut(dfTrain['Age'], 5)).mean()


# ### SibSp & Parch
# 
# SibSp and Parch will be added together to form a new feature called "Family_Size".

# In[ ]:


dfTrain['FamilySize'] = dfTrain['SibSp'] + dfTrain['Parch']
dfTrain['FamilySize'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['FamilySize']).mean()


# ### Ticket
# 
# The Ticket code is hard to understand, so we will have to make some guesses.  
# 
# 1.  The first hypothesis we'll check is to see if ticket name length is a good predictor.
# 2.  The second hypothesis we'll check is to see if tickets with all numbers vs. tickets with numbers and other characters is a good predictor.
# 3.  The third hypothesis we'll check is to see if tickets with spaces vs. not is a good predictor. 

# In[ ]:


dfTrain['Ticket_Len'] = dfTrain['Ticket'].apply(lambda x: len(x))
dfTrain['Ticket_Len'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Ticket_Len']).mean()


# We can see that ticket length is indeed a good predictor.

# In[ ]:


dfTrain['Ticket_Characters'] = dfTrain['Ticket'].apply(lambda x: x.isdigit())
dfTrain['Ticket_Characters'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Ticket_Characters']).mean()


# In[ ]:


del dfTrain['Ticket_Characters']


# We can see that the numerical only vs. numerical and other character tickets is not a good feature to divide on.

# In[ ]:


dfTrain['Ticket_Spaces'] = dfTrain['Ticket'].apply(lambda x: len(x.split()))
dfTrain['Ticket_Spaces'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Ticket_Spaces']).mean()


# We see that tickets with three spaces have slight predictive power.

# ### Fare
# 
# Lets sort the fare into five bins.

# In[ ]:


pd.qcut(dfTrain['Fare'], 3).value_counts()


# In[ ]:


dfTrain['Survived'].groupby(pd.qcut(dfTrain['Fare'], 5)).mean()


# In[ ]:


dfTrain['Survived'].groupby(pd.qcut(dfTrain['Fare'], 3)).mean()


# ### Cabin
# 
# We will see if the cabin letter makes for a good feature.

# In[ ]:


dfTrain['Cabin_Letter'] = dfTrain['Cabin'].apply(lambda x: str(x)[0])
dfTrain['Cabin_Letter'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Cabin_Letter']).mean()


# In[ ]:


dfTrain.head()


# ### Which new features have been added?
# 
# 1.  Name_Len
# 2.  Name_Title
# 3.  Name_Complexity
# 4.  FamilySize
# 5.  Ticket_Len
# 6.  Ticket_Spaces
# 7.  Cabin_Letter

# ## Replace NaN's
# 
# Recall that in the training set, the Age, Fare, Cabin, and Embarked columns have NaN's.
# 
# 1.  Check the test set to see which columns have NaN's.
# 2.  Clean the NaN's.

# In[ ]:


dfTest.info()


# We can see that dfTest has 418 rows, and that the following columns have NaN's:  Age, Fare, Cabin, and Embarked.

# ### Cleaning process
# 
# For Age, we will replace NaN's with the avg passenger age.
# 
# For fare, we will replace the NaN's with the avg passenger faire.
# 
# For Cabin, we will replace the NaN's with 'znan'.
# 
# For Embarked, we will replace the NaN's with 'znan'.

# In[ ]:


dfCombined = pd.concat([dfTrain, dfTest])
dfTrain['Age'] = dfTrain['Age'].fillna(dfCombined['Age'].mean())
dfTrain['Fare'] = dfTrain['Fare'].fillna(dfCombined['Fare'].mean())
dfTrain['Cabin'] = dfTrain['Cabin'].fillna('znan')
dfTrain['Embarked'] = dfTrain['Embarked'].fillna('znan')

dfTest['Age'] = dfTest['Age'].fillna(dfCombined['Age'].mean())
dfTest['Fare'] = dfTest['Fare'].fillna(dfCombined['Fare'].mean())
dfTest['Cabin'] = dfTest['Cabin'].fillna('znan')
dfTest['Embarked'] = dfTest['Embarked'].fillna('znan')

print('done')


# ## Functions (add new new features)
# 
# We will now create a series of functions toadd the aforementioned new features.

# In[ ]:


def manipulateNames(iset):
    iset['Name_Len'] = iset['Name'].apply(lambda x: len(x))
    iset['Name_Title'] = iset['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    iset['Name_Complexity'] = iset['Name'].apply(lambda x: len(x.split()))
    del iset['Name']
    return iset   


# In[ ]:


def manipulateFamily(iset):
    iset['FamilySize'] = iset['SibSp'] + iset['Parch']
    return iset


# In[ ]:


def manipulateTicket(iset):
    iset['Ticket_Len'] = iset['Ticket'].apply(lambda x: len(x))
    iset['Ticket_Spaces'] = iset['Ticket'].apply(lambda x: len(x.split()))
    return iset


# In[ ]:


def manipulateCabin(iset):
    iset['Cabin_Letter'] = iset['Cabin'].apply(lambda x: str(x)[0])
    return iset


# ### Apply functions
# 
# We will now apply these functions to the dfTest data frame.

# In[ ]:


dfTrain = manipulateNames(dfTrain)
dfTrain = manipulateFamily(dfTrain)
dfTrain = manipulateTicket(dfTrain)
dfTrain = manipulateCabin(dfTrain)

dfTest = manipulateNames(dfTest)
dfTest = manipulateFamily(dfTest)
dfTest = manipulateTicket(dfTest)
dfTest = manipulateCabin(dfTest)

dfTrain.head()


# In[ ]:


dfTest.head()


# In[ ]:


dfTrain.info()


# ## Encode data
# 
# Encode all features except PassengerId, as this needs to be untouched for the Kaggle grading script to run properly.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
dfCombined = pd.concat([dfTrain, dfTest])
for feature in list(dfCombined):
    
    le = LabelEncoder()
    le.fit(dfCombined[feature])
    
    if feature in dfTrain:
        if feature != 'PassengerId':
            dfTrain[feature] = le.transform(dfTrain[feature])
    if feature in dfTest:
        if feature != 'PassengerId':
            dfTest[feature] = le.transform(dfTest[feature])


# ## Perform K-Fold Cross Validation
# 
# We are using Stratified K-Fold to keep folds balanced.
# 
# Sklearn can take a dataframe as input to its functions??  Cool!
# 
# We use the Random Forest classifier as it is probably a good choice.

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

X = dfTrain.drop(['Survived', 'PassengerId'], axis=1)
y = dfTrain['Survived']
accuracyList = []

classifierParams = {'n_estimators':[10, 20, 40, 80, 200], 'criterion':['gini', 'entropy'], 'max_features':[.10, .20, .40, .80]}
skf = StratifiedKFold(n_splits=10, shuffle=True)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]

    clf = RandomForestClassifier()
    clf = RandomizedSearchCV(clf, classifierParams[classifierName])

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    accuracyList.append(accuracy)
        
print(np.mean(accuracyList[classifierName]))


# ## Form predictions on test set

# In[ ]:


# Generate predictions
parameters = {'n_estimators':[10, 20, 40, 80], 'criterion':['gini', 'entropy'], 'max_features':[.10, .20, .40, .80]}
clf = RandomForestClassifier()
clf = RandomizedSearchCV(clf, parameters)
clf = AdaBoostClassifier(clf, n_estimators=200)
clf.fit(X, y)
dfTestPredictions = clf.predict(dfTest.drop(['PassengerId'], axis=1))

# Write predictions to csv file
results = pd.DataFrame({'PassengerId': dfTest['PassengerId'], 'Survived': dfTestPredictions})
results.to_csv('results.csv', index=False)
results.head()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Ticket_FirstLetter']).mean()


# We can see that the first letter in the ticket is a useful attribute to consider.

# ### Fare
# 
# In considering fare, we will brake the prices into five bins and see if bin number is a significant indicator.

# In[ ]:


pd.qcut(dfTrain['Fare'], 5).value_counts()


# In[ ]:


dfTrain['Survived'].groupby(pd.qcut(dfTrain['Fare'], 5)).mean()


# ### Cabin
# 
# First, we will check if the initial cabin letter is a good feature.  Then we will check if the cabin number that comes afterward is a good feature.

# In[ ]:


dfTrain['Cabin_Letter'] = dfTrain['Cabin'].apply(lambda x: str(x)[0])
dfTrain['Cabin_Letter'].value_counts()


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Cabin_Letter']).mean()


# ### Embarked
# 
# Lets look at the data related to embarking.

# In[ ]:


dfTrain['Embarked'].value_counts()


# In[ ]:


dfTrain['Embarked'].value_counts(normalize=True)


# In[ ]:


dfTrain['Survived'].groupby(dfTrain['Embarked']).mean()


# In[ ]:


dfTrain['Ticket_FirstLetter'] = dfTrain['Ticket'].apply(lambda x: x[0])
dfTrain['Ticket_FirstLetter'].value_counts()

