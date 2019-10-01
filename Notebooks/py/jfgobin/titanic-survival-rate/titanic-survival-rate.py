#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is my first attempt at a Kaggle Competition. This work is inspired by Megan Risdal's script ([here][1]) and Ben Hammer's intro to Random Forest Benchmark in R ([here][2]).
# 
# Many shouts and tips of the hat to the both of them for the scripts and notebooks, from which I learned a lot.
# 
# # Missing data
# 
# The dataset is not complete: for example, some passengers lacks the cabin number, which includes the deck letter. This could have been useful: the G-deck (2nd and 3rd classes) was below the water level ([Plans][3]). It is less likely that people asleep at that level would have had the time to leave their cabins and rush to the Boat deck, 20 meters above them, where the lifeboats were in time to get in a lifeboat - one of the main contributors to the tragedy was the lack of enough lifeboats compared to the number of passengers on board.
# 
# Some of the values, such as the *Cabin* field won't be reconstructed. The *Embark* and *Age* fields will be reconstructed in the training set then in the test set as well.
# 
# ## Fixing *Embarked*
# 
# Megan uses the median value to determine the port of origin, that is she computes for each class the distribution of the fare paid from each port and assigns the port that is the likeliest to be the origin, that is whose median fare is the closest to the fare paid by the traveler for his or her Pclass. As there are only two values missing in the training set and none in the test set, I won't replicate her code and simply fill in the blanks.
# 
# ## Fixing *Age*
# 
# For this, Megan opted to use the Multiple Imputations using Chained Equations (MICE). This algorithm is provided by the Python package **fancyimpute**. However, after a few tries and some frustrations with the (in)famous "isnan" message, I decided to write something myself. That's not elegant but that does the job.
# 
# Basically, for the each *Pclass*, *Sex* and *Title* variables, I select at random an age in the existing distribution and reassign it. As the imputation is random, there is a fairly large chance that results will change from run to run. A potential workaround is to run this a few times and select the most frequent result for the predicted *Survived* variable.
# 
# ## Fixing *Fare*
# 
# Only one value is missing, so I took the median for people in a relatively similar situation.
# 
# # Trip of the Titanic
# 
# The Titanic was bound for New York, the Titanic left England from Southampton, docked in Cherbourg (France) and the next day in Queenstown (Ireland, nowadays Cobh). It sank five days in its trip, about two days away from its destination.
# 
# While several questions will always remain unanswered, it is certain that a conjunction of factors led to the tragedy: the old mentality of "having lifeboats to carry passengers from a sinking boat to a rescuing boat", the 
# 
#   [1]: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic
#   [2]: https://www.kaggle.com/benhamner/titanic/random-forest-benchmark-r/code
#   [3]: http://titanicwhitestar.e-monsite.com/pages/les-plans-les-ponts-et-interieurs-du-rms-titanic.html

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np                # linear algebra
import pandas as pd               # data processing, CSV file I/O (e.g. pd.read_csv)
#import fancyimpute as fi         # MICE completions
import re                         # Regular expressions
import matplotlib.pyplot as plt   # Plot various stuff
import sklearn                    # Learning algorithm
import sklearn.tree               # Decision trees
import sklearn.preprocessing      # Preprocessors
import sklearn.neighbors          # k-Nearest Neighbours
import sklearn.naive_bayes        # Naive Bayes
import sklearn.svm                # C-Support SVM
import sklearn.linear_model       # Logistic regression
import mlxtend.classifier         # Stacking Classifier
import sklearn.feature_extraction # Extract features through DictVectorizer

# Load the files

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# The columns are
# PassengerID  : The passenger's ID
# Survived     : 1 if the passenger survived, 0 otherwise
# Pclass       : Passenger class, which can be considered a proxy for social status
# Name         : Passenger's name
# Sex          : Passenger's sex
# Age          : Passenger's age
# SibSp        : Number of siblings or spouses aboard
# Parch        : Number of parents or children aboard
# Ticket       : Ticket number
# Fare         : Fare paid, another variable to indicate social status
# Cabin        : Denomination of the cabin
# Embarked     : Port of embarkation

# Some of the data is missing, for example

# 10	1	2	Nasser, Mrs. Nicholas (Adele Achem)	female	14	1	0	237736	30.0708		C
# Which lacks the cabin.

# Which data is complete and useful
# PassengerID, unlikely to be useful (this is a unique ID), but needed to uniquely identify a passenger
# Survived, which is what we need to predict
# Pclass, very useful (Survival rate 1st class is +/-63%, 3rd class is about 24%)
# Name, could be useful for the last name as it will group the families together
# Sex, very useful as this was still a time when "children and ladies first" was applied during naufrage
# Age, would be useful but a 177 rows are missing. Maybe we can reconstruct this.
# SibSp, useful: it shows that single people (SibSp==0) had less chance of survival than couples 
#                or parent-children (53%), but then that the higher the SibSp, the lower the survival rate
# Parch, useful, similar behaviour as SibSp
# Ticket, could be useful, but need some cleanup. Normally numeric but some have characters attached
# Fare, likely to be useful
# Cabin, could have been useful, but there are too many values missing in the training set. Instead the
# Pclass will be used.
# Embarked, useful.

# Augmentation of the dataset
# From the data provided we will derive
# LastName   - the element before the "," in the Name field
# TicketNum  - the numeric part of the ticket

# Prepare the dataset
# Separation of the last name from the name
train['LastName'] = train['Name'].apply(lambda x: x.split(',')[0])
test['LastName'] = test['Name'].apply(lambda x: x.split(',')[0])
# Cleaning of the ticket value
train['TicketNum'] = train['Ticket'].apply(lambda x: x[x.find(' ')+1:])
test['TicketNum'] = test['Ticket'].apply(lambda x: x[x.find(' ')+1:])
# Separate the title from the rest, children were often called "Miss" or "Master" at the time
title_re = re.compile("^.[^,]+, ([^ ]+) .*$")
train['Title'] = train['Name'].apply(lambda x: title_re.search(x).group(1))
test['Title'] = test['Name'].apply(lambda x: title_re.search(x).group(1))

# Check if we have missing data
print("=== BEFORE IMPUTERS ===\n")
print("=== Check for missing values in training set ===")
for iname in train.columns.values.tolist():
    nna = sum(pd.isnull(train[iname]))
    print(repr(iname).rjust(16), repr(nna).rjust(4))
    
print("\n\n=== Check for missing values in test set ===")
for iname in test.columns.values.tolist():
    nna = sum(pd.isnull(test[iname]))
    print(repr(iname).rjust(16), repr(nna).rjust(4))

# Complete / impute the datasets

# In the training set, two rows miss the Embarked value, and 177 the Age value. In the test set, 86
# miss the Age value and 1 the Fare value.
# Megan determined that the two missing Embarked are likely to be "C" (Cherbourg), which we will
# simply fill. If this is against the rule, I will simply port the algorithm.
train.loc[[61,829], 'Embarked']='C'

# Fill the missing ages
# For the missing ages, we will consider that the following variables are representative of the age:
# Pclass, Sex, Title
# Name - unlikely to be helpful as this is likely to be almost unique
# LastName - unlikely to be helpful
# SibSp/Parch - could be useful, but need a lot of work
# Ticket - unlikely to be useful, likely to be unique
# Fare - unlikely to be useful
# Cabin - could have been useful if there wasn't so many missing values
# Embarked - could have been useful, but this chops the dataset into too many fragments
# For each useful variable, I will iterate through the possible values (They are all categoricals)
# and if there are any missing value:
# Find the min, the max and select a value at random between these.
# Now comes the difficult choice of what distribution to use
# Uniform? T-Student? Gamma? Something else?
# Major drawback - as there is an element of randomness, the behavior can change from run to run
# Some data
# In the training set
# Pclass  Title   Sex   Number NA  Number Entries  Percentage NA  Age Min  Age Max  Age Median
#      3  Mr.     male         90             229          39.30       11       74          26
#      3  Mrs.    female        9              33          27.27       15       63          31
#      3  Miss.   female       33              69          47.83        0       45          18
#      3  Master. male          4              24          16.67        0       12           4
#      1  Mr.     male         20              87          22.99       17       80          40
#      1  Mrs.    female        8              34          23.53       17       62          41.5
#      1  Miss.   female        1              45           2.22        2       63          30
#      1  Dr.     male          1               3          33.33       32       50          44
#      2  Mr.     male          9              82          10.98       16       70          31
#      2  Miss.   female        2              32           6.25        2       50          24
# The most problematic imputation will be 3rd Class/Female/Miss which have almost half of 
# the entries missing the age. It seems the term "Miss" was used in a variety of circumstances across
# the three classes. Was that intentional or by mistake?
def age_imputer(df_to_imp, set_name, doplot=False):
    for a1 in df_to_imp['Pclass'].unique():
        for a2 in df_to_imp['Title'].unique():
            for a3 in df_to_imp['Sex'].unique():
                #query = {'Pclass': [a1],
                #         'Sex': [a3],
                #         'Title': [a2]}
                #mask = train[['Pclass','Title','Sex']].isin(query).all(1)
                v1 = df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&(df_to_imp.Sex==a3)]['Age'].count()
                v2 = sum(np.isnan(df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&                                            (df_to_imp.Sex==a3)]['Age']))
                if (v1 > 0) and (v2 > 0):
                    age_dist = np.array(df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&                                                  (df_to_imp.Sex==a3)]['Age'])
                    age_dist = age_dist[np.logical_not(np.isnan(age_dist))]
                    age_dist_before = age_dist.copy()
                    # To make it simple - if will select as many elements as I have NAs to replace
                    # in the list of existing values with respect to distribution. At least the distribution
                    # is respected
                    new_age = np.random.choice(age_dist,v2,replace=True)
                    # Impute the missing values
                    df_to_imp.loc[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&(df_to_imp.Sex==a3)&                                  (pd.isnull(df_to_imp.Age)),'Age'] = new_age
                    if doplot:
                        # Plot distributions (old and new)
                        plt.subplot(2,1,1)
                        plt.title(("Distribution for Pclass: {0:d}, Sex: {1}, Title: {2}\n" +                                   "(imputed: {3:d} value{4})\n(Set name: {5})").format(
                                  a1, a3, a2, v2, 's' if (v2>1) else '', set_name))
                        plt.hist(age_dist_before, normed=True)
                        plt.subplot(2,1,2)
                        plt.hist(df_to_imp[(df_to_imp.Pclass==a1)&(df_to_imp.Title==a2)&                                           (df_to_imp.Sex==a3)]['Age'], normed=True)
                        plt.show()

age_imputer(train, "Train")
age_imputer(test, "Test")


print("=== AFTER IMPUTERS ===\n")
print("=== Check for missing values in training set ===")
for iname in train.columns.values.tolist():
    nna = sum(pd.isnull(train[iname]))
    print(repr(iname).rjust(16), repr(nna).rjust(4))
    
print("\n\n=== Check for missing values in test set ===")
for iname in test.columns.values.tolist():
    nna = sum(pd.isnull(test[iname]))
    print(repr(iname).rjust(16), repr(nna).rjust(4))

# At this point, there is still a row with age == NULL
# She is the only "Ms." "female" "3rd Pclass"
# Given that she is a single lady (Parch==SibSp==0),  let's assign her
# The median age of the single ladies after the coming of age

test.loc[(test.PassengerId==980), 'Age'] = np.median(test.loc[(test.Pclass==3)&                                                              (test.Sex=='female')&                                                              (test.Parch==0)&                                                              (test.SibSp==0)&                                                              (test.Age>17),'Age'])

# At this point, we are done with completing Age. Let's take care of the fare.
# Fare is pretty easy - only one value is missing in the test dataset.
# He is in 3rd Pclass, left from Southampton.
# Let's assign him the median of the tickets of childless single men who embarked in Southampton
# in third class over 40 years old.

test.loc[(test.PassengerId==1044),'Fare'] = test.loc[(test.Pclass==3)&                                                     (test.Embarked=='S')&                                                     (test.Parch==0)&                                                     (test.SibSp==0)&                                                     (test.Age>40)&                                                     -(pd.isnull(test.Fare)),'Fare'].median()

# Some of the entries have a Fare of 0.0. I do not know if this is normal or not, and in doubt
# I will leave them that way. One possible explanation is they were part of the personnel 
# of either White Star Line or Harland and Wolff.

# Well, that is it. The datasets are relatively complete, imputed where a value was missing 
# so we are ready to rock.

# Initially, I had written my own stuff to rank the classifiers, but sklearn has
# its own procedures.
train_set = train.copy()
# Keep only the relevant variables
train_set_data = train_set[['Pclass', 
                            'Sex',
                            'Age',
#                            'SibSp',
#                            'Parch',
                            'Fare']].copy()
train_set_target = train_set['Survived'].copy()
# Classifiers does not use categorical but need labels
le = sklearn.preprocessing.LabelEncoder()
for i_label in ['Sex']:
    train_set_data.loc[:,i_label] = le.fit_transform(train_set_data.loc[:,i_label])
# Dict vectorizer is nice, but that does not guarantee I will have the some feature for both sets
#dv = sklearn.feature_extraction.DictVectorizer(sparse=False)
#train_set_data_dv = dv.fit_transform(train_set_data.to_dict(orient='records'))
# Select and prepare the classifiers
# Decision tree
clf_dt = sklearn.tree.DecisionTreeClassifier()
# k-nearest neighbours (n=5, default)
clf_knn = sklearn.neighbors.KNeighborsClassifier()
# k-nearest neighbours (n=3)
clf_knn3 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
# Naive Bayes
clf_nb = sklearn.naive_bayes.GaussianNB()
# SVM
clf_svc = sklearn.svm.SVC()
# Stacking classifier
lr = sklearn.linear_model.LogisticRegression()
clf_st = mlxtend.classifier.StackingClassifier(classifiers=[clf_dt,clf_knn,clf_nb,clf_svc], 
                                               meta_classifier=lr)
# Look at the scores
print("\n\n")
print("3-fold cross validation")
print("=======================\n")
for clf,clf_name in zip([clf_dt, clf_knn, clf_knn3, clf_nb, clf_svc, clf_st],
                        ['Decision Tree', 'k-Nearest Neighbors (5)',
                         'k-Nearest Neighbors (3)', 'Naive Bayes',
                         'Stacking Classifier (lr)']):
    scores = sklearn.model_selection.cross_val_score(clf,
                                                     train_set_data,
                                                     train_set_target,
                                                     cv=3, 
                                                     scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), clf_name))

# Interestingly, Naive Bayes outperforms everybody else
# Still, I choose the stacked classifier
clf_ch = clf_st
# Prep the test set to retain the features
test_set = test.copy()
test_set_data = test_set[['Pclass', 
                          'Sex',
                          'Age',
#                          'SibSp',
#                          'Parch',
                          'Fare']].copy()
#test_set_data_dv = dv.fit_transform(test_set_data.to_dict(orient='records'))
for i_label in ['Sex']:
    test_set_data.loc[:,i_label] = le.fit_transform(test_set_data.loc[:,i_label])
# And use the model
clf_ch = clf_ch.fit(train_set_data,train_set_target)
test_set['Survived'] = clf_ch.predict(test_set_data)
# As a final verification, check the survival rates

train_sv_rate = (100.0*sum(train['Survived'])) / (1.0*len(train))
test_sv_rate = (100.0*sum(test_set['Survived'])) / (1.0*len(test_set))

print("Survival rates")
print("==============\n")
print("Train set: %2.2f"%(train_sv_rate))
print("Test set:  %2.2f"%(test_sv_rate))

submission = pd.DataFrame({ 'PassengerId': test_set.PassengerId,
                            'Survived': test_set.Survived})
submission.to_csv("titanic_prediction.csv")




# In[ ]:




