#!/usr/bin/env python
# coding: utf-8

# # Predict Survived from Titanic Disaster
# 
# ## August-September 2017, by Jude Moon
# Python3
# 
# 
# # Project Overview
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. 
# 
# In this project, I will analyze what sorts of people were likely to survive. In particular, I will apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# This document is to keep notes as I work through the project and show my thought processes and approaches to solve this problem. It consists of:
# 
# Part1. Data Exploration
# - Missing Value (NaN) Investigation
# - Outliers Investigation
# - Summary of Data Exploration
# 
# Part2. Feature Engineering
# - Creating New Features
# - Converting to Numeric Variables
# - Feature Exploration
# - Scaling Features
# - Feature Scores
# 
# Part3. Algorithm Search
# - Algorithm Exploration
# - Building Pipelines
# 
# 
# ***
# 
# # Part1. Data Exploration
# 

# In[ ]:


get_ipython().magic(u'pylab inline')
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import re
import sys
import pprint
import operator
import scipy.stats
from time import time


# In[ ]:


# load data set
titanic_df = pd.read_csv("../input/train.csv")


# In[ ]:


# the first 5 rows
titanic_df.head()


# In[ ]:


# data type of each column
titanic_df.dtypes


# In[ ]:


# check any numpy NaN by column
titanic_df.isnull().sum(axis=0) # sum by column


# In[ ]:


# statistics of central tendency and variability
titanic_df.describe()


# In[ ]:


# the numbers of survived and dead
titanic_df.groupby(titanic_df['Survived']).count()['PassengerId']


# I learned general idea about the passengers: 
# - total passenger number in training data set is 891
# - survival % is about 38%
# - Pclass is treated as integer, but actually it is category
# - since median of Pclass is 3rd class, passengers are donimated by 3rd class people
# - average age is 29.7 with missing 177 data points
# - sibsp and parch variables are little bit tricky with a lot of zeros
# - mean fare is 32 units
# - cabin has so many missing values
# 
# ## Missing Value (NaN) Investigation
# 
# ### Would NaN introduce bias to 'Cabin'?
# 
# 'Cabin' column has 687 missing values with is 77% of the total. This might introduce bias, so I would like to investigate what is average survival rates of the groups with missing value on 'Cabin' compared to others with 'Cabin' value. 

# In[ ]:


# survival rate by group with missing value on Cabin; True means missing value
titanic_df.groupby(titanic_df['Cabin'].isnull()).mean()['Survived']


# The survival rate of the group with missing value (True) is lower than the average (0.38), while that with value (False) is greater than the average. Missing values of 'Cabin' have a high tendency of introducing bias, meaning that the group of passengers with missing value on 'Cabin' is associated with lower survival rate than those with Cabin value. This would cause that if a supervised classification algorithm was to use 'Cabin' as a feature, it might interpret "NaN" for 'Cabin' as a clue that a person is not survived. So, I have to carefully use 'Cabin' as a feature for supervised classifier algorithms. I am not going to worry about dealing with NaN on 'Cabin' for now because it is not a number. And so I can simply convert NaN to string 'NaN' if I need to.
# 
# ### What about missing value on column 'Age'? How to deal with missing values?

# In[ ]:


# survival rate by group with missing value on Age; True means missing value
titanic_df.groupby(titanic_df['Age'].isnull()).mean()['Survived']


# About 20% data is missing for 'Age'. There is a possibility of NaN-drived bias but not strong as the bias for 'Cabin'. My choice to deal with the missing value is fill NaN with the median of the sample.

# In[ ]:


# replace NaN with the median of Age and create new column called age
titanic_df['age'] = titanic_df['Age'].fillna(titanic_df["Age"].median())

titanic_df['age'].describe()


# ### What about missing value on column 'Embarked'? How to deal with missing values?
# Only two observations are missing for 'Embarked'. I could ignore them or replace them with most frequent port.

# In[ ]:


# survival rate by group with missing value on Age; True means missing value
titanic_df.groupby(titanic_df['Embarked'].isnull()).mean()['Survived']


# In[ ]:


titanic_df.groupby(titanic_df['Embarked']).count()['PassengerId']


# In[ ]:


# replace NaN with the dominant port and create new column called Port
titanic_df['embarked'] = titanic_df['Embarked'].fillna('C')

titanic_df['embarked'].isnull().sum()


# ## Outliers Investigation
# 
# ### Is there an observation who has a lot of NaN?

# In[ ]:


# check any numpy NaN by row
#titanic_df.isnull().sum(axis=1) # sum by row
titanic_df.isnull().sum(axis=1).max() # find the max


# No, there is no observation who has missing values more than two. So, we can keep all the observations.
# 
# ### Are there any outliers in the dataset?

# In[ ]:


# I defined outliers as being above of 99% percentile here
# get lists of people above 99% percentile for each feature
highest = {}
for column in titanic_df.columns:
    if titanic_df[column].dtypes != "object": # exclude string data typed columns
        highest[column]=[]
        q = titanic_df[column].quantile(0.99)
        highest[column] = titanic_df[titanic_df[column] > q].index.tolist()
    
pprint.pprint(highest)


# In[ ]:


# delete 'PassengerId' from dictionary highest
highest.pop('PassengerId', 0)


# ### What are the outliers repeatedly shown among the features?

# In[ ]:


# summarize the previous dictionary, highest
# create a dictionary of outliers and the frequency of being outlier
highest_count = {}
for feature in highest:
    for person in highest[feature]:
        if person not in highest_count:
            highest_count[person] = 1
        else:
            highest_count[person] += 1
             
highest_count


# In[ ]:


# This time, I defined outliers as being below of 1% percentile here
# get lists of people below 1% percentile for each feature
lowest = {}
for column in titanic_df.columns:
    if titanic_df[column].dtypes != "object": # exclude string data typed columns
        lowest[column]=[]
        q = titanic_df[column].quantile(0.01)
        lowest[column] = titanic_df[titanic_df[column] < q].index.tolist()

# delete 'PassengerId' from dictionary highest
lowest.pop('PassengerId', 0)

pprint.pprint(lowest)


# In[ ]:


for person in lowest['age']:
    if person not in highest_count:
        highest_count[person] = 1
    else:
        highest_count[person] += 1
 
highest_count


# Overall, there is no outlier that are repeatedly shown among the features.
# 
# We can focus on age and Fare for continous values and Parch and SibSp for integer values to further investiage outliers. 
# 
# ### Take a look at outliers

# In[ ]:


# fare above 99% percentile
titanic_df.loc[highest['Fare'],['Fare', 'Survived']]


# The mean fare is 32 units but there are outliers who paid 262, 263, or 512 units, which are 8 to 16 times higher than the mean fare. I am going to keep these outliers because this might help to classify survival as extreme cases in such decision tree algorithm.

# In[ ]:


# age above 99% percentile
titanic_df.loc[highest['age'],['age', 'Survived']]


# In[ ]:


# age below 1% percentile
titanic_df.loc[lowest['age'],['age','Survived']]


# The average age is 29, but there are outliers who are 66 to 80 years old, which are 2.3 to 2.8 times higher than the average age, and who are younger than one year old. I am going to keep these outliers for now because the extreme cases of age might be usefull.
# 
# ## Summary of Data Exploration
# 
# - Total number of data points: 891
# - Target: ‘Survived’
# - Total number of data points labeled as survived: 342 (38%)
# - Total number of data points labeled as dead: 549 (62%)
# - Slightly imbalanced classes
# - Number of initial features: 10
# - List of features with missing values or NaN: 
# 
# | Feature  | # of NaN | Survival rate of NaN | Survival rate of non-Nan | Difference in survival rate |
# |----------|----------|----------------------|--------------------------|-----------------------------|
# | Cabin    | 687      | 0.30                 | 0.67                     | 0.37                        |
# | Age      | 177      | 0.29                 | 0.41                     | 0.12                        |
# | Embarked | 2        | 1.00                 | 0.38                     | -0.62                       |
# 
# - Top 3 people repeatedly shown as outliers:
# - The mean fare is 32 units but there are outliers who paid 262, 263, or 512 units, which are 8 to 16 times higher than the mean fare.
# - The average age is 29, but there are outliers who are 66 to 80 years old, which are 2.3 to 2.8 times higher than the average age, and who are younger than one year old.
# - Overall, there is no outlier that are repeatedly shown among the features. 
# 
# ***
# 
# 
# # Part2. Feature Engineering
# 
# ### Brainstorming
# 
# The target is 'Survived', and the rest columns are the candidate features: 
# - 'Fare', 'age', 'Sex', and 'embarked' are ready to go without engineering
# - 'Name', 'Ticket', and 'Cabin' are text variables and might require variation reduction. For example, use only last name instead of using full name with title. 
# - 'SibSp' and 'Parch' have a lot of zeros, where the zero value means truely zero or absence. Squre transformation can be considered. Also, creating a new feature like 'is_family' by combining 'SibSp', 'Parch' and 'family_name' from 'Name'.
# 
# ### Challenges
# 
# The features are mixed with continuous and categorial variables. Most ML algorithms work well with numerical variables and some work with mixed data types. I can think of several approaches:
# - use algorithms that can handle variables with both data types (DT, NB, KNN)
# - use demensionality reduction method to get numerical vectors
# - convert non-ordinal categoical variables to numerical or dummy ([padas.get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html), [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)) variables and use algorithms that works for numerical variables; the limitation would be that not all features can be utilized due to a large number of unique values
# - use ensemble method to combine algorithms for numerical variables and algorithms for categorical variables
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
# 
# ## Creating New Features
# 
# ### family_name

# In[ ]:


# a procedure to create a column with family name only
def get_familyname(name):
    full_name = name.split(',')
    return full_name[0]

# apply get_familyname procedure to the column of 'Name'
familyname = titanic_df['Name'].apply(get_familyname)

# add familyname to the DataFrame as new column
titanic_df['family_name'] = familyname.values


# In[ ]:


# how many unique family name?
len(titanic_df['family_name'].unique())


# ### ticket_prefix

# In[ ]:


# understand 'Ticket' values
titanic_df['Ticket'].head(10)


# In[ ]:


# how many unique family name?
len(titanic_df['Ticket'].unique())


# 'Ticket' variable is not consistant in terms of the format; some are mixed with letters and numbers, some has symbols (/ or .), and the others consist of only numbers. And not all Tickets are different. 

# In[ ]:


# a procedure to create a column with ticket prefix only
def get_prefix(ticket):
    if ' ' in ticket:
        prefix = ticket.split(' ')
        return prefix[0]
    else:
        return 'None'

# apply get_prefix procedure to the column of 'Ticket'
ticketprefix = titanic_df['Ticket'].apply(get_prefix)

# add ticket_prefix to the DataFrame as new column
titanic_df['ticket_prefix'] = ticketprefix.values


# In[ ]:


# count of ticket prefix; False means ticket_prefix == 'None'
titanic_df.groupby(titanic_df['ticket_prefix'] != 'None').count()['PassengerId']


# In[ ]:


# survival rate by group with ticket prefix; False means ticket_prefix == 'None'
titanic_df.groupby(titanic_df['ticket_prefix'] != 'None').mean()['Survived']


# I found no difference in survival rate in the group with vs. without ticket prefix, and both rates are similar to the average of the total.

# In[ ]:


# frequency of ticket_prefix
titanic_df.groupby(titanic_df['ticket_prefix']).count()['PassengerId']


# In[ ]:


# how many unique ticket_prefix?
len(titanic_df['ticket_prefix'].unique())


# I found inconsistency in formatting of prefix. For example, A./5., A.5., A/5, and A/5. could be the same prefix and A/S could be the typo for A/5. I am not sure making the formatting consistent would help to better classify the survived, or the differences in the formatting actually would help to classify them. 

# In[ ]:


# procedure to remove all special characters and change to upper case
def remove_special(initial):
    return (''.join(e for e in initial if e.isalnum())).upper()

titanic_df['ticket_prefix_v2'] = titanic_df['ticket_prefix'].apply(remove_special)

# frequency of ticket_prefix_v2
titanic_df.groupby(titanic_df['ticket_prefix_v2']).count()['PassengerId']


# In[ ]:


# survival rate by ticket_prefix_v2
titanic_df.groupby(titanic_df['ticket_prefix_v2']).mean()['Survived']


# In[ ]:


# how many unique ticket_prefix_v2?
len(titanic_df['ticket_prefix_v2'].unique())


# Now the number of unique ticket prefix was 43 and now it is 29 after cleaning the special characters.
# 
# I think putting the cleaned prefix and the number back togther might help.

# In[ ]:


# a procedure to create a column with ticket number only
def get_number(ticket):
    if ' ' in ticket:
        number = ticket.split(' ')
        return number[1]
    else:
        return ticket

# apply get_number procedure to the column of 'Ticket'
ticketnumber = titanic_df['Ticket'].apply(get_number)

# add ticket_number to the DataFrame as new column
titanic_df['ticket_number'] = ticketnumber.values

titanic_df['ticket_number'].head(10)


# In[ ]:


# add ticket to the DataFrame as new column by concatenating cleaned initial and number
titanic_df['ticket'] = titanic_df['ticket_prefix_v2'] + titanic_df['ticket_number'] 

titanic_df['ticket'].head()


# In[ ]:


# how many unique ticket?
len(titanic_df['ticket'].unique())


# ### cabin_initial

# In[ ]:


# understand 'Cabin' values
titanic_df['Cabin'].head(10)


# The 'Cabin' value consists of a capital letter following by numbers. I am not sure what the letter and numbers mean for but my intuition is that the letter could represent a room location or a room price, so the letter only can be used as a feature. 

# In[ ]:


# how many unique Cabin?
len(titanic_df['Cabin'].unique())


# Out of 204 known 'Cabin', 148 are the unique 'Cabin', and some people share the same 'Cabin' value.

# In[ ]:


def get_initial_letter(cabin):
    cabin = str(cabin) # change data type to string becuase nan is float
    return cabin[0]

titanic_df['cabin_initial'] = titanic_df['Cabin'].apply(get_initial_letter)
titanic_df['cabin_initial'].head(10)


# In[ ]:


# survival rate by group with missing value on Cabin; True means missing value
titanic_df.groupby(titanic_df['cabin_initial']).mean()['Survived']


# In[ ]:


# how many unique 'cabin_initial'?
len(titanic_df['cabin_initial'].unique())


# ### w_family

# In[ ]:


# SibSp and Parch are combined as family by vectoried addition
sibsp = titanic_df['SibSp']
parch = titanic_df['Parch']

family = sibsp + parch

#change datatype to categories with 2 groups
def w_family(family):
    if family != 0:
        return 1
    return 0

# apply w_family procedure to the array
w_family = family.apply(w_family)

# add w_family to the DataFrame as new column
titanic_df['w_family'] = w_family.values


# ## Converting to Numeric Variables
# 
# ### sex

# In[ ]:


# values of Sex
titanic_df["Sex"].unique()


# In[ ]:


# if male, return True or 1 and create new column 'sex'
titanic_df['sex'] = (titanic_df['Sex'] == 'male').astype(int)

# values of sex
titanic_df["sex"].unique()


# ### embarked to C, Q, and S

# In[ ]:


# values of embarked
titanic_df["embarked"].unique()


# In[ ]:


# create dummy variables for embarked
embarked_df = pd.get_dummies(titanic_df["embarked"])


# In[ ]:


# combine two dataframes
titanic_df = [titanic_df, embarked_df]
titanic_df = pd.concat(titanic_df, axis=1, join='inner')

titanic_df.describe()


# In[ ]:


feature_total = np.array(titanic_df.columns)

feature_total


# In[ ]:


feature_numeric = []
for column in titanic_df.columns:
    if titanic_df[column].dtypes != "object" and titanic_df[column].isnull().sum() == 0:
        feature_numeric.append(column)

feature_numeric


# In[ ]:


# remove id and target
feature_numeric = [e for e in feature_numeric if e not in ('PassengerId', 'Survived', 'survived')]

feature_numeric


# In[ ]:


len(feature_numeric)


# In[ ]:


feature_numeric[:5]


# ## Scaling Features
# 
# I will use **MinMaxScaler** to adjust the different units of features to be equally weighted and ranged between 0-1.

# In[ ]:


df_numeric = titanic_df[feature_numeric[:5]]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric),                          index=df_numeric.index, columns=df_numeric.columns)

df_scaled = df_scaled.rename(columns={"Pclass": "pclass_scl", "SibSp": "sibsp_scl",                                       "Parch": "parch_scl", "Fare": "fare_scl",                                       "age": "age_scl"})

df_scaled.describe()


# In[ ]:


df = [titanic_df, df_scaled]
df = pd.concat(df, axis=1, join='inner')
# how to merge two dataframes: https://pandas.pydata.org/pandas-docs/stable/merging.html

df.describe()


# In[ ]:


df.dtypes


# In[ ]:


# define features lists
original_numeric = ['Pclass', 'SibSp', 'Parch', 'Fare', 'age', 'sex']
original_categorical = ['Name', 'Ticket', 'Cabin', 'embarked']
original_total = original_numeric + original_categorical
                    
scaled_numeric = ['pclass_scl', 'sibsp_scl', 'parch_scl', 'fare_scl', 'age_scl', 'w_family', 'sex', 'C', 'Q', 'S']
updated_categorical = ['family_name', 'ticket', 'ticket_prefix_v2', 'cabin_initial']
updated_total = scaled_numeric + updated_categorical                    


# ## Feature Exploration
# 
# | List Name            | Features                                                                                          | # of Features |
# |----------------------|---------------------------------------------------------------------------------------------------|---------------|
# | original_numeric     | ['Pclass', 'SibSp', 'Parch', 'Fare', 'age', sex']                                                 | 6             |
# | scaled_numeric       | ['pclass_scl', 'sibsp_scl', 'parch_scl', 'fare_scl', 'age_scl', 'w_family', 'sex', 'C', 'Q', 'S'] | 10            |
# | original_categorical | ['Name', 'Ticket', 'Cabin', 'embarked']                                                           | 4             |
# | updated_categorical  | ['family_name', 'ticket', 'ticket_prefix_v2', 'cabin_initial']                                    | 4             |
# 
# 
# 
# ## Selecting Features 
# 
# ### What selection process to use?
# 
# - Univariate Selection such as SelectKBest: statistical tests can be used to select the features that have the strongest relationship with the output variable. 
# 
#     For the first trial, I will choose 9 or less features. The number 9 threshold came from the curve of dimensionality, where you may need exponentially more data points as you add more features, that is, 
# 
# >2^(n_featuers) = # of data points 
# 
#     I have 891 data points. 2^9 = 512 and 2^10 = 1024, so 9 is the max feature number. Thus, I will keep in mind to use no more than 9 features if I decide to use SelectKBest.
# 
# - Dimensionality Reduction such as PCA: PCA (or Principal Component Analysis) uses linear algebra to transform the dataset into a compressed form. I think chosing 2-3 dimensions after PCA transformation could be good start.
# 
# ### Which feature scores to compare?
# 
# I choose **f_classif** scoring function for continous variables and **chi2** for categoerical variables. 
# 
# - [Variance](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold) can be useful for unsupervised classification. Since I have already labels, utilizing labels for scoring could be better than soley reling on x-variables. 
# 
# - [The mutual information (MI)](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency. MI can be used for unsupervised clustering.
# 
# - The chi-square distribution arises in tests of hypotheses concerning the independence of two random variables and concerning whether a discrete random variable follows a specified distribution. The F-distribution arises in tests of hypotheses concerning whether or not two population variances are equal and concerning whether or not three or more population means are equal. In other words, chi-square is most appropriate for categorical data, whereas f-value can be used for continuous data [(read more)](https://discussions.udacity.com/t/f-classif-versus-chi2/245226).
# 
# 

# 
# ***
# 
# # Part3. Algorithm Search
# 
# ## Algorithm Exploration
# 
# When dealing with small amounts of data, it’s reasonable to try as many algorithms as possible and to pick the best one since the cost of experimentation is low according to [blog post by Cheng-Tao Chu](http://ml.posthaven.com/machine-learning-done-wrong).
# 
# - SVC
# - KNeighbors 
# - Gaussian Naive Bayes
# - Decision Trees
# - Ensemble Methods
# 
# ## Validation Methods 
# I think a proper validation method for the dataset with imbalanced classes is using cross validation iterators with stratification based on class labels, such as **StratifiedKFold** and **StratifiedShuffleSplit**. This would ensure that relative class frequencies is approximately preserved in each train and test set.

# ## Building Pipelines
# 

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#svc = SVC()
# update svc
svc = SVC(class_weight='balanced')
gnb = GaussianNB()
neigh = KNeighborsClassifier()
dt = tree.DecisionTreeClassifier()
rdf = RandomForestClassifier()
adb = AdaBoostClassifier()

# generate a 1000 train-test pairs iterator with default test set size = 0.1
sss = StratifiedShuffleSplit(n_splits=1000, random_state=44)


# In[ ]:


# a procedure to print out mean scores from cv
def print_scores(clf, data, label):
    scores = ["accuracy", "precision", "recall", "average_precision", "f1", "roc_auc"]
    start = time()
    for score in scores:
        mean_score = cross_val_score(clf, data, label, cv=sss, scoring=score).mean()
        print(score, ':', mean_score)
    
    print("\nThis took %.2f seconds\n" %(time() - start))


# In[ ]:


# svc classifier performance
print_scores(svc, df[original_numeric], df['Survived'])


# In[ ]:


pipeline1 = Pipeline([('selector', SelectKBest()),                       ('clf', svc)])

parameters = {'selector__k':[6,5,4,3],               'clf__C': [0.1, 1, 10, 100, 1000],               'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid_search = GridSearchCV(pipeline1, parameters, scoring='f1')
start = time()
gird_result = grid_search.fit(df[original_numeric], df['Survived']).best_estimator_
print("\nThis took %.2f seconds\n" %(time() - start))

selector = gird_result.named_steps['selector']
k_features = gird_result.named_steps['selector'].get_params(deep=True)['k']
print("Number of features selected: %i" %(k_features))

selected = selector.fit_transform(df[original_numeric], df['Survived'])
scores = zip(original_numeric, selector.scores_, selector.pvalues_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
new_list = list(map(lambda x: x[0], sorted_scores))[0:k_features]

new_clf = gird_result.named_steps['clf']

new_dataset = df[new_list]


# In[ ]:


print(new_list)
print(new_clf)


# In[ ]:


# svc classifier performance after tunning parameters using pipeline
print_scores(new_clf, new_dataset, df['Survived'])

