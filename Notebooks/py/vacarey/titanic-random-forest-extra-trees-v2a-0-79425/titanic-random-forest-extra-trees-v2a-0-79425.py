#!/usr/bin/env python
# coding: utf-8

# # Titanic - Random Forest (Extra Trees) Trial - v2 [0.79425]

# Here, an Extremely Randomized Trees (ExtraTrees) model is used to predict survival on the Titanic.  A new variable is created to hold (essentially) the number of people sharing a ticket, which is similar to cabin share.  Tuning of some ExtraTrees parameters is performed.  The model uses binary encodings for categorial variables.
# 
# The number of samples considered at a split seems to be an important parameter in improving model accuracy.  The ticket share variable does not provide much additional information beyond existing fields.  The final prediction is based on several training runs, as the random state of the system affects outcomes. Binary encoding of categorical variables may slightly improve predictions, but reduces interpretability.
# 
# ### Contents
# 1.  Data Import and Setup
# 2.  Data Exploration, Cleaning, and Preparation
#   1.  Name - extract title 
#   2. Cabin - Extract deck level and examine missing decks
#   3.  Ticket/cabin sharing
#   4. Missing data - Embarcation point
#   5. Fare - missing / zero values
#   6.  Age - Missing data 
# 3.  Gathering Analysis Fields
#   1.  Binary encoding of categorical variables
#   2.  Function to process and clean the data set
# 4.  ExtraTrees Model - Testing and Optimization
#   1.  Preliminary ExtraTrees run and feature importances
#   2.  Random effects on model predictions - train/test split and random state
#   3.  Evaluation of the "ticket share" field.
#   4.  Binary vs. one-hot encoding of categorical fields
#   5.  Model tuning - tree count
#   6.  Model tuning- number of samples used in splits
#   7.  Model tuning - max feature counts
#   8.  Model tuning - max number of leaf nodes
#   9.  Model tuning - tree count with larger split samples
# 5.  Predictions and Submission
#   1.  Function - predictions from tuned model
#   2.  Ensembling - repeating models and combining results
#   3.  Final training and submission data
# 6.  Discussion
# 7.  References

# ## 1.  Data Import and Setup

# ### Import packages

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import sklearn as sk
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import itertools
import scipy
from scipy.interpolate import griddata
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


# ### Import training data
# 
# Create the total family size (Parch + SibSp) here.  Enforce some data types.

# In[ ]:


#
# Import the test and training data.
#

train_data = pd.read_csv("../input/train.csv",
                        dtype={'Cabin':np.object, 'Ticket':np.object, 'Age':np.float,
                              'SibSP':np.int, 'Parch':np.int, 'Fare':np.float,
                              'Embarked':np.object, 'Sex': np.object}, 
                         na_values={'Cabin':'', 'Ticket':'', 
                                    'Embarked':'', 'Sex':''}, 
                         keep_default_na=False)

# Set the index and create a family size field. Make a single a family size of 1
train_data.set_index('PassengerId', inplace=True)

test_data = pd.read_csv("../input/test.csv",
                        dtype={'Cabin':np.object, 'Ticket':np.object, 'Age':np.float,
                              'SibSP':np.int, 'Parch':np.int, 'Fare':np.float,
                              'Embarked':np.object, 'Sex': np.object}, 
                         na_values={'Cabin':'', 'Ticket':'', 
                                    'Embarked':'', 'Sex':''}, 
                         keep_default_na=False)

# Set the index and create a family size field. Make a single a family size of 1
test_data.set_index('PassengerId', inplace=True)


# Create a copy of the data for exploration. Include the training and test data.  This data set will be used to examine missing data, etc.

# In[ ]:


df_analysis = pd.concat([train_data, test_data], axis=0)
df_analysis['family_size'] = df_analysis['SibSp'] + df_analysis['Parch'] + 1

df_analysis.tail(5)


# ## 2.  Data exploration, cleaning and preparation

# ### 2A.  Name - extract title.
# I want to get the title (Mr/Mrs/Miss etc.) I will group uncommon titles

# In[ ]:


def extract_title_str(name_str):
    """Extract a title from a name string"""
    mobj = re.search(", (.*?)[\.| ]", name_str)
    mobj_str = (mobj.group()
                .strip(' ,.').
                lower())
    return(mobj_str)


# Examine title frequencies.  Grouping the "other" category may make sense given low counts.

# In[ ]:


title_ser = df_analysis['Name'].apply(extract_title_str)
title_ser.name ='title'
title_ser.value_counts()


# Get another series of title groups - keep the major categories (mr/mrs/miss/master) and group others

# In[ ]:


title_grp_ser = title_ser.apply(lambda x: x if x in ['mr', 'mrs', 'miss', 'master'] else 'other')
title_grp_ser.name ='title_grp'
title_grp_ser.value_counts()


# Append the title and title group and plot to see how title affects survival

# In[ ]:


df_analysis = pd.concat([df_analysis, title_ser, title_grp_ser], axis=1)

agg_df_title = df_analysis.groupby('title_grp').agg({'Survived':['mean','sem']})
agg_df_title['Survived', 'mean'].plot(kind='bar', yerr=agg_df_title['Survived', 'sem'], alpha = 0.5, error_kw=dict(ecolor='k'));
plt.gcf().suptitle('Survival probability by title');


# ### 2B. Cabin - Extract deck level and examine missing decks
# The deck level can be extracted from the Cabin field.  I will get the deck and examine unknowns

# In[ ]:


#
# Examine the cabin/deck variable.  Many are missing
#

def extract_deck_str(cabin_str):
    """Extract a deck letter from a cabin string"""
    return(cabin_str[0:1].upper())

df_analysis['Cabin'].apply(extract_deck_str).value_counts().sort_values()


# Add a column to the data frame, 'deck_mod', which contains the original deck with T/G grouped into 'TG'. How is survival affected?

# In[ ]:


deck_ser= (df_analysis['Cabin']).apply(extract_deck_str)
deck_ser_mod = deck_ser.apply(lambda x: x if x in ['A','B','C','D','E','F'] 
                              else 'TG' if x in ['T', 'G'] else 'unk')
deck_ser_mod.name = 'deck_mod'

df_analysis = pd.concat([df_analysis, deck_ser_mod], axis=1)
df_analysis.tail()


# In[ ]:


agg_df_deck = df_analysis.dropna().groupby('deck_mod').agg({'Survived':['mean','sem']})
agg_df_deck['Survived', 'mean'].plot(kind='bar', 
                                     yerr=agg_df_deck['Survived', 'sem'], 
                                     alpha = 0.5, error_kw=dict(ecolor='k'));
plt.gcf().suptitle('Survival probability by deck');



# It appears that missing deck cases may have a lower survival probability than cases with deck strings.  Look at fares by deck

# In[ ]:


agg_df_deck2 = df_analysis.groupby('deck_mod').agg({'Fare':['mean','sem']})
agg_df_deck2['Fare', 'mean'].plot(kind='bar', 
                                     yerr=agg_df_deck2['Fare', 'sem'], alpha = 0.5, error_kw=dict(ecolor='k'));
plt.gcf().suptitle('Mean fare by deck');


# Looking at the above information, it seems that the people with missing deck information may be different than those with deck information.  Therefore, instead of trying to infer a deck, I will treat them a separate group. 

# ### 2C. Ticket/cabin sharing
# The ticket number appears to be filled in for all individuals.  It is possible to find out who shares a ticket.  Sometimes, family members seem to share a ticket and sometimes not.  
# 
# I am assuming that people who share a ticket share a cabin.  I will roughly test this by looking at the # of people who share a ticket vs the # sharing a cabin.  This grouping uses the entire data set (test + training), because some people who share a cabin may be in different data sets.
# 
# Ticket numbers are cleaned prior to grouping.

# In[ ]:


#
# Define some functions for calculating ticket and cabin shares
#

def cabin_share_series(cabin_ser):
    """Returns a series containing the # of people sharing each
    passenger's cabin, when this number can be determined"""
    
    # Get the passenger counts per ticket
    agg_ser = cabin_ser[cabin_ser != ''].value_counts()
    agg_ser.name ='cabin_share'
    
    df_ret = pd.merge(cabin_ser.to_frame(), agg_ser.to_frame(), 
                      how='left', left_on='Cabin', right_index=True,
                      indicator=False)
    
    return(df_ret['cabin_share'])


def clean_ticket(ticket_str):
    """Clean up the ticket string by replacing punctuation, converting
    to upper case, and removing whitespace"""
    ticket_str_clean = re.sub(r'[^\w]', '', ticket_str.strip())
    return (ticket_str_clean)

def ticket_count_frame(ticket_ser):
    """Return a data frame containing 2 series:
        ticket_clean: cleaned-up version of ticket string
        ticket_share_count: number of people with a certain ticket"""
    
    ticket_clean_ser = ticket_ser.apply(clean_ticket)
    ticket_clean_ser.name = 'ticket_clean'
    
    # Get the passenger counts per ticket
    agg_ser = ticket_clean_ser.value_counts()
    agg_ser.name ='ticket_share'

    df_ret = pd.merge(ticket_clean_ser.to_frame(), agg_ser.to_frame(), 
                      how='left', left_on='ticket_clean', right_index=True,
                      indicator=False)
    
    return(df_ret)


# Look at whether the cabin and ticket shares are similar, in which case we can use the ticket
# share as an estimate of cabin occupancy

# In[ ]:


#
# Get cabin information from test and training data.
#


share_df = pd.concat([cabin_share_series(df_analysis['Cabin']),
                         ticket_count_frame(df_analysis['Ticket'])], axis=1)


#
# Is the ticket share a good approximation of cabin share?
# Get a grid for ticket vs cabin share, with counts
#

cross_share_df = pd.crosstab(pd.Categorical(share_df['ticket_share']), 
                    pd.Categorical(share_df['cabin_share']))

cross_share_df.index.names= ['Ticket share']
cross_share_df.columns.names=['Cabin share']

cross_share_df.head()
sns.heatmap(cross_share_df, cmap='RdYlGn_r', linewidths=0.5, annot=True,
           cbar_kws={'label': '# of passengers'});
plt.gcf().suptitle('Do people with the same ticket # tend to share a cabin?');

#
# These are not that similar, especially for some tickets
#


# In[ ]:


#
# Add the ticket share to the frame
#

df_analysis = pd.merge(left=df_analysis, right=share_df[['ticket_share', 'cabin_share']],
                       left_index = True, right_index= True, how='left')
df_analysis.tail(5)


# In[ ]:


print('null count: ', df_analysis['ticket_share'].isnull().sum())
df_analysis['ticket_share'].value_counts()


# Look at whether the ticket share is similar to family size.  Do non-family share tickets?  Do families always share?

# In[ ]:


#
# Examine the ticket share vs. family size.
# Leave off upper left corner of singletons to get better heat
#

family_ticket_share_df = pd.crosstab(pd.Categorical(df_analysis['ticket_share']), 
                    pd.Categorical(df_analysis['family_size']))
family_ticket_share_df.index.names= ['Ticket share']
family_ticket_share_df.columns.names=['Family size']
family_ticket_share_df.loc[1,1] = np.nan

sns.heatmap(family_ticket_share_df, 
            cmap='RdYlGn_r', linewidths=0.5, annot=True,
                      cbar_kws={'label': '# of passengers'});
plt.gcf().suptitle('Are ticket shares and family size related?');



# It's clear that ticket and family size are correlated.  However, there are a fair number of non-family ticket sharers. This information could be helpful  There are likely also some family members in separate cabins. 

# Get another heatmap, this time for survival rates by family size and ticket share.

# In[ ]:


agg_df = df_analysis.groupby(['family_size', 'ticket_share']).agg({'Survived':['mean','std', 'count']})
agg_df.columns = agg_df.columns.droplevel()
agg_df.head()

# Remove very low-population cells and unstack
agg_df2 = agg_df[agg_df['count'] >= 3]['mean'].unstack(level=1).T

sns.heatmap(agg_df2, cmap='viridis', linewidths=0.5, annot=True,
                      cbar_kws={'label': 'Survival Probability'});
plt.gcf().suptitle('Survival by family size, ticket share');



# There are some interesting off-diagnoal elements here! Particularly for unrelated people sharing a cabin. In addition, large family may be less of a risk if the family does not share 1 ticket.

# ### 2D. Missing data - Embarcation point
# Infer embarcation point from other information

# In[ ]:


# View missing values
df_analysis['Embarked'].value_counts().sort_values()


# How many are missing embaration points? Very few.  I will set missing values to the most common embarcation point.

# In[ ]:


embarked_ser= df_analysis['Embarked'].apply(lambda x: x if x != '' else 'S')
embarked_ser.name = 'embarked_mod'
embarked_ser.value_counts()


# In[ ]:


# Add the embarcation point to the frame
df_analysis = pd.concat([df_analysis, embarked_ser], axis=1)


# ### 2E. Fare - missing / zero values
# My training data does not contain missing fares.  However, the test data does contain missing fares.  
# 
# Also, Some passengers have a $0 fare.  Is this a proxy for missing data?
# 
# I can't say for sure, but looking at the data, these are all males of working age who share an embarcation point and have a family size 1.  I wonder if they are employees or some other group who have \$0 fare for a reason.  Therefore, I will not fill in \$0 fares.  However, missing fares should be handled.  Since this is so rare, I will simply set this to the median fare.

# In[ ]:


print('training null count: ', df_analysis['Fare'].isnull().sum())
print('training zero count: ', df_analysis[df_analysis['Fare']<= 0]['Fare'].count())

print('test null count: ', test_data['Fare'].isnull().sum())
print('test zero count: ', test_data[test_data['Fare']<= 0]['Fare'].count())


# How does fare affect survival?  Plot

# In[ ]:


sns.boxplot(x='Survived', y='Fare', data = df_analysis);
# Truncate high outliers in plot
plt.ylim([0,140]);


# ### 2F.  Age - Missing data 
# A large number of people have missing (NaN) ages.  Infer age from other information for these people.  All ages are floats

# In[ ]:


print('null count: ', df_analysis['Age'].isnull().sum())


# In[ ]:


# Let's look at an age histogram

ax = sns.distplot(df_analysis['Age'].dropna())

# There are many infants aboard.  I wondered if these were legitimate.
# Looking at titles and ages they seem to be.
# There are no 0.0 ages, for example.  Titles tend to be "miss" or 
# "master".  Therefore, this peak seems real


# I want to fill in missing values of age using other information in the data set.  To this end, I will plot some bar charts to visualize how age depends on various data features.

# In[ ]:



# vector of interesting features
groupby_field_vec= ['SibSp', 'family_size', 'Pclass', 'title_grp', 
                    'Sex', 'Parch', 'embarked_mod', 'deck_mod']

df_list = [df_analysis.groupby(g).agg({'Age':['mean','sem']}) for g in groupby_field_vec]
ind_list = [int('42' + str(x)) for x in range(1,len(groupby_field_vec)+1)]

for i, df in enumerate(df_list):
    df.columns = df.columns.droplevel()
    df['mean'].plot(kind='bar', ax=plt.subplot(ind_list[i]),
                   yerr = df['sem'], figsize=(9,9));

plt.suptitle('Age by various factors')

plt.subplots_adjust(left=None, bottom=0, right=None, top=0.9, wspace=0.3, hspace=0.5)


# Based on the above, I will set the passenger age to the median for  the combination of the following fields:
#     * title_grp
#     * Pclass
#     * family_size (thresholded <=7)
#     * Sex
#     * Parch (thresholded <=7)
#     * deck_mod

# In[ ]:


#
# Look at survival by age
#

df2 = df_analysis.dropna().copy()
df2['age_grp'] = df2['Age'].apply(lambda x: 0 if x < 1 else 1+ int(x/10))

agg_df_age = df2.groupby('age_grp').agg({'Survived':['mean','sem']})
agg_df_age['Survived', 'mean'].plot(kind='bar', yerr=agg_df_age['Survived', 'sem'], alpha = 0.5, error_kw=dict(ecolor='k'));
plt.gcf().suptitle('Survival probability by age');


# In[ ]:


#
# Function to fill in missing ages
#

def age_filled_ser(age_df):
    """Get a series of inferred/actual passenger ages.  If the age is not NaN,
    it is returned.  Otherwise, the age is inferred using the information in age_df.
       The data frame input must contain the following fields:
        Age
        title_grp
        Pclass
        family_size
        Sex
        Parch
        deck_mod
    The mean ages for the above fields (except Age) will be used to fill in 
    missing data. Some values are thresholded.
       It's possible for a passenger to have a unique combination of items.
    In that case, we do backup fills, removing variables in reverse order
    from the list above (e.g. remove deck, then Sex)
       Assume an appropriate index in the passed-in frame.  
    The output series will be named 'age_mod'"""
    
    field_list= ['title_grp', 'Pclass', 'family_size', 
                       'Sex', 'Parch', 'deck_mod']
    
    fill_df = age_df[['Age'] + field_list].copy()
    
    # Threshold some values
    fill_df['family_size'] = (fill_df['family_size']
                              .apply(lambda x: x if (x <=7) else 7))
    fill_df['Parch'] = (fill_df['Parch']
                              .apply(lambda x: x if (x <=7) else 7))

    
    predictor_list = [field_list[0:x] for x in range(len(field_list), 0, -1)]
    fill_df['age_mod'] = fill_df['Age']
    
    # Fill NAs using descending numbers of predictors
    for predictors in predictor_list:
        fill_df['age_mod'] = (fill_df.groupby(predictors)['age_mod']
                          .transform(lambda x: x.fillna(x.mean())))
    
    return(fill_df['age_mod'])


# In[ ]:



age_mod_ser = age_filled_ser(df_analysis)
print('null count:', age_mod_ser.isnull().sum())

df_analysis = pd.concat([df_analysis, age_mod_ser], axis=1)


# ## 3. Gathering Analysis Fields
# 
# Here, I write a function to process a data frame, modifying and adding fields as needed.  I also create binary fields for categorical variables.  Many steps in processing were performed above, but I consolidate them here so that they can easily be applied to both the training and test data sets.  Functions defined above are used in this process.
# 
# Major steps are:
# 1.  Calculate family size
# 2.  Get the title "group" field
# 3.  Get the modified deck string
# 4.  Calculate the ticket share
# 5.  Fill in missing values in embarkation point
# 6.  Fill in missing values for ages
# 7.  Create numeric (binary) fields for categorical variables
# 
# The analysis_df has many of these fields already filled and so will be used to create the frames

# ### 3A.  Binary encoding of categorical variables

# In[ ]:


#
# Get functions for binary encoding of categorical variables in our data sets.
# Assume we pass lists of possible values
#

def binary_encode_dict(value_list):
    """Gets a dictionary for mapping values to a list of binary digits
    for binary encoding of categorical variables"""
    
    # Get the # of items in the list and the binary digits required to encode
    max_bin = len(value_list)
    num_bits = math.ceil(math.log(max_bin, 2))

    
    # Get the binary encodings for each level
    str_dict = {value_list[i]: list("{0:b}".format(i).zfill(num_bits))
                for i in range(0, max_bin)}
    
    # Return the bit count and encoding dict
    return (num_bits, str_dict)

def binary_encode_series(name_list, data_series, col_prefix):
    """Apply binary encoding to values in a series, returning a data
    frame consisting of the encoded values, with sequential columns"""
    
    (col_num, col_dict) = binary_encode_dict(name_list)
    col_df = pd.DataFrame(data_series.apply(lambda x: col_dict[x])
                          .values.tolist(),
                          columns=[col_prefix + str(i) for i in range(0,col_num)],
                         index = data_series.index)
    return(col_df)


# ### 3B.  Function to process and clean the data sets

# In[ ]:


#
# Write the function that processes the data set, cleaning up fields,
# encoding categorial fields, etc.
#

def df_prepare_analysis_binary(input_df, analysis_df):
    """Get a data frame for analysis using the random forest model.
    Use binary encoding of categorical variables.  Assume some
    predictors have been filled in via the analysis_df object"""
    
    # Get the pre-processed fields for the data of interest
    this_analysis_data = analysis_df.loc[input_df.index]
    
    # Add the ticket share
    ret_df = this_analysis_data[['SibSp', 'Parch', 'Pclass', 'ticket_share', 'deck_mod', 
                                'title_grp', 'family_size', 'age_mod',
                                'Sex', 'embarked_mod', 'Fare']]

    # Fill in any NA fares
    fare_mod_ser = ret_df['Fare'].fillna(np.median(analysis_df['Fare'].dropna()))
    fare_mod_ser.name = 'fare_mod'

    # Binary encoding for categorical fields - Sex
    sex_ser_names = ['male', 'female']
    sex_df = binary_encode_series(sex_ser_names, ret_df['Sex'], 'sex_')
    
    # Binary encoding for categorical fields - Deck (modified)
    deck_ser_names = ['A', 'B', 'C', 'D', 'E', 'F', 'TG', 'unk']
    deck_df = binary_encode_series(deck_ser_names, ret_df['deck_mod'], 'deck_mod_')

    # Binary encoding for categorical fields - Title (modified)
    title_grp_names = ['mr', 'mrs', 'miss', 'master', 'other']
    title_df = binary_encode_series(title_grp_names, ret_df['title_grp'], 'title_grp_')
    
    # Binary encoding for categorical fields - Embarkation point (modified)
    embarked_ser_names = ['S', 'Q', 'C']
    emb_df = binary_encode_series(embarked_ser_names, ret_df['embarked_mod'], 'embarked_mod_')

    ret_df = pd.concat([ret_df, fare_mod_ser, emb_df, sex_df, deck_df, title_df], axis=1)
    return(ret_df)


# ## 4.  ExtraTrees Model - Testing and Optimization
# 
# Here, I use the training data only to explore some features of the ExtraTrees model.  I will split our training data into smaller training and "test" portions.  I will look at feature importances, whether the "ticket share" field is really valuable, how changing model parameters (# trees, etc.) affects results, and how much the random state affects the model.  
# 
# The ExtraTrees model does all splits randomly, and so different random states may lead to different accuracy scores.  Moreover, different training data sets may also change results.  I am not using the official "test" data here, but rather assessing performance using the training data set.

# In[ ]:


#
# Create a function for testing the Extra Trees model on the data set.
# This will allow us to vary parameters for the model, including
# the predictor, random state, # of trees, etc.  We will also be
# able to assess the effects of different training/test sets
#  

def test_extra_trees_model(train_test_df, predictor_columns,
                          tt_split_size =0.3, tt_random_state = None,
                          et_n_estimators=10, et_random_state = None,
                          et_max_features = 'auto', et_min_samples_split=2,
                          et_max_leaf_nodes = None):
    '''Test the ExtraTrees model with the training data and certain predictor columns.
    The training data will be split into "training" and "test" portions.  The random state 
    of the split may be passed in as a parameter.  After the split, the ExtraTrees model is run
    and accuracy results obtained.  Extra trees parameters and random states may be passed in as
    parameters also.'''
    
    # Get the test data - split our training set.
    train_test_df_X = train_test_df[predictor_columns]
    train_test_df_Y = train_test_df['Survived']
    X_train, X_test, Y_train, Y_test = train_test_split(train_test_df_X, 
                                                        train_test_df_Y,
                                                        test_size=tt_split_size,
                                                        random_state = tt_random_state)
    
    # Fit the model, and get feature importances
    model = sk.ensemble.ExtraTreesClassifier(n_estimators = et_n_estimators,
                                             random_state = et_random_state,
                                             max_features = et_max_features,
                                             min_samples_split = et_min_samples_split,
                                             max_leaf_nodes = et_max_leaf_nodes)
    fitted_model = model.fit(X_train, Y_train)
    importances = model.feature_importances_
    
    # Process importances into data frame
    importances_dict = {predictor_columns[i]: importances[i] for i in range(0, len(importances))}
    importances_df = (pd.DataFrame.
                  from_dict(importances_dict, orient="index")
                  .rename(columns={0: 'importance'})
                 .sort_values(by='importance', ascending=False))
    
    # Get the predictions, accuracy score, and confusion matrics
    predictions = fitted_model.predict(X_test)
    confusion_matrix = sk.metrics.confusion_matrix(Y_test,predictions)
    accuracy_score = sk.metrics.accuracy_score(Y_test, predictions)
    
    return (importances_df, confusion_matrix, accuracy_score)
    


# ### 4A.  Preliminary ExtraTrees run and feature importances

# In[ ]:


#
# Defne the predictor columns
#

predictor_columns = ['Parch', 'fare_mod', 'Pclass', 'family_size', 'ticket_share',
                     'age_mod', 'sex_0', 
                     'deck_mod_0','deck_mod_1', 'deck_mod_2', 
                     'title_grp_0', 'title_grp_1', 'title_grp_2',
                     'embarked_mod_0', 'embarked_mod_1']


# In[ ]:


#
# Run a simple ExtraTrees test, with default parameters.
# Print the accuracy and confusion matrix
#

train_mod_df = pd.concat([df_prepare_analysis_binary(train_data, df_analysis),
                          train_data['Survived']], axis=1)
(importances_df, confusion_matrix, accuracy_score) = test_extra_trees_model(train_mod_df,
                                                                           predictor_columns)

# print the accuracy info
print('accuracy score: {0:.4g}'.format(accuracy_score))
print('false positives: {0:d}; sensitivity: {1:.3g}'
      .format(confusion_matrix[0,1],
              (confusion_matrix[0,0]/sum(confusion_matrix[0,:]))))
print('false negatives: {0:d}; specificity: {1:.3g}'
      .format(confusion_matrix[1,0],
              (confusion_matrix[1,1]/sum(confusion_matrix[1,:]))))


# In[ ]:


# Print information about feature importances
importances_df[::-1].plot(kind='barh', legend='False');
plt.xlabel('importance');
plt.title('Variable importances in ExtraTrees model');


# The new "ticket share" field I created has a fairly high importance.  Whether this field truly improves the model will be discussed more below.

# ### 4B.  Random effects on model predictions - train/test split and random state
# 
# Test how repeating the fit changes the accuracy score.  First, examine the random state of the sytem, keeping the same cross-validation split for now.  
# 
# Note that the ExtraTrees model does random splits, and so the random state will affect its decisions strongly.  We might expect that repeted runs will result in different forests with different accuracy scores.

# In[ ]:


#
# Use dictionary comprehension to repeat the model tests.  Use the
# same cross-validation split
#

num_items_test = 101

accuracy_dict_et = {x:(test_extra_trees_model(train_mod_df,
                                        predictor_columns,
                                        tt_random_state = 100,
                                        et_random_state = x))[2] for
                x in range(0, num_items_test)}


accuracy_dict_et_val = list(accuracy_dict_et.values())
plt.hist(accuracy_dict_et_val);
plt.title('Accuracy scores from various model random states');
print('mean accuracy: {}'.format(np.mean(accuracy_dict_et_val)))

# Get the index of the median accuracy, for later random state testing
median_rs = sorted(accuracy_dict_et, key = accuracy_dict_et.__getitem__)[int(num_items_test/2)]
print('median accuracy: {}'.format(accuracy_dict_et[median_rs]))


# It appears there's a pretty wide range of results from this model. Also, a large number of samples are needed to get a good histogram.

# In[ ]:


#
# Let's see how different training slices affect the score.
# Keep the ExtraTrees random state constant for now (use the
# median state selected above)
# Keep the train/test split constant for now (set a random state)
#

num_items_test = 101

accuracy_dict_tt = {x:(test_extra_trees_model(train_mod_df,
                                        predictor_columns,
                                        tt_random_state = x,
                                        et_random_state = median_rs))[2] for
                x in range(0, num_items_test)}


accuracy_dict_tt_val = list(accuracy_dict_tt.values())
plt.hist(accuracy_dict_tt_val);
plt.title('Accuracy scores from various training sets');
print('mean accuracy: {}'.format(np.mean(accuracy_dict_et_val)))


# There is a broad peak.  Characteristics of the training/test sets matter a lot!

# ### 4C.  Evaluation of the "ticket share" field.
# 
# Is the ticket_share variable I created helpful?  Try fits with and without this parameter. 
# 
# The "ticket share" field had a fairly high importance (see above).  However, in my (not extensive) experience, highly correlated variables will all tend to have high imporatance, even though adding these fields to the model does not provide additional information nor improve overall accuracy.  Some forests will tend to use one variable, some forests another, leading to votes for both fields.  Therefore, I want to test whether removing the ticket share field from the model affects accuracy.  Given the variability of results, I need to perform multiple tests.

# In[ ]:


#
# Do multiple ET fits, letting both the test slice and the ET random state float
#

num_items_test = 200

st_pred_items = {x:(test_extra_trees_model(train_mod_df,
                                        predictor_columns))[2] for
                x in range(0, num_items_test)}


predictor_columns_no_ticket_share = [x for x in predictor_columns
                                    if x != 'ticket_share']

ns_pred_items = {x:(test_extra_trees_model(train_mod_df,
                                        predictor_columns_no_ticket_share))[2] for
                x in range(0, num_items_test)}


ticket_compare_df = pd.concat([pd.Series(list(st_pred_items.values()), name='With'),
          pd.Series(list(ns_pred_items.values()), name='Without')], axis=1)

# Create a box plot
sns.boxplot(x=['With', 'Without'], y=[ticket_compare_df['With'], ticket_compare_df['Without']]);
plt.title('Comparison of model results with and without ticket share');


# From the above, it doesn't look like the ticket share variable improves predictions in a meaningful way.

# ### 4D.  Binary vs. one-hot encoding of categorical fields
# 
# Did binary encoding of categorical fields help the model? 
# 
# Here, I compare binary to "one hot" (pd.get_dummies) encoding.  I run repeated predictions to get an idea whether the

# In[ ]:


#
# Compare a model with one-hot encoding vs. binary encoding.
#

# Add "one hot" columns to the data frame.
ind_col_list = ['deck_mod', 'Sex', 'title_grp', 'embarked_mod']
one_hot_df = pd.get_dummies(train_mod_df[ind_col_list], 
                            prefix=['deck_hot', 'sex_hot', 'title_hot', 'emb_hot'],
                            columns=ind_col_list)
train_mod_df_hot = pd.concat([train_mod_df, one_hot_df], axis=1)


predictor_columns_hot = ([x for x in predictor_columns
                                    if x not in 
                                     ['sex_0', 'deck_mod_0','deck_mod_1', 
                                      'deck_mod_2', 'title_grp_0', 'title_grp_1', 
                                      'title_grp_2','embarked_mod_0', 'embarked_mod_1']] +
                                     ['sex_hot_female','deck_hot_A','deck_hot_B', 
                                      'deck_hot_C', 'deck_hot_D','deck_hot_E', 'deck_hot_F', 
                                      'deck_hot_TG', 'title_hot_master', 
                                      'title_hot_miss', 'title_hot_mr','title_hot_mrs',
                                       'emb_hot_C', 'emb_hot_Q'])


train_mod_df_hot[predictor_columns_hot].head()

hot_pred_items = {x:(test_extra_trees_model(train_mod_df_hot,
                                        predictor_columns_hot))[2] for
                x in range(0, num_items_test)}

hot_compare_df = pd.concat([pd.Series(list(st_pred_items.values()), name='Binary'),
          pd.Series(list(hot_pred_items.values()), name='OneHot')], axis=1)

sns.boxplot(x=['Binary', 'OneHot'], y=[hot_compare_df['Binary'], hot_compare_df['OneHot']]);
plt.title('Categorical variable encoding effects on model results');



# The results are not very sensitive to encoding.  Binary may be slightly better, and less prone to very low results.

# ### 4E.  Model tuning - tree count
# 
# Does the number of estimators affect model accuracy?

# In[ ]:


#
# Tune the number of trees.  Do multiple trials per tree count, and 
# try a tree count from 1 to 30
#

tree_count = range(1, 30)
num_trials = 20

st_pred_items = {t: [test_extra_trees_model(train_mod_df,
                                           predictor_columns,
                                           et_n_estimators=t)[2] for
                x in range(0,num_trials)] for t in tree_count}

tree_count_df = pd.DataFrame.from_dict(st_pred_items, orient='index')

test_df = pd.concat([tree_count_df.reset_index()['index'], 
                     tree_count_df.mean(axis=1), 
                     tree_count_df.sem(axis=1)], 
                    axis=1).dropna()
test_df.columns = ['trees', 'mean', 'sem']

test_df.plot.scatter(x='trees', y='mean', yerr='sem');
plt.title('Tree count variation in model results');



# It appears that the optimal tree count plateaus at around 5-10 or so (the default).  We could try a slightly larger # of trees or leave at the default.

# ### 4F.  Model tuning- number of samples used in splits
# 
# Does the number of samples required for node splits affect model accuracy?

# In[ ]:


#
# Tune the # of samples required for node splits
#

split_count = range(2, 25)
num_trials = 20

sp_pred_items = {s: [test_extra_trees_model(train_mod_df,
                                           predictor_columns,
                                           et_min_samples_split=s)[2] for
                x in range(0,num_trials)] for s in split_count}

split_count_df = pd.DataFrame.from_dict(sp_pred_items, orient='index')

test_df = pd.concat([split_count_df.reset_index()['index'], 
                     split_count_df.mean(axis=1), 
                     split_count_df.sem(axis=1)], 
                    axis=1).dropna()
test_df.columns = ['split samps', 'mean', 'sem']

test_df.plot.scatter(x='split samps', y='mean', yerr='sem');
plt.title('Split sample count effects on model results');


# More samples (>~10) seem to do better than the default of 2. We may want to increase this parameter to be 5 or more.

# A larger # of samples used for splits makes trees smaller, in effect simplifying models.  This will reduce variance and increase bias.  In previous explorations, it seemed that fits are pretty sensitive to training/test splits; therefore, reducing variance further may be desirable.  
# 
# There doesn't seem to be a point in the above chart where bias becomes too great.  However, choosing a smaller value above the shoulder is probably best.  For this data, a split count of 10 may be good.
# 
# 

# ### 4G.  Model tuning - max feature counts
# 
# The max feature count determines the degree of randomization in the model.  This is the # of predictors examined when deciding how to split a particular node.  When max_features = 1, the attribute to be split is selected entirely at random.  At the other extreme, all attributes could be screend (this defeats the purpose of the extra trees algorithm to some extent, as increased randomization is a feature of this model). 
# 
# The default value of et_max_features is sqrt(n_features), or sqrt(14) ~ 3 or 4 in this model.  Another default set point available to the model is log2(n_features), which would be ~4 also.  The Python implementation of Extra Trees allows tuning this parameter to any integer value, which I attempt below.  

# In[ ]:


#
# Try different # of features considered at the splits.
# The default is sqrt(n_features).
#

#
# Tune the # of features required for node splits
#

feature_count = range(1, len(predictor_columns))
num_trials = 20

fp_pred_items = {f: [test_extra_trees_model(train_mod_df,
                                           predictor_columns,
                                           et_max_features=f,
                                           et_min_samples_split=10)[2] for
                x in range(0,num_trials)] for f in feature_count}

feature_count_df = pd.DataFrame.from_dict(fp_pred_items, orient='index')

test_df = (pd.concat([feature_count_df.mean(axis=1), 
                     feature_count_df.sem(axis=1)], 
                    axis=1).reset_index()
           .dropna())
test_df.columns = ['features', 'mean', 'sem']

test_df.plot.scatter(x='features', y='mean', yerr='sem');
plt.gcf().suptitle('Model tuning: # features compared to split')
plt.title('Model effects for # features compared at splits');



# The model seems pretty insensitive to the number of features, except at the very low end.  Therefore, I will leave this parameter alone.

# ### 4H. Model tuning - max number of leaf nodes

# In[ ]:


#
# Try to tune the # of leaf nodes
#

nodes = range(5, 200, 10)
num_trials = 20

st_pred_items = {t: [test_extra_trees_model(train_mod_df,
                                           predictor_columns,
                                           et_n_estimators=10,
                                           et_min_samples_split=15,
                                           et_max_leaf_nodes = t)[2] for
                x in range(0,num_trials)] for t in nodes}

node_count_df = pd.DataFrame.from_dict(st_pred_items, orient='index')

test_df = (pd.concat([node_count_df.mean(axis=1), 
                     node_count_df.sem(axis=1)], 
                    axis=1).reset_index()
           .dropna())
test_df.columns = ['leaf_nodes', 'mean', 'sem']

test_df.plot.scatter(x='leaf_nodes', y='mean', yerr='sem');
plt.title('Leaf node count effects on model results - higher split count');



# It appears that increasing the leaf nodes past 20 or so is not helpful in the model.  I want to keep trees simpler and will choose 25 as the max leaf count.

# ### 4I. Model tuning - tree count with larger split samples

# In[ ]:


#
# Try the tree-tune again, but with better split samps
#

tree_count = range(1, 200, 10)
num_trials = 20

st_pred_items = {t: [test_extra_trees_model(train_mod_df,
                                           predictor_columns,
                                           et_n_estimators=t,
                                           et_min_samples_split=15,
                                           et_max_leaf_nodes=25)[2] for
                x in range(0,num_trials)] for t in tree_count}

tree_count_df = pd.DataFrame.from_dict(st_pred_items, orient='index')

test_df = (pd.concat([tree_count_df.mean(axis=1), 
                     tree_count_df.sem(axis=1)], 
                    axis=1).reset_index()
           .dropna())
test_df.columns = ['trees', 'mean', 'sem']

test_df.plot.scatter(x='trees', y='mean', yerr='sem');
plt.title('Tree count effects on model results - higher split count');


# It is possible that more trees (~20 or so) may be better here. 

# ## 5.  Predictions and Submission

# ### 5A.  Function - predictions from tuned model

# In[ ]:


#
# Write a function to return predictions for a test set, based
# on a training set
#

def test_extra_trees_model_prediction(train_df, test_df,
                                      predictor_columns,
                                      get_accuracy = False,
                                      et_n_estimators=25, 
                                      et_random_state = None,
                                      et_max_features = 'auto', 
                                      et_min_samples_split=15,
                                     et_max_leaf_nodes=25):
    '''Train an ExtraTrees model, and get a prediction.  Optionlly 
    test the prediction (via an accuracy score).  '''

    # Get the training data
    X_train = train_df[predictor_columns]
    Y_train = train_df['Survived']
    
    # Fit the model
    model = sk.ensemble.ExtraTreesClassifier(n_estimators = et_n_estimators,
                                             random_state = et_random_state,
                                             max_features = et_max_features,
                                             min_samples_split = et_min_samples_split,
                                             max_leaf_nodes = et_max_leaf_nodes)
    fitted_model = model.fit(X_train, Y_train)

    # Get the predictions,
    X_test = test_df[predictor_columns]
    predictions = fitted_model.predict(X_test)
    pred_ser = pd.Series(predictions, index = X_test.index).sort_index()

    # Get an accuracy score if indicated
    if (get_accuracy):
        Y_test = test_df['Survived']
        accuracy_score = sk.metrics.accuracy_score(Y_test, predictions)
    else:
        accuracy_score = None
    
    return (pred_ser, accuracy_score)
    


# ### 5B.  Ensembling - repeating models and combining results
# 
# As the random state affects model outcomes, it may be possible to game the Kaggle scoring by trying different submissions.  In addition, repeated runs of this kernel could result in significantly different scores.  I will look at repeating runs and using a majority vote for final predictions.
# 
# Before creating the final predictions, I test how ensembling affects the results.

# In[ ]:


#
# Test ensembling by splitting the training data again.
# Check that this ensembling brings the overall score near
# the average score for individual runs
#

ens_train, ens_test = train_test_split(train_mod_df, test_size=0.2, random_state = 1)

#
# Repeat modeling for a range of random states
#

num_trials = 21

st_pred_items = [test_extra_trees_model_prediction(ens_train, ens_test, predictor_columns,
                                                    get_accuracy=True,
                                                    et_min_samples_split=15,
                                                    et_random_state = x) 
                 for x in range(0,num_trials)]

predictions = [item[0] for item in st_pred_items]
scores = [item[1] for item in st_pred_items]

#
# Combine predictions - use weighted 
# average majority vote for each observation
#

p_df = pd.DataFrame(predictions).transpose().sort_index()
ens_pred = p_df.sum(axis=1).apply(lambda x: 1 if x > int(num_trials/2) else 0)

ens_Y_test = ens_test['Survived'].sort_index()
accuracy_score = sk.metrics.accuracy_score(ens_Y_test, ens_pred)
print('ensemble accuracy score:{}'.format(accuracy_score))
print('mean score for individual predictions: {}'.format(np.mean(scores)))
print('std dev for individual predictions: {}'.format(np.std(scores)))


# ### 5C.  Final training and submission data
# 
# The final predictions are based on a majority vote of all runs

# In[ ]:


#
# Re-train the model using all the available training data.  
# Then apply the model to the test data, including ensembling
#

test_mod_df = df_prepare_analysis_binary(test_data, df_analysis)


#
# Repeat modeling for a range of random states
#

num_trials = 21

fin_pred_items = [test_extra_trees_model_prediction(train_mod_df, test_mod_df, 
                                                   predictor_columns,
                                                   get_accuracy=False,
                                                   et_min_samples_split=15,
                                                   et_random_state = x) 
                 for x in range(0,num_trials)]

predictions = [item[0] for item in fin_pred_items]

#
# Combine predictions - use weighted 
# average majority vote for each observation
#

final_pred_df = pd.DataFrame(predictions).transpose().sort_index()
ens_pred = (final_pred_df.sum(axis=1)
            .apply(lambda x: 1 if x > int(num_trials/2) else 0))
ens_pred.name = 'Survived'
ens_pred.head(10)


# In[ ]:


#
# Export the predictions to a file
#

ens_pred.to_csv('predictions_vc20171002.csv', header='True')


# Do some very simple sanity checks on the predictions.  Does the overall survival rate make sense?  What about rates by sex and age?

# In[ ]:


print('training data survival rate: {0:.3g}'
      .format(train_mod_df['Survived'].mean()))
print('test predicted survival rate: {0:.3g}'
      .format(ens_pred.mean()))


# In[ ]:


#
# Survival by sex
#

sex_dict = {'train' : train_mod_df[['Sex']], 'test' : test_mod_df[['Sex']]}
sex_comp_df = pd.concat(sex_dict.values(),axis=0,keys=sex_dict.keys())

sur_dict = {'train' : train_mod_df['Survived'], 'test' : ens_pred}
sur_comp_df = pd.DataFrame(pd.concat(sur_dict.values(),axis=0,keys=sur_dict.keys()))

test_df = pd.concat([sex_comp_df, sur_comp_df], axis=1)

t = (test_df.reset_index(0)
     .pivot_table(index='Sex', columns='level_0', aggfunc=[np.mean, scipy.stats.sem]))
t['mean'].plot(kind='bar', yerr=t['sem']);
plt.title('Survival rates by sex: test and training data');


# In[ ]:


#
# Survival by age.
#

age_dict = {'train' : train_mod_df['age_mod'], 
            'test' : test_mod_df['age_mod']}
age_comp_df = pd.concat(age_dict.values(),axis=0,keys=age_dict.keys())

test_df = pd.concat([age_comp_df, sur_comp_df], axis=1)
test_df['age_grp'] = test_df['age_mod'].apply(lambda x: 20*int(x/20)if x < 60 else 60)

t = (test_df.reset_index(0)
     .pivot_table(index='age_grp', values='Survived',
                  columns='level_0', aggfunc=[np.mean, scipy.stats.sem]))
t.head()
t['mean'].plot(kind='bar', yerr=t['sem']);
plt.title('Survival rates by age grp: test and training data');


# In[ ]:


ax = sns.distplot(test_mod_df['age_mod'])


# The test predictions and training data seem to have some differences in survival rates by sex and age.  Presumably interaction terms are a factor.

# ## 6.  Discussion
# 

# This is my first submission.  I was surprised that the test accuracy was less than my training data cross-validation accuracies.  I suspect overfitting as a culprit.
# 
# Data maniplulations:  Binary encoding of categorical variables reduced the degrees of freedom, and may have improved predictions,  but also reduce interpretability.  I believe that interpretability is important in generalizing to other data sources.  The ticket_share field did not seem to improve model accuracy significantly.
# 
# Model tuning: Increasing the number of samples considered at a split seemed to improve predictions.  Ensembling reduced variability due to the random state of the system.
# 
# Usefulness:  Maching learning is relevant when you want to guess an unknown outcome, not when you want to understand factors that contribute to an effect.  There is usually a reason you want the predictons (an intervention).  Interventions I can imagine might be selling insurance to passengers or ships, or perhaps predicting whether you should get on a ship.  The predictions here are based on many variables that are very specific to the titanic, and/or are not actionable.  The characteristics of the ship itself may be more important in predicting survival, compared to passenger characteristics.  I might want a simpler model for predictions for other ships/voyages.

# ## 7.  Referfences
# 
# Extremely Randomized Trees:  http://montefiore.ulg.ac.be/~ernst/uploads/news/id63/extremely-randomized-trees.pdf
# 
# Ensembling: https://mlwave.com/kaggle-ensembling-guide/
# 
