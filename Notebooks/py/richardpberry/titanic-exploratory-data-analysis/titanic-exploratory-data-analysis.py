#!/usr/bin/env python
# coding: utf-8

# # Titanic: Exploratory Data Analysis
# 
# Author: Richard Berry
# 
# Date: 30-July-2018
# 
# Here we perform some initial analysis of the raw data to look for problems that need to be fixed or cleaned. This notebook takes a very structured approach to exploratory analysis that focusses on four key areas:
# 
# 1. Checking data types
# 2. Looking for missing and null values
# 3. Looking for outliers
# 4. Discovering relationships between variables
# 
# The aim is simply to indentify things that should be cleaned, inferred or otherwise used for a predictive model.

# In[ ]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting parameters
get_ipython().magic(u'matplotlib inline')
sns.set(style='whitegrid')   # Use seaborn default syling


# ## Data import

# In[ ]:


df = pd.read_csv('../input/train.csv')
df['IsTrain'] = True

df_test = pd.read_csv('../input/test.csv')
df_test['IsTrain'] = False
df_test.insert(1, 'Survived', np.nan) #  Add a 'Survived' column with all values set to NaN

print('DataFrame Shape: ' + str(df.shape) + '\n')


# In[ ]:


# First get a preview of the data
df.head(10)


# ## Check data types
# 
# Look at characteristics in each column. In particuar, we are looking at:
# 1. Naming conventions for column names. Should be clear and well labelled, as well as space/special character free
# 2. The data type of each column, in particular things that should be numbers should have a numeric data type (rather than object/string)

# In[ ]:


# Examine column labels
print(df.columns)
print(df.info())


# All column names and data types look fine. We could convert some of the categorical data (e.g. Parch) to a category type, but we will leave this for now.
# We can see that we are missing some data though. We explore this next.

# ## Check missing and null values
# 1. Plot missing values to get a visual overview of what data we are missing
# 2. Use series.value_counts methods to look at different values for categorical data (e.g 'missing' instead of NaN)
# 
# In order to plot missing values we first create a missmap function.

# In[ ]:


# The following function was obtained from here
# http://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
# All credit to Tom Augspurger
# Minimal changes were made to make it Python 3 compatible

from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections as collections
from matplotlib.patches import Rectangle


def missmap(df, ax=None, colors=None, aspect=4, sort='descending',
            title=None, **kwargs):
    """
    Plot the missing values of df.

    Parameters
    ----------
    df : pandas DataFrame
    ax : matplotlib axes
        if None then a new figure and axes will be created
    colors : dict
        dict with {True: c1, False: c2} where the values are
        matplotlib colors.
    aspect : int
        the width to height ratio for each rectangle.
    sort : one of {'descending', 'ascending', None}
    title : str
    kwargs : dict
        matplotlib.axes.bar kwargs

    Returns
    -------
    ax : matplotlib axes

    """

    if ax is None:
        fig, ax = plt.subplots()

    # setup the axes
    dfn = pd.isnull(df)

    if sort in ('ascending', 'descending'):
        counts = dfn.sum()
        sort_dict = {'ascending': True, 'descending': False}
        counts = counts.sort_values(ascending=sort_dict[sort])
        dfn = dfn[counts.index]

    # Up to here
    ny = len(df)
    nx = len(df.columns)
    # each column is a stacked bar made up of ny patches.
    xgrid = np.tile(np.arange(nx), (ny, 1)).T
    ygrid = np.tile(np.arange(ny), (nx, 1))
    # xys is the lower left corner of each patch
    xys = (zip(x, y) for x, y in zip(xgrid, ygrid))

    if colors is None:
        colors = {True: '#EAF205', False: 'k'}

    widths = cycle([aspect])
    heights = cycle([1])

    for xy, width, height, col in zip(xys, widths, heights, dfn.columns):
        color_array = dfn[col].map(colors)

        rects = [Rectangle(xyc, width, height, **kwargs)
                 for xyc, c in zip(xy, color_array)]

        p_coll = collections.PatchCollection(rects, color=color_array,
                                             edgecolor=color_array, **kwargs)
        ax.add_collection(p_coll, autolim=False)

    # post plot aesthetics
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    ax.set_xticks(.5 + np.arange(nx))  # center the ticks
    ax.set_xticklabels(dfn.columns)
    for t in ax.get_xticklabels():
        t.set_rotation(90)

    # remove tick lines
    ax.tick_params(axis='both', which='both', bottom=False, left=False,labelleft=False)
    ax.grid(False)

    if title:
        ax.set_title(title)
    return ax


# In[ ]:


# Examine the missing values
fig = plt.figure(figsize=(12.0, 8.0))
ax = fig.subplots()
ax = missmap(df, title='Missing Values', ax=ax)
plt.show(ax)


# Visualising the missing data it's clear that we are missing many age cabin values and age values, as well as a couple from Embarked.

# In[ ]:


# Find passengers with missing Embarked details
df[df.Embarked.isnull()]


# Only two values missing here. Interestingly, both passengers have the same ticket number which seems odd. Let's see if this is the case for other passengers.

# In[ ]:


# Find duplicate ticket numbers
duplicate_index = df.duplicated(subset='Ticket', keep=False)
duplicates = df[duplicate_index].sort_values('Ticket')
print("Total number of duplicate ticket values: " + str(len(duplicates)))
duplicates.head(30)


# There are 572 duplicate ticket numbers. Duplicates often have the same Fare and Cabin numbers. Many of the duplicates are clearly part of a family group (e.g. Mr Engelhart Ostby & Miss Helene Ostby), however this is not always the case and sometimes there is no clear association by name. However, duplicate tickets seem to be 1st class passengers - perhaps wealthy patrons travelling with an entourage?
# 
# This seems an oddity, but probably not significant. Let's move on to looking at values in individual columns. For categorical data let's look at the types of values to make sure nothing is obviously out of place.

# In[ ]:


# Quality check categorical variables to look for missing values or categories out of the ordinary
# There are too many possibilitities for Ticket and Cabin to look at these sensibly

print('Gender values: ')
print(df.Sex.value_counts())

print('\nEmabarked Values: ')
print(df.Embarked.value_counts())

print('\nPclass: ')
print(df.Pclass.value_counts())

print('\nSibSp: ')
print(df.SibSp.value_counts())

print('\nParch: ')
print(df.Parch.value_counts())


# Nothing looks out of the ordinary there. Let's move on to looking at continuous variables.

# ## Check for outliers
# 1. Use df.describe() and look at output for numerical data
# 2. Plot each continuous variable to check for outliers. Use histograms for single continuous variables. Boxplots are also handy and can be split by another categorical column

# In[ ]:


# First look at summary statistics for numerical data
df.describe()


# Again, nothing obviously out of place there, except possibly some outliers for 'Fare', where at least the max and possibly the min values could be suspicous. Let's try plotting. The only continuous variables we have are age and fare.

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(df.Age[df.Age.isnull() == False], ax=ax1)
ax1.set_title("Distribution of age values")
sns.distplot(df.Fare[df.Fare.isnull() == False], ax=ax2)
ax2.set_title("Distribution of fare values")


# In[ ]:


df[df.Fare > 300]


# On the surface it looks like some of the high fare values may be outliers. However, looking at the raw data it is clear that all are first class passengers. The are from the same family and shared the same cabin, the other two also travelled on the same ticket number so perhaps are part of an entourage. It seems unlikely that this is erroneous entry, so let's move onto the final stage of our exploration.

# ## Look for patterns between variables
# 
# This is the final (and most interesting) step. We are looking for two things:
# 1. More detailed breakdown of some of our variables to look for problems with the data
# 2. Correlations that we can use to build a model
# 
# First let's start out with some correlations.

# In[ ]:


# Calculate correlation values between our numerical values
# To include gender details we need to include a new column, assigning 1 for male, 0 for female
# Necessary as pandas correlattion method does not accomodate categorical variables
# We'll also remove PassengerId and IsTrain from our correlation.

df.Sex = df.Sex.astype('category')
df['SexN'] = df.Sex.cat.codes

df_corr = df.corr()
df_corr.drop(labels=['PassengerId', 'IsTrain'], axis=0, inplace=True) # Drop from rows
df_corr.drop(labels=['PassengerId', 'IsTrain'], axis=1, inplace=True) # Drop from columns
df_corr


# In[ ]:


# To make it easier to see the relationships we'll make it into a heatmap

# Generate a mask for the upper triangle so we don't see repeated correlations
mask = np.zeros_like(df_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio. Fix the max and min values for even comparison.
sns.heatmap(df_corr, vmin=-1, vmax=1, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# The most highly correlated values with survival are (in order):
# - Gender (males were less likely to survive)
# - Pclass (lower class were less likely to survive)
# - Fare (passengers paying higher fares were more likely to survive)
# - Parch (higher numbers of parents and children were more likely to survive)
# - Age (younger passengers were more likely to survive)
# - SibSp (passengers with more siblings or spouses were slightly less likely to survive)
#     
# Notable cross-correlations
# - Pclass is inversely associated with fare, which makes sense as higher paying passengers will generally have a higher class
# - Pclass is also inversely associated with associated with age - higher class passengers were likely to be older
# - Parch and SibSp are correlated which also makes sense as families will travel together
# 
# Finally, we visualise some of these relationships in more detail.

# #### Survival by passenger class, age and sex

# In[ ]:


fig = sns.factorplot(x="Sex", y="Age", col="Pclass", kind="swarm", ci=None, data=df)
fig = sns.factorplot(x="Sex", y="Survived", col="Pclass", kind="bar", ci=None, data=df)


# From the swarm plot we can see that there is a general trend toward higher class passengers being older. The bar plot shows us that in all clases males were less likely to survive than females, and that lower class passengers were less likely to survive than higher class passengers. We examine this in more detail next.
# There is also a large increase in the number of low and mid-class passengers around 15-21 years old.

# #### Survival by age bracket and sex

# In[ ]:


# First some data supplementation - create age brackets for easier visualisation
df['AgeF'] = np.nan
df.loc[df.Age >= 0, "AgeF"] = "Infant"
df.loc[df.Age >= 2, "AgeF"] = "Young child"
df.loc[df.Age >=5, 'AgeF'] = 'Child'
df.loc[df.Age >= 16, 'AgeF'] = 'Adult'
df.loc[df.Age >= 50, 'AgeF'] = 'Elderly'

# Set the ordering of columns and then plot
col_order = ['Infant', 'Young child', 'Child', 'Adult', 'Elderly']
fig = sns.factorplot(x="Sex", y="Age", col="AgeF", col_order=col_order, kind="swarm", ci=None, data=df)
fig = sns.factorplot(x="Sex", y="Survived", col="AgeF", col_order=col_order, ci=None, kind="bar", data=df)


# The number of male and female passengers in each age bracket is similar, though higher for males. While survival rate is higher for females at all age brackets, the swarm plot shows there is only a small number of data points for infants (<2 years), young children (2 - 5 years), and children (5-16 years), therefore this comparison may not be meaningful for these age brackets. There is a clear age bias at older age brackets however, with elderly females being highly likely to survive, while elderly males were highly unlikely to survive.

# #### Survival by parents/children
# Here we look at the distribution of pasengers by number of family members, and the relationship between having parents/children onboard and survival.

# In[ ]:


bins = np.arange(0, 6, 1)
g = sns.FacetGrid(df, col="AgeF", row="Sex", hue="Sex", col_order=col_order, legend_out=True, ylim=(0, 500))
g = g.map(plt.hist, "Parch", bins=bins)
fig = sns.factorplot(x="Parch", y="Survived", col="AgeF", hue="Sex", col_order=col_order, ci=None, kind="bar", data=df)


# The histograms show us that the vast majority of passengers were adults unaccompanied by children - especially true for males. Interestingly there were a small number of Child passengers (ages 5-16) travelling without accompanying adults. They tended to be mid-low class and had relatively high survival compared to other children in a similar age group.

# In[ ]:


# Find passengers under the age of 16 travelling unaccompanied by adults
df.loc[(df.Age < 16) & (df.Parch == 0)]


# # Data Exploration Summary
# 
# What have we learned from our data analysis? These are the key observations:
# - The data is very clean. No suspect labels or outliers were observed
# - There are many missing values for Age and Cabin. Fare and Embarked contained 1-2 missing data points only, the rest of the data is complete
# - Many passengers have the same ticket number, suggesting they are travelling together as a group. The significance of this is difficult to determine but it may serve as a useful proxy for filling missing cabin data
# - Age values are skewed to the left, with most passengers (male and female) between around 16-25 years of age
# - Fare values seem to follow an exponential decay. Extremely high fare values do not appear to be outliers but may represent the ticket cost for a group of 1st class passengers
# - Most adult passengers (male and female) were not accompanied by children 
# 
# As for correlations:
# - Age, Pclass, Sex are strongly correlated with survival. Missing age values should be inferred to maximise this correlation
# - Fare is also correlated, but as it is also related to Pclass this may be of secondary importance
# - Parch may only be meaningfully associated with survival for young passengers. Somewhat suprisingly, adult passengers travelling with children do not appear to be more likely to survive than adult passengers without children
# - Logically, cabin may be a very useful predictor of survival, though it needs cleaning work before it can be used. Pclass may actually be serving as a proxy for cabin location
