#!/usr/bin/env python
# coding: utf-8

#  author: André Daniël VOLSCHENK  
# Kaggle project {Titanic: Machine Learning from Disaster}  
# kaggle.com/andredanielvolschenk  
# 
# Welcome to my first Data Science exploration project!  
# The purpose of this kernel is to provide a complete example of a Data Science process, as applied to the simple "Titanic" dataset.
# 
# That being said, I hope you enjoy this journey of data-driven discovery!   
# 
# NOTE: I have decided to hide most of my code, in order to make this notebook easily readable.  
# To view the code for any step, simply click the "Code" button found on the right hand margin of this notebook.

# # Problem statement
# 
# Binary classification:  
# It is your job to predict if a passenger survived the sinking of the Titanic or not.  
# Metric: "accuracy”
# 
# Data Dictionary:  
# 
# |Variable	     | Definition		    | Key
# |-------------------------------------------------------------------------------------------------
# |survival 	                | Survival 		                   |  0 = No, 1 = Yes
# |pclass 		            | Ticket class 		             |  1 = 1st, 2 = 2nd, 3 = 3rd
# |sex 		                  | Sex 	
# |Age 		                 | Age in years 	
# |sibsp 		                 | # of siblings / spouses aboard the Titanic 	
# |parch 		                | # of parents / children aboard the Titanic 	
# |ticket 		             | Ticket number 	
# |fare 		                  | Passenger fare 	
# |cabin 		                | Cabin number 	
# |embarked 	           | Port of Embarkation      |  C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Variable Notes:
# 
# pclass: A proxy for socio-economic status (SES)  
# 1st = Upper  
# 2nd = Middle  
# 3rd = Lower  
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...  
# Sibling = brother, sister, stepbrother, stepsister  
# Spouse = husband, wife (mistresses and fiancés were ignored)  
# 
# parch: The dataset defines family relations in this way...  
# Parent = mother, father  
# Child = daughter, son, stepdaughter, stepson  
# Some children travelled only with a nanny, therefore parch=0 for them.  

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot') # if error, use plt.style.use('ggplot') instead

import seaborn as sns
sns.set_style("whitegrid")     # need grid to plot seaborn plot

import scipy.stats as ss
import math

import sklearn as skl
import sklearn.metrics as sklm
import sklearn.feature_selection as fs
import sklearn.model_selection as ms

import sklearn.preprocessing as prep
import sklearn.decomposition as decomp

import sklearn.pipeline as pipe

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# # Global variables

# In[ ]:


glbl = {}   # define dict
glbl['show_figs'] = 1      # flag to enable printing figures
glbl['n_jobs'] = 1        # -1 to use all available CPUs
glbl['random_state'] = 5   # = None if we dont need demo mode
glbl['n_iter'] = 100        # how many search iterations
glbl['n_splits']=10            # cross validation splits


# # Load
# Lets load the training and testing data.

# In[ ]:


path = '../input/train.csv'
data1 = pd.read_csv(path, sep=',', index_col = 'PassengerId')
del(path)

# load the competition test data
path = '../input/test.csv'
data2 = pd.read_csv(path, sep=',', index_col = 'PassengerId')
del(path)

print('data1 shape:', data1.shape)
print('data2 shape:', data2.shape)


# `data1` is from 1 through 891  
# `data2` is from 892 through 1309

# # Merge
# We need to merge these so that we can clean both simultaneously. This is important to ensure that `data1` and `data2` have even number of features, and that their features are represented in the same way.

# In[ ]:


data = data1.append(data2, sort=False)  # Append rows of data2 to data1

# clean workspace
del(data1, data2)

print('data shape:', data.shape)


# # View/summarize
# View Column DataTypes

# In[ ]:


data.dtypes


# We clearly need to convert some datatypes. For example, `Pclass` is ordinal, `Sex` is nominal, etc.  
# Lets now also look at a sample of the data

# In[ ]:


data.sample(10)     # take a random sample of 10 observations


# Some features look useless, for example: `Ticket`, `Cabin`, `Embarked`. These will not contribute to predictive power.  
# `Name` would be useless, but we can actually extract some info from it, namely, Title. After that, we can delete it.  
# 
# # Cleansing
# Lets drop useless features...
# Now see if we have ?s

# In[ ]:


# drop useless features
data.drop(labels=['Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# see if we have ?s
print ('number of nans in data:', (data.astype(np.object) == '?').any().sum() ) # we have no ?s


# We have 0 ?s in our dataset.  
# Lets see if we have any nans

# In[ ]:


data.isnull().sum() # check each column.


# 418 nans in `Survived`. We can ignore this, since these are the Kaggle testing data, for which we are not given labels.  
# 263 nans in `Age`.  
# 1 nan in `Fare`  
# For the `Age` feature, that is 263/1309 ~ 20% of the data !!!  
# 
# First we consider the missing `Fare`. we will use the median `Fare` for the `Pclass`...  
# Lets see if we still have nans in `Fare`:

# In[ ]:


for Pcl in data.Pclass.unique():   # 1, 2, 3
    med = data[['Fare']].where(data.Pclass==Pcl).median()    # get median over all data for that class
    data.loc[ ((data.Fare.isnull() == True) & (data.Pclass==Pcl)) , 'Fare'] = med[0] # med is series
# clean workspace:
del(Pcl, med)

# see if we have nans:
data.isnull().sum() # check each column.


# Only the `Age` nans remain  
# 
# I hypothesize that none of the current variables strongly correlate with `Age`. I further speculate that a `Title` feature would correlate with `Age`. We first have to create `Title`.  
# 
# # Feature Engineering
# First, we noticed easlier that a dot (period) is placed after titles in the `Name` column.  
# The title is early in the `Name` column, so we use the FIRST occurence of the period.  
# 
# Lets write a function to extract `Title` from `Name`...  
# Lets now see the unique titles in our dataset

# In[ ]:


def extract_title(x):   # x is entire row
    string=x['Title']
    ix=string.find(".")    # use .find to find the first dot
    for i in range(0,ix):
        if (string[ix-i] == ' '):  # if we find space, then stop iterating
            break                   # break out of for-loop
    return string[(ix-i+1):ix]  # return everything after space up till before the dot

data['Title'] = data.Name  # for now copy name directly
data['Title']=data.apply(extract_title, axis=1)     # axis = 1 : apply function to each row
data.drop(labels=['Name'], axis=1, inplace=True)  # we can even drop the 'Name' column now

data.Title.unique()   # lets see the unique titles in our dataset


# Lets keep titles: 'Mr', 'Mrs', 'Miss', 'Master'. All others will be converted to one of these.  
# Notes: Mme is Mrs, and Mlle is Miss, in french. ‘Jonkheer’ is a male honorific for Dutch nobility  
# 
# # Feature engineering
# We declare the function to standardize the `Title` feature...  
# Lets see the unique titles in our dataset

# In[ ]:


# standardize 'Title'
def standardize_title(x):   # x is an entire row
    Title=x['Title']
    
    if x.Sex == 'male':
        if Title != 'Master':   # we can keep 'Master' title, but we want to change all others to Mr
            return 'Mr'
        else:
            return Title
    if x.Sex == 'female':
        if Title in ['Miss', 'Mlle', 'Ms']:
            return 'Miss'
        else:
            return 'Mrs'

data['Title']=data.apply(standardize_title, axis=1)     # axis = 1 : apply function to each row

data.Title.unique()   # lets see the unique titles in our dataset


# Perfect. All titles have been standardized!
# 
# # Visualization
# 
# We have seen earlier taht the NaNs in `Age` corrupt 20% of our data.   
# We dont want to lose 20% of our data; more data is often better than a good classifier.  
# Lets look at how `Title` correlates with `Age`

# In[ ]:


if glbl['show_figs']:
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    data[['Age', 'Title']].boxplot(by = 'Title', ax=ax)


# Indeed titles seem to have significantly different age dispersions.  
#  Are there any other features that correlate with `Age`?

# In[ ]:


if glbl['show_figs']:
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.imshow(data.corr(), cmap=plt.cm.Blues, interpolation='nearest') 	# plots the correlation matrix of data
    plt.colorbar()
    tick_marks = [i for i in range(len(data.columns))]
    plt.xticks(tick_marks, data.columns, rotation='vertical')
    plt.yticks(tick_marks, data.columns)


# `SibSp` correlates somewhat with `Age`. Lets look at the distribution too:

# In[ ]:


if glbl['show_figs']:
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    data[['Age', 'SibSp']].boxplot(by = 'SibSp', ax=ax)


# The `Age` varies greatly within the lower `SibSp` feature classes. It would not be wise to estimate `Age` from `SibSp`.  
# 
# # Clean
# 
# We will replace nans in `Age` with the median of the given `Title` class.  
# Lets declare a function to do that...   
# Lets see if the nans in `Age` have eliminated:

# In[ ]:


for titl in data.Title.unique():
    med = data[['Age']].where(data.Title==titl).median()
    data.loc[ (data.Age.isnull() == True) & (data.Title==titl) , 'Age'] = med[0] # med is a series. must be a scalar
del(titl, med)

# let us see if we have nans:
data.isnull().sum() # check each column.


# The nans in `Age` have been eliminated!
# 
# # Feature engineering
# 
# We have successfully cleaned our features, and created a new one: `Title`.  
# Perhaps we can think of more features to create from existing ones, that may provide predictive power?
# 
# We can make a new feature called `FamSize` to indicate the family size of any passenger. Perhaps this can replace `Parch` and `SibSp`, or add to predictive power?

# In[ ]:


data['FamSize'] = data.SibSp + data.Parch + 1
print(data.columns)


# # View / summarize
# 
# we have noted earlier that the data-types of our features is NOT correct. Lets take a look again:
# 
# Our features are ...

# In[ ]:


data.dtypes


# ...clearly not the right data types  
# 
# Lets list the data types we have and how they should be represented:
# * Nominal (as bool) : `Sex` , `Title` [also `Survived`, but this we cant do now because it has nans]
# * Ordinal (as ordered category) : `Pclass`
# * Count (as int) : `SibSp`, `Parch`, `FamSize`
# * Real-value (as float) : `Age`, `Fare`  
# 
# # Feature representation
# Lets fix our data types and see if they are correct now:

# In[ ]:


# nominal
data.Sex = data.Sex.map({'female':0, 'male':1})

data = pd.get_dummies(data,columns=['Title'])       # turn nominal to bool

# ordinal
ordered_categs = [1, 2, 3]  # categories for Pclass
categs = pd.api.types.CategoricalDtype(categories = ordered_categs, ordered=True)
data.Pclass = data.Pclass.astype(categs) # ordinal
#
del(ordered_categs) # clean workspace

data.dtypes


# `Survived` will remain a float cause it has nans in it.  
# We still need to turn uint8 into bool and store features in their smallest representation.
# 
# # Feature representation
# 
# Let us write a function that converts datatypes to it's lowest numeric representation...
# Lets see if our dtypes are correct now

# In[ ]:


def to_lowest_numeric(x):
    # x is a column
    if (x.apply(np.isreal).all(axis=0)) & ((str(x.dtypes) != 'category')): # if this column is numeric, but NOT categorical
        x = pd.to_numeric(x, errors='coerce', downcast='float') # first downcast floats
        x = pd.to_numeric(x, errors='coerce', downcast='unsigned') # now downcast ints
    
    # now to handle booleans:
    # if x has only the ints 0 and 1  OR  x has only 'True' and 'False' strings
    if set(x.unique()) == set(np.array([0, 1])) :
        x2=x.astype('bool')
        return x2
    elif set(x.unique()) == set(np.array(['True', 'False'])):
        x2 = x=='True'
        return x2
    else:
        return x
#

data = data.apply(to_lowest_numeric, axis=0)

# View Column DataTypes
data.dtypes


# Compare the dtypes to how they were before the function was applied.  
# The data types now look correct, and their encoding is in its lowest numeric form.  
# Note: `Survived` will remain a float due to nans from `data2`.
# 
# Next we will transform our features to attempt to Normalize and Standardize them. we will be using methods, like PCA, which require this. Furthermore, PCA does not handle outliers. There are classifiers which we shall use, that require this type of data.
# 
# # Split
# 
# Before we can transform our features, we have to do data splitting.  
# We should not use tranforms fitted on all our data, to prevent data leakage.
# 
# First we seperate `data1` and `data2` from `data`.  
# recall:
# * `data` is from 1 through 891
# * `data2` is from 892 through 1309  
# Next, we split `data1` into `X` and `y`

# In[ ]:


data1 = data.iloc[0:891, :]     # iloc is incl:excl
data2 = data.iloc[891:1309, :]

# we will split data1 into out train and test sets.
# we will use data2 for the Kaggle submission at the end

X = data1.drop(labels=['Survived'], axis=1)
y=data1['Survived']    # return a series
data2

# clean up Workspace
del(data, data1)

print('X shape:', X.shape)
print('y shape:', y.shape)
print('data2 shape:', data2.shape)


# Lets see if the dtype of `y` is boolean as it should be

# In[ ]:


print ('y dtype:', y.dtypes ) 


# No it is not boolean. Lets downcast it to boolean.

# In[ ]:


y = to_lowest_numeric(y)
print ('y dtype:', y.dtypes ) # = 'bool'     this is correct.


# Yes it is boolean. Good.  
# Next we will split `X` and `y` into new sets. The train:test ratio shall be 80:20.

# In[ ]:


# let us split the train:test ratio at 80:20
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, 
                                                       random_state = glbl['random_state'])

del(X,y) # clean up Workspace

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# We will be looking at transforms, dimensionality reduction, and learning next. In these steps it is important to prevent data leakage, so we will be fitting each step with the X_train, and applying it blindly to X_test.  
# 
# For now, lets just make a copy of X_train to investigate the effects of our transforms. We can alter this copy without actually altering X_train.

# In[ ]:


X_train2 = X_train.copy()
print('X_train2 shape:', X_train2.shape)


# # View / summarize
# Lets look at the mean and standard deviation of `X_train2`

# In[ ]:


X_train2.describe()


# These features have very different means and standard deviations - this clearly needs standardization.   
# Ideally our data should be normally distributed. Lets take a look at that next.
# 
# # Visualize
# 
# Let us look at the distributions of Quantitative data. We will use Violin plots to show distributions of all Quantitative variables.

# In[ ]:


def all_violin(X):  # X is a dataframe
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    ax.set_title('Violin plots of quantitative variables')
    
    # lets store up all the columns that are from Quantitative variables
    # aka we will find columns that are not Nominal (bool dtype) or Ordinal (category dtype)
    dtypes = pd.DataFrame( X.dtypes )
    dtypes = dtypes.astype('str').values.reshape(-1,)
    valid = ( (dtypes != 'bool') & (dtypes != 'category') )
    
    X = X.iloc[:,valid]     # contains only columns that are not bool and not category
    
    ax = sns.violinplot(data=X)     # plot on single axis

if glbl['show_figs']:
    all_violin(X_train2)


# Clearly our ranges are completely different across quantitative variables. Furthermore, the features (except maybe `Age`) do not resemble a normal distribution, so we have established the need to transforms.  
# 
# Lets also look at the boxplots:

# In[ ]:


def all_box(X):  # X is a dataframe
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    ax.set_title('Boxplots of quantitative variables')
    
    # lets store up all the columns that are from Quantitative variables
    # aka we will find columns that are not Nominal (bool dtype) or Ordinal (category dtype)
    dtypes = pd.DataFrame( X.dtypes )
    dtypes = dtypes.astype('str').values.reshape(-1,)
    valid = ( (dtypes != 'bool') & (dtypes != 'category') )
    
    X = X.iloc[:,valid]     # contains only columns that are not bool and not category
    
    ax = sns.boxplot(data=X)     # plot on single axis

if glbl['show_figs']:
    all_box(X_train2)


# From these boxplots, we can see that some of our quantitative variables suffer greatly from outliers.  
# 
# # Transforms: Normalization
# 
# Lets Normalize !
# Done to in hopes of making data:
# * Normally distributed
# * Homogeneity of variance  
# The latter is more important than the former
# 
# Let us apply the Quantile normalizer to the relevent data.  
# The Quantile transformer is robust to outliers, which is perfect, since we have shown that we have serious outliers.  
# Lets look at what those features look like now.

# In[ ]:


norm = prep.QuantileTransformer()
f=X_train2.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'FamSize']]
norm.fit( np.array(f) )
f = norm.transform(np.array(f))
X_train2.loc[:,['Age', 'SibSp', 'Parch', 'Fare', 'FamSize']]=f

del(f)

if glbl['show_figs']:
    all_violin(X_train2)


# `Age` and `Fare` look more normally distributed than they did before.
# Normalization is beneficial for the standardization transformation coming next, but it is not vital.
# 
# # Transforms: Scaling
# Lets remind ourselves again about the mean and standard deviation...

# In[ ]:


X_train2.describe()


# Standard deviation is still totally different, but the data types are still fine.  
# We will use the Standard Scaler to standardize the relevent features.  
# This should make the means=0 and the std=1.

# In[ ]:


scaler = prep.StandardScaler()

f = X_train2.loc[:,['Age','SibSp','Parch','Fare','FamSize']]

scaler.fit( f )
f = scaler.transform( f )
f = pd.DataFrame (f)

X_train2.loc[:,['Age','SibSp','Parch','Fare','FamSize']] = f.values

# clean up workspace
del(f)

X_train2.describe()


# It looks like our quantitative variables have (approximately) zero mean and unit variance. Therefore standardized.  
# Data types are still correct.
# 
# # Visualize
# 
#  Lets see how this affected the relevent features.

# In[ ]:


if glbl['show_figs']:
    all_violin(X_train2)


# Lets also look their boxplots:

# In[ ]:


if glbl['show_figs']:
    all_box(X_train2)


# Looks like we have removed the outliers.  
# Interesting that the `Parch` boxplot shows outliers, however we can rest assured that the data is standardized, as we confirmed in the table of means and stds earlier.
# 
# # Dimensionality reduction
# 
# We want to eliminate redundant or harmful features in the model selection stage. This can be done using some feature selection method or feature projection method.  
# Let us demonstrate each of these approaches.
# * We can select a Feature selection method : RFE (Recursive feature elimination)
# * We select a feature projection method : PCA (Principle Component Analysis)  
# To use PCA, our data must be standardized, and should not have outliers. Our data satisfies these requirements.
# 
# Let's investigate PCA briefly. We will compute the principle components for the training feature subset...  
# Lets look at the shape of `X_train2`, and lets also look at the explained variance per PC graphically:

# In[ ]:


pca = decomp.PCA()
pca = pca.fit(X_train2)

def plotPCA_explained(mod):
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.suptitle('Scree plot of explained variance per principle component')
    ax.set_xlabel('number of components')   # Set text for the x axis
    ax.set_ylabel('explained variance')   # Set text for y axis
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    x = [y + 1 for y in x]          
    plt.plot(x,comps)
#
print ('X_train2 shape:', X_train2.shape)
if glbl['show_figs']:
    plotPCA_explained(pca)


# This curve is often referred to as a scree plot. Notice that the explained variance decreases rapidly until the 6th component and then slowly, thereafter. The first few components explain a large fraction of the variance and therefore contain much of the explanatory information in the data. The components with small explained variance are unlikely to contain much explanatory information. Often the inflection point or 'knee' in the scree curve is used to choose the number of components selected.   
# Now it is time to create a PCA model with a reduced number of components. The code in the cell below trains and fits a PCA model with 6 components, and then transforms the features using that model.

# In[ ]:


pca_6 = decomp.PCA(n_components = 6)
pca_6.fit(X_train2)
X_trainPCA = pca_6.transform(X_train2)
print ('X_train2 shape:' ,X_trainPCA.shape)


# The data shape has been reduced from (712, 11) to (712,6).  
# 
# Now lets rank features by importance in PCA space, and visualize the 1st and 2nd principle components.

# In[ ]:


def drawPCAVectors(transformed_features, components_, columns):
    fig = plt.figure(figsize=(8,6))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.suptitle('Features in principle component space')
    ax.set_xlabel('PC1')   # Set text for the x axis
    ax.set_ylabel('PC2')   # Set text for y axis
    num_columns = len(columns)
    # This funtion will project your *original* feature (columns) onto your principal component feature-space, so that you can visualize how "important" each one was in the multi-dimensional scaling
    # Scale the principal components by the max value in the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])
    ## visualize projections
    # Sort each column by it's length. These are your *original* columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    print("Features by importance:\n", important_features)
    ax = plt.axes()
    for i in range(num_columns):
        # Use an arrow to project each original feature as a labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)
    return ax

if glbl['show_figs']: 
    drawPCAVectors(X_trainPCA, pca_6.components_, X_train2.columns)

# clean workspace
del(X_trainPCA)

# until now we used X_train2, which was a copy of X_train. X_train2 was used to illustrate thet transforms that we will be using in the pipeline. We did not want to alter X_train in any way, but now we can actually implement our transforms as part of our pipeline for real. We can therefore delete X_train2
del(X_train2)


# # Model selection: Pipeline
# In our pipeline we want to include:
# * Feature Normalizer transform : QuantileTransformer
# * Feature Scaler transform  : standardScaler
# * Dimensionality reduction  : RFE or PCA
# * Estimator  : test multiple
# 
# First, we will define a function that returns an estimator and its parameter distributions.  
# For example, the following function requires only an input like `logit` and the y_train data to return a Logistic Regression classfier and parameter distributions that we can search over when we want to optimize it.  
# 
# Next, we declare a function where we can specify which estimator we want to use. The function creates 2 full pipelines for this estimator, one using RFE, and one using PCA.  
# 
# Lets try this out!  
# We will make a pipeline for logit model (using`idx = 0`):

# In[ ]:


def get_estimator(est, y_train):
    #
    ratio_classes =  pd.Series(y_train).value_counts(normalize=True)
    #
    if (est == 'LogisticRegression') | (est == 'logit'):
        from sklearn.linear_model import LogisticRegression as estimator
        parameter_dist = {
                'penalty' : ['l2'], # 'penalty' : ['l1', 'l2'],     # default l2
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’, default: None
                'C': ss.expon(scale=100), # must be a positive float
        }
    elif (est == 'KNeighborsClassifier') | (est == 'knc'):
        from sklearn.neighbors import KNeighborsClassifier as estimator
        parameter_dist = {
                'n_neighbors' : ss.randint(1, 11),
                # 'weights' : ['uniform', 'distance'],
                # 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
    elif (est == 'LinearSVC') | (est == 'lsvc'):
        from sklearn.svm import LinearSVC as estimator
        parameter_dist = {
                'C': ss.expon(scale=10),
                # 'penalty' : ['l1', 'l2'],
                # 'multi_class' : ['ovr', 'crammer_singer'],
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    elif (est == 'SVC') | (est == 'svc'):
        from sklearn.svm import SVC as estimator
        parameter_dist = {
                'C': ss.expon(scale=10),
                'gamma' : ss.expon(scale=0.1), # float, optional (default=’auto’). If gamma is ‘auto’ then 1/n_features will be used instead.
                # 'kernel' : ['rbf', 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    elif (est == 'DecisionTreeClassifier') | (est == 'dtc'):
        from sklearn.tree import DecisionTreeClassifier as estimator
        parameter_dist = {
                'criterion' : ['entropy'], # 'criterion': ['gini', 'entropy'],
                # 'splitter' : ['best', 'random'],
                # 'max_depth': [None, 3],'min_samples_split': ss.randint(2, 11),
                'min_samples_leaf': ss.randint(1, 11),
                'max_features': ss.uniform(0.0, 1.0), # we have to make this a float
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    elif (est == 'RandomForestClassifier') | (est == 'rfc'):
        from sklearn.ensemble import RandomForestClassifier as estimator
        parameter_dist = {
                # 'max_depth': [None, 3],
                'n_estimators' : ss.randint(8, 20), # 'n_estimators' : integer, optional (default=10)
                'criterion': ['gini', 'entropy'],
                'max_features': ss.uniform(0.0, 1.0), # we have to make this a float
                'min_samples_split': ss.randint(2, 11),
                'min_samples_leaf': ss.randint(1, 11),
                "bootstrap": [True, False],
                'class_weight' : [{0:ratio_classes[0], 1:ratio_classes[1]}], #class_weight : dict or ‘balanced’
        }
    #
    estimator=estimator()
    
    if 'random_state' in estimator.get_params():
        estimator.set_params(random_state=glbl['random_state'])
    #
    return estimator, parameter_dist



def createPipes(y_train, idx=0):
    
    '''
    idx specifies which estimator we want to use.
    In this project we chose to demonstrate 6 classifiers:
        Logistic Regression     idx=0
        k-nearest neighbours    idx=1
        linear SVM              idx=2
        SVM                     idx=3
        Decision Tree           idx=4
        Random Forest           idx=5
    
    Outputs: 
        pipe1 { a pipeline using RFE for dimensionality reduction }
        param_dist1 { parameter distributions of the components in pipe1 }
        pipe2 { a pipeline using PCA for dimensionality reduction }
        param_dist2 { parameter distributions of the components in pipe2 }
    '''
    
    # the normalizer
    norm = prep.QuantileTransformer(random_state=glbl['random_state'])
    # the scaler
    scaler=prep.StandardScaler()
    # the classfier estimator as a function of input 'idx'
    classifiers = ['logit', 'knc', 'lsvc', 'svc', 'dtc', 'rfc'] # let us test these classfiers
    estimator, est_param_dist = get_estimator(classifiers[idx], y_train)
    # dimensionality reduction methods
    rfe = fs.RFE(estimator = estimator)
    rfe_param_dist = {
        'n_features_to_select': ss.randint(1,11),   # since we have 11 features
    }
    pca = decomp.PCA()
    pca_param_dist = {
        'n_components': ss.randint(1,10),   # since we have 11 features
        'random_state' : [glbl['random_state']]
    }
    
    #pipe1 uses RFE with estimator, pipe2 uses PCA with estimator
    
    pipe1 = pipe.Pipeline([
            ('norm', norm),
            ('scaler', scaler),
            ('rfe', rfe),
            ('est', estimator)
    ])
    
    pipe2 = pipe.Pipeline([
            ('norm', norm),
            ('scaler', scaler),
            ('pca', pca),
            ('est', estimator)
    ])
    
    pca_param_dist = {f'pca__{k}': v for k, v in pca_param_dist.items()}    # add 'pca__' string to all keys
    rfe_param_dist = {f'rfe__{k}': v for k, v in rfe_param_dist.items()}    # add 'rfe__' string to all keys
    est_param_dist = {f'est__{k}': v for k, v in est_param_dist.items()}    # add 'est__' string to all keys
    # adding the transformer name in the parameter name is required for pipeline
    
    # merge dictionaries
    param_dist1 = {**rfe_param_dist,  **est_param_dist}
    param_dist2 = {**pca_param_dist,  **est_param_dist}
    
    return pipe1, param_dist1, pipe2, param_dist2


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=0)  # create pipelines for logit model

print('pipe1\n',pipe1)
print('estimator of pipe1:', pipe1.named_steps['est'])
print('------------------------------------------------')
print('pipe2\n',pipe2)
print('estimator of pipe2:', pipe2.named_steps['est'])


# We can see that our 2 pipelines were successfully created.
# 
# # Model Selection
# 
# Each pipeline has hyper-parameters that can be optimized. We need to decide how we will decide which hyper-parameter combination sets (aka models) to try. We also have to decide how we will obtain an accurate estimate of each model's performance (aka how will we validate performance). Lastly we need to specify what a good model performance means - this is the objective function
# 
# We need:
# * an objective function
# * a validation method, and
# * a hyper-parameter tuner
# 
# Our objective function is simply the performance metric: 'Accuracy'  
# Our validation method shall be 'Nested CV'  
# Our hyper-parameter tuner shall be 'Random Search'
# 
# Now let us declare a function that does all the above...  
# 
# Recall that we created pipelines for the logit model. We called them `pipe1` and `pipe2`. Their parameter distributions were saved as `param_dist1` and `param_dist2`, respectively.  
# Now lets see how well `pipe1` does and do model selection:

# In[ ]:


def modelSel(pipeline, param_dist, X_train, y_train):
    
    '''
    This function performs model selection given a pipeline and it's distribution
    inputs:
        pipeline { the pipeline to undergo model selection }
        param_dist { the parameter distributions of the pipeline }
    outputs:
        inner { the model selection object. Contains parameters like best_estimator, best_params, best_index }
        outer { the outer CV results }
        this function also prints out the training score, as well as the testing score for the best estimator
    '''
    
    # define the randomly sampled folds for the inner and outer Cross Validation loops:
    insideCV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
    outsideCV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
    
    ## Perform the Random search over the parameters
    inner = ms.RandomizedSearchCV(estimator = pipeline,
                                param_distributions = param_dist,
                                n_iter=glbl['n_iter'], # Number of models that are tried
                                cv = insideCV, # Use the inside folds
                                scoring = 'accuracy',
                                n_jobs=glbl['n_jobs'],
                                return_train_score = True,
                                random_state=glbl['random_state'])
    # The cross validated random search object, 'inner', has been created.
    
    # Fit the cross validated grid search over the data 
    inner.fit(X_train, y_train)
    # we have now scored each of the n_iter models (hyper-param combo) and we have an average score for each. we can use these scores as a model selection step, or we can feed these scores into an optimization algorithm. we wont use optim algo in this project, so we use best_estimator as our selected estimator.
    
    print('best accuracy on inner (train) set', inner.best_score_)
    
    # -------------------------------------------------
    
    # the inner loop evaluates model performance. we decided to let it do our model selection too.
    # the estimate of the classifier is not reliable though. So we need to have an
    # outer CV where we evaluate the 'best_estimator'
    
    outer = ms.cross_val_score(inner.best_estimator_, X_train, y_train, cv = outsideCV,
                               n_jobs=glbl['n_jobs'])
    
    print('For outer (testing) set:')
    #print('Outcomes by cv fold')
    #for i, x in enumerate(outer):
    #    print('Fold %2d    %4.3f' % (i+1, x))
    print('Mean outer performance metric = %4.7f' % np.mean(outer))
    print('stddev of the outer metric       = %4.7f' % np.std(outer))
    #
        
    return inner, outer
#

inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)


# Lets see how well `pipe2` does and do model selection:

# In[ ]:


inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)


# `pipe2` achieved a better accuracy in the outer CV loop.  
# It achived 82.4%, whereas `pipe1` achieved 82.0%.  
# We select `inner2` and `outer2` as our best model.  
# Recall that `pipe1`and `pipe2`use RFE and PCA, respectively. So for Logistic Regression, PCA was the better step.  
# 
# Lets explore this winning pipeline and its best model a little ...  
# * We will look at 3 random models in the pipeline

# In[ ]:


inner=inner2
outer=outer2

# clean worspace
del(pipe1, param_dist1, pipe2, param_dist2)
del(inner1, outer1, inner2, outer2)


# the results of each model in the pipeline
inner_results = pd.DataFrame( inner.cv_results_ )# the score for each model. there are n_iter models
# look at 3 random models in the pipeline
inner_results.sample(3)


# * Print the parameter values of the best model

# In[ ]:


# print the parameter values of the best model
print( inner.best_estimator_ )


# * Print parameters of the best model

# In[ ]:


# print parameters of the best model
print(inner.best_params_)


# * 'Inner' contains many models. Lets print the index of the best model

# In[ ]:


# 'inner' contains many models. What is the index of the best model?
print( inner.best_index_ ) 


# Great!  
# We have a lot of data available to us all wrapped in 'inner' and 'outer'
# 
# 
# # Visualize
# Now we may ask ourselves:  
# What types of analysis /checks can I do with the scores that I get from the outer K folds?  
# 3 checks that we can do that will provide us with insight:  
# * 1  LEARNING CURVE  
# check for stability of the predictions (use iterated/repeated cross-validation)
# {if we use a learning curve to do this, we can also see the effect of number of
# training samples on performance.}
# * 2 VALIDATION CURVE  
# check for the stability/variation of the optimized hyper-parameters.
# For one thing, wildly scattering hyper-parameters may indicate that the inner optimization
# didn't work. For another thing, this may allow you to decide on the hyperparameters without
# the costly optimization step in similar situations in the future. With costly I do not refer
# to computational resources but to the fact that this "costs" information that may better be
# used for estimating the "normal" model parameters.
# * 3 INNER VS OUTER LOOP  
# check for the difference between the inner and outer estimate of the chosen model.
# If there is a large difference (the inner being very overoptimistic), there is a risk that
# the inner optimization didn't work well because of overfitting.  
# 
# Lets look at the simplest check first: check 3

# In[ ]:


if glbl['show_figs']:
    
    # lets look at the simplest check first: check 3
    innerTest_mean, innerTest_std = inner_results.loc[
        inner.best_index_,['mean_test_score','std_test_score']]
    outerTest_mean = np.mean(outer)
    outerTest_std = np.std(outer)
    
    print('The inner CV mean and standard deviation are, respectively:')
    print( innerTest_mean , innerTest_std)
    print('The outer CV mean and standard deviation are, respectively:')
    print( outerTest_mean , outerTest_std)


# What can we summize?  
# The model doesnt seem to be overoptimistic at all! The inner mean is not optimistic relative to the outer mean.  
# The standard deviation of the outer is slightly larger inner CV, though. But this is not high enough to indicate some instability.  
# Based on these results we are not overfitting, and our estimate of instability is likely stable with a stddev of about 5%.  
# 
# 
# Lets now look at Check 1: Plot learning curves

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = ms.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, random_state=glbl['random_state'],
        train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Inner CV (Training) score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Outer CV (Testing) score")
    
    plt.legend(loc="best")
    return plt
#
CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
if glbl['show_figs']:
    # check 1
    # Plot learning curves
    # Learning curves are a good way to see the overfitting effect on the training set and the effect of the training size on the accuracy.
    
    plot_learning_curve(inner.best_estimator_, "logit learning curves", X_train, 
                            y_train, cv=CV, n_jobs=glbl['n_jobs'])


# From this plot, we see that the red training curve is approaching the green testing curve with more training samples, and intersects at about 650 samples. This suggests that we had enough training samples to bring the inner training fold to be a good estimation of the testing score.  
# Up till 650 samples, the red (training score) line is far above the green (testing) line, so with lower samples we have overoptimistic overfitting training loop.  
# The deviation fill around the red line is not very wide... this indicates that the training scores were quite stable.
# The green line has a wide deviation. The standard deviations in the outer CV are large, so the cross validated scores are not stable.  
# This check contradicts the previous check in some ways. This check suggests the stability estimate of the model is inaccurate. The model is unstable with it's prediction score.  
# 
# Now let's look at check 2: Validation curve  
# 
# We will look at the effect of a hyper-parameter on the score.
# 
# Recall that earlier we printed the parameters of the model using `print(inner.best_params_)`  
# The output was  
# `{'est__C': 21.75031832298129, 'est__class_weight': {0: 0.6151685393258427, 1: 0.3848314606741573}, 'est__penalty': 'l2', 'pca__n_components': 9, 'pca__random_state': 5}`  
#     
# Lets look at the effect of varying logit parameter `C`, and the effect of varying PCA parameter `n_components`  
# 
# First we declare a function to plot the validation curve:
# 
# Lets first see the effect of 'C' on score

# In[ ]:


def plot_validation_curve(estimator, X, y, param_name, param_range, cv=10, scoring="accuracy",n_jobs=1):
    plt.figure()
    train_scores, test_scores = ms.validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # the following is used to generate the title (objNmFull) of the figure
    if (  str (type(estimator)) == "<class 'sklearn.pipeline.Pipeline'>"  ):
        objNm = param_name.split('__')   # what is before the '__' ?
        # which object in the pipeline are we considering?
        objStr=str( estimator.named_steps[objNm[0]] )
        # split at '(' and keep what is before
        objNmFull = str( "Validation Curve for Pipeline for " + objStr.split('(')[0] )
        objNm = objNm[1]
    else:
        objNm = param_name
        objStr=str( estimator )
        objNmFull = str( "Validation Curve for " + objStr.split('(')[0] )
    objNmFull = objNmFull + ' for parameter ' + objNm
    
    plt.title(objNmFull)        # make title
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Inner CV (Training) score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Outer CV (Testing) score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
#

# lets first see the effect of 'C' on score
CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
plot_validation_curve(inner.best_estimator_, X_train, y_train,
                      'est__C', np.logspace(-1, 5, 20), cv=CV,
                      scoring="accuracy", n_jobs=glbl['n_jobs'])


# Parameter `C` has virtually no effect at all on score.  
# Now lets see the effect of n_components on score

# In[ ]:


# now lets see the effect of n_components on score
plot_validation_curve(inner.best_estimator_, X_train, y_train,
                      'pca__n_components',
                      np.linspace(1, 10, 10, dtype = int),
                      cv=CV, scoring="accuracy",n_jobs=glbl['n_jobs'])


# We see significant variation in performance for `n_components`.  
# 
# What value does this add to our investigation?  
# From the validation curves, we can summize that next time we dont really need to tune `C`. but we have to tune `n_components`. we also observe that the hyper-parameters do not scatter wildly, the curve is smooth, so the inner CV provided a good estimation of hyper-parameter effects.  
# 
# 
# So for the logit (Logistic Regression) estimator, we have created and compared pipelines 1 and 2, and we have explored the best model of `pipe2`.  
# Lets save the best logit-based model as `best_logit`:

# In[ ]:


best_logit = inner.best_estimator_

# clean workspace
del(inner, outer)

print('best_logit\n', best_logit)


# Now lets evaluate pipelines with other estimators:
# 
# ### knc (k-nearest neighbours classifier):
# Note that `pipe1` fails because RFE wont work with knc since the knc does not expose "coef_" or "feature_importances_" attributes.

# In[ ]:


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=1)  # create pipelines for knc model

# inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
# this fails, because RFE wont work with knc:
# RuntimeError: The classifier does not expose "coef_" or "feature_importances_" attributes

print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)
# lets name rename inner2 as:
best_knc = inner2.best_estimator_

print('\nbest pipeline saved.')


# ### lsvc (linear support vector classifier):

# In[ ]:


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=2)  # create pipelines for lsvc model
print('pipe1')
inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)

# lets name rename inner1 as:
best_lsvc = inner1.best_estimator_
print('\nbest pipeline saved.')


# ### svc (support vector classifier):
# Note that `pipe1` fails because RFE wont work with svc since the svc does not expose "coef_" or "feature_importances_" attributes.

# In[ ]:


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=3)  # create pipelines for svc model

#inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
# this fails, because RFE wont work with svc:
# RuntimeError: The classifier does not expose "coef_" or "feature_importances_" attributes

print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)

# lets name rename inner2 as:
best_svc = inner2.best_estimator_
print('\nbest pipeline saved.')


# ### dtc (decision tree classifier):

# In[ ]:


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=4)  # create pipelines for dtc model
print('pipe1')
inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)
# lets name rename inner1 as:
best_dtc = inner1.best_estimator_
print('\nbest pipeline saved.')


# ### rfc (random forest classifier):

# In[ ]:


pipe1, param_dist1, pipe2, param_dist2 = createPipes(y_train, idx=5)  # create pipelines for rfc model

print('pipe1')
inner1, outer1 = modelSel(pipe1, param_dist1, X_train, y_train)
print('pipe2')
inner2, outer2 = modelSel(pipe2, param_dist2, X_train, y_train)

# lets name rename inner1 as:
best_rfc = inner1.best_estimator_
print('\nbest pipeline saved.')

# clean workspace
del(pipe1, param_dist1, pipe2, param_dist2)
del(inner1,outer1,inner2,outer2)


# # Model blending
# Next we can do model mixing / model blending  
# We have many models now. we can see if combining them could perhaps produce an even stronger classifier!  
# 
# We will be using 2 'Voting Classifiers'
# *     one with Hard Vote or majority rules   (votingC_hard aka VCH)

# In[ ]:


from sklearn.ensemble import VotingClassifier

# hard voting classifier
votingC_hard = VotingClassifier(
        estimators=[('logit', best_logit), ('knc', best_knc), 
                    ('lsvc', best_lsvc), ('svc', best_svc), 
                    ('dtc',best_dtc), ('rfc',best_rfc)],
        voting='hard', n_jobs=glbl['n_jobs'])
#
CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
cv_estimate2 = ms.cross_val_score(votingC_hard, X_train, y_train, cv = CV, n_jobs=glbl['n_jobs'])
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate2):
    print('Fold %2d    %4.3f' % (i+1, x))
print('Mean performance metric = %4.3f' % np.mean(cv_estimate2))
print('stddev of the metric       = %4.3f' % np.std(cv_estimate2))
#


# *     one with Soft Vote or weighted probabilities   (votingC_soft aka VCS)

# In[ ]:


# soft voting classifier
votingC_soft = VotingClassifier(
        estimators=[('logit', best_logit), ('knc', best_knc),
                    #('lsvc', best_lsvc),    # AttributeError: 'LinearSVC' object has no attribute 'predict_proba'. we need to take this out.
                    ('svc', best_svc),      # predict_proba is not available when  probability=False   --- we will set this soon ...
                    ('dtc',best_dtc), ('rfc',best_rfc)],
        voting='soft', n_jobs=glbl['n_jobs'])
#

# for soft vote we need probabilities. the SVC can be set to have probabilities...
# probability : boolean, optional (default=False)
#    Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
# lets see the parameters i can set in VotingC_soft:
votingC_soft.get_params().keys()
# we need to set that 'svc__est__probability' to True
votingC_soft.set_params(svc__est__probability=True)

CV = ms.KFold(n_splits=glbl['n_splits'], shuffle = True, random_state=glbl['random_state'])
cv_estimate3 = ms.cross_val_score(votingC_soft, X_train, y_train, cv = CV)
print('Outcomes by cv fold')
for i, x in enumerate(cv_estimate3):
    print('Fold %2d    %4.3f' % (i+1, x))
print('Mean performance metric = %4.3f' % np.mean(cv_estimate3))
print('stddev of the metric       = %4.3f' % np.std(cv_estimate3))
#

del(cv_estimate2, cv_estimate3, i, x)


# # Evaluation on test set
# 
# Until now we have only used training set `X_train`.  
# Let us now evaluate each model on the entirely unseen `X_test`.   
# 
# First, we fit each model with ENTIRE train-set...  
# Now lets score each model on the test set  
# This is the best estimate of performance we have, without actually submitting to Kaggle   
# 
# Lets visualize the performance of each model:  
# We create a column chart of the scores for each estimator used

# In[ ]:


# First, we fit each model with ENTIRE train-set.
best_logit.fit(X_train, y_train)
best_knc.fit(X_train, y_train)
best_lsvc.fit(X_train, y_train)
best_svc.fit(X_train, y_train)
best_dtc.fit(X_train, y_train)
best_rfc.fit(X_train, y_train)
votingC_hard.fit(X_train, y_train)
votingC_soft.fit(X_train, y_train)


# now lets score each model on the test set

scores=[]
classifiers=pd.DataFrame( ['logit', 'knc', 'lsvc', 'svc', 'dtc', 'rfc', 'vch', 'vcs'] )
classifiers.columns = ['classifier']

scores.append( best_logit.score(X_test, y_test) )
scores.append( best_knc.score(X_test, y_test) )
scores.append( best_lsvc.score(X_test, y_test) )
scores.append( best_svc.score(X_test, y_test) ) 
scores.append( best_dtc.score(X_test, y_test) ) 
scores.append( best_rfc.score(X_test, y_test) )
scores.append( votingC_hard.score(X_test, y_test) ) 
scores.append( votingC_soft.score(X_test, y_test) )

scores = pd.DataFrame( scores )
scores.columns = ['score']

scoresdf = pd.concat([classifiers, scores], axis = 1)   # concatenate columns
# clean workspace
del(classifiers, scores)


# Lets sort the `scoresdf` dataframe
scoresdf.sort_values(by='score', inplace=True)
scoresdf.score = (scoresdf.score*100000).astype(int)/1000   # give each 3 decimal points



# we create a column chart of the scores for each estimator used
fig = plt.figure(figsize=(8,6))   # define (new) plot area
ax = fig.gca()   # define axis
sns.barplot(x = 'score', y='classifier', data = scoresdf, palette="Blues_d", ax=ax)
plt.title('Accuracy Score on the test-set by different classifiers \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Classifier')

for i, v in enumerate( scoresdf.score ):  # give each 3 decimal points
    ax.text(v + 0.5, i + .25, str(v), color='black', fontweight='bold')
# clean workspace
del(i, v, scoresdf)


# From this figure we see that our best classfier was the voting classifier with soft voting. From the non-blended models, the pipeline using the Decision Tree Classifier was the strongest model.
#  
#  # Feature importances
# It is convenient that the Decision Tree Classifier scored so well. The DTC is a highly useful in that it ranks features on importance.  
# Suppose we wanted to know the relative importance of each feature. The DTC and RFC are highly useful in that it ranks features on importance. Lets consider the DTC:  
# Recall that we use RFE with our DTC. This means that the starting features are 'intact' -  they are not in PCA space. Some of them are, however, excluded from the DTC.

# In[ ]:


# lets collect the names of the oringinal features
featureRanks = pd.DataFrame(X_train.columns)
featureRanks.columns=['feature']
# first we need to figure out which features were deleted by the RFE
featureRanks['support'] = best_dtc.named_steps['rfe'].support_
featureRanks['ranking'] = best_dtc.named_steps['rfe'].ranking_

# lets look at the feature importance ranking as per the RFC
featureRanks['importance'] = 0     # initialize
# now set the importance of features that were included
featureRanks.loc[featureRanks.support==True,'importance'] = best_dtc.named_steps['est'].feature_importances_

# lets sort by feature importances
featureRanks.sort_values(by='importance', inplace=True)
featureRanks.importance = (featureRanks.importance*100000).astype(int)/1000   # give each 3 decimal points

# we create a column chart of the importances for each feature
fig = plt.figure(figsize=(8,6))   # define (new) plot area
ax = fig.gca()   # define axis
sns.barplot(x = 'importance', y='feature', data = featureRanks, palette="Blues_d", ax=ax)
plt.title('Importance as per Decision Tree Classifier with RFE \n')
plt.xlabel('Importance [%]')
plt.ylabel('Feature')

for i, v in enumerate( featureRanks.importance ):  # give each 3 decimal points
    ax.text(v + 0.5, i + .25, str(v), color='black', fontweight='bold')

del(i,v, featureRanks)


# We will discuss the feature importance ranking soon...
# 
# # Visualization
# 
# If we think about a potential product for this project, we may suggest a flow-chart that accurately summarizes the probability of a given passenger to survive. The classifier that is designed for such transparency and wonderful interpretability is the DTC (Decision Tree Classifier). Lets biuld a Decision Tree Graph to showcase the decision-making process behind the Decision Tree Classifier:

# In[ ]:


# we will need to name the features used by the DTC

featureRanks2 = pd.DataFrame(X_train.columns)
featureRanks2.columns=['feature']
# first we need to figure out which features were deleted by the RFE
featureRanks2['support'] = best_dtc.named_steps['rfe'].support_
featureRanks2['ranking'] = best_dtc.named_steps['rfe'].ranking_
# now isolate feature names taht were used
names = featureRanks2.loc[featureRanks2.support==True,['feature']]
names = list(names.feature)

from sklearn.base import clone
dtc = clone ( best_dtc.named_steps['est'] )     # copies estimator wihtout pointing to it

dtc.fit(X_train.loc[:,names], y_train)   # refit the classifier with included data only
# note that DTC does not need transforms on X_train, so we can use it directly

import graphviz 

# Create DOT data
dot_data = skl.tree.export_graphviz(dtc, out_file=None,
                                    max_depth = None,
                                    feature_names = names,
                                    class_names = True,     # in ascending numerical order
                                    impurity=False,
                                    proportion=True,
                                    filled = True, rounded = True)
#

# Draw graph
graph = graphviz.Source(dot_data) 
# show graph
graph


# How to interpret:  
# For example consider the first node:
# *  the conditional tells us where to go next
# *  the 'samples' indicate how many samples we have narrowed our search down to. Since this is the first node, we will always have samples=100%
# *  'value' indicate the proportion of the target taht have reached this node. so in this case Survived 0 and 1. In node 1 the proportion is 0.704 and 0.296. this means 70.4% of all passengers died. The ratio in 'value' sets the color intensity of the node.
# *  'class' indicate that at this node, what is the most likely result. in node 1 it is y[0]: Death. This can be derived from 'value'.
# *  if the condition is True, then go left. If the condition is False, then go right.
# 
# 
# What does the graph above tell us?  
# Consider the most important feature: `Title_Mr`.  
# "Women and children first" is a code of conduct, whereby the lives of women and children were to be saved first in a life-threatening situation, typically abandoning ship, when survival resources such as lifeboats were limited.  
# This saying was in-fact made famous by the 1997 film 'Titanic'. From our examination, it seems that this saying had quite a lot of weight in reality!  
# The most important factor in the survival of a given passenger was whether their title (barring royal, academic, and naval titles) was "Mr". We can clearly see from the Decision Tree above that if you were a 'Mr' (go right on the first node) then your prospects are dim at best... the colour of the right half of the Decision Tree is red compared to the left half.
# 
# The 2 next most important features are`Fare` and `Pclass`. For the majority of third-class passengers, the overwhelming problem they faced after the collision was navigating the labyrinth of passageways, staircases and public rooms to get up to the boat deck, which was mostly in the first-class area of the ship. There was no single staircase leading all the way up through the six to eight decks they would have to traverse to get near the lifeboats, and no handy maps of the ship they could use. Even the staff had trouble finding their way around.  
# There is a myth that 3rd class passengers were even locked out of the boat deck - a fabrication by 1997's Titanic - however 3rd class passengers who survived declared these rumours to be false.

# # Kaggle submission
# Our best model was the Voting Classifier with Soft voting (VCS). We will submit these results to Kaggle to score our method.  
# Lets take a sample to see what our data looks  like

# In[ ]:


# First, clean workspace
del(featureRanks2, dot_data, names)

# we dont need the train-set or test-set data at all anymore
del(X_train, y_train, X_test, y_test)


# we now use the testing data from Kaggle, which we have named 'X2'

y2 = data2.Survived
X2 = data2.drop(columns=['Survived'])
del(data2)

# we use our best classifier: the VCS
y_pred = votingC_soft.predict(X2)     # make the predictions based on X2

# now lets make the submission

submit = pd.DataFrame(y2)  # just quickly create a new dataFrame called submit
submit['PassengerId'] = X2.index
submit['Survived'] = y_pred

# lets reorder our columns... just to make it look nice
submit = submit[['PassengerId','Survived']]

# lets take a look at what our data looks  like
submit.head(5)


# Recall that survived needs to be numeric 0 or 1  
# Lets fix that and look at the data again:

# In[ ]:


submit['Survived'] = submit.Survived.astype('uint8')
# lets take a look at what our data looks  like
submit.head(5)


# Looks correct now   
# 
# Finally we can output the results. The results will be saved under **Output**  
# Scroll to the top of this notebook to see the headings:
# Notebook, Code, Data, **Output**, Comments, Log, Versions, Fork

# In[ ]:


#submit file
submit.to_csv("../working/submit.csv", index=False)

print("Submitted to 'Output'")


# # Final comments
# My intention with this project was to *tell a story using data*.  
# I hope that this kernel was insightful and enjoyable!  
# 
# I have some 'TODO's in mind:
# * Implement parallel computing
# * Try different Dimensionality Reduction methods. For example: isomap
# * Try different classifiers. Gradient Boosting methods are popular on Kaggle, especially the XGB implementation.
# * I am currently only performing `n_iter` = 100 iterations in my cross-validated search. i would like to extend this to be more thorough.
# * Right now I am using RandomizedSearch. Instead, I would love to implement an optimization algorithm like Bayesian optimization / Gradient-based optimization / Evolutionary algorithm (Genetic algorithm). This would have the benefit of intelligently guiding the hyper-parameter search and implemting a search-stop criterion.
# * The `weights` parameter in the Voting Classifiers can be set. In fact, I wish to optimize the weights. This will exclude classfiers that hurt the Voting Classfier, and promote those that contribute more to its predictive power.
# 
# I welcome comments and suggestions for improvement!
