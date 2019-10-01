#!/usr/bin/env python
# coding: utf-8

# ## This is a practice to solve problems using notebook
# 
# this nb is based on a set of previous kernels, list as follow:
# 
# + [/sachinkulkarni/titanic/an-interactive-data-science-tutorial](/sachinkulkarni/titanic/an-interactive-data-science-tutorial)
# + [/mariammohamed/titanic/training-different-models](/mariammohamed/titanic/training-different-models)
# + [/shivendra91/titanic/rolling-in-the-deep](/shivendra91/titanic/rolling-in-the-deep)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
# from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV


# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# Helpers for plot
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["head", "../input/train.csv"]).decode("utf8"))
print(check_output(["head", "../input/test.csv"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
# test data
test_data = pd.read_csv('../input/test.csv')


# let's have a look at the dataset
#drop unnecessary columns
#train_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True, axis=1)
train_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True, axis=1)
test_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True, axis=1)

# type of data structure
# http://pandas.pydata.org/pandas-docs/stable/dsintro.html


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


plt.hist(train_data['Pclass'], color='lightblue')
plt.tick_params(top='off', bottom='on', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0, 4])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.set_xticks([1, 2, 3])
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()


# In[ ]:


plot_correlation_map( train_data )


# In[ ]:


plot_distribution( train_data , var = 'Age' , target = 'Survived' , row = 'Sex' )


# In[ ]:


plot_categories( train_data , cat = 'Pclass' , target = 'Survived' )


# In[ ]:


plot_categories( train_data , cat = 'Sex' , target = 'Survived' )


# In[ ]:


plot_categories( train_data , cat = 'Age' , target = 'Survived' )


# In[ ]:


plot_categories( train_data , cat = 'SibSp' , target = 'Survived' )


# In[ ]:


plot_categories( train_data , cat = 'Parch' , target = 'Survived' )


# In[ ]:


plot_categories( train_data , cat = 'Fare' , target = 'Survived' )


# In[ ]:


plot_categories( train_data , cat = 'Embarked' , target = 'Survived' )


# In[ ]:


# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( train_data.Sex == 'male' , 1 , 0 ) , name = 'Sex' )


# In[ ]:


# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( train_data.Embarked , prefix='Embarked' )
embarked.head()


# In[ ]:


# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( train_data.Pclass , prefix='Pclass' )
pclass.head()

