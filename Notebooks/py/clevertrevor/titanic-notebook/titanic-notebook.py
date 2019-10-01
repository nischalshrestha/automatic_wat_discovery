#!/usr/bin/env python
# coding: utf-8

# **Stage One: Business Understanding**
# 
# 1.1 Objective: To predict survival on the Titanic
# 
# This notebook has been derived from https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial
# It has been adapted(shortened) for personal learning and improved simplicity.
# Stages have been created using https://www.sv-europe.com/crisp-dm-methodology/

# **Stage Two: Data Understanding**
# Loading data into the report

# In[ ]:


# warnings
import warnings
warnings.filterwarnings('ignore')

# data handling
import numpy as np
import pandas as pd

# modelling algo
from sklearn.svm import SVC, LinearSVC

# modelling helpers
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().magic(u'matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

train_data = pd.read_csv('../input/train.csv')


# **Load Data**

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
full = train.append(test, ignore_index=True)
titanic = full[ :891 ]
del train, test


# **Helper functions**

# In[ ]:


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
    corr = titanic.corr()
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


# **Analyze Data**
# 
# Variable Descriptions:
#  - Survived: Survived (1) or died (0)
#  - Pclass: Passenger's class
#  - Name: Passenger's name
#  - Sex: Passenger's sex
#  - Age: Passenger's age
#  - SibSp: Number of siblings/spouses aboard
#  - Parch: Number of parents/children aboard
#  - Ticket: Ticket number
#  - Fare: Fare
#  - Cabin: Cabin
#  - Embarked: Port of embarkation

# In[ ]:


titanic.head()


# In[ ]:


titanic.describe()


# **Section 3: Data Preparation**

# In[ ]:


sex = pd.Series( np.where( full.Sex == 'male', 1, 0), name = 'Sex')
embarked = pd.get_dummies( full.Embarked, prefix='Embarked')
pclass = pd.get_dummies( full.Pclass, prefix='Pclass')

# cols with some derived values
imputed = pd.DataFrame()
imputed['Age'] = full.Age.fillna( full.Age.mean() )
imputed['Fare'] = full.Fare.fillna( full.Fare.median() )


# **Variable Selection**

# In[ ]:


full_X = pd.concat( [sex, embarked, pclass, imputed], axis=1)
full_X.head()


# **Create Datasets**

# In[ ]:


train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X, valid_X, train_y, valid_y = train_test_split( train_valid_X, train_valid_y, train_size=0.7 )


# **Stage 4: Model Selection and Fitting**

# In[ ]:


model = LogisticRegression()
model.fit(train_X, train_y)
print( model.score( train_X, train_y), model.score( valid_X, valid_y))


# **Stage 5: Evaluation**
# This simple model does quite well on the data set. It has a relatively high accuracy with only slight overfitting. Model parameters could be tweaked and features improved to amend the overfitting.

# In[ ]:


# deployment
test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
global me
me = "Trevor"
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )

