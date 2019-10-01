#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import AdaBoostClassifier
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
    
    # get titanic & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]

del train , test

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)

# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )


# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )



# Create dataset
age = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)
age[ 'Age' ] = full.Age.fillna( full.Age.mean() )

fare = pd.DataFrame() 

# Fill missing values of Fare with the average of Fare (mean)
fare[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()

# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )


# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket

full_X = pd.concat( [ age,fare, sex,pclass,family[ 'FamilySize' ],embarked,ticket] , axis=1 )
full_X.head()
# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .14 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)



model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )
scores = cross_val_score(model, train_valid_X ,  train_valid_y, cv=10)
print("rfm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'random_all_feat.csv' , index = False )

model = LogisticRegression()
model.fit( train_X , train_y )
scores = cross_val_score(model, train_valid_X ,  train_valid_y, cv=10)
print("log Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'logis_all_feat.csv' , index = False )

model = SVC()
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_rbf_all_feat.csv' , index = False )


model = SVC(kernel='linear')
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_lin_all_feat.csv' , index = False )


model = SVC(kernel='poly')
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_poly_all_feat.csv' , index = False )

model = SVC(kernel='sigmoid')
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_sigmo_all_feat.csv' , index = False )

full_X = pd.concat( [ age,fare, sex,pclass,family[ 'FamilySize' ],embarked ] , axis=1 )
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .14 )

##feature importance
plot_variable_importance(train_X, train_y)

full_X = pd.concat( [ age,fare, sex] , axis=1 )
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .14 )


##trying only most important


model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'random_age_fare_sex_feat.csv' , index = False )

model = LogisticRegression()
model.fit( train_X , train_y )


test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'logis_age_fare_sex_feat.csv' , index = False )

model = SVC(kernel='linear')
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_lin_age_fare_sex_feat.csv' , index = False )

full_X = pd.concat( [ age,fare, sex,pclass,family[ 'FamilySize' ],embarked ] , axis=1 )
full_X.head()
# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )
plot_model_var_imp(model, train_X, train_y)

model = SVC(kernel='linear')
model.fit( train_X , train_y )

clf = Pipeline([
  ('feature_selection', SelectFromModel(SVC(kernel='linear'))),
  ('classification', SVC(kernel='linear'))
])
clf.fit(train_X , train_y)

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_lin_selction.csv' , index = False )
scores = cross_val_score(model, train_valid_X ,  train_valid_y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )
plot_categories( titanic , cat = 'Sex' , target = 'Survived' )
plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )
plot_categories( titanic , cat = 'SibSp' , target = 'Survived' )
plot_categories( titanic , cat = 'Parch' , target = 'Survived' )


title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )


cabin = pd.DataFrame()


# dummy encoding ...
cabin = pd.get_dummies( full['Cabin'] , prefix = 'Cabin' )

full_X = pd.concat( [age,fare,pclass, embarked , cabin , sex,ticket ] , axis=1 )
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )
model = SVC(kernel='linear')
model.fit( train_X , train_y )

clf = Pipeline([
  ('feature_selection', SelectFromModel(SVC(kernel='linear'))),
  ('classification', SVC(kernel='linear'))
])
clf.fit(train_X , train_y)


model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'svm_lin_selction.csv' , index = False )


full_X = pd.concat( [ age,fare, sex,pclass,family[ 'FamilySize' ],embarked,ticket,cabin] , axis=1 )
full_X.head()
# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

model = RandomForestClassifier(n_estimators=10)
model.fit( train_X , train_y )

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'rfm10.csv' , index = False )


