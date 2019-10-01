#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# The main objective of this data exploration is to predict the survival or the death of a given passenger based on a set of variables such as age and gender.

# # 2. Data Exploration
# 
# # 2.1  Import Important Libraries
# 
#  We need to import python libraries with all the functionality that we will need.

# In[ ]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')

#Import Important Libraries
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_columns = 100

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

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
pd.options.display.max_rows = 100
# Input data files are available in the "../input/" directory.
# The script below lists all  the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # 2.2 Helper functions setup

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
    


# # Available Data
# 
# There are two datasets available; a training set and a test set. We build predictive model using the training data set, and evaluate our model using the test dataset .
# 

# # 2.3  Load training and test data sets.

# In[ ]:


# get titanic & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]

del train , test

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)


# # 2.4 Statistics and visualisations
# 
# For better understanding of our data,we consider some important facts about various variables including their relationship with the target variable. In this case, our target variable is survival.
# 
# Let's begin by having a look at our data. Pandas allows us to have a sneak peak at our data.

# In[ ]:




titanic.head()


# # 2.4.1  Check some key information about the variables
# 
# Survived column is our target variable. The value 1 indicates that the passenger survived and 0 means that the passenger died.
# 
# 
# 
# Other descriptive variables are:
#  1. Age
#  2. List item
#  3. Sex
#  4. PassengerId: and id given to each traveler on the boat
#  5. Pclass: the passenger class. It has three possible values: 1,2,3
#  6. Name
#  7. SibSp: number of siblings and spouses traveling with the passenger
#  8. Parch: number of parents and children traveling with the passenger
#  9. The ticket number
#  10. The ticket Fare
#  11. The cabin number
#  12. The embarkation. Which has three possible values S,C,Q

# In[ ]:


titanic.describe()


# # Check if there are null values in Age column. 

# In[ ]:


sum(pd.isnull(titanic['Age']))


# We find that there are 177 values missing the Age column.This needs to be fixed in order to avoid errors later
# 
# To fix this, we  replace the null values with the median age which is more robust to outliers than the mean.

# In[ ]:


titanic['Age'].fillna(titanic['Age'].median(), inplace=True)


# # 2.4.2 Plot a heatmap 
# 
# This will help us to determine the most important variables in our data.

# In[ ]:


plot_correlation_map( titanic )


# # 2.4.3  Further Analysis
# 
# Explore the relationship between features and survival of passengers.
# 
# Check the relationship between age and survival.

# In[ ]:



# Plot distributions of Age of passangers who survived or did not survive
plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )


# # Visualize survival based on the gender.

# In[ ]:


survived_sex = titanic[titanic['Survived']==1]['Sex'].value_counts()
dead_sex = titanic[titanic['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))


# Sex variable seems to be an important feature. More women were more likely to survive than men.

# # Correlate the survival with the age variable.

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([titanic[titanic['Survived']==1]['Age'],titanic[titanic['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# We can see that passengers who were less than 10 were more likely to survive than older passengers of ages between 12 and  50. 

# # Investigate numeric variables
# 
# Plot the distributions of Fare of passengers who survived or did not survive.This could be a good predictive variable.

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([titanic[titanic['Survived']==1]['Fare'],titanic[titanic['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# We can observe that Passengers with cheaper ticket fares were more likely to die. In other words , passengers with more expensive tickets, seemed to have been rescued first.

# # Combine Age, Fare, and Survival variables in a single chart.

# In[ ]:


plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(titanic[titanic['Survived']==1]['Age'],titanic[titanic['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(titanic[titanic['Survived']==0]['Age'],titanic[titanic['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


# A distinct cluster of dead passengers (the red one) appears on the chart. Those people are adults (aged between 15 and 50) of lower class (lowest ticket fares).
# In fact, the ticket fare correlates with the class as we see it in the chart below.

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
titanic.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)


# Now let's  see how the embarkation sites affected survival.

# In[ ]:


survived_embark = titanic[titanic['Survived']==1]['Embarked'].value_counts()
dead_embark = titanic[titanic['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))


# We can observe that there is no distinct correlation between embarkation and survival 

# # Investigating categorical variables

# In[ ]:


# Plot distributions of Fare of passangers who survived or did not survive
plot_categories( titanic , cat = 'Fare' , target = 'Survived' )


# In[ ]:


# Plot survival rate by Embarked
plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )


# In[ ]:


# Plot survival rate by Sex
plot_categories( titanic , cat = 'Sex' , target = 'Survived' )


# In[ ]:


# Plot survival rate by Pclass
plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )


# In[ ]:



# Plot survival rate by SibSp
plot_categories( titanic , cat = 'SibSp' , target = 'Survived' )


# In[ ]:


# Plot survival rate by Parch
plot_categories( titanic , cat = 'Parch' , target = 'Survived' )


# # 3. Feature engineering
# 
# From previous observations, we noticed some interesting correlations between variables. However, we could not analyze more some features like the names or the tickets because these features requires further processing. In the next section we will transform these specific features so that they can easily fed into machine learning algorithms.

# # 3.1  Transform Categorical variables into numeric variables

#  Transform Categorical variables into numeric variables
# 
# But first, let's define a print function that asserts whether or not a feature has been processed.

# In[ ]:


# Define a function that check if a feature has been processed or not
def status(feature):
    print ('Processing',feature,': ok')


# # 3.2 Combine training and test data sets
# It is always advisable to combine the training data set and the test data sets. This is particularly useful especially if your test data set appears to have a feature that doesn't exist in the training set. Therefore, if we don't combine the two sets, testing our model on the test set will fail.

# In[ ]:


def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined


# In[ ]:


combined = get_combined_data()


# In[ ]:


combined.shape


# # 3.3 Extracting the passenger titles
# 
# The names variable has some additional information that can help us determine the social status of a passenger. For example, a name with a title such as “Peter, Master. Michael J” can tell us that the passenger is a master. We therefore need to we introduce additional information about the social status of a passenger by simply parsing the name and extracting its title.

# In[ ]:



# Function to parse and exract titles from passanger names
def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
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
    combined['Title'] = combined.Title.map(Title_Dictionary)


# The above function parses the names and extracts titles from passenger names. It then maps the extracted titles to categories of titles we selected : Officer,Royalty,Mr,Mrs,Miss, and Master.

# In[ ]:


# Check the new titles feature
get_titles()
combined.head()


# # 3.4 Processing Age
# 
# We noticed earlier that Age variable was missing 177 values. We fixed this issue by replacing the missing values with the median age. This is not be the best solution because age may differ between groups and categories of passengers.
# 
# To illustrate this, let's group our data by sex, Title, and passenger class and for each subset and compute the median age.

# In[ ]:


grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()


# From the above data, it is clear that the median age in the Age column is different based on the Sex, Pclass, and Title put together.
# 
# For example, if the passenger is female, from Pclass 1, with royalty title, the median age is 39.Whereas if the passenger is male, from Pclass 3, with a title Mr., the median age is 26.
# 
# We therefore create a function that fills in the missing age in the combined data set based on the different attributes of the passengers.

# In[ ]:


def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
    status('age')


# In[ ]:


process_age()


# In[ ]:


combined.info()


# We can see that the missing ages have been replaced. However, we notice a missing value in Fare, two missing values in Embarked and a lot of missing values in Cabin. We'll come back to these variables later.
# 
# Let's now process the names.

# In[ ]:


def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
    
    status('names')


# The above function drops the Name column since we won't be using it anymore because we created a Title column.
# 
# It then we encodes the title values using a dummy encoding

# In[ ]:


process_names()


# In[ ]:


combined.head()


# # 3.5 Processing Fare

# In[ ]:


#This function  replaces one missing Fare value by the mean
def process_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
    
    status('fare')


# In[ ]:


process_fares()


# # 3.6 Processing Embarked

# In[ ]:


# This functions replaces the two missing values of Embarked with the most frequent Embarked value.
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
    
    status('embarked')


# In[ ]:


process_embarked()


# # 3.7 Processing Cabin

# In[ ]:


# This function replaces NaN values with U (for Unknow). 
# It then maps each Cabin value to the first letter. 
#Then it encodes the cabin values using dummy encoding .
def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
    
    status('cabin')


# In[ ]:


process_cabin()


# In[ ]:


combined.info()


# We can see that we don't have a ny missing values now.

# In[ ]:


combined.head()


# # 3.8 Processing Sex

# In[ ]:


#This function maps the string values male and female to 1 and 0 respectively.
def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
    
    status('sex')


# In[ ]:


process_sex()


# # 3.9 Processing Pclass

# In[ ]:


# This function encodes the values of Pclass (1,2,3) using a dummy encoding.
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    combined.drop('Pclass',axis=1,inplace=True)
    
    status('pclass')


# In[ ]:


process_pclass()


# # 3.10 Processing Ticket

# In[ ]:


def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket=''
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))       
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)

    status('ticket')


# In[ ]:


process_ticket()


# # 3.11 Processing Family
# This part includes creating new variables based on the size of the family.
# 
# We create these variable with the assumption that large families are grouped together, hence they are more likely to get rescued than people traveling alone.

# The above function introduces 4 new features;
# 
#  1. FamilySize : the total number of relatives including the passenger (him/her)self.
#  2. Sigleton : a boolean variable that describes families of size = 1
#  3. SmallFamily : a boolean variable that describes families of 2 <= size <= 4
#  4. LargeFamily : a boolean variable that describes families of 5 < size

# In[ ]:


def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)


# In[ ]:


process_family()


# In[ ]:


combined.shape


# We end up with a total of 68 features.

# In[ ]:


combined.head()


# As you can see, the features range in different intervals. Let's normalize all of them in the unit interval. All of them except the PassengerId that we'll need for the submission.

# In[ ]:


def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print ('Features scaled successfully !')


# In[ ]:


scale_all_features()


# # 4. Modeling
# 
# In this part, we use our knowledge of the passengers based on the features we created and then build a statistical model. You can think of this model as a black box that crunches the information of any new passenger and decides whether or not he survives.
# There is a wide range of models to use, from logistic regression to decision trees and more sophisticated ones such as random forests and gradient boosted trees.
# 
# We'll be using Random Forests because ensemble methods work well with most machine learning problem. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set.
# 
# Back to our problem, we now have to:
# 
#  1. Break the combined dataset in train set and test set.
#  2. Use the train set to build a predictive model.
#  3. Evaluate the model using the train set.
#  4. Test the model using the test set and generate and output file for the submission.
# 
# Let's start by importing the useful libraries.
# 

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score


# To evaluate our model we'll be using a 5-fold cross validation with the Accuracy metric.
# 
# To do that, we'll define a small scoring function.

# In[ ]:


def compute_score(clf, X, y,scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5,scoring=scoring)
    return np.mean(xval)


# Recover the train set and the test set from the combined dataset.

# In[ ]:


def recover_train_test_target():
    global combined
    
    train0 = pd.read_csv('../input/train.csv')
    
    targets = train0.Survived
    train = combined.ix[0:890]
    test = combined.ix[891:]
    
    return train,test,targets


# In[ ]:


train,test,targets = recover_train_test_target()


# # 5.0 Feature selection
# 
# We have 68 features so far. This number is quite large.
# 
# When feature engineering is done, we usually tend to decrease the dimensionality by selecting the "right" number of features that capture the essential.
# 
# Feature selection comes with many benefits:
# 
#  1. It decreases redundancy among the data
#  2. It speeds up the training process
#  3. It reduces overfitting
# 
# Tree-based estimators can be used to compute feature importance, which in turn can be used to discard irrelevant features.

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, targets)


# # 5.1 Check the importance of each feature.

# In[ ]:


features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_


# In[ ]:


features.sort(['importance'],ascending=False)


# As you you can see, there is a great importance linked to Title_Mr, Age, Fare, and Sex.
# 
# There is also an important correlation with the Passenger_Id.
# 
# Let's now transform our train set and test set in a more compact datasets.

# In[ ]:


model = SelectFromModel(clf, prefit=True)
train_new = model.transform(train)
train_new.shape


# In[ ]:



test_new = model.transform(test)
test_new.shape


# We have now reduced our features to 8 features.

# # 6.0  Hyperparameters tuning
# 

# In[ ]:



forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# # 7.0 Generate an output file to submit on Kaggle.

# In[ ]:


output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('titanic_pred.csv',index=False)

