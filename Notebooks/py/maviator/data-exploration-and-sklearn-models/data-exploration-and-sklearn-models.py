#!/usr/bin/env python
# coding: utf-8

# # Kaggle Titanic Competition

# ### Data exploration of the Kaggle Titanic competition
# ### Data set can be found at https://www.kaggle.com/c/titanic/data

# In[ ]:


# Handle table-like data and matrices
import numpy as np
import pandas as pd

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
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# get titanic & test csv files as a DataFrame
train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")
full = train.append(test, ignore_index=True)
print (train.shape, test.shape, full.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


full = train.append( test , ignore_index = True )
full.shape


# In[ ]:


train.describe()


# In[ ]:


survived_sex = train[train['Survived']==1]['Sex'].value_counts()
dead_sex = train[train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))


# ## Plotting survival against Fare

# In[ ]:


figure = plt.figure(figsize=(15,8))
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# In[ ]:


test = train[:10]
test


# In[ ]:


test.groupby('Survived').mean()


# ## Average Fare for survival

# In[ ]:


train.groupby('Survived').mean()


# ## Average Fare per class

# In[ ]:


train.groupby('Pclass').mean()


# ## Survival rate per sex

# In[ ]:


train.groupby('Sex').mean()


# ## Survival rate per age

# In[ ]:


train.groupby('Age').mean()


# In[ ]:


plt.plot(train.groupby('Age').mean()['Survived'])


# ## Grouping by SibSp

# In[ ]:


train.groupby('SibSp').mean()


# ## Grouping by Parch

# In[ ]:


train.groupby('Parch').mean()


# ## Transforming Categorical values to binary

# In[ ]:


# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
sex.head()


# In[ ]:


# Create a new variable for every unique value of Embarked
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
embarked.head()


# In[ ]:


# Create a new variable for every unique value of Embarked
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
pclass.head()


# ## Filling missing Age and Fare values

# In[ ]:


# Create dataset
imputed = pd.DataFrame()

# Fill missing values of Age with the average of Age (mean)
imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )

# Fill missing values of Fare with the average of Fare (mean)
imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )

imputed.head()


# ## Adding new Feature : Family Size

# In[ ]:


# Adding a new feature: family size

family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
family.head()


# ## Assembling data set for modeling

# In[ ]:


full_X = pd.concat( [ imputed , embarked , family, sex ] , axis=1 )
full_X.head()


# In[ ]:


# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = train.Survived
test_X = full_X[ 891: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)


# ## Modelling

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
#model = SVC()
#model = GradientBoostingClassifier()
#model = KNeighborsClassifier(n_neighbors = 3)
#model = GaussianNB()
#model = LogisticRegression()


# In[ ]:


model.fit( train_X , train_y )


# In[ ]:


# Score the model
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))


# ## Deployment

# In[ ]:


test_Y = model.predict( test_X )
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )

