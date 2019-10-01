#!/usr/bin/env python
# coding: utf-8

# **Goal of this Notebook**
# 
# In this notebook, our goal is to predict the survival of a passenger from the sinking of Titanic.
# For that first, we will import the python libraries necessary for the data analysis and visualizing. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# After importing the libraries, we load the datasets available to us and there are 2 datasets given to us training data and test data. we'll be using training data for building the predictive model and test data to evaluate it.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head() #to get the sneak peak of the data


# Lets have look at the information about the variables in training data

# In[ ]:


train.describe()


# we observe that for the 'Age' attribute the count is just 714 out of 819 and most machine learning alghorims require all variables to have values in order to use it for training the model. The simplest method is to fill missing values with the average of the variable across all observations in the training set, but we can fill in the missing values based on th mean of the Passenger class.

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass',y='Age',data = train,palette = 'winter')


# From the above visualization, we can observe the mean ages for the different classes like for class 1 the mean age is around 37 etc...to get the exact average age for each class we use pandas 

# In[ ]:


train.groupby('Pclass').mean()['Age']


# Now we know the average age for the three classes and try to fill in these values for the missing data

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# Now we'll apply this function for the age column

# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Lets  check the train data again

# In[ ]:


train.describe()


# Also observing the training data, we can figure out that features Fare and Cabin interrelated in such a way that one's whose fare is high had a cabin allocation and null values for low fare, with these many null values it doestn't work well with our analysis.
# 
# The feature ticket also have null values and it doestn't have much importance in the analysis. To handle this we'll drop these two features from our dataset.

# In[ ]:


train = train.drop(['Ticket','Cabin'],axis = 1)
train.head()


# Now lets visualize the survival of data based on Sex, Age, Fare and Embarked

# In[ ]:


survived_sex = train[train['Survived']==1]['Sex'].value_counts()
dead_sex = train[train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))


# From the data above, it's obvious that female are more likely to survive 

# In[ ]:


figure = plt.figure(figsize=(13,8))
plt.hist([train[train['Survived']==1]['Age'],train[train['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# From the graph, passengers of age less than 10 and older people like age 79 or 80 are more likely to survive than the remaining passengers between age 10 to 65.

# In[ ]:


figure = plt.figure(figsize=(13,8))
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# Passengers with higher priced tickets are more likely to survive than with the lower priced tickets.

# In[ ]:


survived_embarked = train[train['Survived']==1]['Embarked'].value_counts()
dead_embarked = train[train['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarked,dead_embarked])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(13,8))


# Passengers from Southampton seems to survive the most, but comparing the number of passengers boarded the titanic passengers from Cherbourg are most likely to survive.
# 
# let's visualise the data based on 'Age' and 'Fare' together

# In[ ]:


plt.figure(figsize=(13,8))
ax = plt.subplot()
ax.scatter(train[train['Survived']==1]['Age'],train[train['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(train[train['Survived']==0]['Age'],train[train['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)


# From the plot, we can observe that passengers of age between 15 to 50 are most likely not to survive.
# 
# Now we will merge both train data and test data, but before we'll observe the test data.

# In[ ]:


test.describe()


# From the test data, we observe that some of the values for features Age and Fare are null values and we fill with the null values with the mean values.

# In[ ]:


test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)


# In[ ]:


test.describe()


# For feature engineering, we'll combine the training and test data because the process for feature selection on training data has to be done later while testing on test data. For that we make sure that both datasets have the same features.

# In[ ]:


# extracting and then removing the targets from the training data
targets = train.Survived
train.drop('Survived',1,inplace=True)

#merging train data and test data for future feature engineering
titanic = train.append(test)
titanic.reset_index(inplace=True)
titanic.drop('index',inplace=True,axis=1)


# In[ ]:


titanic.head()


# In[ ]:


titanic.describe()


# The simplest method for feature engineering is to create seperate dataframes for each features necessary for feature selection and later combine all the generated dataframes. We'll start with generating dataframe for features 'Age' and 'Fare' and continue with 'Embarked', 'Name', 'Pclass', 'Sex', 'FamilySize'

# In[ ]:


age_fare = pd.DataFrame()

age_fare['Age'] = titanic['Age']
age_fare['Fare'] = titanic['Fare']

age_fare.head()


# In[ ]:


embarked = pd.get_dummies( titanic.Embarked , prefix='Embarked' )
embarked.head()


# Titles before the Passengers Names reflect social status and may predict survival probability. So we will extract titles from their names.
# 
# 

# In[ ]:


title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = titanic[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

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
title.head()


# In[ ]:


PClass = pd.get_dummies(titanic.Pclass, prefix = 'Pclass')
PClass.head()


# Convert the Sex Atrribute into numerical values where 1 represents 'male' and 0 represents 'female'

# In[ ]:


sex = pd.Series( np.where( titanic.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
sex.head()


# We'll create new feature 'FamilySize', the size of the family is obtained by adding two attributes Parch and SibSp, because family size might also considered in saving the passengers from the titanic.

# In[ ]:


family = pd.DataFrame()

# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = titanic[ 'Parch' ] + titanic[ 'SibSp' ] + 1

# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )

family.head()


# Now we will combine all the data frames we created 

# In[ ]:


combined = pd.concat( [ age_fare, title, PClass, sex, embarked, family] , axis=1 )
combined.head()


# we use our knowledge of the passengers based on the features we created and then build a statistical model. For that we will first import the libraries we need

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier


# Divide the data for training and testing purposes.

# In[ ]:


titanic_train = combined[:891]
titanic_test = combined[891:]
titanic_test.describe()


# Tree-based estimators can be used to compute feature importances, which in turn can be used to discard irrelevant features.

# In[ ]:


clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(titanic_train, targets)


# In[ ]:


features = pd.DataFrame()
features['feature'] = titanic_train.columns
features['importance'] = clf.feature_importances_


# Let's have a look at the importance of each feature.

# In[ ]:


features.sort_values(['importance'],ascending=False)


# We'll define a function ML_Algol to save the time, because the pattern is same for every ML Algorithm.

# In[ ]:


def ML_Algol(x):
    algorithm = x
    algorithm.fit(titanic_train,targets)
    print (algorithm.score( titanic_train , targets ))


# select which model you want to run to run the algorithm

# In[ ]:


model = RandomForestClassifier(n_estimators=100)
ML_Algol(model)


# In[ ]:


model = SVC()
ML_Algol(model)


# In[ ]:


model = KNeighborsClassifier(n_neighbors = 3)
ML_Algol(model)


# In[ ]:


model = GaussianNB()
ML_Algol(model)


# In[ ]:


model = LogisticRegression()
ML_Algol(model)


# generate the output

# In[ ]:


test_Y = model.predict(titanic_test)
passenger_id = titanic[891:].PassengerId
test_new = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test_new.shape
test_new.head()
test_new.to_csv( 'titanic_pred.csv' , index = False )


# In[ ]:




