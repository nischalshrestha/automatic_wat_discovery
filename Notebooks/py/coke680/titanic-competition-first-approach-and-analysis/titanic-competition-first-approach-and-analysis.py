#!/usr/bin/env python
# coding: utf-8

# # Titanic competition: First approach and analysis
# 
# ### ** Hi! This is my first competition (kernel) and i'm just a beginner in Data Science field. Any questions or comments are welcome!**

# In[ ]:


#useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#ML algorithms
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})

# Any results you write to the current directory are saved as output


# Now, we gonna read the file and see the dataset.

# In[ ]:


train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

full = train_df.append( test_df , ignore_index = True )
titanic = full[ :891 ]

train_df.head()


# Number of rows

# In[ ]:


len(train_df)


# ### **Working on data**

# Watching null values 

# In[ ]:


train_df.isnull().sum().sort_values(ascending=False)


# Cabin and age have severals null values. We have to modify our dataset and try to input our missing data.

# In[ ]:


train_df.dtypes


# As a good praxis, we gonna input the mean in 'Age' (cause it's a numerical variable) in the null values.

# In[ ]:


train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df.isnull().sum().sort_values(ascending=False)


# 'Cabin' is a variable that have almost all values on null. I decide discard it and just work with the other variables.

# In[ ]:


train_df = train_df.drop(['Cabin'], axis=1)
train_df.head()


# Watching the observations on 'Embarked'

# In[ ]:


train_df.Embarked.unique()


# Some values are null. Now, we have to count the number of observations to get the data moda.

# In[ ]:


train_df['Embarked'].value_counts()


# Inserting new data (mode, cause it's a categorical data).

# In[ ]:


train_df['Embarked'].fillna('S', inplace=True)
train_df['Embarked'].value_counts()


# And now, our dataset is ready to explore

# ## **First analysis**

# In[ ]:


train_df.describe()


# Let's see the target distribution

# In[ ]:


print('Distribution of target variable\n',train_df['Survived'].value_counts())
ax = sns.countplot(train_df['Survived'],palette="viridis")

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/len(train_df)), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text


#  61,6% of the people don't survived.
#  
#  Let's see the correlations in variables.

# In[ ]:


mean_correlation = train_df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(mean_correlation,vmax=1,square=True,annot=True,cmap='Oranges')


# 'Parch' and 'SibSp' have a good correlation. 'Fare' and 'Pclass' have a correlation but it's negative. It means that at a higher fare, the class is lower (in this case,  first class).

# In[ ]:


sns.barplot(x='Pclass', y='Fare', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Class and Fare')


# In average, we found the higher fare in the first class and more important, the people who survive belong mostly to the first class.

# Fare have a good correlation with Survived. It means, if you have money, you can survive D:
# 
# Let's see this in a bar plot:

# In[ ]:


sns.barplot(x='Survived', y='Fare', data = train_df, palette = 'viridis')
plt.title('Relationship between Fare and be alive')


# Let's see with the age.

# In[ ]:


sns.barplot( x='Survived', y='Age', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Age (mean)')


# Nothing relevant.

# In[ ]:


sns.countplot( x='Sex', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Sex')


# Most of the men did not survive, while the majority of the women did. Even more:

# In[ ]:


def people(passenger):
    age, sex = passenger
    return 'child' if age < 15 else sex
    
train_df['People'] = train_df[['Age','Sex']].apply(people, axis=1)


# In[ ]:


sns.countplot( x='People', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and People')


# Womens and children first!
# 
# Now, working on percentage.

# In[ ]:


sns.barplot( x='People', y='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and People')


# We see that women and children are more likely to survive than men.

# In[ ]:


sns.countplot( x='Embarked', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Embarked')


# Mostly of people who embarked on Southampton don't survived.

# In[ ]:


sns.barplot( x='Embarked', y='Fare', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Fare and Embarked')


# Ok, this is interesting. People who embarked on Cherbourg, have a higher probability of survive and they paid a higher fare.
# 
# We can infer that people who lives in Cherbourg have a better economic situation (in average), and we can confirm it with the high correlation that exist between 'Pclass' and 'Fare'.

# Now, working on SibSp.

# In[ ]:


sns.countplot( x='SibSp', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived | Siblings and Spouses')


# In[ ]:


sns.countplot( x='Parch', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Sex')


# Now, we gonna combine 'SibSP' and 'Parch' to know the number of family members of each passenger.

# In[ ]:


train_df['family_members'] =  train_df['Parch'] + train_df['SibSp']
sns.countplot( x='family_members', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and and Family members')


# In[ ]:


sns.barplot( x='family_members', y='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Family members')


# We see that passengers who have 3 family members are more likely to survive.

# Now we gonna work on name feature and we look for something interesting.

# In[ ]:


train_df['Lastname'], train_df['Name'] = train_df['Name'].str.split(',', 1).str


# In[ ]:


train_df.head()


# In[ ]:


train_df['Nclass'], train_df['Name'] = train_df['Name'].str.split('.', 1).str
train_df.head()


# In[ ]:


sns.countplot( x='Nclass', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Nclass')


# Mostly of Mr don't survived :(
# 
# Ok, we have a lot of categories and we gonna try to normalize.

# In[ ]:


train_df.Nclass.unique()


# In[ ]:


train_df['Nclass'] = train_df['Nclass'].map({' Jonkheer': 'Other', ' the Countess': 'Other', ' Col': 'Other', ' Rev': 'Other', ' Mlle': 'Mrs', ' Mme': 'Mrs', ' Capt': 'Other', ' Ms': 'Miss', ' Lady': 'Miss', ' Major': 'Other', ' Sir': 'Mr', ' Dr': 'Mr', ' Don': 'Mr', ' Master': 'Mr', ' Miss': 'Miss', ' Mrs': 'Mrs', ' Mr': 'Mr',})


# In[ ]:


train_df.Nclass.unique()


# In[ ]:


sns.countplot( x='Nclass', hue='Survived', data = train_df, palette = 'viridis')
plt.title('Relationship between Survived and Nclass')


# Ok, this bargraph is more clear!

# ## **Preprocessing**

# Let's see our dataset again:

# In[ ]:


train_df.head()


# Now we have a difficult choice:
# 
# First at all, we have numerical and categorical variables.
# 
# We can't categorize 'Name' and 'Ticket' into a group in order to convert all the categorical values into numeric. But we can do it with 'Sex' and 'Embarked'.

# In[ ]:


train_df.Embarked.unique()


# In[ ]:


train_df.Sex.unique()


# So, in 'Embarked' we have:
# 
# 1 = S (Southampton)
# 2 = C (Cherbourg)
# 3 = Q (Queenstown)

# In[ ]:


train_df['Embarked'] = train_df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})


# In 'People':
# 1 = Male, 2 = Female, 3 = Child

# In[ ]:


train_df['People'] = train_df['People'].map({'male': 1, 'female': 2, 'child': 3})
#Now on Sex Feature
train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 2})
#Now on Nclass
train_df['Nclass'] = train_df['Nclass'].map({'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Other': 4})


# In[ ]:


train_df.head()


# Discarting some labels we are ready to put the data in the juicer and play!

# In[ ]:


train_df = train_df.drop(['PassengerId'], axis=1)
train_df = train_df.drop(['Name'], axis=1)
train_df = train_df.drop(['Ticket'], axis=1)
train_df = train_df.drop(['Lastname'], axis=1)
#train_df = train_df.drop(['Sex'], axis=1)
train_df.head()


# ## **Attribute selection**

# Just for fun, we gonna make an attribute selection. The end of this is see the variables that have more weight in our model prediction.
# 
# Let's start with Fisher.

# In[ ]:


# F - Fisher.
target = train_df['Survived']
k = 7  # Number of attributes
train = train_df.drop(['Survived'], axis=1)
atributes = list(train.columns.values)
selected = SelectKBest(f_classif, k=k).fit(train, target)
atrib = selected.get_support()
final = [atributes[i] for i in list(atrib.nonzero()[0])]
final


# In[ ]:


# F - Fisher.
target = train_df['Survived']
k = 6  # Number of attributes
train = train_df.drop(['Survived'], axis=1)
atributes = list(train.columns.values)
selected = SelectKBest(f_classif, k=k).fit(train, target)
atrib = selected.get_support()
final = [atributes[i] for i in list(atrib.nonzero()[0])]
final


# In[ ]:


# F - Fisher.
target = train_df['Survived']
k = 5  # Number of attributes
train = train_df.drop(['Survived'], axis=1)
atributes = list(train.columns.values)
selected = SelectKBest(f_classif, k=k).fit(train, target)
atrib = selected.get_support()
final = [atributes[i] for i in list(atrib.nonzero()[0])]
final


# We see that 'Pclass' and 'Fare' have high weight in our model.
# 
# ### **ExtraTreesClassifier**
# 
# Let's confim it with one more powerful algorithm:

# In[ ]:


# ExtraTrees
model = ExtraTreesClassifier()
era = RFE(model, 7)  # Number of attributes
era = era.fit(train, target)
atrib = era.support_
final = [atributes[i] for i in list(atrib.nonzero()[0])]
final


# In[ ]:


# ExtraTrees
model = ExtraTreesClassifier()
era = RFE(model, 6)  # Number of attributes
era = era.fit(train, target)
atrib = era.support_
final = [atributes[i] for i in list(atrib.nonzero()[0])]
final


# In[ ]:


# ExtraTrees
model = ExtraTreesClassifier()
era = RFE(model, 5)  # Number of attributes
era = era.fit(train, target)
atrib = era.support_
final = [atributes[i] for i in list(atrib.nonzero()[0])]
final


# In this case, 'Pclass' and 'Sex' have a high weight in our model. 
# This is an artesan job. No one have a good answer. You have to try and try again.

# ### **Recursive feature elimination with cross validation and random forest classification**

# Now, we will find how many atributtes do we need for best accuracy

# In[ ]:


X = train_df.drop(['Survived'],axis = 1 )
y = train_df.Survived

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3)


# In[ ]:


rf = RandomForestClassifier() 
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])


# In[ ]:


X = train_df.drop(['Survived','Embarked'],axis = 1 )


# ## **Model selection**

# Ok, we have many models and algorithms. First, we gonna try with all features.

# In[ ]:


classifiers = [
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression(),
    LinearSVC()]

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test), name)


# Using xgb (Xgboost)
# 
# http://xgboost.readthedocs.io/en/latest/python/python_intro.html

# In[ ]:


params = {
    'eta': 1,
    'max_depth': 15,
    'objective': 'binary:logistic',
    'lambda' : 3,
    'alpha' : 3
}    
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 100, verbose_eval=50)
predictions = model.predict(xgb.DMatrix(X_test))
survived = [int(round(value)) for value in predictions]
accuracy = accuracy_score(survived, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Feature importance in xgb:

# In[ ]:


xgb.plot_importance(booster=model)
plt.show()


# Validating our xgb model with a confusion matrix:

# In[ ]:


print(classification_report(survived, y_test))
print('\n')
print(confusion_matrix(survived, y_test))


# ## **Prediction on test data**

# ### First at all, let's work on test data:

# Looking for missed values.

# In[ ]:


test_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


#mapping to int
test_df['Embarked'] = test_df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
#filling missed values in Age and Fare
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)


# Creating new features

# In[ ]:


def people(passenger):
    age, sex = passenger
    return 'child' if age < 15 else sex
    
test_df['People'] = test_df[['Age','Sex']].apply(people, axis=1)


# In[ ]:


#working on People
test_df['People'] = test_df['People'].map({'male': 1, 'female': 2, 'child': 3})


# In[ ]:


#Now on Sex
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 2 })


# In[ ]:


#Add family members
test_df['family_members'] =  test_df['Parch'] + train_df['SibSp']


# Looking for Nclass in test.

# In[ ]:


test_df['Lastname'], test_df['Name'] = test_df['Name'].str.split(',', 1).str
test_df['Nclass'], test_df['Name'] = test_df['Name'].str.split('.', 1).str
test_df.head()


# Looking for unique values

# In[ ]:


test_df.Nclass.unique()


# In[ ]:


test_df['Nclass'] = test_df['Nclass'].map({' Col': 'Other', ' Rev': 'Other', ' Ms': 'Miss', ' Dr': 'Mr', ' Dona': 'Mrs', ' Master': 'Mr', ' Miss': 'Miss', ' Mrs': 'Mrs', ' Mr': 'Mr',})


# In[ ]:


test_df.Nclass.unique()


# Mapping to int

# In[ ]:


test_df['Nclass'] = test_df['Nclass'].map({'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Other': 4})


# In[ ]:


test_df.isnull().sum().sort_values(ascending=False)


# Drop unnecesary columns

# In[ ]:


test_df = test_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Lastname'], axis=1)
test_df = test_df.drop(['Embarked'], axis=1)


# In[ ]:


test_df.head()


# Train and make prediction with GradientBoostingClassifier

# In[ ]:


#prediction with GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X, y)
# make prediction
prediction = model.predict(test_df)
test_df['Survived'] = prediction


# In[ ]:


id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': id , 'Survived': prediction } )


# In[ ]:


test.head()


# In[ ]:


test.to_csv( 'titanic.csv' , index = False )


# ### Conclusions:
# 
# - People who embarked on Cherbourg have the highest probability of surviving, in addition to paying higher fare. We can say they belong in greater proportion to first class.
# 
# - Women and childrens have the highest probability to survive.
# 
# - Passengers who have 3 family members are more likely to survive.
# 
# -  In average, we found the higher fare in the first class and more important, the people who survive belong mostly to the first class.
# 
# - GradientBoostingClassifier is the better option to predict results.

# **Any  questions or comments are welcome!**

# In[ ]:




