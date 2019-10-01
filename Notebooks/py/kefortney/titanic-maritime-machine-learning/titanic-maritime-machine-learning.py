#!/usr/bin/env python
# coding: utf-8

# Summary
# -------
# 
# On April 15, 1912 the RMS Titanic carrying 2,224 passengers and crew struck an iceberg on the ship's maiden voyage. Over the next two hours slid below the icy waves along with over 1,500 dead by the time the RMS Carpathia over 6 hours later. A recent coal strike had ensured that the Titanic was not booked with her full complement of 3,339 passengers and crew otherwise the disaster would have been much deadlier.
# 
# Oddly enough, the RMS Titanic was actually carrying more lifeboats than required by law which was based on gross tonnage, not number of passengers. If every life boat has been successfully launched at full capacity (most were not fully loaded and two drifted away as she sunk) there would have only been room for 1,178 in total, still well shy the number of people aboard.
# 
# This is an exploration of using machine learning to determine factors in survival and predict missing values. The dataset is of passengers only (crew numbered about 885 people and had a survival rate of around 24%). Titanic's passengers numbered approximately 1,317 people: 324 in First Class, 284 in Second Class, and 709 in Third Class but this data set has a total pf 1,309 passenger records. 
# 
# The Data
# --------
# In order to solve this problem there are several steps I need to go through:
# 
# *  Import the data
# *  Analyze for trends
# *  Fill missing values
# *  Build features
# *  Apply machine learning
# *  Submit results
# 
# Python is brand new to me and this is my first foray into machine learning, please comment below with any suggestions. 

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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Visualisation
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().magic(u'matplotlib inline')
sns.set(style="ticks")


# Data Importation and Cleaning
# ---------------------------
# 
# First I wanted to get an understanding of the data and see how many are missing values.  At first I need to import from csv's both the training and the test data sets. 
# 
# * PassengerId - Unique Identifier
# * Survival - Survival (0 = No; 1 = Yes)
# * Pclass 1 - First Class, 2 = Second Class, 3 = Third Class
# * Name - Last Name, Surname First Name and additional qualifier if needed
# * Sex - Male or Female
# * Age - Age, Fractional if Age less than One (1) If the Age is Estimated, it is in the form xx.5
# * SibSp - Number of Siblings/Spouses Aboard
# * Parch - Number of Parents/Children Aboard
# * Ticket - Ticket Number
# * Fare - Passenger Fare
# * Cabin - Cabin with the letter being deck and number is cabin, decks should be A-G
# * Embarked - Port where person board (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[ ]:


# get titanic & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# if you want to see where values are missing
print("THIS IS THE TRAIN_DF INFO")
train_df.info()
print("-------------------------")
print("THIS IS THE TEST_DF INFO")
test_df.info()


# In[ ]:


# Just a quick check of the train_df data, errors are from the NaNs under Age
train_df.describe()


# Initial View
# ------------------
# Before getting to far into the depths there are a couple of quick visualizations that we should do just to get a feel for the data.  Sex and Age seem to factor heavily into survival but what else can be done to make modeling it more accurate?

# In[ ]:


# histogram of Sex and Age split by survival
g = sns.FacetGrid(train_df, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist,"Age",color="lightblue")
plt.show()

# distribution of age across different classes
train_df.Age[train_df.Pclass == 1].plot(kind='kde')    
train_df.Age[train_df.Pclass == 2].plot(kind='kde')
train_df.Age[train_df.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")    
plt.title("Age Distribution within Classes")
plt.xlim(0,80)
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
plt.show()


# Feature Analysis
# -------------------
# A quick check of features shows that some have greater viability than others.  Class and Sex looks to have stronger correlation and SibSp shows that being single was not in your favor.

# In[ ]:


sns.stripplot(x="Pclass", y="Age", hue="Survived", data=train_df, jitter=True)
plt.show()


# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[ ]:


sns.stripplot(x="Sex", y="Age", hue="Survived", data=train_df, jitter=True)
plt.show()


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# In[ ]:


sns.stripplot(x="SibSp", y="Age", hue="Survived", data=train_df, jitter=True)
plt.show()


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()


# In[ ]:


sns.stripplot(x="Parch", y="Age", hue="Survived", data=train_df, jitter=True)
plt.show()


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()


# In[ ]:


train_df.corr()["Survived"]


# Cleaning Data
# -------------
# There are some missing values, some are simpler than others. The first one is a quick fill for the missing single fare and embarkment points with the median value.

# In[ ]:


# Plot values for embarkment
train_df.Embarked.value_counts().plot(kind='bar', alpha=0.55)
plt.title("Passengers per Boarding Location")

# Embarked only in train_df, fill the two missing values with the most occurred value, which is "S".
train_df["Embarked"].value_counts() 
train_df["Embarked"] = train_df["Embarked"].fillna("S")

# Fill in the single missing fare with median value
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
                                           
# Convert fare from float to int
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)


# Family Size
# -------
# 
# Before tackling the missing Age information, it makes sense to do a little feature engineering now. First we start by the simple creation of FamilySize by adding SibSp and Parch together.  

# In[ ]:


# Create a family size variable
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]


# Name
# ----
# Next is taking a look at the Names and see what can be extracted. As you can see below there is the last name, a comma, title, first, middle name and then anything additional in parenthesis.

# In[ ]:


train_df['Name'].head(5)


# Title
# --------
# 
# The next step is to split out the title and simplfy the possible iterations.  This will replace the multitude of titles with just five: Mr, Mrs, Miss, Master and Rare Title.  Poonan's work was very helpful and I recommend taking a look at her work: https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline 

# In[ ]:


import re

# function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles
titles = train_df["Name"].apply(get_title)

#Add in the title column with all the current values so we can then manually change them
train_df["Title"] = titles

# Titles with very low cell counts to be combined to "rare" level
rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

# Also reassign mlle, ms, and mme accordingly
train_df.loc[train_df["Title"] == "Mlle", "Title"] = 'Miss'
train_df.loc[train_df["Title"] == "Ms", "Title"] = 'Miss'
train_df.loc[train_df["Title"] == "Mme", "Title"] = 'Mrs'
train_df.loc[train_df["Title"] == "Dona", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Lady", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Countess", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Capt", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Col", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Don", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Major", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Rev", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Sir", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
train_df.loc[train_df["Title"] == "Dr", "Title"] = 'Rare Title'

titles = train_df["Name"].apply(get_title)
# print(pd.value_counts(titles))

#Add in the title column.
test_df["Title"] = titles

# Titles with very low cell counts to be combined to "rare" level
rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

# Also reassign mlle, ms, and mme accordingly
test_df.loc[test_df["Title"] == "Mlle", "Title"] = 'Miss'
test_df.loc[test_df["Title"] == "Ms", "Title"] = 'Miss'
test_df.loc[test_df["Title"] == "Mme", "Title"] = 'Mrs'
test_df.loc[test_df["Title"] == "Dona", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Lady", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Countess", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Capt", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Col", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Don", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Major", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Rev", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Sir", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
test_df.loc[test_df["Title"] == "Dr", "Title"] = 'Rare Title'

print(train_df['Title'].value_counts())
print(test_df['Title'].value_counts())


# Title and Survival
# ------------------
# A quick plot of the titles and survival distribution (1=survived, 0=perished)shows there are certain trends that are easy to see. As we discovered earlier Sex was a factor in survival so it is no surprise that the title Mr. was hardest hit. Master was a term for a young boy but that did not help as much as you would think. 

# In[ ]:


sns.swarmplot(x="Title", y="Age", hue="Survived", data=train_df)
plt.show()


# In[ ]:


train_df[["Title", "Survived"]].groupby(['Title'], as_index=False).mean()


# Filling Missing Ages
# --------------------
# 
# There is a fair number of missing age.  Plotting the distribution before and after adding in ages allowed me to see what the impact would be.
# 
# I read through a bunch of other's work on this, some used random forest, or filled using the mean.  I felt like mean produced a spike rather than a distribution that mimiced what was there so I tried a couple of interpolation methods I settled on linear for a fairly even distribution. 
# 
# After the fact I went back and compared interpolation against mean and mean produced a more accurate model so I ended up returning to that but I left the alternative there if you want to try it.

# In[ ]:


# Plot of Age before filling missing values to visualize the distribution
plt.hist(train_df['Age'].dropna(),bins=80)
plt.title('Before Correcting Missing Ages')
plt.show()

# Fill in all missing values with linear interpolation
# train_df['Age']= train_df.Age.fillna(train_df.Age.interpolate(method='linear')) 
train_df['Age']= train_df.Age.fillna(train_df.Age.mean())

# Plot of Age again after linear interpolation was completed
plt.hist(train_df['Age'],bins=80)
plt.title('After Correcting Missing Ages')
plt.show()


# In[ ]:


# Plot of Age before filling missing values to visualize the distribution
plt.hist(test_df['Age'].dropna(),bins=80)
plt.title('Before Correcting Missing Ages')
plt.show()

# Fill in all missing values with linear interpolation
#test_df['Age']= test_df.Age.fillna(test_df.Age.interpolate(method='linear')) 

test_df['Age']= test_df.Age.fillna(test_df.Age.mean())

# Plot of Age again after linear interpolation was completed
plt.hist(test_df['Age'],bins=80)
plt.title('After Correcting Missing Ages')
plt.show()


# Age Bins
# ---------------------------
# In order to get  little more out of the Age, I decided to bin them into subsets with splits and then spent a little time working on refining them. I tried a couple of different buckets and ended up settling on 0-10, 10-21, 21-55 and 55-81 and applied labels to the grouping.

# In[ ]:


# Used to bin the ages at the points
agepercentile = [0, 10, 21, 55, 81]

# Creates a new column binning the ages in to brackets and labeling them with numbers.  
train_df["AgeBin"] = pd.cut(train_df['Age'],agepercentile, labels=["child","youth","adult","elder"])
test_df["AgeBin"] = pd.cut(test_df['Age'],agepercentile, labels=["child","youth","adult","elder"])

sns.swarmplot(x="AgeBin", y="Age", hue="Survived", data=train_df)
plt.show()


# In[ ]:


train_df[["AgeBin", "Survived"]].groupby(['AgeBin'], as_index=False).mean()


# Fare Binning
# ---------
# Likewise there is a percentile insight into the fares. I decided to break it into thirds as 2nd Class and 3rd Class overlapped quite a bit in fares. The resulting swarmplot really says it all.

# In[ ]:


# This takes the age and breaks it into precentiles
print(np.percentile(train_df['Fare'],[0,33,66,100]))
print(np.percentile(test_df['Fare'],[0,33,66,100]))
farepercentile = [0, 8, 26, 513]

# Creates a new column binning the ages in to brackets and labeling them with numbers.  
# Prencentiles In this case are 0-7, 7=14, 14-31 and 31-513)
train_df["FareBin"] = pd.cut(train_df['Fare'],farepercentile, labels=["Low","Mid","High"])
test_df["FareBin"] = pd.cut(test_df['Fare'],farepercentile, labels=["Low","Mid","High"])

# plot the result
sns.swarmplot(x="FareBin", y="Age", hue="Survived", data=train_df)
plt.show()


# In[ ]:


train_df[["FareBin", "Survived"]].groupby(['FareBin'], as_index=False).mean()


# Assembly of Training Set
# -------
# Now it is time to take all that hard work and bring it all into a data set that can be used to predict survival. I dropped Cabin and Ticket information but kept everything else.  Splitting the binned values into individual columns with Boolean values helped with the accuracy. 

# In[ ]:


AgeBin = pd.get_dummies( train_df['AgeBin'] , prefix = 'AgeBin')
FareBin = pd.get_dummies( train_df['FareBin'] , prefix = 'FareBin')
Embarked = pd.get_dummies( train_df['Embarked'] , prefix = 'Embarked')
Title = pd.get_dummies( train_df['Title'] , prefix = 'Title')
Sex = pd.get_dummies( train_df['Sex'] , prefix = 'Sex' )
Pclass = pd.get_dummies( train_df['Pclass'] , prefix = 'Pclass')
Age = train_df['Age']
Fare = train_df['Fare']
SibSp = train_df['SibSp']
Parch = train_df['Parch']
Survived = train_df['Survived']


# In[ ]:


train_X = pd.concat([Age , Fare, SibSp , Parch, Pclass, AgeBin, FareBin, Embarked, Title, Sex], axis=1)
train_X.head()

# This is just to determing correlation with survived to see how well it worked
train_corr = pd.concat([Survived, Age , Fare, SibSp , Parch, Pclass, AgeBin, FareBin, Embarked, Title, Sex], axis=1)


# Assembly of Test Set
# -------
# Now time repeat this step for the test data using the same features. 

# In[ ]:


AgeBin = pd.get_dummies( test_df['AgeBin'] , prefix = 'AgeBin')
FareBin = pd.get_dummies( test_df['FareBin'] , prefix = 'FareBin')
Embarked = pd.get_dummies( test_df['Embarked'] , prefix = 'Embarked')
Title = pd.get_dummies( test_df['Title'] , prefix = 'Title')
Sex = pd.get_dummies( test_df['Sex'] , prefix = 'Sex' )
Pclass = pd.get_dummies( test_df['Pclass'] , prefix = 'Pclass')
Age = test_df['Age']
Fare = test_df['Fare']
SibSp = test_df['SibSp']
Parch = test_df['Parch']


# In[ ]:


test_X = pd.concat([Age , Fare, SibSp , Parch, Pclass, AgeBin, FareBin, Embarked, Title, Sex], axis=1)
test_X.head()


# Correlation
# --------------------------
# A quick plot of the training data with Survived included shows the corelation. Work on the Title seems to have paid off.

# In[ ]:


corr = train_corr.corr()
sns.heatmap(corr)
plt.show()


# In[ ]:


train_corr.corr()['Survived']


# Time to create all the datasets that get feed into model. 
# 
# train_valid_X is all the features on the training data for the model to learn from
# train_valid_y is the list of correlating values for the same set with whether or not they survived 
# test_X is all the features on the testing data and has been already defined
# 

# In[ ]:


# Create all datasets that are necessary to train, validate and test models
train_valid_X = train_X
train_valid_y = train_df.Survived
# test_X = test_X
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )


# Models
# ------------
# Here are a selection of models, uncomment whichever you want to run.   

# In[ ]:


model = RandomForestClassifier(n_estimators=700,min_samples_leaf=3)
# model = SVC()
# model = GradientBoostingClassifier()
# model = KNeighborsClassifier(n_neighbors = 3)
# model = GaussianNB()
# model = LogisticRegression()


# Now apply the selected model with the datasets and see what you get

# In[ ]:


model.fit( train_X , train_y )


# This is an interesting way to score the model by comparing both the training and test data to make sure you are not over fitting.

# In[ ]:


# Score the model
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))


# Submission
# Finally time to submit it all. I am sure I will be back for revisions as I learn more but it was a great way to get my hands wet with machine learning. 
# Here are some of the other kernels that were very helpful:
# https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
# https://www.kaggle.com/helgejo/titanic/an-interactive-data-science-tutorial
# https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline

# In[ ]:


test_Y = model.predict( test_X )
passenger_id = test_df.PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head(10)
test.to_csv( 'titanic_pred.csv' , index = False )


# Revision History Notes: Submitted
