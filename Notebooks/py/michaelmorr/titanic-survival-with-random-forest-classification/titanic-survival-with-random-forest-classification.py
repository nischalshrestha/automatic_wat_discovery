#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load Packages
import numpy as np # linear algebra
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import re 
from IPython.display import display_html
import itertools
import math
import random
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib

#Load sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn import linear_model

# Import Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

#Learning curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

from IPython.display import display_html
import warnings

# for inline plots
get_ipython().magic(u'matplotlib inline')
warnings.filterwarnings('ignore')

mpl.rcParams['figure.figsize'] = (8, 6)
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["axes.labelsize"] = 15
mpl.rc('xtick', labelsize = 15) 
mpl.rc('ytick', labelsize = 15)
sns.set(style = 'whitegrid', palette = 'muted', font_scale = 2)
    
print('Libraries Imported')


# In[ ]:


# get data from csv files
test  = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

#determine sizes of datasets
n_train, m_train = train.shape
n_test, m_test = test.shape


# divide into X and y data
X_train = pd.DataFrame(train.iloc[:,1: m_train])
y_train = pd.DataFrame(train.iloc[0:, 1])

X_test_original  = test
X_test = test

print('Data Imported')


# ## DATA EXPLORATION

# In[ ]:


# determint the size of the data sets

# print a summary of loaded results
print('FULL DATA')
print('Number of features (m): %.0f'%(m_train))
print('Number of traing samples (n): %.0f'%(n_train))

print('\n\nTest DATA')
print('Number of features (m): %.0f'%(m_test))
print('Number of traing samples (n): %.0f'%(n_test))

cnt = 0
# print out the features
print('\n\nFeatures: ')
for feature in X_train.columns:
    cnt += 1
    print('%d. '%(cnt), feature,'\t\t')


# VARIABLE DESCRIPTIONS:
# 
#     Survived: Survived (1) or died (0)
#     Pclass: Passenger's class
#     Name: Passenger's name
#     Sex: Passenger's sex
#     Age: Passenger's age
#     SibSp: Number of siblings/spouses aboard
#     Parch: Number of parents/children aboard
#     Ticket: Ticket number
#     Fare: Fare
#     Cabin: Cabin
#     Embarked: Port of embarkation
# 

# In[ ]:


# take a sample of what the data looks like
X_train.head(10)


# ### Helper Functions

# In[ ]:


#sets up the parametes for plotting.. size and font
def PlotParams(Font, sizex, sizey):
    mpl.rcParams['figure.figsize'] = (sizex,sizey)
    plt.rcParams["legend.fontsize"] = Font
    plt.rcParams["axes.labelsize"] = Font
    mpl.rc('xtick', labelsize = Font) 
    mpl.rc('ytick', labelsize = Font)

#sets up Seaborn parametes for plotting
def snsParams(font, colour_scheme):
    #eaborn.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.set(style = 'whitegrid', palette = colour_scheme, font_scale = font)

#determined ht emissing data
def Missing (X):
    total = X.isnull().sum().sort_values(ascending = False)
    percent = round(X.isnull().sum().sort_values(ascending = False)/len(X)*100, 2)
    missing = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
    return(missing) 

#plots number of dataframes side by side
def SideSide(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw = True)

#makes heat map of correllations
def PlotCorr(X):
    corr = X.corr()
    #fig , ax = plt.figure( figsize = (6,6 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    sns.heatmap(
        corr, cmap = cmap, square = True, cbar = False, cbar_kws = { 'shrink' : 1 }, 
     annot = True, annot_kws = { 'fontsize' : 14 }
    )
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90) 
    
#plot top correlatins in a heat map
def TopCorr(X, lim):
    corr = X.corr()
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    #fig , ax = plt.subplots( figsize = (6,6 ) )
    sns.heatmap(corr[(corr >= lim) | (corr <= -lim)], 
         vmax = 1.0,  cmap = cmap, vmin = -1.0, square = True, cbar = False, linewidths = 0.2, annot = True, 
                annot_kws = {"size": 14})
    plt.yticks(rotation = 0)
    plt.xticks(rotation = 90)


# In[ ]:


# provide information about the types of data we are dealing with
print('ORIGINAL TRAINING DATA:')
X_train.info()

print('\n\n\nORIGINGAL TEST DATA:')
X_test.info()

#summarise the types of data
print('\ndata types of features:')

cnt = 0
d_type = ['float64', 'int64','object','dtype']
print('\n\tTRAIN \t\t TEST')
for c1, c2 in zip(X_train.get_dtype_counts(), X_test.get_dtype_counts()):
    cnt += 1
    print("%s:\t%-9s \t%s"%(d_type[cnt],c1, c2))
    


# the floats will need to be converted into int64 values, some of the objects (e.g. sex) will ned to be converted into numberics, all NaN and null values will need to be filled, 

# In[ ]:


X_train.describe(include = "all")


# Average Age is 29 years and ticket price is 32. As there are 681 unique tickets and there is no way to extract less detailed information we exclude this variable. There are 891 unique names but we could take a look on the title of each person to understand if the survival rate of people from high society was higher

# ### Missing Data

# as can be seen from the above data, there are several featured with NaN missing values. Lets take a look at the features that have missing values missing

# In[ ]:


# Fill empty values with NaN
X_train = X_train.fillna(np.nan)
X_test = X_test.fillna(np.nan)

#finds missing values
missing_train = Missing(X_train)
missing_test = Missing(X_test)
    
print('TRAIN DATA','\t\t','TEST DATA')
SideSide(missing_train, missing_test)

#plot missing data in heatmap for visualisation
print('\n\n  MISSING TRAINING DATA \t\t\t MISSING TEST DATA')
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
plt.figure(figsize = (10,5));
plt.subplot(1, 2, 1)
sns.heatmap(X_train.isnull(), yticklabels = False, cbar = False, cmap = cmap)
plt.subplot(1, 2, 2)
sns.heatmap(X_test.isnull(), yticklabels = False, cbar = False,cmap = cmap);


# Age, embarked and cabin have the most missing values. Both of these terms can be used to determin the survival (see later). thus these missing values will need to be filled in based on their relationship with other featured
# 
# The Embarked (test) feature has only 2 missing values, which can easily be filled. 
# The 'Age' feature, which has 177 missing values, will be filled with values based on its relationship with other features. 

# 

# ### Feature Correlation
# 

# In[ ]:


#show the correlations between all the featured in a heatmap
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
PlotCorr(X_train);
plt.subplot(1,2,2)
TopCorr(X_train, 0.2)


# In[ ]:


# highest correlated with correlation of features with 'Survived'
print('Featured hights correlation with survival')
print('Feature\tCorrelation')
Survive_Corr = X_train.corr()["Survived"]
Survive_Corr = Survive_Corr[1:9] # remove the 'Survived'
Survive_Corr= Survive_Corr[np.argsort(Survive_Corr, axis = 0)[::-1]] #sort in descending order
print(Survive_Corr)


# In[ ]:


# Plot the top correlationin a bar chart for east visualisation.
width = 0
fig, ax = plt.subplots(figsize = (10,6))
rects = ax.barh(np.arange(len(Survive_Corr)), np.array(Survive_Corr.values), color = 'red')
ax.set_yticks(np.arange(len(Survive_Corr)) + ((width)/1))
ax.set_yticklabels(Survive_Corr.index, rotation ='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Survival",fontsize = 14);
ax.grid(True)


# it can be seen that tha Pclass and the Fare have the stongest correlation with the survival rate. these parameters themselves are also highly correlated with eachther. 

# ### Survival by sex

# In[ ]:


snsParams(2, 'muted')
# plot survival count for male and female
plt.figure(figsize = (20,5))
plt.subplot(1, 3, 1)
b = sns.countplot(x = 'Survived',hue = 'Sex', data = X_train);
b.set_xlabel("Survived",fontsize = 15)
b.set_ylabel("Count",fontsize = 15)
b.legend(fontsize = 14)
snsParams(1.5, 'muted')

#survival probability of males and females
plt.subplot(1, 3, 2)
g = sns.barplot(x = "Sex", y = "Survived",data = X_train)
g = g.set_ylabel("Survival Probability")

plt.subplot(1, 3, 3)
sns.violinplot(y = 'Survived', x = 'Sex', data = X_train, inner = 'quartile')


# This first bar plot above shows the distribution of female and male survived and died.
# This count plot shows the actual distribution of male and female passengers that survived and did not survive. It shows that among all the females ~ 230 survived and ~ 70 did not survive. While among male passengers ~110 survived and ~480 did not survive. 
# 
# 
# second plot. The second plot reinforces this idea where, it shows that ~74% female passenger survived while only ~19% male passenger survived
# 
# 
# the violin plot also reinforces the fact that more males die and more women survive
# 
# 
# 
# from this is is evident that  Males have less chance to survive than Female. this is probably due to the "Women and children first" mentality
# 
# 

# ### AGE

# 

# In[ ]:


# plot survival number for age dependandcy
fig, axes = plt.subplots(figsize = (20,6), nrows = 1, ncols = 3)

g = sns.distplot(X_train[X_train['Survived'] == 1].Age.dropna(), bins=20, label = 'Survived')
g = sns.distplot(X_train[X_train['Survived'] == 0].Age.dropna(), bins=20, label = 'Not Survived')

g = sns.kdeplot(X_train["Age"][(X_train["Survived"] == 0) & (X_train["Age"].notnull())], color = "Green", shade = False)
g = sns.kdeplot(X_train["Age"][(X_train["Survived"] == 1) & (X_train["Age"].notnull())], ax = g, color = "Blue", shade= False)

g.set_xlabel("Age",fontsize = 15)
g.set_ylabel("Frequency",fontsize = 15)
g = g.legend(["Not Survived","Survived"],fontsize = 15)
plt.xlim(0,80)
plt.ylim(0,0.04)
plt.grid(True)

women = X_train[X_train['Sex'] == 'female']
men = X_train[X_train['Sex'] == 'male']

#For womwn
ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins = 20, label = 'survived', ax = axes[0], kde = False)
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 20, label = 'not survived', ax = axes[0], kde = False)
ax.set_xlabel("Age",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
ax.legend(fontsize = 15)
ax.set_title('Female', fontsize = 15)
ax.set(xlim = (0, X_train['Age'].max()));
ax.set(ylim = (0, 50));
    
    
#For men
ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins = 20, label = 'survived', ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins = 20, label = 'not survived', ax = axes[1], kde = False)
ax.set_xlabel("Age",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
ax.legend(fontsize = 15)
ax.set_title('Male', fontsize = 15)
ax.set(xlim = (0, X_train['Age'].max()))
ax.set(ylim = (0, 50));

g = sns.factorplot(x = "Survived", y = "Age",data = X_train, kind="box")
g = sns.factorplot(x = "Survived", y = "Age",data = X_train, kind="violin")



# FEMALES: have a much higher survival count than men. there are 2 intervals with a high survival count: infants 0 - 5 year, and adults 16 - 38 years old
# 
# MEN: have a much lower survival count than women. again there are 2 intervals with relatively high survival counts, infants 0 - 5 year, and adults 20 - 32 years old
# 
# 
# When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 5) for babies and very young childrens.
# 
# 
# The age distribution for survivors and non-survivors are very similar. One notable difference is that, of the survivors, a larger proportion were children. 

# 

# ### PClass

# In[ ]:


plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
sns.barplot(x = 'Pclass', y = 'Survived', data = X_train)

# Explore Pclass vs Survived by Sex
plt.subplot(1, 3, 2)
g = sns.barplot(x = "Pclass", y = "Survived", hue = "Sex", data = X_train)
#g = g.set_ylabels("survival probability")

plt.subplot(1, 3, 3)
sns.countplot(x = 'Survived',hue = 'Pclass',data = X_train);

plt.figure(figsize = (16,6))
plt.subplot(1, 2, 1)
sns.violinplot(y = 'Survived', x = 'Pclass', data = X_train, inner = 'quartile')
plt.subplot(1, 2, 2)
sns.violinplot(x='Pclass', y = 'Age', hue = 'Survived', data = X_train, split = True)


ax = sns.factorplot(y = "Age", x = "Pclass", hue = "Sex", data = X_train, kind = "box")
sns.factorplot(y = "Age", x = "Sex", hue = "Pclass", data = X_train, kind = "box")



# Here we see clearly, that Pclass is contributing to a persons chance of survival, especially if this person is in class 1.
# 
# This trend is conserved when we look at both male and female passengers. it can also be seen that a far large percentage of women survive compared to men
# 

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(X_train, col = 'Survived', row = 'Pclass', size = 3.2, aspect = 1.2)
grid.map(plt.hist, 'Age', alpha = 0.8, bins=20)
grid.add_legend();


# The plot above confirms our assumption about pclass 1, but we can also spot a high probability that a person in pclass 3 will not survive.
# 

# In[ ]:


PlotParams(15,6,6)
X_train.Age[X_train.Pclass == 1].plot(kind = 'kde')    
X_train.Age[X_train.Pclass == 2].plot(kind = 'kde')
X_train.Age[X_train.Pclass == 3].plot(kind = 'kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes", fontsize = 15)
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'), loc = 'best') ;
plt.xlim(0,80)
plt.ylim(0,0.04)


# the median age increases with decreasing class the third class will have a a large number of infants

# ### Embarked

# In[ ]:


# Explore Embarked vs Survived 
plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
g = sns.barplot(x = "Embarked", y = "Survived",  data = X_train)

# Explore Pclass vs Survived by Sex
plt.subplot(1, 3, 2)
g = sns.barplot(x = "Embarked", y = "Survived", hue = "Sex", data = X_train)
#g = g.set_ylabels("survival probability")
plt.subplot(1, 3, 3)
sns.countplot(x = 'Survived',hue = 'Embarked',data = X_train);


# In[ ]:


sns.factorplot(y = "Age", x = "Embarked", hue = "Pclass", data = X_train, kind = "box")


# It seems that passenger coming from Cherbourg (C) have more chance to survive. Below we can see that this is not related to class. perhaps it is related to Derck level... see later

# In[ ]:


# Explore Pclass vs Embarked 
PlotParams(15, 8, 6)
snsParams(2,'muted')

g = sns.factorplot("Pclass", col = "Embarked",  data = X_train, size = 8, 
                   kind = "count", palette = "muted")
g = g.set_ylabels("Count")
g = sns.factorplot("Pclass", col = "Embarked",  data = X_train,
                   hue = "Sex", size = 8, kind = "count", palette = "muted")

g = g.set_ylabels("Count")


# ### SibSP & Parch

# In[ ]:


PlotParams(15, 10, 6)
plt.figure(figsize = (16,5))
plt.subplot(1, 2, 1)
g = sns.barplot(x = "Parch", y = "Survived",  data = X_train, palette = "muted")
plt.subplot(1, 2, 2)
g = sns.barplot(x = "SibSp", y = "Survived",  data = X_train, palette = "muted")

plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
sns.violinplot(y = 'Survived', x = 'Parch', data = X_train, inner = 'quartile')
plt.subplot(1, 2, 2)
sns.violinplot(y = 'Survived', x = 'SibSp', data = X_train, inner = 'quartile')


# 
# 
# Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6 ).
# 
# Be carefull there is an important standard deviation in the survival of passengers with 3 parents/children
# 

# # Fare

# In[ ]:


PlotParams(15, 8, 6)
plt.figure()
sns.kdeplot(X_train["Fare"][X_train.Survived == 1])
sns.kdeplot(X_train["Fare"][X_train.Survived == 0])
plt.legend(['Survived', 'Died'])
plt.xlabel('Fare')
plt.ylabel('Survival Probability')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(0,200)
plt.ylim(0,.060)
plt.show()


fig, ax = plt.subplots(figsize=(16,4),ncols=2)
ax1 = sns.boxplot(x = "Embarked", y = "Fare", hue = "Pclass", data = X_train, ax = ax[0]);
ax2 = sns.boxplot(x = "Embarked", y = "Fare", hue = "Pclass", data = X_test, ax = ax[1]);
ax1.set_title("Training Set", fontsize = 15)
ax2.set_title('Test Set',  fontsize = 15)
fig.show()


# 
# 
# As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model. Passengers who paid lower fare appear to have been less likely to survive. This is probably strongly correlated with Passenger Class, which we'll look at next.
# 

# 

# # Feature Engineering

# In[ ]:


#combine the tets and training data so that operations can be performed together
full_data = [X_train, X_test] 


# As we see, Age column contains 256 missing values in the whole dataset.
# 
# Since there is subpopulations that have more chance to survive (children for example), it is preferable to keep the age feature and to impute the missing values.
# 
# To adress this problem, i looked at the most correlated features with Age (Sex, Parch , Pclass and SibSP).

# ### Missing data and Combine

# In[ ]:


#fill in Embarked datta with S as it is the most common
for X in full_data:
    X['Embarked'] = X['Embarked'].fillna("S")


# In[ ]:


X_train.head()


# ### Cabin number to Deck level

# ater looking closely at the cabin number, it can be seen that it is an alpha-numeric identity. The letter indicates the deck and the number represents the cabin number on this deck. We will therefore subsitute this 'Cabin' category for a 'Deck" categor and simply extract the deck letter
# 
# recall most cabin number are missing so lets see how may people have cabine and if it is related to surviving

# In[ ]:


# cabine Vrs no cabine survival rates
for X in full_data:
    X["CabinBool"] = (X["Cabin"].notnull().astype('int'))
    
#draw a bar plot of CabinBool vs. survival
sns.barplot(x = "CabinBool", y = "Survived", data = X_train)
plt.show()


# In[ ]:


# Extract deck 
def extract_cabin(x):
    return x != x and 'Other' or x[0]

for X in full_data:
    X['Cabin'] = X['Cabin'].apply(extract_cabin)
    X['Deck'] = X['Cabin']

train_deck = pd.DataFrame(X_train.groupby('Deck').size())
test_deck = pd.DataFrame(X_test.groupby('Deck').size())

print('TRAIN \t\t TEST')
SideSide(train_deck,test_deck )


# In[ ]:


snsParams(1.2, 'muted')
plt.figure(figsize = (16,5))

plt.subplot(1, 3, 1)
g = sns.countplot(X_train["Cabin"], palette = "muted")
plt.subplot(1, 3, 2)
g = sns.barplot(x = "Deck", y = "Survived",  data = X_train, palette = "muted")

plt.subplot(1, 3, 3)
sns.countplot(x = 'Survived',hue = 'Deck',data = X_train, palette = "muted");

snsParams(2, 'muted')
plt.figure(figsize = (16,5))
g = sns.factorplot("Deck", col = "Pclass",  data = X_train, size = 8, 
                   kind = "count", palette = "muted")
g = g.set_ylabels("Count")
g = sns.factorplot("Deck", col = "Embarked",  data = X_train,
                   hue = "Sex", size = 8, kind = "count", palette = "muted")
g = g.set_ylabels("Count")


# we can see that passengers with a cabin have generally more chance to survive than passengers without (X).
# 
# It is particularly true for cabin B, C, D, E and F.

# ### Families or Alone

# In[ ]:


# To get the full family size of a person, added siblings and parch.

PlotParams(15, 8, 6)
# determine size of family on board
for X in full_data:
    X['Family'] = X['SibSp'] + X['Parch'] + 1 
    
axes = sns.factorplot('Family','Survived', hue = 'Sex', data = X_train, aspect = 2)
plt.grid(True)
axes = sns.factorplot('Family','Survived',  data = X_train, aspect = 2)
plt.grid(True)

for X in full_data:
    X['Alone'] = [1 if i<2 else 0 for i in X['Family']]
    

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(18,6))
sns.barplot(x = "Family", y = "Survived", hue = "Sex", data = X_train, ax = axis1);
sns.barplot(x = "Alone", y = "Survived", hue = "Sex", data = X_train, ax = axis2);
sns.barplot(x = "Alone", y = "Survived", data = X_train)
plt.show()
   


# 
# 
# Assumption: the less people was in your family the faster you were to get to the boat. The more people they are the more managment is required. However, if you had no family members you might wanted to help others and therefore sacrifice.
# 
# The females traveling with up to 2 more family members had a higher chance to survive. However, a high variation of survival rate appears once family size exceeds 4 as mothers/daughters would search longer for the members and therefore the chanes for survival decrease.
# 
# Alone men might want to sacrifice and help other people to survive.
# 
# 

# ### AGE Categories

# HOw do we fill in the missing age??? lets investigate
# 

# In[ ]:


# Explore Age vs Sex, Parch , Pclass and SibSP
sns.factorplot(y = "Age", x = "Sex", data = X_train, kind = "box")
sns.factorplot(y = "Age", x = "Sex", hue = "Pclass", data = X_train, kind = "box")
sns.factorplot(y = "Age", x = "Parch", data = X_train, kind = "box", palette = "muted")
sns.factorplot(y = "Age", x = "SibSp", data = X_train, kind = "box", palette = "muted")


# 
# 
# Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.
# 
# However, 1rst class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
# 
# Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.
# 
# Lets take a closer look at the correlations
# 

# In[ ]:



PlotCorr(X_train[["Age","Sex","SibSp","Parch","Pclass",'Family','Alone', 'Fare']])

#correlation of features with target variable
Age_Corr = X_train.corr()["Age"]
#Age_Corr= Age_Corr[np.argsort(Age_Corr, axis = 0)[::-1]] #sort in descending order
Age_Corr = Age_Corr[1:10] # remove the 'Survived'
print(Age_Corr)


# The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.
# 
# In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.
# 
# So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.
# 
# The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.
# 

# In[ ]:


#Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows

def ReplaceAge(X):
    index_NaN_age = list(X["Age"][X["Age"].isnull()].index)

    for i in index_NaN_age :
        age_med = X["Age"].median()
        
        age_pred = X["Age"][((X['SibSp'] == X.iloc[i]["SibSp"]) & 
                                        (X['Parch'] == X.iloc[i]["Parch"]) &
                                        (X['Pclass'] == X.iloc[i]["Pclass"]) &
                                        (X['Family'] == X.iloc[i]["Family"]) &
                                        (X['Alone'] == X.iloc[i]["Alone"]) &
                                         (X['Alone'] == X.iloc[i]["Alone"])
                                        )].median()
        if not np.isnan(age_pred) :
            X['Age'].iloc[i] = age_pred
        else :
            X['Age'].iloc[i] = age_med
    return (X)

for X in tqdm(full_data):
     X = ReplaceAge(X)
    
print('Done')


# In[ ]:


#sort the ages into logical categories
## create bins for age
def AgeCategory(age):
    a = ''
    if age <= 3:
        a = 'Baby'
    elif age <= 12: 
        a = 'Child'
    elif age <= 18:
        a = 'Teenager'
    elif age <= 35:
        a = 'Young Adult'
    elif age <= 65:
        a = 'Adult'
    elif age == 'NaN':
        a = 'NaN'
    else:
        a = 'Senior'
    return a
        
for X in full_data:
    X['Age Group'] = X['Age'].map(AgeCategory)


plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
g = sns.barplot(x = "Age Group", y = "Survived",  data = X_train)
plt.xticks(rotation = 90)

plt.subplot(1, 3, 2)
sns.countplot(x = 'Survived', hue = 'Age Group',data = X_train)

plt.subplot(1, 3, 3)
sns.boxplot(data = X_train, x = "Age Group", y = "Age");
plt.xticks(rotation = 90)


# ### Fare Feature

# In[ ]:


# fill missing Fare with median fare for each Pclass
for X in full_data:
    X["Fare"].fillna(X.groupby("Pclass")["Fare"].transform("median"), inplace = True)
    
for X in full_data:
    X.loc[ X['Fare'] <= 7.91, 'Fare'] = 0
    X.loc[(X['Fare'] > 7.91) & (X['Fare'] <= 14.454), 'Fare'] = 1
    X.loc[(X['Fare'] > 14.454) & (X['Fare'] <= 31), 'Fare']   = 2
    X.loc[(X['Fare'] > 31) & (X['Fare'] <= 99), 'Fare']   = 3
    X.loc[(X['Fare'] > 99) & (X['Fare'] <= 250), 'Fare']   = 4
    X.loc[X['Fare'] > 250, 'Fare'] = 5
    X['Fare'] = X['Fare'].astype(int)


# ### Titles

# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names
for X in full_data:
    X['Title'] = X['Name'].apply(get_title)
    
# Group all non-common titles into one single grouping "Rare"
for X in full_data:
    X['Title'] = X['Title'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'Noble')
    X['Title'] = X['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'Officer')
    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms', 'Miss')
    X['Title'] = X['Title'].replace('Mme', 'Mrs')

    
print('TRAIN TITLE \t TEST TITLES')
train_titles = pd.DataFrame(X_train.Title.value_counts())
test_titles = pd.DataFrame(X_test.Title.value_counts())

SideSide(train_titles,test_titles)

plt.figure(figsize = (16,6))
plt.subplot(1, 3, 1)
g = sns.barplot(x = "Title", y = "Survived",  data = X_train)
plt.xticks(rotation = 90)

plt.subplot(1, 3, 2)
sns.countplot(x = 'Survived', hue = 'Title',data = X_train);
plt.xticks(rotation = 90)

plt.subplot(1, 3, 3)
sns.boxplot(data = X_train, x = "Title", y = "Age");
plt.xticks(rotation = 90)

tab = pd.crosstab(X_train['Title'], X_train['Pclass'])
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)

tab_prop.plot(kind = "bar", stacked = True)
plt.xticks(rotation = 90)


# 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.

# ### Mapping and removal of features

# In[ ]:


#map each Sex value to a numerical value
sex_map = {"male": 0, "female": 1}
Embark_map = {"C": 1,"S": 2, "Q": 3}
deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Other": 9}
age_map = {"Baby": 1, "Child": 2, "Teenager": 3, "Young Adult": 4, "Adult": 5, "Senior": 6}
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, "Noble": 5}

for X in full_data:
    X["Sex"] = X["Sex"].map(sex_map)
    X["Embarked"] = X["Embarked"].map(Embark_map)
    X["Deck"] = X["Deck"].map(deck_map)
    X["Age Group"] = X["Age Group"].map(age_map)
    X["Title"] = X["Title"].map(title_mapping)


# In[ ]:


X_train = X_train.drop("Name", axis = 1) 
X_test = X_test.drop("Name", axis = 1) 
X_train = X_train.drop("Ticket", axis = 1) 
X_test = X_test.drop("Ticket", axis = 1) 
X_train = X_train.drop("Cabin", axis = 1) 
X_test = X_test.drop("Cabin", axis = 1) 
X_train = X_train.drop("Age", axis = 1) 
X_test = X_test.drop("Age", axis = 1) 
X_test = X_test.drop("PassengerId", axis = 1)  


# ### Look at hte prepared Data

# In[ ]:


X_test.head(10)


# In[ ]:


plt.figure(figsize = (20,12))
PlotCorr(X_train);


# In[ ]:


#correlation of features with target variable
Survive_Corr = X_train.corr()["Survived"]
Survive_Corr = Survive_Corr[np.argsort(Survive_Corr, axis = 0)[::-1]] #sort in descending order
Survive_Corr = Survive_Corr[1:15] # remove the 'Survived'
print(Survive_Corr)


# In[ ]:


X_train = X_train.drop("Survived", axis = 1)


# In[ ]:


missing_train = Missing(X_train)
missing_test = Missing(X_test)


print('TRAIN DATA','\t\t','TEST DATA')
SideSide(missing_train, missing_test)

print('\n\nMISSING TRAINING DATA \t\t\t MISSING TEST DATA')
plt.figure(figsize = (10,5));
plt.subplot(1, 2, 1)
sns.heatmap(X_train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
plt.subplot(1, 2, 2)
sns.heatmap(X_test.isnull(), yticklabels = False, cbar = False,cmap = 'viridis');


# In[ ]:


print('TRAINING')
print(X_train.info())
print('\n\nTEST')
print(X_train.info())

X_train.head(0)
X_test.head(0)

cnt = 0
d_type = ['float64', 'int64','object','dtype']
print('\n\tTRAIN \t\t TEST')
for c1, c2 in zip(X_train.get_dtype_counts(), X_test.get_dtype_counts()):
    cnt += 1
    print("%s:\t%-9s \t%s"%(d_type[cnt],c1, c2))
    


# ## PREDICTION MODELS

# in the following section several classification methods with be tested to determine which is the best, logistical Regression, Stochastic Gradient Descent, Random Forest, Support Vector Machine, K-Nearest Neighbour, Gausssian Naive Bayes, Multi-Layer Perceptron, Linear Support Vector Machine, Decisison Tree.

# In[ ]:



# grid search
def GridSearchModel(X, Y, model, parameters, cv):
    CV_model = GridSearchCV(estimator = model, param_grid = parameters, cv = cv)
    CV_model.fit(X, Y)
    CV_model.cv_results_
    print("Best Score:", CV_model.best_score_," / Best parameters:", CV_model.best_params_)
    
# Learning curve
def LearningCurve(X, y, model, cv, train_sizes):

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = cv, n_jobs = 4, 
                                                            train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std  = np.std(train_scores, axis = 1)
    test_scores_mean  = np.mean(test_scores, axis = 1)
    test_scores_std   = np.std(test_scores, axis = 1)
    
    train_Error_mean = np.mean(1- train_scores, axis = 1)
    train_Error_std  = np.std(1 - train_scores, axis = 1)
    test_Error_mean  = np.mean(1 - test_scores, axis = 1)
    test_Error_std   = np.std(1 - test_scores, axis = 1)

    Scores_mean = np.mean(train_scores_mean)
    Scores_std = np.mean(train_scores_std)
    
    _, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X, y, model)
    
    plt.figure(figsize = (16,4))
    plt.subplot(1,2,1)
    ax1 = Confuse(y, y_pred, classes)
    plt.subplot(1,2,2)
    plt.fill_between(train_sizes, train_Error_mean - train_Error_std,train_Error_mean + train_Error_std, alpha = 0.1,
                     color = "r")
    plt.fill_between(train_sizes, test_Error_mean - test_Error_std, test_Error_mean + test_Error_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_Error_mean, 'o-', color = "r",label = "Training Error")
    plt.plot(train_sizes, test_Error_mean, 'o-', color = "g",label = "Cross-validation Error")
    plt.legend(loc = "best")
    plt.grid(True)
     
    return (model, Scores_mean, Scores_std )

def ApplyModel(X, y, model):
    
    model.fit(X, y)
    y_pred  = model.predict(X)

    Accuracy = round(np.median(cross_val_score(model, X, y, cv = cv)),2)*100
 
    Error   = 1 - Accuracy
    
    precision = precision_score(y_train, y_pred) * 100
    recall = recall_score(y_train, y_pred) * 100
    f1score = f1_score(y_train, y_pred) * 100
    
    return (model, y_pred, Accuracy, Error, precision, recall, f1score)  
    
def Confuse(y, y_pred, classes):
    cnf_matrix = confusion_matrix(y, y_pred)
    
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis = 1)[:, np.newaxis]
    c_train = pd.DataFrame(cnf_matrix, index = classes, columns = classes)  

    ax = sns.heatmap(c_train, annot = True, cmap = cmap, square = True, cbar = False, 
                          fmt = '.2f', annot_kws = {"size": 20})
    return(ax, c_train)

def PrintResults(model, X, y, title):
    
    model, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X, y, model)
    
    _, Score_mean, Score_std = LearningCurve(X, y, model, cv, train_size)
    Score_mean, Score_std = Score_mean*100, Score_std*100
    
    
    print('Scoring Accuracy: %.2f %%'%(Accuracy))
    print('Scoring Mean: %.2f %%'%(Score_mean))
    print('Scoring Standard Deviation: %.4f %%'%(Score_std))
    print("Precision: %.2f %%"%(precision))
    print("Recall: %.2f %%"%(recall))
    print('f1-score: %.2f %%'%(f1score))
    
    Summary = pd.DataFrame({'Model': title,
                       'Accuracy': Accuracy, 
                       'Score Mean': Score_mean, 
                       'Score St Dv': Score_std, 
                       'Precision': precision, 
                       'Recall': recall, 
                       'F1-Score': f1score}, index = [0])
    return (model, Summary)


# In[ ]:


classes = ['Dead','Survived']
cv = ShuffleSplit(n_splits = 100, test_size = 0.25, random_state = 0)
train_size = np.linspace(.1, 1.0, 15)


# Results will be printed and sumarised using the follow parameters
# 
# Confusion Matrix: 
#     -  True Negatived: passengers were correctly classified as not survived 
#     -  False positives: where wrongly classified as not survived 
#     -  True postivies: correctly classified as survived 
#     -  False negative: passengers where wrongly classified as survived 
# 
# Precision: a passengers survival correctly  
# 
# Recall: The recall tells us that it predicted the survival 
# 
# F-Score: The F-score is computed with the harmonic mean of precision and recall. 
# 
# OOB: out-of-bag samples to estimate the generalization accuracy.
# 
# 
#  Learning curves allow us to diagnose if the is overfitting or underfitting.
# 
# 
# OVERFITTING:
# When the model overfits, it means that it performs well on the training set, but not not on the validation set. Accordingly, the model is not able to generalize to unseen data. If the model is overfitting, the learning curve will present a gap between the training and validation scores. Two common solutions for overfitting are reducing the complexity of the model and/or collect more data.
#     
#     
# UNDERFITTING: 
# underfitting means that the model is not able to perform well in either training or validations sets. In those cases, the learning curves will converge to a low score value. When the model underfits, gathering more data is not helpful because the model is already not being able to learn the training data. Therefore, the best approaches for these cases are to improve the model (e.g., tuning the hyperparameters) or to improve the quality of the data (e.g., collecting a different set of features).
# 

# In[ ]:


#Logistic Regresion
model = LogisticRegression()
model, Summary_LR = PrintResults(model, X_train, y_train, 'Logistic Regression')

y_train_LR = pd.Series(model.predict(X_train), name = "LR")
y_test_LR = pd.Series(model.predict(X_test), name = "LR")


# In[ ]:


# stochastic gradient descent (SGD) learning
model = linear_model.SGDClassifier(max_iter = 200, tol = None)
model,Summary_SGD = PrintResults(model, X_train, y_train, 'SGD')
y_train_SGD = pd.Series(model.predict(X_train), name = "SGD")
y_test_SGD = pd.Series(model.predict(X_test), name = "SGD")


# In[ ]:


# Random Forest
model = RandomForestClassifier(n_estimators = 10)
model,Summary_RF = PrintResults(model, X_train,y_train, 'Random Forest')
y_train_RF = pd.Series(model.predict(X_train), name = "RF")
y_test_RF = pd.Series(model.predict(X_test), name = "RF")


# In[ ]:


#SVM
model = SVC()
model,Summary_SVM = PrintResults(model, X_train, y_train, 'SVM')
y_train_SVM = pd.Series(model.predict(X_train), name = "SVM")
y_test_SVM = pd.Series(model.predict(X_test), name = "SVM")


# In[ ]:


# KNN
model = KNeighborsClassifier(n_neighbors = 3)
model,Summary_KNN = PrintResults(model, X_train, y_train,'KNN')
y_train_KNN = pd.Series(model.predict(X_train), name = "KNN")
y_test_KNN = pd.Series(model.predict(X_test), name = "KNN")


# In[ ]:


# Gaussian Naive Bayes
model = GaussianNB()
model,Summary_GNB = PrintResults(model, X_train, y_train, "GNB")
y_train_GNB = pd.Series(model.predict(X_train), name = "GNB")
y_test_GNB = pd.Series(model.predict(X_test), name = "GNB")


# In[ ]:


# Perceptron
model = Perceptron(max_iter = 5)
model,Summary_MLP = PrintResults(model, X_train, y_train, 'MLP')
y_train_MLP = pd.Series(model.predict(X_train), name = "MLP")
y_test_MLP = pd.Series(model.predict(X_test), name = "MLP")


# In[ ]:


# Linear SVC
model = LinearSVC()
model,Summary_LSVM = PrintResults(model, X_train, y_train,"LSVM")
y_train_LSVM = pd.Series(model.predict(X_train), name = "LSVM")
y_test_LSVM = pd.Series(model.predict(X_test), name = "LSVM")


# In[ ]:


# Decision Tree
model = DecisionTreeClassifier()
model,Summary_DT = PrintResults(model, X_train, y_train, 'DT')
y_train_DT = pd.Series(model.predict(X_train), name = "DT")
y_test_DT = pd.Series(model.predict(X_test), name = "DT")


# In[ ]:


#Which is the best Model ?

Class_Results = pd.concat([Summary_LR, Summary_SGD, Summary_RF, 
                           Summary_SVM, Summary_KNN, Summary_GNB,
                           Summary_MLP, Summary_LSVM, Summary_DT], ignore_index = True)
    

Class_Results = Class_Results.sort_values(by = 'Accuracy', ascending=False)
Class_Results = Class_Results.set_index('Accuracy')
Class_Results.head(10)


# In[ ]:


# Concatenate all classifier results
y_test_Results = pd.concat([y_test_LR, y_test_SGD, y_test_RF, y_test_SVM, y_test_KNN,y_test_GNB,
                              y_test_MLP, y_test_LSVM, y_test_DT], axis=1)

y_train_Results = pd.concat([y_train_LR, y_train_SGD, y_train_RF, y_train_SVM, y_train_KNN, y_train_GNB,
                              y_train_MLP, y_train_LSVM, y_train_DT], axis=1)

plt.figure(figsize = (14, 7))
plt.subplot(1,2,1)
PlotCorr(y_train_Results)
plt.title('Training data')
plt.subplot(1,2,2)
PlotCorr(y_test_Results)
plt.title('Test data')



# ## Closer Look at Random Forest

# From the above analysis the Random Forest algorithm provides results (Score, precision, recall, f-score). so lets take a closer look at htis to see if even furth improvements can be made
# 
# Sklearn measure a features importance by looking at how much the treee nodes, that use that feature, reduce impurity on average (across all trees in the forest).

# In[ ]:


# Random Forest
model = RandomForestClassifier(n_estimators = 10, oob_score = True)
model, y_pred, Accuracy, Error, precision, recall, f1score = ApplyModel(X_train, y_train, model)
Priority = pd.DataFrame({'Feature': X_train.columns,'Importance':np.round(model.feature_importances_,3)})
Priority  = Priority .sort_values('Importance',ascending = False).set_index('Feature')


# In[ ]:


Priority.head(15)


# In[ ]:



width = 0
fig, ax = plt.subplots(figsize = (10,6))
rects = ax.barh(np.arange(len(Priority)), np.array(Priority.values), color = 'red')
ax.set_yticks(np.arange(len(Priority)) + ((width)/1))
ax.set_yticklabels(Priority.index, rotation ='horizontal')
ax.set_xlabel("Importance")
ax.set_title("Feature Importance for Random Forrest w.r.t Survival",fontsize = 14);
ax.grid(True)


# A general rule is that, the more features you have, the more likely your model will suffer from overfitting and vice versa.'Alone', 'Parch', 'CabinBool', 'SibSp' don't play a significant role in our random forest classifiers prediction process. Because of that I will drop them from the dataset and train the classifier again. 

# In[ ]:


X_train = X_train.drop("Alone", axis = 1) 
X_train = X_train.drop("CabinBool", axis = 1)
X_train = X_train.drop("Parch", axis = 1) 
X_train = X_train.drop("SibSp", axis = 1) 

X_test = X_test.drop("Alone", axis = 1) 
X_test = X_test.drop("CabinBool", axis = 1)
X_test = X_test.drop("Parch", axis = 1) 
X_test = X_test.drop("SibSp", axis = 1) 


# In[ ]:


# Random Forest again after droppingn parameters

model = RandomForestClassifier(n_estimators = 200, oob_score = True)
_, Summary_RF = PrintResults(model, X_train,y_train,'Random Forest')
y_test_RF = pd.Series(model.predict(X_test), name = "Survived")
print("oob score:", round(model.oob_score_, 4) * 100, "%")


# Our random forest model predicts the same as before. Dropping the features did not make much of a difference except to decrease the stdv
# 
# Cnnfusion Matrix: 
# 95% passengers were correctly classified as not survived (called true negatives),
# 5% where wrongly classified as not survived (false positives).
# 85% 249 where correctly classified as survived (true positives).
# 15% were passengers where wrongly classified as survived (false negatives)
# 
# Precision:
# Our model predicts 91% of the time, a passengers survival correctly  
# 
# Recall: The recall tells us that it predicted the survival of 85 % of the people who actually survived.
# 
# F-Score: The F-score is computed with the harmonic mean of precision and recall. 
# 
# OOB: out-of-bag samples to estimate the generalization accuracy.

# In[ ]:


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label = "precision", linewidth = 5)
    plt.plot(threshold, recall[:-1], "b", label = "recall", linewidth = 5)
    plt.xlabel("threshold", fontsize = 19)
    plt.legend(loc = "upper right", fontsize = 19)
    plt.ylim([0, 1])
    
    
# getting the probabilities of our predictions
y_scores = model.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
plt.figure(figsize = (10, 6))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# Above you can clearly see that the recall is falling of rapidly at a precision of around 85%. Because of that you may want to select the precision/recall tradeoff before that - maybe at around 75 %.
# 
# You are now able to choose a threshold, that gives you the best precision/recall tradeoff for your current machine learning problem. If you want for example a precision of 80%, you can easily look at the plots and see that you would need a threshold of around 0.4. Then you could train a model with exactly that threshold and would get the desired accuracy.
# 
# 

# ### Hyperparameter Tuning

# Below you can see the code of the hyperparamter tuning for the parameters criterion, min_samples_leaf, min_samples_split and n_estimators.
# 
# I put this code into a comments because it takes a long time to run it. 

# In[ ]:


#this section is commente out as it takes too long to run. the results are shown at the bottom

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import make_scorer, accuracy_score
#from sklearn.model_selection import GridSearchCV

## Choose the type of classifier. 
#model = RandomForestClassifier()

## Choose some parameter combinations to try
#parameters = {'n_estimators': [10,100,200,400,600], 
#              'max_features': ['log2', 'sqrt','auto'], 
#              'criterion': ['entropy', 'gini'],
#              'max_depth': [2, 3, 5, 10, 20], 
#              'min_samples_split': [2, 3, 5, 10, 20, 30],
#              'min_samples_leaf': [1,5,10,20,30,50]
#             }

## Type of scoring used to compare parameter combinations
#acc_scorer = make_scorer(accuracy_score)

## Run the grid search
#grid_obj = GridSearchCV(model, parameters, scoring = acc_scorer, n_jobs = 4, verbose = 1)
#grid_obj = grid_obj.fit(X_train, y_train.values.ravel())

#3 Set the clf to the best combination of parameters
#model_rf_final = grid_obj.best_estimator_

## Fit the best algorithm to the data. 
#model_rf_final.fit(X_train, y_train)


##The Following areh the results.... took several hourse to run
#RandomForestClassifier(bootstrap = True, class_weight = None, criterion = 'entropy',
#            max_depth = 10, max_features = 'log2', max_leaf_nodes = None,
#            min_impurity_decrease = 0.0, min_impurity_split = None,
#            min_samples_leaf = 1, min_samples_split = 30,
#            min_weight_fraction_leaf = 0.0, n_estimators = 100, n_jobs = 1,
#            oob_score = False, random_state = None, verbose = 0, warm_start = False)


# In[ ]:


model_rf_final = RandomForestClassifier(bootstrap = True, class_weight = None, criterion = 'entropy',
            max_depth = 10, max_features = 'log2', max_leaf_nodes = None,
            min_impurity_decrease = 0.0, min_impurity_split = None,
            min_samples_leaf = 1, min_samples_split = 30,
            min_weight_fraction_leaf = 0.0, n_estimators = 100, n_jobs = 1,
            oob_score = False, random_state = None, verbose = 0, warm_start = False)


# ### TESTING FINAL MODEL

# In[ ]:


_, Summary_LR = PrintResults(model_rf_final, X_train, y_train, 'Logistic Regression')
y_train_pred = pd.Series(model_rf_final.predict(X_train), name = 'Survived')
y_test_pred = pd.Series(model_rf_final.predict(X_test), name = "Survived")


# In[ ]:


y_test_pred = pd.Series(model_rf_final.predict(X_test), name = "Survived")


# In[ ]:


y_test_pred_final = pd.DataFrame(y_test_pred)


# In[ ]:


y_test_pred_final.head()


# # Prepare Submission Data

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": X_test_original["PassengerId"],
        "Survived": y_test_pred_final['Survived']
    })
submission.to_csv('Titanic Submission.csv', index = False)

print('Done')


# In[ ]:


submission.head()

