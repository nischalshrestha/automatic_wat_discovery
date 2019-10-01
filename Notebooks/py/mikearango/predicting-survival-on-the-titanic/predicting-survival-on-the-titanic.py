#!/usr/bin/env python
# coding: utf-8

# # Predicting Survival on the Titanic
# This notebook is my first attempt at classification and machine learning in Python. For this reason, I welcome all comments and recommendations. 
# 
# ## Competition Description 
# Kaggle describes the competition as follows: 
# 
# > "The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. . .One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. . .In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy."
# 
# ## Goal 
# Predict if a passenger survived the sinking of the Titanic or not. For each PassengerId in the test dataset , you must predict a 0 (did not survive) or 1 (survived) value for the Survived variable.
# 
# ### Metric 
# *Accuracy* - the percentage of passengers you correctly predict.
# 
# ###Submission File Format
# 
# - A .csv file with exactly *418* entries plus a header row.
# - The file should have exactly 2 columns:
#  - PassengerId (sorted in any order)
#  - Survived (binary predictions: 0 for did not survive, 1 for survived)

# #Getting Started
# 
# **Initializing Libraries**

# In[ ]:


# Data Analysis and Wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split , StratifiedKFold
import xgboost as xgb


# Configure Visualizations
get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 8 , 6


# **Loading the data into python**

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# **The variables in the training dataset are:**

# In[ ]:


print(train_df.columns.values)
train_df.head()
# train_df.tail()


# **Variable Types:**
#  
# - Categorical - Survived, Pclass, Sex, Embarked, Name, PassengerId, 
# - Quantitative - Age, Fare, SibSp, Parch
# - Mixed - Cabin, Ticket
# 
# **Variable Description:**
# 
# - Survived: survival of passenger (0 = did not survive; 1 = survived)
# - Pclass: Passenger's class
# - Name: Passenger's name
# - Sex: Passenger's sex
# - Age: Passenger's age
# - SibSp: Number of siblings/spouses the passenger had aboard
# - Parch: Number of parents/children the passenger had aboard
# - Ticket: Ticket number
# - Fare: Ticket fare
# - Cabin: Cabin number
# - Embarked: Port of embarkation
# 
# **Structure of train and test datasets**

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# **Variables with missing data**

# In[ ]:


print('In the train dataset:')
print('Age has ' + str(891 - 714) + ' missing values') 
print('Cabin has ' + str(891 - 204) + ' missing values') 
print('Embarked has ' + str(891 - 889) + ' missing values') 
print('_'*40)
print('In the test dataset:')
print('Age has ' + str(418 - 332) + ' missing values') 
print('Fare has ' + str(418 - 417) + ' missing values')
print('Cabin has ' + str(418 - 91) + ' missing values')


# ##Defining Functions to be Used in Exploratory Data Analysis

# In[ ]:


def plotHist(df, x, col, row = None, bins = 20): 
    grid = sns.FacetGrid(df, col = col, row = row)
    grid.map(plt.hist, x, bins = bins)
    
def plotDistribution(df, var, target, **kwargs):
    rowVar = kwargs.get('rowVar', None)
    colVar = kwargs.get('colVar' , None)
    grid = sns.FacetGrid(df, hue = target, aspect = 4, row = rowVar, col = colVar)
    grid.map(sns.kdeplot, var, shade = True)
    grid.set(xlim = (0, df[var].max()))
    grid.add_legend()

def plotCategorical(df, cat, target, **kwargs):
    rowVar = kwargs.get('rowVar', None)
    colVar = kwargs.get('colVar', None)
    grid = sns.FacetGrid(df, row = rowVar, col = colVar)
    grid.map(sns.barplot, cat, target)
    grid.add_legend()

def plotCorrelation(df):
    corr = df.corr()
    heat, ax = plt.subplots(figsize = (12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    heat = sns.heatmap(
        corr, 
        cmap = cmap,
        square = True, 
        cbar_kws = {'shrink': .9}, 
        ax = ax, 
        annot = True, 
        annot_kws = {'fontsize': 12})

def describeMore(df):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by = 'Levels', inplace = True)
    return levels

def analyzeByPivot(df, grouping_var, sorting_var):
    place = df[[grouping_var, sorting_var]].groupby([grouping_var], 
               as_index = False).mean().sort_values(by = sorting_var, ascending = False)
    return place


# #Exploratory Data Analysis (EDA)

# **General info on the variables**

# In[ ]:


describeMore(train_df)


# **Summary statistics of the variables**

# In[ ]:


train_df.describe(percentiles = [.01, .05, .10, .25, .50, .75, .90, .95, .99])


# In[ ]:


train_df.describe(include=['O'])


# **Quantitative Variables:**
# 
# - There were passengers of 88 different ages in the training dataset. The average age was about 29.7, 90% of passengers were 50 years old or younger, 99% were less than 66 years old and at least 25% of passengers were less than 21 years old.  
# - At least 25% of passengers in the training data had a sibling or spouse on board 
# - At least 75% of passengers did not have parents or children on board. 
# - The distribution of ticket fares is heavily skewed to the right as evidenced by a mean of about 32, but a median of about 14.5. About 75% of people paid a fare of 31 or less, while others paid up to 512. 
# 
# **Categorical Variables:**
# 
# - All passengers in the training data have a unique PassengerId
# - Approximately 38 % of passengers in the training dataset survived in comparison to the true population survival rate of about 32.5% (specified in the description).  
# - There are 3 levels of Passenger classes and at least 50% of passenger had a Pclass of 3.  
# - 577 of the 891 passengers in the training set are males (about 65%). 

# ## Correlation of Variables

# In[ ]:


plotCorrelation(train_df)


# The variables that are more strongly correlated (positively or negatively) are represented by darker colors in the heat map. 
# 
# **Analyze by pivoting variables with classes/levels against each other**

# In[ ]:


analyzeByPivot(train_df, 'Pclass', 'Survived')


# It appears passengers in Pclass 1 survived at much higher rates in the training data than the other classes. 

# In[ ]:


analyzeByPivot(train_df, 'Sex', 'Survived')


# It appears females survived at much higher rates than males in the training data.

# In[ ]:


analyzeByPivot(train_df, 'SibSp', 'Survived')


# Passengers traveling with 0-2 SibSp had the highest survival rates in the training data. 

# In[ ]:


analyzeByPivot(train_df, 'Parch', 'Survived')


# Passengers traveling with 0-3 Parch had the highest survival rates in the training data. But those traveling with 1, 2, or 3 Parch had about a 15% higher survival rate than those with 0 parents or children on board. 

# ## Data Visualization

# **Mean survival rate of passengers by Pclass**

# In[ ]:


plotCategorical(train_df, 'Pclass', 'Survived')


# **Distribution of passenger age by survival status**

# In[ ]:


plotHist(train_df, 'Age', 'Survived')


# **Mean survival rate of passengers by SibSp**

# In[ ]:


plotCategorical(train_df, 'SibSp', 'Survived')


# **Mean survival rate of passengers by Parch**

# In[ ]:


plotCategorical(train_df, 'Parch', 'Survived')


# **Distribution of passenger age by Pclass and survival status**

# In[ ]:


plotHist(train_df, 'Age', 'Survived', 'Pclass', bins = 20)


# **Distribution of passenger age by sex and survival status**

# In[ ]:


plotHist(train_df, 'Age', 'Survived', row = 'Sex')
plotDistribution(train_df, var = 'Age', target = 'Survived', rowVar = 'Sex')


# **Mean Survival by Pclass of Males and Females**

# In[ ]:


sns.pointplot(x = "Pclass", y = "Survived", hue = "Sex", data = train_df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"])


# **Mean survival rate of passengers by Pclass, point of embarkation, and sex**

# In[ ]:


grid = sns.FacetGrid(train_df, col='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# **Price of fare by sex, point of embarkation, and survival**

# In[ ]:


plotCategorical(train_df, 'Sex', 'Fare', rowVar = 'Embarked', colVar = 'Survived')

# grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', size = 2.2, aspect = 1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', ci = None)
# grid.add_legend()


# # Data Cleaning
# This section will consist of dropping extraneous variables/missing observations , imputing missing values where possible, manipulating old variables to make new ones,  and pre-processing. 

# ## Dropping Ticket and Cabin Variables
# 
# I am going to drop Ticket and Cabin. The ticket variable is alpha-numeric and 681 of the 891 passengers have a unique ticket. I am going to assume that some couples/families shared tickets since a ticket ID can be the same and there is only 1 missing fare in both the train and test datasets. Moreover, I am dropping Cabin because of the 891 observations in the training dataset there are only 204 observations where a cabin is listed and 147 of them are unique. 

# In[ ]:


train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After dropping Ticket and Cabin -- ", 
'Train:', train_df.shape, ', Test:', test_df.shape)


# ##Creating Title Variable
# 
# The following code uses a regular expression to extract a title ending in a period from the name variable for both training and test dataset.

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# **Now let's look at the new title column for males and females:**

# In[ ]:


pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


analyzeByPivot(train_df, 'Title', 'Survived')


# Since there are a lot of similar titles and a few titles that fall into certain categories, let's group them accordingly. 

# In[ ]:


for dataset in combine:
    # Standardize officer titles 
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major', 'Dr'], 'Officer')
    # Standardize royal titles
    dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don', 'Sir', 'Countess', 'Dona', 'Lady'],
                                                'Royalty')
    # Mlle stands for Mademoiselle which is French for Miss
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    # Standardize Ms to Miss
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    # Mme stands for Madame which is French for Mrs
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# Note: I left Rev on its own based on the assumption they will let others take their spot on the lifeboat
# 
# **Mean survival rate by passenger title**

# In[ ]:


analyzeByPivot(train_df, 'Title', 'Survived')


# Now that we have each passenger's title, we can drop Name and PassengerId from the training dataset and Name from the test dataset. 

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ###Converting Title to Categorical

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty": 5, "Officer": 6, "Rev": 7}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # We only checked train_df for titles, so we don't know if there are other non-mapped titles in 
    # test_df. For this reason we fill NA's with 0 and reserve that as an "Other" level.
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()


# ##Converting Sex to Binary

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
train_df.head()


# ## Imputing Missing Age Values
# 
# In accordance with Seghal's work on this topic, I impute the missing age values by iterating over Sex and Pclass. The goal is is to find the mean age (after excluding null values) for each combination of sex and Pclass. Recall that Sex has 2 levels (male, female) and Pclass has 3 levels (1, 2, 3) so we will have 6 combinations. 

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# I will iterate over Sex and Pclass and populate the guess_ages array with the mean age of each combination at the end of each iteration. 

# In[ ]:


# For each dataset
for dataset in combine:
    # For each sex
    for i in range(0, 2):
        # For each Pclass
        for j in range(0, 3):
            # Populate guess_age dataframe with each combo of Sex and Pclass for each iteration
            # and drop all NA values
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            # Set age_guess equal to the median age of each Sex and Pclass combo
            age_guess = guess_df.median()
            # Populate guess_ages array with each age_guess 
            guess_ages[i,j] = age_guess          
    # For each sex 
    for i in range(0, 2):
        # For each Pclass
        for j in range(0, 3):
            # For each combo of Sex and Pclass, find the rows where Age is null and impute with 
            # the corresponding age guess from the guess_ages array
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]
    # Convert age from float to int 
    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# **Checking to see the distribution of age now that it has no missing values**

# In[ ]:


plotDistribution(train_df, var = 'Age', target = None)
train_df.Age.describe()


# It seems that age goes from about 0 to 80 and the median age is around 26. 
# 
# ##Creating AgeBin Variable
# The AgeBin variable will separate ages into 5 separate bins 

# In[ ]:


# Cuts Age into 5 bins
train_df['AgeBin'] = pd.cut(train_df['Age'], 5)
analyzeByPivot(train_df, 'AgeBin', 'Survived')


# **Now we use the AgeBin variable to encode the Age variable as a categorical variable**

# In[ ]:


combine = [train_df, test_df]

for dataset in combine: 
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df.head()


# ###Dropping AgeBin

# In[ ]:


train_df = train_df.drop(['AgeBin'], axis=1)
combine = [train_df, test_df]
train_df.head()


# ##Creating GroupSize Variable
# 
# Note we have to add 1 to this number to account for the passenger. 

# In[ ]:


for dataset in combine:
    dataset['GroupSize'] = dataset['SibSp'] + dataset['Parch'] + 1
analyzeByPivot(train_df, 'GroupSize', 'Survived')


# **Next, I encode GroupSize as alone (GroupSize = 0), small (GroupSize = 1), medium (GroupSize = 2), and large (GroupSize = 3).**

# In[ ]:


for dataset in combine: 
    # GroupSize = 1 --> Alone
    dataset.loc[dataset['GroupSize'] == 1, 'GroupSize'] = 0
    # GroupSize = 2, 3, or 4 --> Small   
    dataset.loc[(dataset['GroupSize'] > 1) & (dataset['GroupSize'] <= 4), 'GroupSize'] = 1
    # GroupSize = 5, 6, or 7 --> Medium   
    dataset.loc[(dataset['GroupSize'] > 4) & (dataset['GroupSize'] <= 7), 'GroupSize'] = 2
    # GroupSize = 8+ --> Large  
    dataset.loc[dataset['GroupSize'] > 7, 'GroupSize'] = 3
analyzeByPivot(train_df, 'GroupSize', 'Survived')


# In[ ]:


train_df.head()


# ##Dropping SibSp and Parch

# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

train_df.head()


# ##Imputing Missing Values in Embarked
# 
# Before, we noted that Embarked is missing 2 values in the train dataset and will use the mode to impute missing values. 

# In[ ]:


# Drop NA values and report to mode of Embark
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    # Impute missing values with "S"
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
analyzeByPivot(train_df, "Embarked", "Survived")


# ##Converting Embarked to Categorical

# In[ ]:


embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

train_df.head()


# ##Imputing Missing Values in Fare Variable (test_df)
# 
# The code below specifies that we want to impute the single missing value with the median fare amount in the test dataset (inplace argument specifies that we want to update the Fare variable). 

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# ##Creating FareBin Variable
# 
# Below is the distribution of Fare by percentiles. I eventually chose to break Fare up into quintiles.

# In[ ]:


train_df.Fare.describe(percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9])


# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 5)
analyzeByPivot(train_df, 'FareBand', 'Survived')


# ###Converting Fare to Categorical

# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]


# ##Cleaned Datasets

# In[ ]:


train_df.head(10)


# In[ ]:


test_df.head(10)


# **Correlation matrix of cleaned training dataset**

# In[ ]:


plotCorrelation(train_df)


# #Preparing Data for Modeling
# 
# Drop predictor variable (Survived) from the list of independent variables (X). 

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape


# #Modeling
# 
# ##Logistic Regression

# In[ ]:


logreg = LogisticRegression() # Define Logistic Regression model
logreg.fit(X_train, Y_train) # Run the model using the training data
Y_pred = logreg.predict(X_test) # Use the model to predict Y values given the test data
acc_log = round(logreg.score(X_train, Y_train) * 100, 2) # Accuracy 
print("Accuracy:", acc_log)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = logreg, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# **Correlation Coefficients**

# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Variable']
coeff_df["Correlation Coeff"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by = 'Correlation Coeff', ascending=False)


# ##Support Vector Classifier (SVM)

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("Accuracy:", acc_svc)


# ###Linear SVC

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("Accuracy:", acc_linear_svc)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = linear_svc, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# ##k-Nearest Neighbors (k-NN)

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_knn)


# ##Naive Bayes

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_gaussian)


# ##Perceptron

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_perceptron)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = perceptron, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# ##Stochastic Gradient Descent

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_sgd)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = sgd, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# ##Decision Tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_decision_tree)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = decision_tree, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# In[ ]:


def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values([ 'Importance'], ascending = True)
    imp[:10].plot(kind = 'barh')

plot_model_var_imp(decision_tree, X_train, Y_train)


# ##Random Forest
# 

# In[ ]:


random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_random_forest)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = random_forest, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# ##Gradient Boosting Classifer

# In[ ]:


grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train, Y_train)
Y_pred = grad_boost.predict(X_test)
grad_boost.score(X_train, Y_train)
acc_grad_boost = round(grad_boost.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_grad_boost)


# **Optimal Number of Variables**

# In[ ]:


rfecv = RFECV(estimator = linear_svc, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')
rfecv.fit(X_train,Y_train)
print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))
print("Optimal number of variables: %d" % rfecv.n_features_)

# Plot number of variables VS. cross-validation scores
plt.figure()
plt.xlabel("Number of Variables Used")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)
plt.show()


# ##XGBoost

# In[ ]:


xgboost = xgb.XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate = 0.05)
xgboost.fit(X_train, Y_train)
Y_pred = xgboost.predict(X_test)
xgboost.score(X_train, Y_train)
acc_xgboost = round(xgboost.score(X_train, Y_train) * 100, 2)
print("Accuracy: ", acc_xgboost)


# #Ranking Models

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Classifier', 'Linear SVC', 'k-NN', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree', "Gradient Boosting Classifier", "XGBoost"],
    'Score': [acc_log, acc_svc, acc_linear_svc, acc_knn, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree, acc_grad_boost, acc_xgboost]})
models.sort_values(by = 'Score', ascending = False)


# #Submission

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_rf})
submission.to_csv('submission.csv', index = False)
submission.head()


# ### Acknowledgements
# I would like to credit a number of people as their notebooks on this topic make up a large portion of the code and analysis used to produce this notebook: 
# 
# - [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions) - Manav Seghal
# - [Scikit-Learn ML from Start to Finish](https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish) - Jeff Delaney
# - [An Interactive Data Science Tutorial](https://www.kaggle.com/helgejo/titanic/an-interactive-data-science-tutorial) - Helge Bjorland
# - [XGBoost example (Python)](https://www.kaggle.com/datacanary/titanic/xgboost-example-python/code) - DataCanary
# 
