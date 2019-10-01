#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic Survivors
# 1 Introduction
# 
# 2 Load and check data
# - 2.1 load data
# - 2.2 Outlier detection
# - 2.3 Join train and test set
# - 2.4 Null and missing values
# 
# 3 Feature analysis
# - 3.1 Numerical values
# - 3.2 Categorical values
# 
# 4 Filling missing Values
# - 4.1 Age
# 
# 5 Feature engineering
# - 5.1 Family Size
# - 5.2 Cabin
# - 5.3 Ticket
# 
# 6 Modeling
# - 6.1 Simple modeling
#  - 6.1.1 Cross validate models
#  - 6.1.2 Hyperparamater tunning for best models
#  - 6.1.3 Plot learning curves
#  - 6.1.4 Feature importance of the tree based classifiers
# - 6.2 Ensemble modeling
#  - 6.2.1 Combining models
# - 6.3 Prediction
#  - 6.3.1 Predict and Submit results

# ## 1. Introduction
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# ### Evaluation
# The historical data has been split into two groups, a 'training set' and a 'test set'. For the training set, we provide the outcome ( 'ground truth' ) for each passenger. You will use this set to build your model to generate predictions for the test set.
# 
# For each passenger in the test set, you must predict whether or not they survived the sinking ( 0 for deceased, 1 for survived ). Your score is the percentage of passengers you correctly predict.
# 
# The Kaggle leaderboard has a public and private component. 50% of your predictions for the test set have been randomly assigned to the public leaderboard ( the same 50% for all users ). Your score on this public portion is what will appear on the leaderboard. At the end of the contest, we will reveal your score on the private 50% of the data, which will determine the final winner. This method prevents users from 'overfitting' to the leaderboard.

# In[ ]:


#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')


# ## 2. Load and check data

# ### 2.1 Load Data

# In[ ]:


# load the databse
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
IDtest = test["PassengerId"]


# In[ ]:


# see the training data
train.head()


# In[ ]:


# See the testing data
test.head()


# ### 2.2 Outliers Detection

# In[ ]:


# import required libraries
from collections import Counter

# Outlier detection 
def detect_outliers(df,n,features):
    """
    Takes a dataframe of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# Since outliers can have a dramatic effect on the prediction (espacially for regression problems), i choosed to manage them.
# 
#  The Tukey method (Tukey JW., 1977) is used to detect ouliers which defines an interquartile range comprised between the 1st and 3rd quartile of the distribution values (IQR). An outlier is a row that have a feature value outside the (IQR +- an outlier step).
# 
# Detected the outliers from the numerical values features (Age, SibSp, Sarch and Fare). Then, considered outliers as rows that have at least two outlied numerical values.

# In[ ]:


#Outliers in train
train.loc[Outliers_to_drop]


# 10 outliers are detected 3 of them have very high tickets price and 7 of them have very high SibSp values.

# In[ ]:


# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# ### 2.3 Join Train and Test Set
# Join train and test datasets in order to obtain the same number of features during categorical conversion.

# In[ ]:


train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# ### 2.4 Null and missing values

# In[ ]:


# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()


# Age and Cabin features have most missing values.
# 
# Survived missing values correspond to the join testing dataset (Survived column doesn't exist in test set and has been replace by NaN values when concatenating the train and test set)

# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# The train set only have 170 Age an 680 Cabing missing values

# In[ ]:


train.head()


# In[ ]:


# Lets see the data types of Train data
train.dtypes


# In[ ]:


# Lets take a look into the Summarie and statistics
train.describe()


# ## 3. Feature analysis

# ### 3.1 Numerical values

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, 
                fmt = ".2f", cmap = "coolwarm");


# In the heatmap we can see that fare feature have a very good coorelation with survival.

# #### Sibsp

# In[ ]:


# Explore SibSp feature vs Survived
m = sns.catplot(x="SibSp",y="Survived",data=train,kind="bar", height = 7)
m.despine(left=True)
m = m.set_ylabels("survival probability")


# The person having 0-2 siblings/spouse have good chance of survival and person have 3-5 have very less chance pf survival

# #### Parch

# In[ ]:


m = sns.catplot(x='Parch', y='Survived', data=train, kind='bar', height=7)
m.despine(left=True)
m = m.set_ylabels("survival probability")


# Here we can see that having 1-3 parents/children have higher probability of survival.
# This is also a good feature.

# #### Age

# In[ ]:


# Explore Age vs Survived
m = sns.FacetGrid(train, col='Survived')
m = m.map(sns.distplot, "Age")


# It looks like that passengers aged between 20-35 survived the most and passenger aged between 55-70 have very less survival rate.

# In[ ]:


# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 5) to babies and very young childrens with survival rate and a peak corresponsing (between 25-30) to young adults with no survival rate.

# #### Fare

# In[ ]:


dataset.Fare.isnull().sum()


# one missing value let's fill it with the median.

# In[ ]:


#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# In[ ]:


# Explore Fare distribution 
# Flexibly plot a univariate distribution of observations.
m = sns.distplot(dataset["Fare"], color="r", label="Skewness : %.2f"%(dataset["Fare"].skew()))
m = m.legend(loc="best")


# As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled.
# 
# In this case, it is better to transform it with the log function to reduce this skew.

# In[ ]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


m = sns.distplot(dataset["Fare"], color="r", label="Skewness : %.2f"%(dataset["Fare"].skew()))
m = m.legend(loc="best")


# skewness is reduced

# ### 3.2 Categorical values

# #### Sex

# In[ ]:


g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")


# Females have a high rate of Survival

# In[ ]:


# See the two groups data ratio
train[["Sex","Survived"]].groupby('Sex').mean()


# It shows clearly that Female have more chance to survive than Male.
# So Sex, will play an important role in the prediction of the survival.

# #### Pclass

# In[ ]:


# Explore Pclass vs Survived
g = sns.catplot(x="Pclass",y="Survived",data=train, kind="bar", height = 6, palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# The person having a 1st class ticket have high probability of survival as compared to people with 3rd class tickets.

# In[ ]:


# Let's explore Pclass vs Survived by Sex
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# As we can see that females of class 1 and 2 tickets have higher rate of survival than females of 3rd class ticket.
# We can also see that ticket class is an important factor for the survival rate for male and female both.

# #### Embarked(Port of Embarkation)

# In[ ]:


# we have seen some missing values in embarkation lets see
dataset.Embarked.isnull().sum()


# In[ ]:


# Lets explore the embarked to fill the missing value
dataset.Embarked.value_counts()


# In[ ]:


# S have the most number of Embarkation so lets fill the missing values with this
dataset.Embarked.fillna("S", inplace=True)


# In[ ]:


dataset.Embarked.isnull().sum()


# In[ ]:


# Explore Embarked vs Survived 
g = sns.catplot(x="Embarked", y="Survived",  data=train,
                   height=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passenger coming from Cherbourg (C) have more chance to survive.
# 
# My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).
# 
# Let's see the Pclass distribution vs Embarked

# In[ ]:


# Explore Pclass vs Embarked 
g = sns.catplot("Pclass", col="Embarked",  data=train,
                   height=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# 1st class is higher in Cherbourg (C) and the 3rd class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q).
# At this point, we can say that the first class has an higher survival rate. My hypothesis is that first class passengers were prioritised during the evacuation due to their influence.

# ## 4. Filling missing values

# In[ ]:


dataset.isnull().sum()


# ### 4.1 Age

# We can see that Age have 256 missing values. So let's explore it a bit to find the best way to fill the missing values

# In[ ]:


# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.catplot(y="Age", x="Sex", data=dataset, kind="box")
g = sns.catplot(y="Age", x="Sex", hue="Pclass", data=dataset, kind="box")
g = sns.catplot(y="Age", x="Parch", data=dataset, kind="box")
g = sns.catplot(y="Age", x="SibSp", data=dataset, kind="box")


# Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.
# 
# However, 1st class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
# 
# Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.

# In[ ]:


# First lets change the sex into categorical numbers
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[ ]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.
# 
# In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.
# 
# So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.
# 
# The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

# In[ ]:


# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med


# In[ ]:


dataset.Age.isnull().sum()


# In[ ]:


g = sns.catplot(x="Survived", y = "Age",data = train, kind="box")
g = sns.catplot(x="Survived", y = "Age",data = train, kind="violin")


# No difference between median value of age in survived and not survived subpopulation.
# 
# But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.

# ## 5. Feature Engineering

# ### 5.1 Family Size
# We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, i choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger).

# In[ ]:


# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[ ]:


g = sns.catplot(x="Fsize",y="Survived",data = dataset, kind='point')
g = g.set_ylabels("Survival Probability")


# We can see that people with large family size have less survival rate.
# Additionally let's create 4 categories according to family size.

# In[ ]:


# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


dataset.head()


# In[ ]:


# Let's again analyse family size with survival rate
g = sns.catplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


# Factorplots of family size categories show that Small and Medium families have more chance to survive than single passenger and large families.

# In[ ]:


dataset.shape


# ### 5.2 Cabin

# In[ ]:


dataset['Cabin'].head()


# In[ ]:


dataset["Cabin"].describe()


# In[ ]:


dataset["Cabin"].isnull().sum()


# In[ ]:


dataset["Cabin"][dataset["Cabin"].notnull()].head()


# In[ ]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# The first letter of the cabin indicates the Desk, i choosed to keep this information only, since it indicates the probable location of the passenger in the Titanic.

# In[ ]:


# Show the counts of observations in each categorical bin using bars.
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[ ]:


g = sns.catplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# Because of the low number of passenger that have a cabin, survival probabilities have an important standard deviation and we can't distinguish between survival probability of passengers in the different desks.
# 
# But we can see that passengers with a cabin have generally more chance to survive than passengers without (X).
# 
# It is particularly true for cabin B, C, D, E and F.

# In[ ]:


# Create Dummy variables
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# ### 5.3 Tickets

# In[ ]:


dataset['Ticket'].head()


# > It could mean that tickets sharing the same prefixes could be booked for cabins placed together. It could therefore lead to the actual placement of the cabins within the ship.
# 
# > Tickets with same prefixes may have a similar class and survival.
# 
# > So i decided to replace the Ticket feature column by the ticket prefixe. Which may be more informative.

# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[ ]:


dataset.Ticket.value_counts()


# In[ ]:


dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# In[ ]:


dataset.head()


# In[ ]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")


# In[ ]:


# Drop useless variables 
dataset.drop(labels = ["PassengerId", "Name", "Embarked"], axis = 1, inplace = True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# ## 6. Modeling

# In[ ]:


## Separate train dataset and test dataset

train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)


# ### 6.1 Simple Modeling

# #### 6.1.1 Cross Validation Models
# I compared 10 popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
# 
# - SVC
# - Decision Tree
# - AdaBoost
# - Random Forest
# - Extra Trees
# - Gradient Boosting
# - Multiple layer perceprton (neural network)
# - KNN
# - Logistic regression
# - Linear Discriminant Analysis

# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[ ]:


# Modeling step Test differents algorithms 
random_state = 7
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# I decided to choose the SVC, AdaBoost, RandomForest , ExtraTrees and the GradientBoosting classifiers for the ensemble modeling.

# #### 6.1.2 Hyperparameter tunning for best models
# > I performed a grid search optimization for AdaBoost, ExtraTrees , RandomForest, GradientBoosting and SVC classifiers.
#    I set the "n_jobs" parameter to 4 since i have 4 cpu . The computation time is clearly reduced.
#   

# In[ ]:


### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING

# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_


# In[ ]:


gsadaDTC.best_score_


# In[ ]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[ ]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[ ]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[ ]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# #### 6.1.3 Plot learning curves
# > Learning curves are a good way to see the overfitting effect on the training set and the effect of the training size on the accuracy.

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


# GradientBoosting and Adaboost classifiers tend to overfit the training set. According to the growing cross-validation curves GradientBoosting and Adaboost could perform better with more training examples.
# 
# SVC and ExtraTrees classifiers seem to better generalize the prediction since the training and cross-validation curves are close together.

# #### 6.1.4 Feature importance of tree based classifiers
# In order to see the most informative features for the prediction of passengers survival, i displayed the feature importance for the 4 tree based classifiers.

# In[ ]:


nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1


# I plot the feature importance for the 4 tree based classifiers (Adaboost, ExtraTrees, RandomForest and GradientBoosting).
# 
# We note that the four classifiers have different top features according to the relative importance. It means that their predictions are not based on the same features. Nevertheless, they share some common important features for the classification , for example 'Fare', 'Pc-3', 'Age' and 'Sex'.
# 
# We can say that:
# 
# - Pc_1, Pc_2, Pc_3 and Fare refer to the general social standing of passengers.
# 
# - Sex  refer to the gender.
# 
# - Age  refer to the age of passengers.
# 
# - Fsize, LargeF, MedF, Single refer to the size of the passenger family.
# 
# **According to the feature importance of this 4 classifiers, the prediction of the survival seems to be more associated with the Age, the Sex, the family size of the passengers more than the location in the boat.***

# In[ ]:


test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)


# The prediction seems to be quite similar for the 5 classifiers except when Adaboost is compared to the others classifiers.
# 
# The 5 classifiers give more or less the same prediction but there is some differences. Theses differences between the 5 classifier predictions are sufficient to consider an ensembling vote.

# ### 6.2 Ensemble modeling

# #### 6.2.1 Combining models
# I choosed a voting classifier to combine the predictions coming from the 5 classifiers.
# 
# I preferred to pass the argument "soft" to the voting parameter to take into account the probability of each vote.

# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# ### 6.3 Prediction

# #### 6.3.1 Predict and Submit results

# In[ ]:


test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)


# In[ ]:




