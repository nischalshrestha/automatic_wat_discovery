#!/usr/bin/env python
# coding: utf-8

# ## Titanic: Machine Learning from Disaster
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# This notebook presents a profound exploratory analysis of the dataset in order to provide understanding of the dependencies and interesting facts. We will then use these dependencies to create several models that will be able to predict whether a person survived the shipwreck or not. We will analyse these models and find out the best among them.

# ## Contents of the notebook
# 
# * ### EDA
#     - Feature Analysis 
#     - Finding trends in the data
#     
# * ### Data Cleaning and Feature Engineering
#     - Interpolating Empty/ Nan Values 
#     - Creating new features
#     - Removing redundant/ unusable features
#  
# * ### Feature Transformation
#     - Removing skeweness
#     - Binning Features
#     - Hot encoding features, etc    
#     
# * ### Predictive Modelling
#     - Finding the best Algorithm
#     - Tuning the best Algorithm
#     - Submitting results

# ## EDA (Exploratory Data Analysis)

# In[ ]:


# Importing important libraries for EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
sns.set_style('whitegrid')

from IPython.display import display
get_ipython().magic(u'matplotlib inline')

# Ignore warnings

import warnings
warnings.filterwarnings('ignore')

rand = 7


# In[ ]:


# Import the datasets and have a peek at the data

train_df = pd.read_csv("./../input/train.csv")
test_df = pd.read_csv("./../input/test.csv")
passenger_ids = test_df["PassengerId"]

display(train_df.sample(n = 3))
display(test_df.sample(n = 3))


# In[ ]:


# Are there any missing values?, any variables that need to be converted to another type?

print("Training Data Info\n")
display(train_df.info())

print("Testing Data Info\n")
display(test_df.info())


# <u>We get the following observations after looking at the data :</u>
# 
# #### Continious Features
# 
# 1. Age
# 2. Fare
# 
# We need to check for any skeweness in these continious feature distributions and scale/ bin them so that they may work well with models that use distances.
# 
# #### Categorical Feature/ Nominal Features
# 
# 1. Sex
# 2. Embarked
# 
# We would need to hot encode them.
# 
# #### Ordinal/ Interval Features
#  
# 1. Pclass
# 2. SibSp
# 3. Parch
# 
# #### Alphanumeric
# 
# 1. Name
# 2. Cabin
# 
# #### The following columns have null values and need to be dropped or interpolated :
# 
# 1. Age (Both in training and testing data)
# 2. Cabin (Both)
# 3. Embarked (Training data)
# 4. Fare (Testing data)
# 
# #### The following are features we need to transform to get any information out of or need to drop them.
# 
# 1. Name
# 2. Cabin
# 3. Ticket

# ##### Lets look at the bigger picture first. How many people actually survived the accident?

# In[ ]:


train_df['Survived'].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (4, 4), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Survival Percentage")
plt.legend(["Did not Survive", "Survived"])
plt.show()


# It is evident from the above pie chart that a lot of people didn't survive the accident. 
# The survival rate is 38.4% which is pretty low.
# A not so good model can be a model that predicts everyone to not survive the accident. 
# Lets deep dive into the data and see which kind of people did/didn't survive the accident

# ##### Also we have an Imbalanced dependent variable for our problem since there are unequal occureneces of the dependent variable survived. This could potentially lead to a flawed model. We will use stratified train_test_split and stratified cross validation split to overcome this problem. 

# ##### Pclass is the socio economic status of the people who onboarded the titanic. Did it affect the survival rate by any chance?

# In[ ]:


survival_by_class = train_df.groupby("Pclass")["Survived"].mean()
display(survival_by_class)

f, ax = plt.subplots(1, 2, figsize = (10, 4))


sns.barplot(x = "Pclass", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = 'Pclass', hue='Survived', data = train_df, ax = ax[1])
ax[1].set_xlabel("Survived Count")
ax[1].set_ylabel("Pclass")

plt.show()


# Looks like better the socio economic status, more are the chances of survival.
# Is the socio economic status by any way related to the Fare these passengers paid?

# In[ ]:


sns.barplot(x = "Pclass", y = "Fare", data = train_df)
plt.ylabel("Average Fare")
plt.show()


# Looks like better the Socio Economic Class( Since they paid a higher fare ), more the chances of survival.

# ##### Did gender had any impact on the survival of the passengers ?

# In[ ]:


survival_by_sex = train_df.groupby("Sex")["Survived"].mean()
display(survival_by_sex)

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "Sex", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Sex", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_xlabel("Survived Count")

plt.show()


# Looking at the above statistics, women had a higher chance of surviving then men.

# ##### Does the Port of Embarkation have any relation to Survival rate?

# In[ ]:


survival_by_port = train_df.groupby("Embarked")["Survived"].mean()
display(survival_by_port)

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "Embarked", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Embarked", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_ylabel("Survived Count")

plt.show()


# Looks like there is correlation here. People who embarked from Cherbourg had a higher chance of survival than the other ports. This dosen't make sense on its own. Maybe there is another factor at play here. 
# Lets study the survival rates taking into account multiple features

# ##### Lets check for Survival rate with Pclass and Sex together

# In[ ]:


pd.crosstab([train_df["Sex"], train_df["Survived"]], train_df["Pclass"], margins = True).style.background_gradient(cmap = 'summer_r')


# In[ ]:


sns.factorplot(x = "Pclass", y = "Survived", hue = "Sex", data = train_df)
plt.ylabel("Survived Fraction")
plt.show()


# The survival rate for women is better than men irrespective of class. Though the survival rate in general decreases as with decrease in socio economic status. Therefore both of these features are necessary for our analysis.

# ##### We earlier figured out that Cherbourg had the highest survival rate. Did majority of people from class 1 embarked from Cherbourg ?

# In[ ]:


pd.crosstab([train_df["Pclass"], train_df["Survived"]],train_df["Embarked"], margins = True).style.background_gradient(cmap = 'summer_r')


# In[ ]:


f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.factorplot(x = "Embarked", y = "Survived", hue = "Pclass", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Embarked", hue = "Pclass", data = train_df, ax = ax[1])
ax[1].set_xlabel("People Count")

plt.show()


# In[ ]:


sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Sex', col = 'Embarked', data = train_df)
plt.ylabel("Survived Fraction")

plt.show()


# ##### From Graph 2 (Count Plot):
# 
# Majority of class 1 people actually boarded from Southampton but the proportion of class 1 people is highest in Cherbourg and is almost 50%. Maybe thats why majority of people who boarded from Cherbourg had a decent rate of survival.
# 
# ##### From Graph 1 (Factor Plot):
# 
# People of the same class have almost the same rate of survival among the different ports(There are some exceptions :- Class 3 people for Southampton had a lower rate of survial than the same class people who boarded from other ports and Class 2 people from Q had a higher rate of survival, even higher than class 1 people(Gender may be at play here as we can see from the last graph))
# 
# ##### From Graph 3 (Multi-column-factor-plot):
# 
# Port of embarkation may not then be a good feature to determine the survival rate since once you break it down to the level of Pclass and Gender, the survival rate is almost the same irrespective of Embarkation.

# ##### Did Parch (Number of parents onboard)/ SibSp (Number of siblings/ spouses onboard) had any affect on Survival?

# In[ ]:


print(train_df.groupby("SibSp")["Survived"].mean())

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "SibSp", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "SibSp", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_xlabel("People Count")

plt.show()


# In[ ]:


print(train_df.groupby("Parch")["Survived"].mean())

f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "Parch", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "Parch", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_xlabel("People Count")

plt.show()


# More Sibling or spouses, less the chances of survival. But people who had no sibling/ spouses had an even lesser chance of survival.
# Were most of these people were males? Did these people help others before helping themselves ?
# Same is the case with parch.
# In a nutshell, Bigger families had less survival rate, but there survival rate is better then people who travelled alone.

# ## Data Cleaning and Feature Engineering  

# Lets do a little bit of feature engg and study the affects of these features on the survival rate
# 
# This step is heavily taken from 'Sina'(https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)
# and 'Ash' (https://www.kaggle.com/ash316/eda-to-prediction-dietanic)

# In[ ]:


import math

# Has a cabin ?

def hasCabin(x):
    return int((x["Cabin"] is not np.nan))

train_df["HasCabin"] = train_df.apply(hasCabin, axis = 1)
test_df["HasCabin"] = test_df.apply(hasCabin, axis = 1)

# Combine SibSp & Parch to create a new variable FamilySize

train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Use family size to create a new feature isAlone.

def isAlone(x):
    familySize = x["FamilySize"]
    return int(familySize == 1)

train_df["IsAlone"] = train_df.apply(isAlone, axis = 1)
test_df["IsAlone"] = test_df.apply(isAlone, axis = 1)

display(train_df.sample(n = 3))


# ##### Lets study the effects of these newly created features on survival rate

# ##### Lets first start with cabin.

# In[ ]:


f, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.barplot(x = "HasCabin", y = "Survived", data = train_df, ax = ax[0])
ax[0].set_ylabel("Survived Fraction")

sns.countplot(y = "HasCabin", hue = "Survived", data = train_df, ax = ax[1])
ax[1].set_ylabel("People Count")

plt.show()


# People who had cabins are more likely to survive. Are most of these people from better socio economic background?

# In[ ]:


sns.barplot(x = "Pclass", y = "HasCabin", data = train_df)
plt.ylabel("Fraction of people who had cabins")
plt.show()


# Looks like it! Maximum people who had cabins belong to class 1 which already has a higher rate of survival. Maybe these two features are correlated?

# ##### Did being alone have any impact on the survival rate of the passengers?

# In[ ]:


sns.barplot(x = "IsAlone", y = "Survived", data = train_df)
plt.ylabel("Survived Fraction")
plt.show()


# People who are alone are less likely to survive. This may go on to be a useful feature.

# ##### Extract info about titles from Name. This may in itself turn out to be an important feature.
# ##### Also, we have many missing values for age. This feature will also help in us interpolating age.

# In[ ]:


import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""    
    
train_df["Title"] = train_df["Name"].apply(get_title)
test_df["Title"] = test_df["Name"].apply(get_title)

train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')


# ##### Now that we have extracted titles out of names. Lets study the affect of these titles on the survival rate.

# In[ ]:


sns.barplot(x = "Title", y = "Survived", data = train_df)
plt.ylabel("Survived Fraction")
plt.show()


# So, men had it rough ha?

# ##### Lets now fill in the missing age values using this new title feature and study the affect age had on survival rate

# In[ ]:


train_df["Age"] = train_df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))
test_df["Age"] = test_df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15, 7))

sns.violinplot("Pclass", "Age", hue = "Survived", data = train_df, split = True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex", "Age", hue="Survived", data = train_df, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()

From the above graphs we can see that children had a higher rate of survival, irrespective of Class and Gender.
In general women had a higher survival rate. Also for men, as the age increases, the survival rate decreases.
# ## Feature Transformation 

# ##### Lets start by binning age. We could have also Scaled the age to be b/w 0 - 1.

# In[ ]:


bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)

sns.barplot(x = "AgeGroup", y = "Survived", data = train_df)


# Babies are more likely to survive than any other group.

# In[ ]:


train_df[train_df["Embarked"].isnull()]


# In[ ]:


# Since the ticket is first class and the persons survived, lets fill it with C
train_df["Embarked"] = train_df["Embarked"].fillna("C")


# In[ ]:


test_df["Fare"] = test_df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.mean()))


# ##### Lets bin Fare. We again could have used a scaler instead of binning.

# In[ ]:


train_df['FareRange'] = pd.qcut(train_df['Fare'], 4)

train_df['Farecat'] = 0
train_df.loc[train_df['Fare'] <= 7.91,'Farecat'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare']<=14.454),'Farecat'] = 1
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare']<=31),'Farecat'] = 2
train_df.loc[(train_df['Fare'] > 31) & (train_df['Fare']<=513),'Farecat'] = 3


test_df['FareRange'] = pd.qcut(test_df['Fare'], 4)

test_df['Farecat'] = 0
test_df.loc[test_df['Fare'] <= 7.91,'Farecat'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare']<=14.454),'Farecat'] = 1
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare']<=31),'Farecat'] = 2
test_df.loc[(test_df['Fare'] > 31) & (test_df['Fare']<=513),'Farecat'] = 3


# In[ ]:


train_df = train_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "FareRange", "Age"], axis=1)
test_df = test_df.drop(labels=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "FareRange", "Age"], axis=1)


# In[ ]:


# Lets also solidify our findings using the correlation matrix
corr = train_df.corr()
sns.heatmap(corr, annot = True, cmap = 'RdYlGn', linewidths = 0.2)
fig = plt.gcf()
fig.set_size_inches(12, 12)
plt.show()


# ##### Hot encode the categorical features so that they work well with certain algorithms

# In[ ]:


train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)


# In[ ]:


display(train_df.sample(n = 5))


# ## Model Application 

# In[ ]:


# Split into features and target variable
features = train_df.iloc[:, 1:]
target = train_df.iloc[:, 0]


# In[ ]:


# Create a simple decision tree to see important features

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(min_samples_split = 10)
classifier.fit(features, target)

pd.Series(classifier.feature_importances_, features.columns).sort_values(ascending = True).plot.barh(width = 0.6)
fig = plt.gcf()
fig.set_size_inches(12, 12)
plt.show()


# Looks like Title has the greatest feature importance. I am not suprised the Sex is not given any importance since title in itself would be highly correlated to sex. 

# ##### Helper Functions

# In[ ]:


# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, stratify = target, random_state = rand)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[ ]:


modelResults = pd.DataFrame(columns = ['Model_Name', 'Model', 'Params', 'Test_Score', 'CV_Mean', 'CV_STD'])

def save(grid, modelName, calFI):
    global modelResults
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    test_score = grid.score(X_test, y_test)
    
    print("Best model parameter are\n", grid.best_estimator_)
    print("Saving model {}\n".format(modelName))
    print("Mean Cross validation score is {} with a Standard deviation of {}\n".format(cv_mean, cv_std))
    print("Test Score for the model is {}\n".format(test_score))
    
    if calFI:
        pd.Series(grid.best_estimator_.feature_importances_, features.columns).sort_values(ascending = True).plot.barh(width = 0.6)
        fig = plt.gcf()
        fig.set_size_inches(12, 12)
        plt.title("{} Feature Importance".format(modelName))
        plt.show()
    
    
    cm = confusion_matrix(y_test, grid.best_estimator_.predict(X_test))
    
    cm_df = pd.DataFrame(cm, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
    sns.heatmap(cm_df, annot = True)
    plt.show()
        
    
    modelResults = modelResults.append({'Model_Name' : modelName, 'Model' : grid.best_estimator_, 'Params' : grid.best_params_, 'Test_Score' : test_score, 'CV_Mean' : cv_mean, 'CV_STD' : cv_std}
                                       , ignore_index=True)

def norm_save(model, modelName):
    global modelResults
    cv_scores = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    
    print("Saving model {}\n".format(modelName))
    print("Mean Cross validation score is {} with a Standard deviation of {}\n".format(cv_mean, cv_std))
    print("Test Score for the model is {}\n".format(test_score))
    
    cm = confusion_matrix(y_test, y_pred)
    
    cm_df = pd.DataFrame(cm, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
    sns.heatmap(cm_df, annot = True)
    plt.show()
        
    
    modelResults = modelResults.append({'Model_Name' : modelName, 'Model' : model, 'Params' : None, 'Test_Score' : test_score, 'CV_Mean' : cv_mean, 'CV_STD' : cv_std}
                                       , ignore_index=True)


# In[ ]:


def doGridSearch(classifier, params):
    cv = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, random_state = rand)
    score_fn = make_scorer(accuracy_score)
    grid = GridSearchCV(classifier, params, scoring = score_fn, cv = cv)
    grid = grid.fit(X_train, y_train)
    
    return grid


# ##### Training different models

# ##### K-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
params = {"n_neighbors" : np.arange(5, 21, 1),
         "weights" : ["uniform", "distance"]}

grid = doGridSearch(KNN, params)

save(grid, "K-Nearest Neighbor", False)


# ##### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = rand)

params = {"min_samples_split" : np.arange(5, 20, 1),
         "max_features" : np.arange(3, 25, 1)}

grid = doGridSearch(tree, params)
save(grid, "Decision Tree", False)


# ##### Ada Boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

adaBoostModel = AdaBoostClassifier(random_state = rand)
params = {"n_estimators" : [50, 75, 100, 125, 150, 200],
         "learning_rate" : [0.5, 0.75, 1, 1.25, 1.5]}

grid = doGridSearch(adaBoostModel, params)
save(grid, 'ADABoost', True)


# ##### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomForestModel = RandomForestClassifier(random_state = rand)
params = {"n_estimators" : [50, 75, 100, 125, 150, 200],
         "max_features" : [3, 4, 5, 6, 7, 8],
         "min_samples_split" : [2, 4, 6, 8, 10]}

grid = doGridSearch(randomForestModel, params)
save(grid, 'RandomForest', True)


# ##### SVM

# In[ ]:


from sklearn.svm import SVC

svc = SVC(random_state = rand)
params = {"C" : [0.1, 1, 1.1, 1.2], "gamma" : [0.01, 0.02, 0.03, 0.04, 0.08, 0.1, 1], 
          "kernel" : ["linear", "poly", "rbf", "sigmoid"]}

grid = doGridSearch(svc, params)
save(grid, 'SVC', False)


# ##### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gradientModel = GradientBoostingClassifier(random_state = 0)
params = {"learning_rate" : [0.03, 0.035, 0.04, 0.45], 
          "n_estimators" : [90, 100, 110], 
          "max_depth" : [2, 3],
          "min_samples_split" : [7, 8, 9]}

grid = doGridSearch(gradientModel, params)
save(grid, "GradientBoost", True)


# ##### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logisticModel = LogisticRegression(random_state = 0)
params = {}

grid = doGridSearch(logisticModel, params)
save(grid, 'LogisticRegression', False)


# ##### Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

naiveModel = GaussianNB()
params = {}

grid = doGridSearch(naiveModel, params)
save(grid, 'NaiveBayes', False)


# ##### Voting Classifier

# In[ ]:


from sklearn.ensemble import VotingClassifier
votingClassifier = VotingClassifier(estimators = [(modelResults.loc[2]["Model_Name"], modelResults.loc[2]["Model"]),
                                                  (modelResults.loc[4]["Model_Name"], modelResults.loc[4]["Model"]), 
                                                  (modelResults.loc[5]["Model_Name"], modelResults.loc[5]["Model"])], voting = 'hard')

votingClassifier.fit(X_train, y_train)
norm_save(votingClassifier, "Voting-Top-3")


# ##### Bagging Classifier

# In[ ]:


from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator = modelResults.loc[4]["Model"], n_estimators = 10)
bag.fit(X_train, y_train)

norm_save(bag, "Bagged KNN")


# In[ ]:


sns.barplot(x = "Model_Name", y = "Test_Score", data = modelResults.sort_values(by = "Test_Score", ascending = False))
fig = plt.gcf()
fig.set_size_inches(18, 10)
plt.ylabel("Test Score")
plt.xlabel("Model")
plt.show()


# In[ ]:


display(modelResults)


# In[ ]:


# Submission File
# get the adaboost model
submissionModel = modelResults.loc[2]["Model"]
submissions = submissionModel.predict(test_df)
submissions = pd.Series(submissions, name="Survived")

submission = pd.concat([passenger_ids, submissions],axis = 1)

submission.to_csv("titanic.csv",index=False)




# In[ ]:


print("Done")


# In[ ]:




