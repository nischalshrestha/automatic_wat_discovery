#!/usr/bin/env python
# coding: utf-8

# Predicting Surviving the Sinking of the Titanic
# -----------------------------------------------
# 
#  
# This represents my first attempt at training up some classifiers for the titanic dataset.

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set_style("whitegrid")

# machine learning
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# In[ ]:


# get titanic & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
combine = [train_df, test_df]


# # Data exploration #
# 
# First get some summary statistics about the datasets.

# In[ ]:


# view column labels
print(train_df.columns.values)


# In[ ]:


# preview the data
train_df.head()


# Now transpose the first few rows in order to see all attributes more easily as row labels.

# In[ ]:


train_df.head(3).T


# In[ ]:


# missing values, data types
train_df.info()
print('-'*40)
test_df.info()


# The above info shows that columns (from training data) with missing/empty values are:
# 
#  - Age (177 missing values)
#  - Cabin (687 missing values)
#  - Embarked (2 missing values)

# In[ ]:


# describe numeric columns
train_df.describe()


# In the training dataset there are 891 passengers with an overall survival rate of 38.4%.
# The oldest person is 80 years and the youngest is 5 months (0.42*12). The average fare is 32.20 dollars but the median fare is 14.45. This suggests outliers at the upper end of the fare, and indeed the maximum fare is $512.33.

# In[ ]:


# describe categorical columns
train_df.describe(include=['O'])


# In[ ]:


# just for fun, examine the records of ten year olds (there are only two) 
train_df[train_df.Age == 10].stack()


# # Detailed data investigation #
# 
# A closer look at each of the attributes (columns) and their relationship to survival.

# ##Sex##
# 
# Sex is a *nominal* attribute with two categories (i.e. it is dichotomous). Let's plot some counts and survival rates by sex. Note that survival values are 0/1, thus rates can be be calculated simply via the mean survive value.

# In[ ]:


# count passengers by sex
plt.subplot(211) # 3 digit convenience notation for arguments (last digit represents plot number)
sns.countplot(x='Sex', data=train_df, palette='Greens_d')

# survival rate by sex
# note that barplot plots mean() on y by default
plt.subplot(212)
sns.barplot(x='Sex', y='Survived', data=train_df, palette='Greens_d') 


# **Observations:**
# 
#  - Many more males than females
#  - Survival rate of females much greater than males
# 
# Let's get the actual numbers below using pandas.

# In[ ]:


# count passengers by sex
train_df.groupby('Sex').size()


# In[ ]:


# survival rates by sex
train_df.groupby(['Sex'])['Survived'].mean().sort_values()


# Thus, 18.9% of males (from the training set) survived compared to 74.2% of females.

# ##Passenger class##
# 
# Passenger class (Pclass) is an *ordinal* attribute with three categories, 1, 2 and 3. The three categories have an order (representing socioeconomic status) but although the categories are given numeric labels, this attribute *is not* numeric! To see this, consider that 3rd class = 1st + 2nd class is a nonsense. This will be important later when we construct features. Again, let's plot some counts and survival rates.

# In[ ]:


# size of groups in passenger class
plt.subplots(figsize=(8,6))
plt.subplot(211) 
sns.countplot(x='Pclass', data=train_df, palette='Purples_d') # _d = dark palette

# survival rate by sex
plt.subplot(212)
sns.barplot(x='Pclass', y='Survived', data=train_df, palette='Purples_d') 


# **Observations:**
# 
#  - Three classes
#  - Most passengers travelled by 3rd class (more than half; see below)
#  - Survival rate increases with class
# 
# Again, let's get the actual numbers below using pandas.

# In[ ]:


# count passengers by passenger class
train_df.groupby(['Pclass']).size()


# In[ ]:


# survival rates by passenger class
train_df.groupby(['Pclass'])['Survived'].mean().sort_values(ascending=False)


# ##Age##
# 
# Age is a *ratio* attribute (it is properly numeric, see [Types of data measurement scales][1]). Ages < 1 indicate age in months.
# 
# 
#   [1]: http://www.mymarketresearchmethods.com/types-of-data-nominal-ordinal-interval-ratio/

# In[ ]:


# count the number of passengers for first 25 ages
train_df.groupby('Age').size().head(25)

# another way to do the above
#train_df['Age'].value_counts().sort_index().head(25) 


# In[ ]:


# convert ages to ints
age = train_df[['Age','Survived']].dropna() # returns a copy with blanks removed
age['Age'] = age['Age'].astype(int) # floors floats

# count passengers by age (smoothed via gaussian kernels)
plt.subplots(figsize=(18,6))
plt.subplot(311)
sns.kdeplot(age['Age'], shade=True, cut=0)

# count passengers by age (no smoothing)
plt.subplot(312)
sns.countplot(x='Age', data=age, palette='GnBu_d')

# survival rates by age
plt.subplot(313)
sns.barplot(x='Age', y='Survived', data=age, ci=None, palette='Oranges_d') # takes mean by default


# Observations:
# 
#  - Under 16s tend to have the highest survival rates
#  - Very high survival rates at 53, 63 and 80
#  - Survival of over 16s is fairly noisy. Possible that survival might increase with age.

# ## Survival by age group and sex ##
# 
# Now let's look at survival by age groups *and* sex to see if any patterns become clearer.

# In[ ]:


# bin age into groups
train_df['AgeGroup'] = pd.cut(train_df['Age'],[0,4,15,25,35,45,65,100])
test_df['AgeGroup'] = pd.cut(test_df['Age'],[0,4,15,25,35,45,65,100])

# survival by age group
train_df.groupby('AgeGroup')['Survived'].mean()


# In[ ]:


# survival by age group and sex
train_df[['Survived','AgeGroup', 'Sex']].groupby(['Sex', 'AgeGroup']).mean()


# In[ ]:


# count passengers by age group and sex
sns.factorplot(x='AgeGroup', col='Sex', data=train_df, kind='count')

# survival by age group and sex
sns.factorplot(x='AgeGroup', y='Survived', col='Sex', data=train_df, kind='bar')


# The relationship between survival and age group looks very different for males and females:
# 
# - Males: survival rates increase *inversely* with age for (0, 25] and (25, 100). That is, younger boys fare better than older boys and younger men survive more than older men.  
# - Females: no obvious relationship between surviving and age. In particular, girls and baby girls do not fare better than women; in fact, girls (4, 15] have the *lowest* survival rates of females. 
# 
# A feature space containing (child, man, woman) would do a decent job of representing this relationship to survivability. 
# 
# Non-linear classifiers (e.g. decision trees, multi-layer nn, nearest neighbour) applied to both sex and age group might do even better because of the noticeable relationship between survivability and age group for males.  

# ## Family Size##
# 
# We create a new feature, FamilySize, that sums Parch and SibSp. This will enable us to drop Parch and SibSp from the datasets.

# In[ ]:


# calculate family size
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# count passengers by age group and sex
plt.subplot(211)
sns.countplot(x='FamilySize', data=train_df)

# survival by age group and sex
plt.subplot(212)
sns.barplot(x='FamilySize', y='Survived', data=train_df)


# Survival increases with family size, until families of size 4. Family sizes of 5 and above have reduced survival.

# Deck
# ----
# 
# Cabin might be conceivably be related to survival, but unfortunately most values are missing. Nevertheless, by way of an exercise, we will extract the feature, Deck, from cabin by taking the first character of the label and analyze survival rates by deck.

# In[ ]:


# deck is the first letter of cabin
train_df['Deck'] = train_df['Cabin'].dropna().apply(lambda x: str(x)[0])
train_df[['PassengerId','Name', 'Cabin', 'Deck']].head(2).T


# In[ ]:


# count passengers by the deck their cabin is on
plt.subplots(figsize=(8,6))
plt.subplot(211) 
sns.countplot(x='Deck', data=train_df)

# survival rate by deck
plt.subplot(212)
sns.barplot(x='Deck', y='Survived', data=train_df) 


# ## Other attributes ##
# For this first attempt, I am ignoring the attributes below as they seem unlikely to be related to survival:
# 
#  - PassengerId
#  - Name (however, extracting titles from names might be informative)
#  - Ticket
#  - Fare (could be related to socioeconomic status but we already have a class attribute)
#  - Embarked

# # Data wrangling - Age group#
# 
# Fill missing age group values. We don't want to drop them as this would lose many rows. Instead, we will randomly generate age groups according to the frequency that they occur in the data. We will calculate the frequency separately for males and females.

# In[ ]:


# number of males/females without an age
def get_na(dataset):
    na_males = dataset[dataset.Sex == 'male'].loc[:,'AgeGroup'].isnull().sum()
    na_females = dataset[dataset.Sex == 'female'].loc[:,'AgeGroup'].isnull().sum()
    return {'male': na_males, 'female': na_females}

# number of males and females by age group
def get_counts(dataset):
    return dataset.groupby(['Sex', 'AgeGroup']).size()

# randomly generate a list of age groups based on age group frequency (for each sex separately) 
def generate_age_groups(num, freq):
    age_groups = {}
    for sex in ['male','female']:
        relfreq = freq[sex] / freq[sex].sum()
        age_groups[sex] = np.random.choice(freq[sex].index, size=num[sex], replace=True, p=relfreq)    
    return age_groups

# insert the new age group values
def insert_age_group_values(dataset, age_groups):
    for sex in ['male','female']:
        tmp = pd.DataFrame(dataset[(dataset.Sex == sex) & dataset.Age.isnull()]) # filter on sex and null ages 
        tmp['AgeGroup'] = age_groups[sex] # index age group values
        dataset = dataset.combine_first(tmp) # uses tmp to fill holes
    return dataset

# fill holes for train_df
na = get_na(train_df)
counts = get_counts(train_df)
counts['female']
age_groups = generate_age_groups(na, counts)
age_groups['female']
train_df = insert_age_group_values(train_df, age_groups)
train_df.info() # check all nulls have been filled    
print('-'*40)

# repeat for test_df
na = get_na(test_df)
counts = get_counts(train_df) # reuse the frequencies taken over the training data as it is larger
age_groups = generate_age_groups(na, counts)
test_df = insert_age_group_values(test_df, age_groups)
test_df.info() # check all nulls have been filled     


# # Feature engineering #
# 
# Now that we've explored the data let's create some features:
# 
#  - **Sex:** Convert to a single binary feature, Female. No need to create a feature for Male, that would be redundant.
#  - **Pclass:** Convert to two binary features, PClass_1 and PClass_2. Similar to Male above, having a PClass_3 would be redundant.
#  - **Age group:** The age attribute binned using separators [0, 4, 15, 25, 35, 45, 65, 100]. Convert to a number of binary features, one for each age group.
#  - **Family size:** The sum of SibSp and Parch plus 1.

# In[ ]:


# Sex -> Female

# training set
dummy = pd.get_dummies(train_df['Sex'])
dummy.columns = ['Female','Male']
train_df = train_df.join(dummy['Female'])

# test set
dummy = pd.get_dummies(test_df['Sex'])
dummy.columns = ['Female','Male']
test_df = test_df.join(dummy['Female'])

train_df[['Name', 'Sex', 'Female']].head(2).T
#train_df.columns


# In[ ]:


# Pclass -> PClass_1, PClass_2

# training set
dummy  = pd.get_dummies(train_df['Pclass'])
dummy.columns = ['PClass_1','PClass_2','PClass_3']
train_df = train_df.join(dummy[['PClass_1', 'PClass_2']])

# test set
dummy  = pd.get_dummies(test_df['Pclass'])
dummy.columns = ['PClass_1','PClass_2','PClass_3']
test_df = test_df.join(dummy[['PClass_1', 'PClass_2']])

train_df[['Name', 'Pclass', 'PClass_1', 'PClass_2']].head(2).T
#train_df.columns


# In[ ]:


# AgeGroup -> binary features

# training set
dummy  = pd.get_dummies(train_df['AgeGroup'])
dummy.columns = ['Ages_4','Ages_15','Ages_25','Ages_35','Ages_45','Ages_65','Ages_100']
train_df = train_df.join(dummy)

# test set
dummy  = pd.get_dummies(test_df['AgeGroup'])
dummy.columns = ['Ages_4','Ages_15','Ages_25','Ages_35','Ages_45','Ages_65','Ages_100']
test_df = test_df.join(dummy)


# ## Experimental features ##
# Some additional features to explore.

# In[ ]:


# Fare

# there is a single missing "Fare" value
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# convert from float to int (floor)
#train_df['Fare'] = train_df['Fare'].astype(int)
#test_df['Fare'] = test_df['Fare'].astype(int)


# In[ ]:


# Embarked -> PortC, PortQ

# Fill missing values with the most occurred value
print(train_df.groupby('Embarked').size().sort_values())
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# training set
dummy = pd.get_dummies(train_df['Embarked'])
#dummy.columns
dummy.columns = ['Port_C','Port_Q','Port_S']
#train_df = train_df.join(dummy[['Port_C','Port_Q']])

# test set
dummy  = pd.get_dummies(test_df['Embarked'])
dummy.columns = ['Port_C','Port_Q','Port_S']
#test_df = test_df.join(dummy[['Port_C','Port_Q']])


# ## Dropping attributes ##
# Drop unused attributes to avoid detecting spurious relationships.

# In[ ]:


# drop the attributes that will be unused
train_df.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 
                   'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare', 
                   'Embarked', 'Deck', 'AgeGroup'], axis=1, inplace=True)

test_df.drop(['Pclass', 'Name', 'Sex', 'Age', 
                   'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare',
                   'Embarked', 'AgeGroup'], axis=1, inplace=True)

train_df.head(10).T


# The sample above shows the features and their values for the first ten training examples.

# # Modeling #
# 
# Our task is a binary classification problem: we want to formulate a relationship that predicts an output (Survived or not) from engineered features (Sex, Age group, Family size...). This is type of learning is supervised learning, since a model will be trained on a dataset containing pairs of inputs and outputs. 
# 
# Suitable methods for performing classification include:
# 
#  - Logistic Regression*
#  - Perceptron*
#  - Support Vector Machines (SVMs)* 
#  - Naive Bayes classifier* 
#  - KNN or k-Nearest Neighbors
#  - Decision Tree
#  - Random Forrest
#  - Artificial neural network
#  - Relevance Vector Machine
# 
# The methods marked * either discover linear classification boundaries (logistic regression, perceptron, and SVMs if using linear kernels) or assume no relationship between features (naive bayes) and thus are not expected to perform as well (see the section above on the relationship between survival, age group and sex).

# ## Training data ##
# Let's use cross validation to perform the evaluation. This method will give a reasonable indication of predictive accuracy as evaluation will take place on data that is not seen during training. The package **`sklearn.model_selection`** includes support for cross validation.

# In[ ]:


# split the datasets into matched input and ouput pairs
X_train = train_df.drop("Survived", axis=1) # X = inputs
Y_train = train_df["Survived"] # Y = outputs
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# Model fitting
# ----------
# (Some of this section is based on [this titanic tutorial][1].)
# 
# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. See [Logistic regression on Wikipedia][2].
# 
# Note the confidence score generated by the model based on our training dataset.
# 
# 
#   [1]: https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
#   [2]: https://en.wikipedia.org/wiki/Logistic_regression

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
scores = cross_val_score(logreg, X_train, Y_train, cv=10)
acc_log = round(scores.mean() * 100, 2)
acc_log
#Y_pred = logreg.predict(X_test)


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# In[ ]:


logreg.fit(X_train, Y_train)
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
scores = cross_val_score(gaussian, X_train, Y_train, cv=10)
acc_gaussian = round(scores.mean() * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron (a single layer neural net)

perceptron = Perceptron()
scores = cross_val_score(perceptron, X_train, Y_train, cv=10)
acc_perceptron = round(scores.mean() * 100, 2)
acc_perceptron


# In[ ]:


# Neural Network (a multi layer neural net)

neural_net = MLPClassifier()
scores = cross_val_score(neural_net, X_train, Y_train, cv=10)
acc_neural_net = round(scores.mean() * 100, 2)
acc_neural_net


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
scores = cross_val_score(sgd, X_train, Y_train, cv=10)
acc_sgd = round(scores.mean() * 100, 2)
acc_sgd


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
scores = cross_val_score(linear_svc, X_train, Y_train, cv=10)
acc_linear_svc = round(scores.mean() * 100, 2)
acc_linear_svc


# In[ ]:


# Support Vector Machine

svc = SVC() # uses a rbf kernel by default (i.e. can discover non-linear boundaries)
scores = cross_val_score(svc, X_train, Y_train, cv=10)
acc_svc = round(scores.mean() * 100, 2)
acc_svc


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
scores = cross_val_score(decision_tree, X_train, Y_train, cv=10)
acc_decision_tree = round(scores.mean() * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest - an ensemble model

random_forest = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(random_forest, X_train, Y_train, cv=10)
acc_random_forest = round(scores.mean() * 100, 2)
acc_random_forest


# In[ ]:


# AdaBoost - an ensemble method

ada_boost = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(ada_boost, X_train, Y_train, cv=10)
acc_ada_boost = round(scores.mean() * 100, 2)
acc_ada_boost


# In[ ]:


# k-Nearest Neighbors - a non-parametric method

knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn, X_train, Y_train, cv=10)
acc_knn = round(scores.mean() * 100, 2)
acc_knn


# Model evaluation
# ----------------
# 
# We now rank the models and choose a high performing one for our problem. The Support Vector Machine consistently tops the chart. 
# 
# Decision Tree and Random Forest also both score high, but we prefer Random Forest as it avoids overfitting to the training set better than a decision tree and is therefore likely to perform better on the test dataset.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machine', 'kNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Descent', 'Linear SVC', 
              'Decision Tree', 'AdaBoost', 'Neural Network'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, 
              acc_ada_boost, acc_neural_net]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


# using random forest for submission
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_submission_1.csv', index=False)
#pd.set_option('display.max_rows', len(submission))
#submission


# Use cross validation to assess predictive accuracy
# --------------------------------------------------
# 
# We can easily improve the above scores by evaluating on the training data (compare the random forest scores above and below). However, scores produced like this are not truly indicative of predictive accuracy and should be avoided. To see why, consider that a classifier that simply memorizes each input and output pair will score perfectly but be unable to generalise to other examples. 
# 

# In[ ]:


# Random Forest : scoring on training data

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# What next? 
# -------------------------------
# 
# **_More feature exploration:_**
# Including *Fare* significantly increases the best accuracy to about 92% when *fare* is floored and 94% otherwise. Additionally including *Embarked* brings it up to 95%. It may worth be investigating if any relationship between these attributes and survival can be detected, especially for *fare*.
# 
# Other possibilities for features include *Deck* and *Title*, which can be extracted from *Cabin* and *Name* respectively.
# 
# Could also try two or more overlapping binnings for age groups (e.g. bins as defined by cutting on [0,4,15,25,35,45,65,100] and [10,20,30,40,55,100]). If going down this path, focus on introducing extra bins for age groups that contain many passengers and have a steeper gradient on the survival curve (such as for the twenties, e.g. cut on [10,20,30]).
# 
# **_Refitting:_**
# Most of the models above used their default parameters. Choose a few promising models and attempt to optimize their (hyper-)parameters. The sklearn library used above offers a couple of ways to do this automatically (via grid search and cross-validated models, see [Model selection][1] and [Tuning the hyper-parameters of an estimator][2]).
# 
# 
#   [1]: http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
#   [2]: http://scikit-learn.org/stable/modules/grid_search.html#grid-search
