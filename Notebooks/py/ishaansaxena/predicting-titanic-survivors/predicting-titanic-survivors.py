#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Classification
# This notebook is my step by step approach to predict if a passenger survived the titanic disaster or did not.
# 
# ## Setting Up
# ### Importing Required Libraries
# Import numpy and pandas to setup our problem.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt


# ### Importing & Cleaning Training Data
# Read training data and remove NA values. If the number of rows before after dropping NA values is something that is useable, we will proceed, otherwise we will look at other ways to treat missing data or select specific rows.

# In[ ]:


df = pd.read_csv("../input/train.csv")
print("Before Cleaning:\t", df.shape)
nona = df.dropna()
print("After Cleaning:\t\t", nona.shape)


# Clearly, since there are a lot of rows with missing values, let's check if they come from specific columns.

# In[ ]:


df.info()


# For now, let's drop the cabin column, remove NA values, and proceed to further analysis. In the future, we might try to fill in cabin values based on correlations with Fares, etc. 

# In[ ]:


df = df.drop("Cabin", axis=1).dropna()


# Let's see what that gives us:

# In[ ]:


print(df.head())
print(df.shape)
ns = df.loc[df["Survived"] == False] # Didn't survive :(
s  = df.loc[df["Survived"] == True]  # Survivors


# 712 rows feels like a good place to start.

# ## Analyzing the Data for Classification
# ### Choosing Columns for Classification
# #### 1. Numerical Variables
# To start, let's check how survival correlates with numerical columns. Out of all the data we have for a passenger, we know that the ticket fare, age, siblings/spouses on-board, and parents/children on-board are the only numerical data types.
# #### 1.1  Fare
# First, we start by checking if there is any visible difference between ticket for those who survive and those who did not.

# In[ ]:


sns.boxplot(x="Survived", y="Fare", data=df)


# This does seem a little informative, but just so we can conduct a hypthesis test, let's log transform the data and recreate the same plot.

# In[ ]:


df["logFare"] = np.log(df["Fare"]) # We are ignoring 0 value fares since they are outliers.
sns.boxplot(x="Survived", y="logFare", data=df)


# Let's conduct a hypothesis test to check if this difference is of any significance.

# In[ ]:


t, p = stats.ttest_ind(ns["Fare"],s["Fare"])
print("p-value: %s" % p)


# Clearly, since the p-value is very small, we have reason to believe that the fares of the survivors differed significantly from that of those who didn't, and that the difference was more than 0. Thus, this should be picked as a classification parameter.
# 
# #### 1.2  Age
# Since the response to ages is expected to be different in comparison to fares (in that, while people who paid more would be thought to have been priorities, people of both very small and very big ages would be thought to have been considered as an evacuation priority similarly). So, for this metric, let's check the distribution of survivors wrt age instead.

# In[ ]:


sns.distplot(ns["Age"])
sns.distplot(s["Age"])


# As we expected, for the survivors, there's a significant bump in the number of children. The distribution of the survivors, as expected, is also more variant than that of those who did not due to the age bias. This too, seems to be an important estimator. Let's split it into discrete classes and see if we can do any better.

# In[ ]:


df.loc[df['Age'] >= 0, 'AgeC'] = 0
df.loc[df['Age'] > 16, 'AgeC'] = 1
df.loc[df['Age'] > 50, 'AgeC'] = 2
df.loc[df['Age'] > 70, 'AgeC'] = 3
ns = df.loc[df["Survived"] == False] # Didn't survive :(
s  = df.loc[df["Survived"] == True]  # Survivors
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'AgeC', bins=7)


# Numerically:

# In[ ]:


print(df[["AgeC", "Survived"]].groupby(['AgeC'], as_index=False).mean())


# #### 1.3  Number of Siblings or Number of Parents/Children on Board
# Let's see how number of siblings or parents/children correlates with survival. Let's see the distributions, starting with siblings on board.

# In[ ]:


sns.distplot(ns["SibSp"])
sns.distplot(s["SibSp"])


# Now for parents/children on board:

# In[ ]:


sns.distplot(ns["Parch"])
sns.distplot(s["Parch"])


# These two features seem to have little correlation with survival rates.  However, let's see if we can combine these in some way to get a significant difference.

# In[ ]:


df["Family"] = df["Parch"] + df["SibSp"]
ns = df.loc[df["Survived"] == False] # Didn't survive :(
s  = df.loc[df["Survived"] == True]  # Survivors


# We have combined parents/children siblings into one single column space which gives us the count of family members on board. Let's see if this has any effect on the passengers' survival rates.

# In[ ]:


sns.distplot(ns["Family"])
sns.distplot(s["Family"])


# It seems now as though people with no family members on board didn't fare very well. Let's look at the same numerically and see if we can use this (number of family members) as a feature.

# In[ ]:


print(df[["Family", "Survived"]].groupby(['Family'], as_index=False).mean())


# It seems as though this still needs a little modification. Let's change this column to 0 for 0 family members and 1 for 0 family members. Let's see if this has any more merit.

# In[ ]:


df.loc[df['Family'] == 0, 'HasFamily'] = 0
df.loc[df['Family'] >= 1, 'HasFamily'] = 1
print(df[["HasFamily", "Survived"]].groupby(['HasFamily'], as_index=False).mean())
ns = df.loc[df["Survived"] == False] # Didn't survive :(
s  = df.loc[df["Survived"] == True]  # Survivors


# This seems much more usable!

# #### 2. Categorical Variables
# We should also look how survival is affected by different categories that a person might fall in.
# 
# #### 2.1 Sex
# Most obviously, as I remeber from Titanic, would be this difference, "ladies and children first". Let's see how the sex has an effect on survival:

# In[ ]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Sex', bins=3)


# This seems to be conclusive enough, but let's also look at this difference numerically.

# In[ ]:


print(df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# This is a very important result, as this shows that Sex is going to be an important feature for an algorithm. Since we will be using this in our model, let's convert this into a numeric value.

# In[ ]:


df['SexN'] = df['Sex']
df['SexN'].replace({'male': 0, 'female': 1},inplace=True)


# #### 2.2 Passenger Class
# Again, as I know from the film, the class seems to be something of an important factor. Let's check this out in a simlar way as above.

# In[ ]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Pclass', bins=5)


# Again, numerically:

# In[ ]:


print(df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# Clearly, there's a significant bias in the treatment of people from different classes. *Sigh*. Early 1900s surely weren't great. We have a good indicator here though.
# 
# #### 2.3 Embarkment Port
# Finally, let's look at the embarkment port and see if it has any effect on the survival rate.

# In[ ]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Embarked', bins=5)


# This doesn't show us much other tthan the fact that a lot of people boarded the ship from Southampton. Go Saints. Let's look at these numerically.

# In[ ]:


print(df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# However, this still does not tell us very much, since there seems to be no intuitive correlation between the port of embarkment and survival. Let's look for confounding variables, starting with the most obvious - passenger class.

# In[ ]:


g = sns.FacetGrid(df, col='Pclass', row='Sex')
g.map(plt.hist, 'Embarked', bins=5)


# This shows us that the port of emarkment is not all that useful when we now come to think of it. For instance, consider Queenstown - there seem to be very few people from higher classes who boarded the Titanic from Queenstown, thus, a smalled expected survival rate. Similarly, when we look at Cherbourg, the number of people from higher classes who have boarded is higher, specifically women of higher classes. Clearly, the port is surplus to our requirements, as it does not contribute any extra information than we already have.

# ## Splitting the Data into Test and Train Sets
# While Kaggle has already done this for us, we will still split into 80:20 test:train sets just to check the performance of the difference models which we will test.

# In[ ]:


from random import randint
from sklearn import model_selection

Features = ["Pclass", "SexN", "Age", "AgeC", "Fare", "HasFamily", "Family"]

X = df[Features].values
Y = df["Survived"].values.ravel()

validation_size = 0.40
seed = randint(0, 100)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y,
    test_size=validation_size,
    random_state=seed
)

print("Train size:\t", Y_train.shape[0])
print("Test size:\t", Y_test.shape[0])


# ## Training and Testing Different Models
# ### Importing Models
# Now, let's import the various models we want to train and validate.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import models
models = []
models.append(("LR", LogisticRegression))
models.append(("CART", DecisionTreeClassifier))
models.append(("KNN", KNeighborsClassifier))
models.append(("LDA", LinearDiscriminantAnalysis))
models.append(("NB", GaussianNB))
models.append(("SVC", SVC))
models.append(("RF", RandomForestClassifier))
models.append(("XGB", XGBClassifier))


# ### Training Models
# Let's proceed to train these models and check how they perform.

# In[ ]:


from sklearn.metrics import accuracy_score

for mtuple in models:
    name, model = mtuple
    m = model()
    m.fit(X_train, Y_train)
    Y_pred = m.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)
    print(name, score)


# It seems from these values that the Logistical Regression, Decision Tree Classifier, and Random Forest Classifier are the ones giving the best performance. However, with many different seed values for the test/train split, Random Forest consistently offers the best performance, and as a result, we shall proceed with it.
# 
# ### Hyperparameter Optimization
# #### Random Hyperparameter Grid
# First, let's use a randomized hyperparam grid to tune the following hyper params:

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rf = RandomForestClassifier()
cv = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
cv.fit(X_train, Y_train)

best_params = cv.best_params_


# Let's check the best params that this search has yielded:

# In[ ]:


print(best_params)


# Now, let's true to check the accuracy difference when using these hyperparams. First let's revisit our base model.

# In[ ]:


base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, Y_train)
Y_pred = base_model.predict(X_test)
base_score = accuracy_score(Y_test, Y_pred)
print(base_score)


# Next, the same classifier with our hyperparams.

# In[ ]:


model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
new_score = accuracy_score(Y_test, Y_pred)
print(new_score)


# That's an improvement of:

# In[ ]:


print("%.4f%%" % ((new_score-base_score)/base_score * 100))


# #### Grid Search with Cross Validation
# Now, let's use the random hyperparams to narrow down our grid search params, and conduct the search again.

# In[ ]:


# NOTE: This was created manually on the last run kernel code. Should ideally be done on the basis of best_params.
param_grid = {
    'bootstrap': [True],
    'max_depth': [60, 70, 80],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [8, 9, 10, 11, 12],
    'n_estimators': [1500, 1600, 1700]
}


# Optimize!

# In[ ]:


from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
cv = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
cv.fit(X_train, Y_train)

best_params = cv.best_params_


# And finally:

# In[ ]:


print(best_params)


# Let's see how this has had an effect on our model's accuracy.

# In[ ]:


print("Base Model:\t", base_score)
print("New Model:\t", new_score)
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
opt_score = accuracy_score(Y_test, Y_pred)
print("Optimized:\t", opt_score)


# ## Final Steps & Submission
# Now that we have selected the features, selected the model, and optimized the hyper parameters, we can finally train our model and make the submission file. Let's train our model on the entire training data first:

# In[ ]:


seed = randint(0, 100)
# best_params = {}
model = RandomForestClassifier(**best_params, random_state=seed)
# model = RandomForestClassifier(random_state=seed)
model.fit(X, Y)


# Reading Kaggle Test Dataset:

# In[ ]:


df_test = pd.read_csv("../input/test.csv")
df_test.info()


# Fill in null values for age and fare by mean from train data:

# In[ ]:


df_test['Age'].fillna((df['Age'].mean()), inplace=True)
df_test['Fare'].fillna((df['Fare'].mean()), inplace=True)


# Let's prepare test data:

# In[ ]:


df_test['SexN'] = df_test['Sex']
df_test['SexN'].replace({'male': 0, 'female': 1}, inplace=True)

df_test['Family'] = df_test['Parch'] + df_test['SibSp']
df_test.loc[df_test['Family'] == 0, 'HasFamily'] = 0
df_test.loc[df_test['Family'] >= 1, 'HasFamily'] = 1

df_test.loc[df_test['Age'] >= 0, 'AgeC'] = 0
df_test.loc[df_test['Age'] > 16, 'AgeC'] = 1
df_test.loc[df_test['Age'] > 50, 'AgeC'] = 2
df_test.loc[df_test['Age'] > 70, 'AgeC'] = 3

test = df_test[Features].values


# And finally, predicting!

# In[ ]:


Y_pred = model.predict(test)


# ### Create Submission Dataframe
# Let's take this prediction and create our submission file!

# In[ ]:


submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": Y_pred
})
print(submission.head())
print(submission.shape)


# In[ ]:


submission.to_csv('submission_RF.csv', index=False)


# #### Other Models
# Although we only optimized the hyper parameters of the RandomForest model due to it's consistently better performance, we will also run some other models just to check how accurate the submissions are in comparison. This is just to see how these models perform in comparison to each other, and isn't intended for the competition performance.

# In[ ]:


models = []
models.append(("LR", LogisticRegression))
models.append(("LDA", LinearDiscriminantAnalysis))
models.append(("NB", GaussianNB))
models.append(("XGB", XGBClassifier))


# In[ ]:


for mtuple in models:
    name, model = mtuple
    m = model()
    m.fit(X, Y)
    Y_pred = m.predict(test)
    submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv('submission_' + name + '.csv', index=False)


# 

# 

# #### More on Hyperparameter Optimization:
# [William Koehrsen's Tutorial on Hyperparameter Optimization](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
