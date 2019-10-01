#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic survivors
# Théo Painvin
# 
# ### 1. Data overview
#     a. Data importation and first analyses
#     b. Numerical features
#     c. Categorical features
# ### 2. Features engineering
#     a. Features creation
#     b. Filling in missing data
#     c. Ordered and non-ordered categorical features
# ### 3. Predicting the survivors
#     a. Models selection and simple modelling
#     b. Hyperparameters tunings
#     c. Test set predictions

# # 1. Data overview
# ## 1.a Data importation and first analyses

# In[1]:


get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt

# Import the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Combine the two datasets
entire = pd.merge(train, test, how='outer')


# We will work on the entire dataset all along the data analyses phase. It is indeed good to consider all the available data in order to get representative statistics (for example, to fill in missing data), the training dataset's number of rows being quite small.
# In the end of the data analyses and engineering, we will simply select the training set considering the rows of the entire dataset that have a non null value for `Survived` feature. isnull and notnull methods are very useful for that. 
# Let's get into lists the names of the numerical and categorical features.

# In[2]:


num_feat = list(entire.dtypes[entire.dtypes != 'object'].index)
categ_feat = list(entire.dtypes[entire.dtypes == 'object'].index)
print("Categorical features:", categ_feat)
print("Numerical features:", num_feat)


# In[3]:


features_without_Survived = [f for f in num_feat + categ_feat if f != 'Survived']
# Call describe method on the two main sets. First, the training set.
print(entire[entire.Survived.notnull()].describe())


# In[4]:


# We learn here that around 38% on passengers in the training set survived.
# Let's describe the test set:
print(entire[entire.Survived.isnull()].describe())


# The train and test sets contain respectfuly 891 and 418 rows. `PassengerId` values from 1 to 891 represent train set passengers. Test set passengers are from 892 to 1309. First of all, create a function that get the number of missing values for each features.

# In[5]:


def show_nan(df):
    if True not in df[features_without_Survived].isnull().values:
        print("The entire dataset does not contain missing value.")
    else:
        print("The entire dataset contains:")
        for c in df.columns:
            if (c != 'Survived') & (True in df[c].isnull().values):
                l = [df[c].isnull().value_counts()[True], 
                     round((df[c].isnull().value_counts()[True] /\
                            df.PassengerId.count()), 5),
                     c]
                msg = ("\t- {} ({:.1%}) missing values for feature"
                       " '{}'".format(*l))
                print(msg)


# In[6]:


show_nan(entire)


# `Age` and `Cabin` features contain loads of missing values. `Age` being a numerical feature, a simple way of solving the issue can be to fill in the missing values with the mean value of all the entire set. A more elegant approach could be to apply the mean value of a certain category. For example, we can extract the passenger's status from the `Name` feature: the average "Mrs" age value will be probably a bit higher than the "Miss" one. Let's investigate it later.

# In[7]:


entire.Name[entire.Age.isnull()].head(3)


# We can indeed extract from `Name` the status between ", " and "." using regular expressions. We'll do it in §2.a. `Cabin` feature contains loads of missing values, which is not surprising because some people may simply not have travelled in a cabin. Only two missing values for `Embarked` feature, which we can fill in with the most frequent value. Let's build a function to plot features:

# In[8]:


def plot_bar(df, feat_x, feat_y, normalize=True):
    """ Plot with vertical bars of the requested dataframe and features"""
    
    ct = pd.crosstab(df[feat_x], df[feat_y])
    if normalize == True:
        ct = ct.div(ct.sum(axis=1), axis=0)
    return ct.plot(kind='bar', stacked=True)


# ## 1.b Numerical features
# Using the `plot_bar` function written above, we can plot and visualize the correlation between the survival rate and another feature.
# ### Plotting Survived versus Pclass

# In[9]:


plot_bar(train, 'Pclass', 'Survived')
plt.show()


# The survival rate clearly decreases as the class decreases.
# ### Plotting `Survived` versus `SibSp`: number of siblings or spouses on board

# In[10]:


plot_bar(train, 'SibSp', 'Survived')
plt.show()


# ### Plotting `Survived` versus `Parch`: number of children or parents on board

# In[11]:


plot_bar(train, 'Parch', 'Survived')
plt.show()


# An interesting aspect of `SibSp` and `Parch` is that these features both design family members. We can then create a feature representing the family size on board of the passenger, which may have a correlation with the survival rate. Indeed, a single passenger may receive less help from other passengers than somebody with family members onboard.
# ## 1.c Categorical features
# ### Plotting `Survived` versus `Sex`

# In[12]:


# Plot the survival rates of the different "Sex" feature
plot_bar(train, 'Sex', 'Survived')
plt.show()


# 
# There seems to be a strong correlation between gender and survival: women are clearly more likely to survive than men.
# ### Plotting `Survived` versus `Embarked`

# In[13]:


plot_bar(train, 'Embarked', 'Survived')
plt.show()


# ### Plotting `Survived` versus `Cabin`

# In[14]:


plot_bar(train, 'Cabin', 'Survived')
plt.show()


# The `Cabin` feature clearly needs a bit of feature engineering before being plotted..!
# # 2. Feature engineering
# ## 2.a Features creation
# As stated previously, we will create new features based on `Name` and (`SibSp`, `Parch`) respectfully: the `Status` and `FamilySize` features.
# ### `Status` creation

# In[15]:


def create_status(name):
    """From "Name" feature extract the status"""
    m = re.search(" [A-Za-z]+\.", name)
    if m:
        status = re.sub('\.', '', re.sub(" ", "", m.group(0)))
        return status
    else:
        return "None"

for df in [entire, train]:
    df['Status'] = df['Name'].apply(create_status)
    df = df.drop(['Name'], axis=1)


# In[16]:


# Let's see the different status categories and their corresponding frequencies using value_counts method
entire.Status.value_counts()


# In[17]:


# List the rare status to group in one "Rare" group
list_rare = ['Dr', 'Rev', 'Col', 'Major', 'Sir', 'Lady',
             'Dona', 'Don', 'Jonkheer', 'Countess', 'Capt']

for df in [entire, train]:
    # "Mme", is the French equivalent of "Mrs"
    df['Status'] = df['Status'].replace(["Mme", "Ms"], "Mrs")
    # "Mlle": "Miss"
    df['Status'] = df['Status'].replace(["Mlle"], "Miss")
    # Create "Rare" status
    df['Status'] = df['Status'].replace(list_rare, "Rare")
# Print the final 'Status' for each dataset
print("\nNumber of counts per 'Status' value in the entire dataset:")
print(entire['Status'].value_counts())


# `Status` feature being created, we can now plot its correlation with the survival rate using the training set.

# In[18]:


plot_bar(train, 'Status', 'Survived')
plt.show()


# ### `FamilySize` creation
# We will now create the `FamilySize` feature, which is equal to the sum of `SibSp` and `Parch` and 1 (the current passenger). Then, if the passenger has no children/spouse/husband/parent on board, `FamilySize == 1`, meaning he is single. 

# In[19]:


def family_size(number):
    if number == 1:
        return 'single'
    if 2 <= number <= 3:
        return 'small_family'
    if 4 <= number <= 5:
        return 'medium_family'
    if number > 5:
        return 'large_family'
    
for df in [entire, train]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySize'] = df['FamilySize'].apply(family_size)


# In[20]:


plot_bar(train, 'FamilySize', 'Survived')
plt.show()


# Looks like large families and single passengers were more likely not to survive.
# ## 2.b Missing data
# Four features have missing data. `Embarked` and `Fare` have only few missing data which can be filled in without much efforts. `Cabin` and `Age` however contain loads of missing data, we then need to fill them in with more advanced technics. In order to fill in missing data with data linked to certain groups, the groupby method coupled with mean method will be quite useful.
# ### `Embarked` missing data
# `Embarked` feature contains two missing values. There are only 3 unique values for this feature: ['C', 'Q', 'S'].
# Missing values names does not sound French, so they probably come from "S": Southampton (EN), or "Q": Queenstown (NZ). Let's fill it with the most frequent values: 'S'.

# In[21]:


entire['Embarked'] = entire['Embarked'].fillna("S")


# ### `Fare` missing data
# Let's fill in the missing `Fare` value of the test set with the mean value of its `Pclass` category. Indeed we saw that `Pclass` is the feature most correlated to `Fare`.

# In[22]:


entire[entire.Fare.isnull()]


# In[23]:


# Mr. Storey travelled in 3rd class. Let's find out the median value on the entire dataset:
link_fare_pclass = entire.groupby('Pclass')['Fare'].median()
link_fare_pclass


# In[24]:


# We can then fill in the missing value
entire.Fare = entire.Fare.fillna(link_fare_pclass[3])


# ### `Cabin` missing data
# The `Ticket` feature can help us manage missing `Cabin` data. Indeed, we may miss the `Cabin` information of a passenger that has the same `Ticket` value of a passenger who has a known `Cabin` value. They will very probably have the same `Cabin` value. Let's dig into `Ticket` feature.
# First, I decide to change the `Cabin` name by it's first letter, in order to reduce the number of `Cabin` categories without losing interesting information. Due to missing values, on which a function like apply cannot operate well, I need to fill in missing values by "Unknown". We can then apply a lambda function enabling getting the first character of `Cabin` feature, then we can eventually replace "Unknown" values ("U", actually) by `np.nan`.

# In[25]:


entire.Cabin = entire.Cabin.fillna('Unknown')
entire.Cabin = entire.Cabin.apply(lambda cabin_name: cabin_name[0])

# Return to np.nan values thanks to replace method
entire.Cabin = entire.Cabin.replace('U', np.nan)

# Create a groupby object grouping the entire dataset by "Ticket"
gbyTicket = entire.groupby('Ticket')


# In[26]:


# Now we will focus on the groups of gbyTicket that contain both missing and non missing
# values for Cabin feature
# The missing values will eventually be replaced by the value of the others
dict_ticket_cabin = {}
for ti, gr in gbyTicket:
    if (True in gr['Cabin'].isnull().values) and       (False in gr['Cabin'].isnull().values):
        print("\n******************** Ticket n°{} ********************".format(ti))
        print(gr[['Cabin', 'Pclass', 'Fare', 'SibSp', 'Parch',
                  'FamilySize', 'Embarked']])
        # Fill in dict_ticket_cabin with the corresponding cabin values,
        # when a "Ticket" value is shared
        dict_ticket_cabin[ti] = gr['Cabin'].describe()['top']
    else:
        # Fill in dict_ticket_cabin when no "Ticket" value is shared in the group
        # grouped by "Ticket"
        dict_ticket_cabin[ti] = 'U'


# We can see that 16 `Cabin` missing values can be filled with their corresponding group value. `dict_ticket_cabin` was created in the previous cell in order to link the ticket number to the corresponding cabin name.
# Every `Cabin` missing value linked to a same `Ticket` value is linked to the corresponding shared `Cabin` value. If no `Ticket` value is shared, then "U" will be provided to the `Cabin` missing value.

# In[27]:


# Function allowing filling missing values with the ones contained in "dict_mean_ages" dictionary
# Groups have a "name" attribute set internally, we then can use that:
fill_cabin_from_dict = lambda g: g.fillna(dict_ticket_cabin[g.name])


# In[28]:


#Before filling in the "Cabin" missing data, let's use "show_nan" function
show_nan(entire)


# In[29]:


entire['Cabin'] = entire.groupby('Ticket')['Cabin'].apply(fill_cabin_from_dict)

# Use show_nan again to see if "fill_cabin_from_dict" worked
show_nan(entire)


# Now let's see the impact of `Cabin` feature on survival rate:

# In[30]:


train['Cabin'] = entire.groupby('Ticket')['Cabin'].apply(fill_cabin_from_dict)
plot_bar(train, 'Cabin', 'Survived')
plt.show()


# There is no more missing data in `Cabin` feature. The last feature to consider is `Age`.
# ### `Age` missing data
# Now that the `Status` feature is created, we can fill in missing `Age` values with the median value of the passengers per status and class features. Let's build a `groupby` object displaying the median value and the number of passengers per status and class:

# In[31]:


age_median = entire.groupby(['Status', 'Pclass'])['Age'].agg(['median', 'count'])
age_median = age_median.reset_index()
age_median


# We can now fill in the `Age` missing values with these median values.

# In[32]:


for index in entire[entire.Age.isnull()].index:
    median = age_median[(age_median.Status == entire.iloc[index]['Status']) &                        (age_median.Pclass == entire.iloc[index]['Pclass'])                       ]['median'].values[0]
    entire.set_value(index, 'Age', median)


# In[33]:


show_nan(entire)


# Now that we don't have missing values on `Age` anymore, it's possible to categorize this feature into different groups. qcut function discretizes variables using equal-sized buckets.

# In[34]:


categ_age = pd.qcut(entire.Age, 8, labels=range(8), retbins=True)
entire['Age_category'] = categ_age[0]


# We treated all the numerical and categorical features. We can now categorize `Fare` using qcut method:

# In[35]:


categ_fare = pd.qcut(entire.Fare, 8, labels=range(8), retbins=True)
entire['Fare_category'] = categ_fare[0]


# We are all good with missing categorical or numerical values. Let's have a last overview of all cleaned entire dataset without the features that won't be useful for predictions: `Ticket`, `Name`, `PassengerId`. `Parch` and `SibSp` can also be dropped since they were used to create the `FamilySize` feature. `Age` and `Fare` have been categorized, they can then be dropped too.

# In[36]:


entire = entire.drop(['Ticket', 'PassengerId',  'Name',
                      'SibSp', 'Parch', 'Age', 'Fare'],
                     axis=1)


# ## 2.c Ordered and non-ordered categorical features
# Now let's finish §2 with ordered and non-ordered categorical features: non-ordered ones need to be transformed into dummy variables, thought for ordered ones we simply need to set them as categorical type with their sorted values. `Age_category` and `Fare_category` are already seen as ordered categorical features.

# In[37]:


# Ordered features transformation
ordered = {
    'Pclass': [3, 2, 1],
    'FamilySize': ['single', 'small_family', 'medium_family', 'large_family'],
}
for feat, val in ordered.items():
    entire[feat] = entire[feat].astype('category',
                                       categories=val,
                                       ordered=True).cat.codes

# Non-ordered features transformation
non_ordered = ['Sex', 'Status', 'Embarked', 'Cabin']
entire = pd.get_dummies(entire, columns=non_ordered, drop_first=True)


# `drop_first` option in `pandas.get_dummies` function allows keeping only (n-1) features over the n created features. For example for `Sex` feature, instead of creating `Sex_female` **and** `Sex_male`, only one of them will be created. It avoids ending up with highly correlated features.
# # 3. Predicting the survivors
# ## 3.a Models selection and simple modelling
# Let's create the training and testing vectors, choose three classifier models and run them on the training set without hyperparameters tuning. I choose three famous ensemble modelling classifiers: `RandomForestClassifier` (bagging), `AdaBoostClassifier` and `GradientBoostingClassifier` (boosting).

# In[38]:


#List of features for train set
feat_train = entire.columns.values
# List of features for test set
feat_test = [feat for feat in feat_train if feat != 'Survived']


# In[39]:


# Create the (X, y) training vectors to be injected in ours classifiers
X_train = entire[entire.Survived.notnull()].drop(['Survived'], axis=1)
y_train = entire[entire.Survived.notnull()]['Survived']

# Split the training set into a development and an evaluation sets
from sklearn.model_selection import train_test_split
X_dev, X_eval, y_dev, y_eval = train_test_split(X_train,
                                                y_train,
                                                test_size=0.2,
                                                random_state=42)
# And the test set
X_test = entire[entire.Survived.isnull()].drop(['Survived'], axis=1)


# In[40]:


# Import classifiers
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)

from sklearn.metrics import accuracy_score


# In[41]:


# Use them without any hyperparameters tuning
models = [RandomForestClassifier(random_state=77),
          GradientBoostingClassifier(random_state=77),
          AdaBoostClassifier(random_state=77)]

from sklearn.model_selection import cross_val_score, GridSearchCV

for model in models:
    score = cross_val_score(model, X_dev, y_dev, cv=5)
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.3f} "
           "(+/- {2:.3f})".format(model.__class__.__name__,
                                  score.mean(),
                                  score.std()))
    print(msg)
    
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_dev, y_dev)
    pred_eval = model.predict(X_eval)
    acc_eval = accuracy_score(y_eval, pred_eval)
    print("\tAccuracy on evaluation set\t\t= {0:.3f}".format(acc_eval))


# The models having the lowest standard deviation are RandomForest and AdaBoost, which indicates that the variance of these models is quite low, which is what we're looking for in order to finally obtain a model that can generalize well. Over these three models, the least biased is the AdaBoost model as the difference between its development and evaluation sets accuracies is the smallest. The best accuracies over the evaluation set are obtained with RandomForest and GradientBoosting.
# Scaling isn't necessary as these three models are not sensitive to data scaling.
# ## 3.b Hyperparameters tuning
# Let's tune the hyperparameters with our three classifiers. The `dict_clf` dictionary will be filled in with the estimators names, their best parameters combination, best score and fitting time.
# Nota: the calculation time being limited for kernels, I reduced the size of my grid of parameter values, which will affect the final accuracy. You can expand the range of paramgrid keys values in order to improve your score!

# In[42]:


# Import time module in order to get the time spent by GridSearchCV with all the
# different classifiers
import time
dict_clf = {}


# In[43]:


# 1. Random Forest
paramgrid = {
    'n_estimators':      [100, 200, 500, 750, 1000],
    'criterion':         ['gini', 'entropy'],
    'max_features':      ['auto', 'log2'],
    'min_samples_leaf':  list(range(2, 7))
}
GS = GridSearchCV(RandomForestClassifier(random_state=77),
                  paramgrid,
                  cv=4)
t0 = time.time()
GS.fit(X_dev, y_dev)
t = time.time() - t0
best_clf = GS.best_estimator_
best_params = GS.best_params_
best_score = GS.best_score_
name = 'RF'
best_clf.fit(X_dev, y_dev)
acc_eval = accuracy_score(y_eval, best_clf.predict(X_eval))
dict_clf[name] = {
    'best_par': best_params,
    'best_clf': best_clf,
    'best_score': best_score,
    'score_eval': acc_eval,
    'fit_time': t,
}


# In[44]:


# 2. GradientBoosting
paramgrid = {
    'n_estimators':      [100, 200, 500, 750, 1000],
    'max_features':      ['auto', 'log2'],
    'min_samples_leaf':  list(range(2, 7)),
    'loss' :             ['deviance', 'exponential'],
    'learning_rate':     [0.05, 0.1, 0.2],
}
GS = GridSearchCV(GradientBoostingClassifier(random_state=77),
                  paramgrid,
                  cv=4)
t0 = time.time()
GS.fit(X_dev, y_dev)
t = time.time() - t0
best_clf = GS.best_estimator_
best_params = GS.best_params_
best_score = GS.best_score_
name = 'GB'
best_clf.fit(X_dev, y_dev)
acc_eval = accuracy_score(y_eval, best_clf.predict(X_eval))
dict_clf[name] = {
    'best_par': best_params,
    'best_clf': best_clf,
    'best_score': best_score,
    'score_eval': acc_eval,
    'fit_time': t,
}


# In[45]:


# 3. AdaBoost
paramgrid = {
    'n_estimators':  [100, 200, 500, 750, 1000],
    'learning_rate': [0.05, 0.1, 0.5, 1, 2]
}
GS = GridSearchCV(AdaBoostClassifier(random_state=77),
                  paramgrid,
                  cv=4)
t0 = time.time()
GS.fit(X_dev, y_dev)
t = time.time() - t0
best_clf = GS.best_estimator_
best_params = GS.best_params_
best_score = GS.best_score_
name = 'ADB'
best_clf.fit(X_dev, y_dev)
acc_eval = accuracy_score(y_eval, best_clf.predict(X_eval))
dict_clf[name] = {
    'best_par': best_params,
    'best_clf': best_clf,
    'best_score': best_score,
    'score_eval': acc_eval,
    'fit_time': t,
}


# Print all these informations about our classifiers after hyperparameters tuning:

# In[46]:


for clf in dict_clf.keys():
    print("{0} classifier:\n\t- Best score = {1:.2%}".format(clf, dict_clf[clf]['best_score']))
    print("\t- Score on evaluation set = {0:.2%}".format(dict_clf[clf]['score_eval']))
    print("\t- Fitting time = {0:.1f} min".format(round(dict_clf[clf]['fit_time']/60, 1)))
    print("\t- Best parameters:")
    for par in sorted(dict_clf[clf]['best_par'].keys()):
        print("\t\t* {0}: {1}".format(par, dict_clf[clf]['best_par'][par]))


# ## 3.c. Test set predictions
# Our three classifiers have equivalent accuracy over the evaluation set. We can then let them vote using `VotingClassifier`.

# In[47]:


from sklearn.ensemble import VotingClassifier

estimators = [('RF', dict_clf['RF']['best_clf']),
              ('GB', dict_clf['GB']['best_clf']),
              ('ADB', dict_clf['ADB']['best_clf'])]

# Instanciate the VotingClassifier using the soft voting
voter = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
voter.fit(X_train, y_train)

pred = voter.predict(X_test).astype(int)


# In[48]:


# Calculate the known survival rate in the training set
known = train.Survived.values
nb_survived = 0
for i in known:
    if i == 1:
        nb_survived += 1
print("Number of survivors in training set: {0} over {1} "
      "({2:.2%})".format(nb_survived, len(known), nb_survived/len(known)))

# Calculate the predicted survival rate in the test set
nb_survived = 0
for i in pred:
    if i == 1:
        nb_survived += 1
print("Number of survivors in predicted set: {0} over {1} "
      "({2:.2%})".format(nb_survived, len(pred), nb_survived/len(pred)))


# The survival rate in the training and predicted sets are both around 37/38%, which is a good thing. Now, we only need to eventually create the final csv file to submit the predictions:

# In[49]:


# Build the requested DataFrame
ids = test['PassengerId']

dict_pred = {'PassengerId': ids, 'Survived': pred}
df_pred = pd.DataFrame(dict_pred).set_index(['PassengerId'])

# Get the date to append to the 'Predictions_*.csv' file
from datetime import datetime
date_str = datetime.now().strftime('%Y-%m-%d_%Hh%M')

df_pred.to_csv('Predictions_' + date_str + '.csv')


# That leads to a final accuracy of around 80%. The running time on Kaggle being limited to 60min, the exhaustivity of parameters in `GridSearchCV` in §3.b had to be restricted. Enjoy and let's predict!

# 
