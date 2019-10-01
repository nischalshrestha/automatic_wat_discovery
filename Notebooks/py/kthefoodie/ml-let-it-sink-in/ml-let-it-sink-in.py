#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# 
# Objective of this notebook is to implement many algorithms for the purpose of demonstration. Intention is to write instructive, clear code, rather than efficient one. Focus will be on having clear reasoning for all the steps, which will highlight a general startegy to tackle any problem. There will also be links directing to the relevnt material, if you don't know (or need to revisit) a particular topic.
# 
# I hope this will be useful to you. If you have any doubts or suggestion, feel free to comment.
# 
# Link to notebooks referred to:
# 1. [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook)
# 2. [Deep Visualisations - Simple Methods](https://www.kaggle.com/jkokatjuhha/deep-visualisations-simple-methods)
# 3. [A Journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# 4. [Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier)
# ----
# # Index
# 
# 1. [Getting our tools ready](#1)
# 2. [Understanding data](#2)<br>
# &nbsp;&nbsp;2.1 [Filling up missing values](#2.1)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.1.1 ['Fare'](#2.1.1)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.1.2 ['Age'](#2.1.2)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.1.3 ['Fare'](#2.1.3)<br>
# &nbsp;&nbsp;2.2 [Feature engineering and Data visualisation](#2.2)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.2.1 ['Cabin'](#2.2.1)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.2.2 ['family size](#2.2.2)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.2.3 ['title'](#2.2.3)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.2.4 ['Converting data to suitable numeric values'](#2.2.4)<br>
# &nbsp;&nbsp;&nbsp;&nbsp;2.2.5 ['Correlations'](#2.2.5)<br>
# 3. [Models and predictions](#3)<br>
# &nbsp;&nbsp;3.1[First level model](#3.1)<br>
# &nbsp;&nbsp;3.2[Second level model](#3.2)<br>

# <a id='1'></a>
# ## 1. Getting our tools ready
# If anyone is wondering why python is one of the most preferred language by Data Scientists, simple answer is 'great libraries'. Python has fantastic tools to handle data systematically ([pandas](http://pandas.pydata.org/)), tools to do computations efficiently and easily ([numpy](http://www.numpy.org/)), tools to visualise data ([matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [plotly](https://plot.ly/python/)) and last but not the least tools for machine learning ([scikit-learn](http://scikit-learn.org/)).
# 
# Our first step is to import required libraries and set few options.

# In[ ]:


# Importing required libraries

import pandas as pd
import numpy as np
import re
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# <a id='2'></a>
# ## 2. Understanding data
# Before we start doing any kind of analysis, we need to know what kind of data is available to us. Here, we have data in form of 2 files, train.csv and test.csv. Let us first load the data and store it in a variable for easy access. We will use pandas.read_csv for this.
# Reason we are given train and test data seperately is that everyone makes the prediction on same test data. So it is easy to compare results amongest submissions.

# In[ ]:


# Reading data into a dataframe
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Now let us look at first few rows of the data.

# In[ ]:


train.head()


# Next step is to understand columns and datatypes in the dataframe. Typically, such description is available along with data. ([Titanic Data description](https://www.kaggle.com/c/titanic/data)). (If it isn't, we can simply use pandas methods, dataframe.dtypes and dataframe[column_name].value_counts().)
# 
# 

# ### Titanic Data Dictionary
# 
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes<br>
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd<br>
# sex	Sex	<br>
# Age	Age in years	<br>
# sibsp	# of siblings / spouses aboard the Titanic	<br>
# parch	# of parents / children aboard the Titanic	<br>
# ticket	Ticket number	<br>
# fare	Passenger fare	<br>
# cabin	Cabin number	<br>
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton <br>
# <br>
# <br>
# Variable Notes<br>
# <br>
# pclass: A proxy for socio-economic status (SES)<br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# <br>
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>
# <br>
# sibsp: The dataset defines family relations in this way...<br>
# Sibling = brother, sister, stepbrother, stepsister<br>
# Spouse = husband, wife (mistresses and fiancés were ignored)<br>
# <br>
# parch: The dataset defines family relations in this way...<br>
# Parent = mother, father<br>
# Child = daughter, son, stepdaughter, stepson<br>
# Some children travelled only with a nanny, therefore parch=0 for them.

# ----
# Let's check for missing values in each column.

# In[ ]:


train.info()
# can also use train.isnull().sum()


# In[ ]:


test.info()


# Both datasets have a lot of missing values for 'Cabin' column. Few walues for 'Age' column are also missing. In train dataset, 'Embarked' column has 2 missing values. In test dataset, one value for 'fare' is missing.
# 
# Missing values in 'Cabin' can be interpreted as not having a cabin.
# For remaining columns, we will use imputation techniques to fill up the values.

# <a id='2.1'></a>
# ## 2.1 Filling up missing values
# There are 2 ways main approaches to deal with missing data. One is to ignore rows with missing data. Other is to try to fill up these values. This is called 'Data Imputation' ([Imputation](https://en.wikipedia.org/wiki/Imputation_(statistics)). There are several data imputation techniques.

# <a id='2.1.1'></a>
# ### 2.1.1 'Embarked'
# This column is categorical type. As data description suggests, there are only 3 values viz., 'S', 'C' and 'Q'. We will take the most common value and use that to fill up missing values.

# In[ ]:


train['Embarked'] = train['Embarked'].fillna(np.argmax(train['Embarked'].value_counts()))


# In[ ]:


train['Embarked'].unique()


# <a id='2.1.2'></a>
# ### 2.1.2 'Age'
# 
# Using same value for every missing point will introduce bias. We'll use random integers, but in a such a way that it doesn't affect the distribution very much. One way to achieve this is to draw raandom integers from the range mean +- standard deviation.

# In[ ]:


age_combined = np.concatenate((test['Age'].dropna(), train['Age'].dropna()), axis=0)
mean = age_combined.mean()
std_dev = age_combined.std()
train_na = np.isnan(train['Age'])
test_na = np.isnan(test['Age'])
impute_age_train = np.random.randint(mean - std_dev, mean + std_dev, size = train_na.sum())
impute_age_test = np.random.randint(mean - std_dev, mean + std_dev, size = test_na.sum())
train["Age"][train_na] = impute_age_train
test["Age"][test_na] = impute_age_test
new_age_combined = np.concatenate((test["Age"],train["Age"]), axis = 0)


# In[ ]:


# Check the effect of imputation on the distribution
_ = sns.kdeplot(age_combined)
_ = sns.kdeplot(new_age_combined)


# <a id='2.1.3'></a>
# ### 2.1.3 'Fare' 
# 
# There is only one missing value for 'Fare' column in test dataset. So we will use median value.

# In[ ]:


print(test['Fare'].isnull().sum())
test['Fare'] = test['Fare'].fillna(np.median(train['Fare']))
print(test['Fare'].isnull().sum())


# <a id='2.2'></a>
# ## 2.2 Feature engineering and Data visualisation
# 
# #### Feature engineering
# 
# The features in your data will directly influence the predictive models you use and the results you can achieve. Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models. ([Feature engineering](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/))
# 
# #### Data visualisation
# 
# Plots and information graphics are powerful way to understand data. They make complex data more accessible, understandable and usable. Rather than a list of numbers, a simple distribution plot conveys a lot more information lot more quickly. That is why data visualisations are extremely important in any ML project.

# <a id='2.2.1'></a>
# ### 2.2.1 'Cabin'
# 
# As mentioned earlier, missing cabin values will be interpreted as not having a cabin.
# 1 will represent having a cabin, 0 wil represent otherwise.

# In[ ]:


train["Cabin"] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test["Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


# In[ ]:


mean_survival_cabin = train[["Cabin", "Survived",'Sex']].groupby(['Cabin','Sex'],as_index=False).mean()
sns.set(font_scale=1.7)
ax = sns.barplot(x='Cabin', y='Survived', hue = 'Sex', data=mean_survival_cabin)
ax.legend(loc = 'upper left')


# Women have much higher rate of survival. Also, those who had cabin, have much higher rate of survival.

# <a id='2.2.2'></a>
# ### 2.2.2 'family size'
# 
# From the description,<br>
# sibsp - # of siblings / spouses aboard the Titanic<br>
# parch - # of parents / children aboard the Titanic
# 
# This can be combined into one variable, viz., 'family size'.

# In[ ]:


train['family_size'] = train['SibSp'] + train['Parch'] + 1
test['family_size'] = test['SibSp'] + test['Parch'] + 1
cols_to_drop = ['SibSp','Parch']
train.drop(cols_to_drop, inplace = True, axis = 1)
test.drop(cols_to_drop, inplace = True, axis = 1)


# In[ ]:


sns.factorplot('family_size','Survived', hue = 'Sex', data=train, size = 5, aspect = 3)


# The females traveling alone or with up to 3 more family members had a higher chance to survive. The chances for survival decrease once family size exceeds 4.<br>
# For men, survival rate is low but increases as family size increases till upto 3 other members. After that it drops again.<br>
# Overall, one can say that, after family size exceeds 4, survival rate is pretty low for both men and women, presumably because one would spend time searching for family members.

# <a id='2.2.3'></a>
# ### 2.2.3 'title'
# 
# Title of a person contains a lot of useful information. It is a combination of gender, martial status (for women) and age.

# In[ ]:


# Lets try to find a pattern in 'Name' column
train[['Name']].head(20)


# Name seems to have format - last name, title. name<br>
# So our pattern is ', (title). '. Note that spaces are important.

# In[ ]:


titles_train = train['Name'].str.extract(' ([A-Za-z]+)\.')


# In[ ]:


print(titles_train.value_counts(),'\n')
print('Null value count is ', titles_train.isnull().sum())


# Luckily, there are no missing values. Otherwise, we might have had to predict  it using gender and age.<br>
# Note that 'Countess','Dona','Lady' and 'Mme' all are used for married women, so those will be replaced with 'Mrs'.<br>
# Similarly, 'Mlle' and 'Ms' will both be mapped to 'Miss'.<br>
# All titles with lesss than 10 entries will be mapped to 'rare'. Reason is to avoid creating too many categories.

# In[ ]:


titles_train.replace(['Countess', 'Dona', 'Lady', 'Mme'], 'Mrs', inplace = True)
titles_train.replace(['Mlle', 'Ms'], 'Miss', inplace = True)
rare_titles = []
temp = titles_train.value_counts()
for title in temp.index:
    if temp[title] < 10:
        rare_titles.append(title)
        
titles_train.replace(rare_titles, 'rare', inplace = True)


# In[ ]:


train['title'] = titles_train
del titles_train


# In[ ]:


# Repeat the procedure with test dataset.
titles_test = test['Name'].str.extract(' ([A-Za-z]+)\.')
print(titles_test.value_counts(),'\n')
print('Null value count is ', titles_test.isnull().sum())


# In[ ]:


titles_test.replace(['Countess', 'Dona', 'Lady', 'Mme'], 'Mrs', inplace = True)
titles_test.replace(['Mlle', 'Ms'], 'Miss', inplace = True)
rare_titles = []
temp = titles_test.value_counts()
for title in temp.index:
    if temp[title] < 10:
        rare_titles.append(title)
        
titles_test.replace(rare_titles, 'rare', inplace = True)
test['title'] = titles_test
del titles_test


# In[ ]:


test['title'].value_counts()


# There are some more possible features that we can create. But these three should be good enough. You can refer to noebooks linked at the top, if you are curious about other possible feature.

# <a id='2.2.4'></a>
# ### 2.2.4 Converting data to suitable numerical values
# 
# We need to convert all the data to numerical values, so that we will be able to use ML algorithms. Also, some continuous data will be converted to discrete data. Reason for doing so is that, exact value of the variable isn't relevant, but in what range or group that value belongs to is relevant.<br>
# We will also [dummy code](https://en.wikiversity.org/wiki/Dummy_variable_(statistics) categorical variables. This will be done for variables which doesn't have numeric relationship and/or meaning amongest it's possible values. ([why do we need to dummy code categorical variables](https://stats.stackexchange.com/questions/115049/why-do-we-need-to-dummy-code-categorical-variables))

# In[ ]:


# put train and test datasets in one list for the ease of doing operations.
data = [train, test]

# delete 'Ticket' and 'Name' columns
for df in data:
    df.drop(['Ticket','Name'], inplace = True, axis = 1)


# In[ ]:


# 'Sex' column - straightforward 0 and 1 mapping
train['Sex'] = train['Sex'].apply(lambda x:1 if x == 'female' else 0)
test['Sex'] = test['Sex'].apply(lambda x:1 if x == 'female' else 0)


# #### 'Age' column
# 
# Let us first see the distribution.

# In[ ]:


f, [ax1,ax2] = plt.subplots(1,2, figsize = (20,5))
sns.distplot(train['Age'][train['Survived'] == 1][train['Sex'] == 0], hist = False, ax = ax1, norm_hist = True, 
             label = 'Survived')
sns.distplot(train['Age'][train['Sex'] == 0], hist = False, ax = ax1, norm_hist = True, label = 'Male age distribution')
sns.distplot(train['Age'][train['Survived'] == 0][train['Sex'] == 0], hist = False, ax = ax2, norm_hist = True, 
             label = 'Didn\'t Survive')
sns.distplot(train['Age'][train['Sex'] == 0], hist = False, ax = ax2, norm_hist = True, label = 'Male age distribution')


# In a given plot, if both distributions are identical, it means that survival isn't dependant on variable under consideration. Here, both distributions differ considerably only for age < 15 and some difference for age group 15 to 30.

# In[ ]:


f, [ax1,ax2] = plt.subplots(1,2, figsize = (20,5))
sns.distplot(train['Age'][train['Survived'] == 1][train['Sex'] == 1], hist = False, ax = ax1, norm_hist = True, 
             label = 'Survived')
sns.distplot(train['Age'][train['Sex'] == 1], hist = False, ax = ax1, norm_hist = True, label = 'Female age distribution')
sns.distplot(train['Age'][train['Survived'] == 0][train['Sex'] == 1], hist = False, ax = ax2, norm_hist = True, 
             label = 'Didn\'t Survive')
sns.distplot(train['Age'][train['Sex'] == 1], hist = False, ax = ax2, norm_hist = True, label = 'Female age distribution')
ax1.legend(loc = 'upper right')
ax2.legend(loc = 'upper right')


# There is very little diffrence between actual age distribution and distribution of survived females. Again, we see slightly higher survival rate for small values of age. This is suggests that children were given priority on the life boats.

# In[ ]:


# We will now create age groups and check survival rate.
cut_offs = [0,15,30,80]
temp = pd.DataFrame(columns = ['Sex','Survived','age_group'])
for i in range(1,len(cut_offs)):
    df = train[["Survived",'Sex']][train['Age']>cut_offs[i-1]][train['Age']<=cut_offs[i]].groupby(['Sex'],as_index=False).mean()
    df['age_group'] = 'less than ' + str(cut_offs[i])
    temp = temp.append(df, ignore_index = True)


# In[ ]:


ax = sns.barplot(x = 'age_group', y = 'Survived', hue = 'Sex', data = temp)
ax.legend(bbox_to_anchor=(1.25, 1))


# In[ ]:


# Let us map values in age column to appropriate age groups.
train['Age'] = train['Age'].apply(lambda x: 1 if x <= 15 else 2 if x <= 30 else 3)
test['Age'] = test['Age'].apply(lambda x: 1 if x <= 15 else 2 if x <= 30 else 3)

train = pd.get_dummies(data = train, columns = ['Age'])
test = pd.get_dummies(data = test, columns = ['Age'])

# 2nd age group has lowest survival rate overall, so we will treat that as a base case and delete that column.
train.drop(['Age_2'], axis = 1, inplace = True)
test.drop(['Age_2'], axis = 1, inplace = True)


# #### 'Fare' column
# 
# Fare is a combination of Passenger class, where the passenger embarked and also, whether or not they had a cabin. Let us try to see 

# In[ ]:


# 'Fare' is expected to correlate with 'Pclass'
f, [ax1, ax2] = plt.subplots(1,2, figsize = (20,5))
sns.barplot(hue = 'Embarked', y = 'Fare', x = 'Pclass', data = train[train['Cabin'] == 0], ax = ax1, hue_order = ['S','C','Q'])
sns.barplot(hue = 'Embarked', y = 'Fare', x = 'Pclass', data = train[train['Cabin'] == 1], ax = ax2, hue_order = ['S','C','Q'])
ax1.set_title('Doesn\'t have Cabin')
ax2.set_title('Has Cabin')
ax2.set_ylim([0,180])
plt.show()


# As expected, there is correlation between Pclass and Fare. Additionally, 'Embarked' and 'Cabin' seems to have some impact as well, but there is no obvious trend.

# In[ ]:


def map_fare(x, cut_offs = None):
    if cut_offs == None:
        cut_offs = train['Fare'].describe()[['min','25%','50%','75%','max']]
    cut_offs = np.sort(cut_offs)
    for i in range(1,len(cut_offs)):
        if x <= cut_offs[i]:
            return i


# In[ ]:


# Let us find Pearson correlation coefficient between Pclass and mapped values of 'Fare'
mapped_fares = train['Fare'].apply(map_fare)
mapped_fares.corr(train['Pclass'])


# Correlation coefficient value is negative because numerically lower value of class means higher fare.<br>
# Note that having correlated variables doesn't affect the model, rather it affects the interpretation of the results.

# In[ ]:


test['Fare'] = test['Fare'].apply(map_fare)
train['Fare'] = mapped_fares


# #### 'Embarked' and 'title' columns
# These are still categorical type. We will use dummy variables to encode them.

# In[ ]:


train = pd.get_dummies(train, columns = ['Embarked', 'title'])
test = pd.get_dummies(test, columns = ['Embarked', 'title'])

train.drop(['Embarked_S','title_rare'], inplace = True, axis = 1)
test.drop(['Embarked_S','title_rare'], inplace = True, axis = 1)


# In[ ]:


train.drop('PassengerId', axis = 1, inplace = True)
test_passenger_id = test['PassengerId']
test.drop('PassengerId', axis = 1, inplace = True)


# In[ ]:


# Let us check dtypes to ensure every variable is numeric.
train.dtypes


# In[ ]:


test.dtypes


# #### 2.2.5 Correlations
# Let us explore through correlation values.
# <a id='2.2.5'></a>

# In[ ]:


correlations = train.corr()


# In[ ]:


d = {}
for col in correlations:
    temp = correlations[col].drop(col)
    for row in temp.index:
        if abs(temp[row]) > 0.5:
            if row + ' - ' + col not in d:
                d[col + ' - ' + row] = float('{:.4f}'.format(temp[row]))
d


# Possible explanations for these observations :
# 
# 1. Title 'Master' is used for young age boys.
# 2. 'Fare', 'Cabin', 'Pclass' are expected to be correlated to each other.
# 3. Title does depend on Gender of the person.
# 4. Correlation between Survival and gender indicates that females had higher chances of survival.

# In[ ]:


# If you wish to see diagramatic representation of correlations, you can 'uncomment' code in this cell.

# f, ax = plt.subplots(1,1,figsize = (12,12))
# sns.set(font_scale = 1)
# sns.heatmap(correlations,square=True, annot=True, ax = ax, cmap = 'PuBu', cbar=True,
#             cbar_kws={"shrink": 0.75}, fmt = '.2f')
# plt.setp(ax.get_xticklabels(), fontsize=14)
# plt.setp(ax.get_yticklabels(), fontsize=14)
# plt.show()


# <a id='3'></a>
# ## 3. Models and predictions
# 
# Now that we have created and modified our features, it is time to train various models and obtain predictions on test dataset. We will first train few models and then use model ensembling ([kaggle-ensembling-guide](https://mlwave.com/kaggle-ensembling-guide/)). Ensembled models typically have a lower generalisation error. It is a strategy used by many Kaggle compitition winners. There is a simple reasoning why ensembling reduces error. Say you have 2 fairly uncorrelated models. Then probability of same example being misclassified is low.<br>
# [This notebook](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/notebook) is excellent example of ensembling.

# <a id='3.1'></a>
# ### 3.1 First level models
# There are many classifier algorithms available in sklearn library. We will train 8 first level models. Note that, typically you will not train so many models. It is done here for the purpose of demonstartion. You are advised to go through all of them and play around with various parameters to see what effect they have on the output, accuracy, time taken from training etc.<br>

# In[ ]:


seed = 0  # Seed to use when calling functions involving random selection. Important for reproducibility
kf = KFold(n_splits = 4, random_state = seed)
survived = train['Survived']
train.drop('Survived', axis = 1, inplace = True)


# KFold ([sklearn.model_selection.KFold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)):<br>
# Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default). <br>
# Each fold is then used once as a validation while the k - 1 remaining folds form the training set.<br>

# In[ ]:


list_of_indices = []
for (_, temp) in kf.split(train.index):
    for index in temp:
        list_of_indices.append(index)
train_predictions = pd.DataFrame(index = list_of_indices)
test_predictions = pd.DataFrame()


# In[ ]:


def train_model(clf_name, clf, prediction_df, train_df, test_df):
    prediction_df[clf_name] = [-1]*prediction_df.shape[0]
    temp = pd.DataFrame()
    
    for i, (train_index, test_index) in enumerate(kf.split(train_df.index)):
        x = train_df.loc[train_index]
        y = survived.loc[train_index]
        test_values = train_df.loc[test_index]
        
        clf.fit(x,y)
        
        prediction_df[clf_name].loc[test_index] = list(clf.predict(test_values))
        temp[i] = list(clf.predict(test_df))
        
    test_predictions[clf_name] = temp.apply(lambda x: x.value_counts().index[0], axis = 1)


# ### Logistic regression

# In[ ]:


# Initialize the model with desired parameters.
lr = LogisticRegression(random_state = seed)
train_model(clf_name = 'logistic_regression', clf = lr, prediction_df = train_predictions, train_df = train, test_df = test)


# ### SVC

# In[ ]:


# Initialize the model with desired parameters.
svc = SVC(random_state = seed, kernel = 'linear', C = 0.025)
train_model(clf_name = 'SVC', clf = svc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### Decision Tree Classifier

# In[ ]:


# Initialize the model with desired parameters.
dtc = DecisionTreeClassifier(random_state = seed, max_depth = 10, min_samples_split = 30)
train_model(clf_name = 'decision_tree_classifier', clf = dtc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### Random Forest Classifier

# In[ ]:


# Initialize the model with desired parameters.
rfc = RandomForestClassifier(random_state = seed, n_estimators = 500, warm_start = True,
                             max_depth = 5, min_samples_leaf = 5)
train_model(clf_name = 'random_forest_classifier', clf = rfc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### Extra Trees Classifier

# In[ ]:


# Initialize the model with desired parameters.
etc = ExtraTreesClassifier(random_state = seed, n_estimators = 500, warm_start = True,
                             max_depth = 8, min_samples_leaf = 5)
train_model(clf_name = 'extra_trees_classifier', clf = etc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### Gradient Boosting Classifier

# In[ ]:


# Initialize the model with desired parameters.
gbc = GradientBoostingClassifier(random_state = seed, n_estimators = 50, warm_start = True, learning_rate = 0.1,
                                 max_depth = 5, min_samples_leaf = 25)
train_model(clf_name = 'gradient_boosting_classifier', clf = gbc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### Ada Boost Classifier

# In[ ]:


# Initialize the model with desired parameters.
abc = AdaBoostClassifier(random_state = seed, n_estimators = 500)
train_model(clf_name = 'ada_boost_classifier', clf = abc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### K-neighbours Classifier

# In[ ]:


# Initialize the model with desired parameters.
knc = KNeighborsClassifier(p = 2, n_neighbors = 3)
train_model(clf_name = 'k_neighbors_classifier', clf = knc, prediction_df = train_predictions, train_df = train, test_df = test)


# ### Accuracy and AUC scores
# 
# There are various objective criterias to judge how good the model is. Confusion matrix ([insert link here](wiki confusion matrix)) provides list of many such indicators, which can be calculated easily. Depending upon the problem at hand, appropriate indicators must be selected to decide usefullness of the model.<br>
# 
# #### Accuracy
# 
# Accuracy of the model is simply number of correct predictions divided by number of total predictions.
# 
# #### AUC score
# 
# AUC is area under ROC curve. Higher AUC score usually means better model. Although it is not always true.

# In[ ]:


accuracy = {}
for col in train_predictions.columns:
    accuracy[col] = sum([1 if train_predictions[col].loc[i] == survived.loc[i] else 0 for i in survived.index])/791


# In[ ]:


fig, ax = plt.subplots(1,1, figsize = (10,5))
sns.barplot(x = sorted(accuracy, key = accuracy.get, reverse = True), y = np.sort(list(accuracy.values()))[::-1],
            ax = ax, color = 'c')
for label in ax.get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(15)


# In[ ]:


auc_score = {}
for col in train_predictions.columns:
    auc_score[col] = roc_auc_score(survived, train_predictions[col])


# In[ ]:


fig, ax = plt.subplots(1,1, figsize = (10,5))
sns.barplot(x = sorted(auc_score, key = auc_score.get, reverse = True), y = np.sort(list(auc_score.values()))[::-1],
            ax = ax, color = 'c')
for label in ax.get_xticklabels():
    label.set_rotation(90)
    label.set_fontsize(15)


# In[ ]:


corr = train_predictions.corr()
f, ax = plt.subplots(1,1,figsize = (12,12))
sns.set(font_scale = 1)
sns.heatmap(corr,square=True, annot=True, ax = ax, cmap = 'PuBu', cbar=True,
            cbar_kws={"shrink": 0.75}, fmt = '.2f')
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.show()


# <a id = '3.2'></a>
# ### 3.2 Second level model
# 
# Second level models train on predictions of first level model. Such models usually have lower generalization error. This is explained in more detail with a simple model below.

# In[ ]:


first_level_models = train_predictions.columns


# #### Majority voting
# 
# Now we have 7 first level models and their predictions. Chances that a particular example will be classified into wrong category by majority of the models are low. So, our simple second level model will be to look at the predictions from all 7 models and take majority vote.

# In[ ]:


train_predictions['majority_voting_all_models'] = train_predictions[first_level_models].apply(lambda x: x.value_counts().index[0], axis = 1)
test_predictions['majority_voting_all_models'] = test_predictions[first_level_models].apply(lambda x: x.value_counts().index[0], axis = 1)
accuracy['majority_voting_all_models'] = sum([1 if train_predictions['majority_voting_all_models'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791
accuracy['majority_voting_all_models']


# Note that there is strong correlation between many of the selected models. This means they will dominate the voting and will not allow us to benefit from the differences between models. So, let us select 3 models with highest accuracy but low correlation.

# In[ ]:


# 'logistic_regression','extra_trees_classifier','gradient_boosting_classifier' and 'random_forest_classifier' are top 4 models.
# But 'extra_trees_classifier' and 'random_forest_classifier' have high correlation, so we'' choose only 1 out of these 2.
selected_cols = ['logistic_regression','extra_trees_classifier','gradient_boosting_classifier']
train_predictions['majority_voting_selected_cols'] = train_predictions[selected_cols].apply(lambda x: x.value_counts().index[0],
                                                                                            axis = 1)
test_predictions['majority_voting_selected_cols'] = test_predictions[selected_cols].apply(lambda x: x.value_counts().index[0],
                                                                                            axis = 1)
accuracy['majority_voting_selected_cols'] = sum([1 if train_predictions['majority_voting_selected_cols'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791
accuracy['majority_voting_selected_cols']


# #### Logistic regression
# Let us try to train Logistic regression model as a second level model. See if it performs better than majority voting.

# In[ ]:


lr_second_level = LogisticRegression(random_state = seed)
train_model(clf_name = 'logistic_regression_second_level', clf = lr_second_level, prediction_df = train_predictions,
            train_df = train_predictions[first_level_models], test_df = test_predictions[first_level_models])
accuracy['logistic_regression_second_level'] = sum([1 if train_predictions['logistic_regression_second_level'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791
accuracy['logistic_regression_second_level']


# In[ ]:


lr_second_level_selected_cols = LogisticRegression(random_state = seed)
train_model(clf_name = 'logistic_regression_second_level_selected_cols', clf = lr_second_level, prediction_df = train_predictions,
            train_df = train_predictions[selected_cols], test_df = test_predictions[selected_cols])
accuracy['logistic_regression_second_level_selected_cols'] = sum([1 if train_predictions['logistic_regression_second_level_selected_cols'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791
accuracy['logistic_regression_second_level_selected_cols']


# In[ ]:


most_accurate_clfs = sorted(accuracy, key = accuracy.get, reverse = True)
most_accurate_clfs


# In[ ]:


submission = pd.DataFrame({'PassengerId' : test_passenger_id,
                          'survived' : test_predictions['logistic_regression_second_level_selected_cols']})
submission.to_csv('titanic.csv', index = False)

