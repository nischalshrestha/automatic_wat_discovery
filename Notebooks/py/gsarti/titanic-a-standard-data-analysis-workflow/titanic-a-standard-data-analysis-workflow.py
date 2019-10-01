#!/usr/bin/env python
# coding: utf-8

# # Titanic Competition - An Introduction to Data Analytics
# ## By Gabriele Sarti
# 
# * URL of the Kaggle competition: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
# 
# * Find it here on Github: **_TODO_**
# 
# ## Table of Contents
# 
# 0. [Defining the problem](#problem)
# 1.  [Preliminary steps](#first-steps)
# 2. [Data cleaning](#clean)
# 3. [Data exploration](#explore)
# 4. [Data visualization](#viz)
# 5. [Data modeling](#model)
# 6. [Parameter tuning](#tune)
# 7. [Final results](#end)

# ## 1 - Preliminary steps<a name="first-steps"></a>
# 
# ### Load data analysis libraries
# 
# 
# I will use the standard tools for performing data analysis in Python, namely **numpy** for linear algebra and arrays,  **pandas** for managing the dataset and I/O data operations, **scipy** for advanced mathematics operations and **matplotlib** for plotting the results and gaining insights.
# 
# I will also use **sklearn** for data cleaning and data modeling, along with **seaborn** for additional visualization purposes.

# In[ ]:


# General libraries for scientific purposes
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import sklearn
import os
import sys

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data models
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
from xgboost import XGBClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Libraries versions
print('Python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))
print('pandas: {}'.format(pd.__version__))
print('scipy: {}'.format(sp.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('-'*30)

# Print local folder content
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### Load data
# 
# Both the datasets are found inside the **input** folder, accessible as it follows.
# 
# I create a copy of the training set in order to set up a data cleaning pipeline and try different transformations.

# In[ ]:


# Importing local datasets
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

# Reference of both train and test, for cleaning purposes
datasets = [train, test]


# ### First observations
# 
# I use the functions `head`, `info` and `describe` to acquire some basic useful information about the training set.
# 

# In[ ]:


train.head()


# In[ ]:


train.info()
print('-'*30)
test.info()
train.describe(include = 'all')


# We are interested in particular in the null values present inside both the train and the test dataset, since we will have to adjust them in order to perform our analysis.

# In[ ]:


print('Null training values:\n', train.isnull().sum())
print("-"*30)
print('Test/Validation columns with null values:\n', test.isnull().sum())


# We can recap the data dictionary by adding this new information:
# 
# | Type    | Variable | Definition                                  | Key                                            | # train values | # train null values | # test values | # test null values |
# |---------|----------|---------------------------------------------|------------------------------------------------|----------------|---------------------|---------------|--------------------|
# | Categorical   | survived | Survival                                    | 0 = No, 1 = Yes                                | 891            | 0                   | 418           | 0                  |
# | Ordinal   | pclass   | Ticket class                                | 1 = 1st, 2 = 2nd, 3 = 3rd                      | 891            | 0                   | 418           | 0                  |
# | String  | name     | Name of the passenger                       |                                                | 891            | 0                   | 418           | 0                  |
# | Categorical  | sex      | Sex                                         |                                                | 891            | 0                   | 418           | 0                  |
# | Continuous | age      | Age in years                                |                                                | 712            | 177                 | 332           | 86                 |
# | Categorical  | sibsp    | # of siblings / spouses aboard the  Titanic |                                                | 891            | 0                   | 418           | 0                  |
# | Categorical  | parch    | # of parents / children aboard the  Titanic |                                                | 891            | 0                   | 418           | 0                  |
# | String  | ticket   | Ticket number                               |                                                | 891            | 0                   | 418           | 0                  |
# | Continuous | fare     | Passenger fare                              |                                                | 891            | 0                   | 417           | 1                  |
# | String  | cabin    | Cabin number                                |                                                | 204            | 687                 | 91            | 327                |
# | Categorical  | embarked | Port of embarkation                         | C = Cherbourg, Q = Queenstown, S = Southampton | 889            | 2                   | 418           | 0                  |
# 
# #### Variable notes:
# 
# 
# * **pclass:** A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# * **age:** Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * **sibsp:** The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * **parch:** The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# ### Some considerations from preliminary phase
# 
# Since some of the models I will benchmark for this competition require the entries to be non-null, I must take care of null values. 
# 
# The most prominent categories having null values are `age` and `cabin`, with a consistent number of null values, followed by `embarked` (only in the train set) and `fare` (only in the test set), having just a couple of nulls.
# 
# 

# ## 2 - Data Cleaning <a name="clean"></a>
# 
# ### Removing null values
# 
# The approach I choose for dealing with those variables is the following:
# * Since age and fare are a float values, I will fill null entries with the respective medians across the dataset.
# * Since embarked is categorical, I will fill null embarked entries with the mode of the variable across the dataset.
# * Since cabin is not relevant for my analysis, I will drop it along with `PassengerId` and `Ticket` to reduce the noise in my training set.

# In[ ]:


drop_column = ['Cabin', 'Ticket']

for d in datasets:    
    d['Age'].fillna(d['Age'].median(), inplace = True)
    d['Fare'].fillna(d['Fare'].median(), inplace = True)
    d['Embarked'].fillna(d['Embarked'].mode()[0], inplace = True)
    d.drop(drop_column, axis=1, inplace = True)    

print(train.isnull().sum())
print("-"*30)
print(test.isnull().sum())


# ### Experimenting with feature combinations
# 
# By observing the data and other people kernels, I got some ideas for performing feature combination:
# 
# * Merging `Sibsp` and `Parch` in a single numerical discrete attribute, called `FamilySize`.
# * Creating a `IsAlone` categorical attribute that is true when `FamilySize = 0`.
# * Creating a `Title` categorical attribute, with titles extracted from passengers' names.
# * Converting `Fare` and `Age` attributes into ordinal ones by grouping values into categories.

# In[ ]:


for d in datasets:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

    d['IsAlone'] = 1
    d['IsAlone'].loc[d['FamilySize'] > 1] = 0
    
    # Extract titles from names
    d['Title'] = d['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group uncommon titles under "other" label
    d['Title'] = d['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    
    # Group synonyms together 
    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')

    # Grouping continuous values into categories
    d['FareBand'] = pd.qcut(d['Fare'], 4)
    d['AgeBand'] = pd.cut(d['Age'].astype(int), 5)

train.info()
test.info()
train.head()


# ### Converting formats
# 
# We can now convert all the categorical literal values in numerical values thanks to the `LabelEncoder` class.
# 
# We create two clones before applying the transformation for plotting and understanding purposes.

# In[ ]:


le = LabelEncoder()

titanic_train = train.copy(deep = 'True')
titanic_test = test.copy(deep = 'True')
titanic_train['Age'] = titanic_train['AgeBand']
titanic_train['Fare'] = titanic_train['FareBand']
titanic_test['Age'] = titanic_test['AgeBand']
titanic_test['Fare'] = titanic_test['FareBand']

column_transformed = ['Sex', 'Embarked', 'Title', 'AgeBand', 'FareBand']
column_transform = ['Sex', 'Embarked', 'Title', 'Age', 'Fare']

for d in datasets:
    for i in range(len(column_transform)):
        d[column_transform[i]] = le.fit_transform(d[column_transformed[i]])

datasets.append(titanic_train)
datasets.append(titanic_test)


# ### Remove unnecessary columns
# 
# Now that we have engineered our new columns and converted the old ones, we can finally drop unnecessary columns
# * `Name`, since it is a simple string from which we already extracted the title
# * `SibSp` and `Parch` since they have been merged in `FamilySize`, and they don't seem very correlated with the survival when taken alone.
# * `FareBand` and `AgeBand`, since we used them to change `Age` and `Fare` values.
# * For the training set, we also want to drop the `PassengerId`

# In[ ]:


drop_column = ['Name', 'SibSp', 'Parch', 'FareBand', 'AgeBand']
for d in datasets:
        d.drop(drop_column, axis=1, inplace = True)
train.drop('PassengerId', axis=1, inplace = True)
titanic_train.drop('PassengerId', axis=1, inplace = True)


# In[ ]:


train.head()


# In[ ]:


titanic_train.head()


# ## 3 - Data exploration <a name="explore"></a>
# 
# ### Looking for correlations
# 
# We'll use the `corr` method to compute the standard correlation coefficient between the numerical features and the survival of our passengers.
# 
# For categorical values, we'll simply observe the survival rate compared to each category.

# In[ ]:


feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
corr_matrix = train.corr()

print(corr_matrix["Survived"].sort_values(ascending=False))
print("-"*30)

for feature in feature_names:
    print('Correlation between Survived and', feature)
    print(titanic_train[[feature, 'Survived']].groupby([feature], as_index=False).mean())
    print("-"*30)


# Thoughts on the correlation patterns:
# * Being female is definitely a significant factor, shown both by `Sex` and `Title` attributes.
# * Being married, or at least in company of someone, seems relevant, as both shown by `FamiliSize` and `Title` attributes
# * Social rank and financial capability seems to play a role, as we can see in `Title` (Master), `Pclass` and `Fare`. 
# 
# All those points are quite intuitive.
# 

# ## 4 - Data Visualization <a name="viz"></a>
# 
# # TODO

# ## 5 - Data Modeling <a name="model"></a>
# 
# ### Creating train and test copies for models

# In[ ]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# The set of models I am going to compare
models = [
            LinearRegression(),
            LogisticRegressionCV(),
            Perceptron(),
            GaussianNB(),
            KNeighborsClassifier(),
            SVC(probability=True),
            DecisionTreeClassifier(),
            AdaBoostClassifier(),
            RandomForestClassifier(),
            XGBClassifier()    
        ]

# Create a table of comparison for models
models_columns = ['Name', 'Parameters','Train Accuracy', 'Validation Accuracy', 'Execution Time']
models_df = pd.DataFrame(columns = models_columns)
predictions = pd.DataFrame(columns = ['Survived'])

cv_split = ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 )

index = 0
for model in models:
    models_df.loc[index, 'Name'] = model.__class__.__name__
    models_df.loc[index, 'Parameters'] = str(model.get_params())
    
    scores = cross_validate(model, X_train, Y_train, cv= cv_split)

    models_df.loc[index, 'Execution Time'] = scores['fit_time'].mean()
    models_df.loc[index, 'Train Accuracy'] = scores['train_score'].mean()
    models_df.loc[index, 'Validation Accuracy'] = scores['test_score'].mean()   
    
    index += 1

models_df.sort_values(by = ['Validation Accuracy'], ascending = False, inplace = True)
models_df


# ## 6 - Parameter Tuning <a name="tune"></a>
# 
# ### Grid Search Tuning

# In[ ]:


param_grid = {
              'criterion': ['gini', 'entropy'],
              'max_depth': [2,4,6,8,10,None],
              'random_state': [0]
             }

tree = DecisionTreeClassifier(random_state = 0)
score = cross_validate(tree, X_train, Y_train, cv  = cv_split)
tree.fit(X_train, Y_train)

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
grid_search.fit(X_train, Y_train)

print('Before GridSearch:')
print('Parameters:', tree.get_params())
print("Training score:", score['train_score'].mean()) 
print("Validation score", score['test_score'].mean())
print('-'*30)
print('After GridSearch:')
print('Parameters:', grid_search.best_params_)
print("Training score:", grid_search.cv_results_['mean_train_score'][grid_search.best_index_]) 
print("Validation score", grid_search.cv_results_['mean_test_score'][grid_search.best_index_])


# ## 7 - Final Results <a name="end"></a>
# 
# The final achieved result using a `DecisionTreeClassifier` is 79.425 % of accuracy on the test set.

# In[ ]:


final_tree = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
final_tree.fit(X_train, Y_train)
test['Survived'] = final_tree.predict(X_test)

submission = test[['PassengerId','Survived']]
submission.to_csv("submission.csv", index=False)


# In[ ]:




