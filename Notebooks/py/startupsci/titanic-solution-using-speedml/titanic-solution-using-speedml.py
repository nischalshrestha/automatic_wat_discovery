#!/usr/bin/env python
# coding: utf-8

# # Titanic Solution Using Speedml
# 
# ### This notebook uses the [Speedml](https://speedml.com)  Python package to speed start machine learning projects.
# 
# Speedml integrates best ML packages and popular strategies used by top data scientists in an easy to use Python package.
# 
# > Using Speedml on Titanic dataset we **quickly jumped from low 80% rank to top 20% rank**, within just a few feature engineering iterations.
# 
# Speedml is under active development and Kaggle version of the API may not be the latest. For demonstrating the latest Speedml features we have also created the same notebook on GitHub.
# 
# ### Download the latest version of [this notebook from GitHub](https://github.com/Speedml/notebooks/blob/master/titanic/titanic-solution-using-speedml.ipynb).
# 
# ### Table of Contents
# 
# - Loading the datasets.
# - Differentiating between numerical, categorical, high-cardinality, and continuous features.
# - Feature correlation heatmap matrix.
# - Feature distribution for outliers detection.
# - Plotting continuous features for outliers detection.
# - Plotting categorical features for outliers detection.
# - Fixing outliers with a single line of code.
# - Plotting continuous features against categorical features.
# - Feature engineering for high-cardinality.
# - New feature extraction from existing features.
# - Hyper-parameters tuning for model classifier.
# - Model evaluation and accuracy ranking.
# - Model prediction and feature selection.
# - Saving the results.

# ## Getting Started
# 
# To get started all you need to do is include one package in your project. Speedml includes pandas, sklearn, numpy, xgboost, by default so you do not need to import these.

# In[ ]:


from speedml import Speedml

get_ipython().magic(u'matplotlib inline')


# It takes one line of code to initialize train, test datasets, define the target and unique id variables. This also initializes wrapper components for EDA (sml.plot), XGBoost (sml.xgb), modeling (sml.model), feature engineering (sml.feature) and more...

# In[ ]:


sml = Speedml('../input/train.csv', 
              '../input/test.csv', 
              target = 'Survived',
              uid = 'PassengerId')


# You can access pandas directly as a Speedml component.

# In[ ]:


sml.train.head()


# ## Feature Correlations
# 
# You can quickly check feature correlations using a plot. Learn how to interpret this plot at https://speedml.com/plot-correlation-of-features/ 

# In[ ]:


sml.plot.correlate()


# ## Outliers Detection and Fix
# 
# We can use distributions to understand skew (left/right) for determining outliers.

# In[ ]:


sml.plot.distribute()


# Continuous or high-cardinality numerical features are better plotted using scatter plot for determining outliers.
# 
# We do not expect outliers in case of Age feature as the distribution plot is fairly close to normal (rising in the middle of x-axis and falling on either sides evenly).

# In[ ]:


sml.plot.continuous('Age')


# The method clearly shows some outliers in case of Fare feature. This coincides with our observation from the distribution plot for the Fare feature which is skewed towards left.

# In[ ]:


sml.plot.continuous('Fare')


# To correct the outliers we fix only values in upper range of the 99th percentile. As the results show these constitute around 1% of overall samples.

# In[ ]:


sml.feature.outliers('Fare', upper=99)


# While we impact only a few samples, the outliers fix is fairly significant as shown by the same plot after the fix.

# In[ ]:


sml.plot.continuous('Fare')


# Let us fix a categorical feature this time.

# In[ ]:


sml.plot.ordinal('Parch')
print(sml.feature.outliers('Parch', upper=99))
sml.plot.ordinal('Parch')


# ## Feature Engineering For High-Cardinality
# 
# High-cardinality features like Ticket and Age are candidates for feature engineering. We use the density method to create a new feature based on Age and Ticket and drop the Ticket feature in turn. This simple iteration improves our model significantly and helps us jump 100s of positions on the Kaggle leaderboard.

# In[ ]:


sml.feature.density('Age')
sml.train[['Age', 'Age_density']].head()


# In[ ]:


sml.feature.density('Ticket')
sml.train[['Ticket', 'Ticket_density']].head()


# In[ ]:


sml.feature.drop(['Ticket'])


# ## Extracting New Features
# 
# We will now extract new features Deck from Cabin and FamilySize from Parch and SibSp.

# In[ ]:


sml.plot.crosstab('Survived', 'SibSp')


# In[ ]:


sml.plot.crosstab('Survived', 'Parch')


# This cell demonstrates linear, concise workflow Speedml API enables within few lines of code.

# In[ ]:


sml.feature.fillna(a='Cabin', new='Z')
sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')
sml.feature.drop(['Cabin'])
sml.feature.mapping('Sex', {'male': 0, 'female': 1})
sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')
sml.feature.add('FamilySize', 1)


# In[ ]:


sml.plot.crosstab('Survived', 'Deck')


# In[ ]:


sml.plot.crosstab('Survived', 'FamilySize')


# In[ ]:


sml.feature.drop(['Parch', 'SibSp'])


# Here is a single line of code to impute all empty features values (numerical and text) with numerical median or most common text value.

# In[ ]:


sml.feature.impute()


# In[ ]:


sml.train.info()
print('-'*50)
sml.test.info()


# In[ ]:


sml.plot.importance()


# In[ ]:


sml.train.head()


# In[ ]:


sml.feature.extract(new='Title', a='Name', regex=' ([A-Za-z]+)\.')
sml.plot.crosstab('Title', 'Sex')


# In[ ]:


sml.feature.replace(a='Title', match=['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], new='Rare')


# In[ ]:


sml.feature.replace('Title', 'Mlle', 'Miss')


# In[ ]:


sml.feature.replace('Title', 'Ms', 'Miss')
sml.feature.replace('Title', 'Mme', 'Mrs')
sml.train[['Name', 'Title']].head()


# In[ ]:


sml.feature.drop(['Name'])
sml.feature.labels(['Title', 'Embarked', 'Deck'])
sml.train.head()


# In[ ]:


sml.plot.importance()


# In[ ]:


sml.plot.correlate()


# In[ ]:


sml.plot.distribute()


# ### For next steps in the workflow including hyper-parameter tuning, model evaluation, feature selection.
# 
# ### Download the latest version of [this notebook from GitHub](https://github.com/Speedml/notebooks/blob/master/titanic/titanic-solution-using-speedml.ipynb).
