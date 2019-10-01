#!/usr/bin/env python
# coding: utf-8

# ## Titanic EDA Using Speedml
# 
# This notebook performs Exploratory Data Analysis (EDA) on the Titanic dataset using the [Speedml](https://speedml.com) package.
# 
# > Speedml is a Python package for speed starting machine learning projects.
# 
# Speedml imports and initializes popular packages like pandas, xgboost, and sklearn, so you only need to import one package. Simple.

# In[ ]:


from speedml import Speedml

get_ipython().magic(u'matplotlib inline')


# ## Initialize
# 
# Let us load the datasets, identify target variable `Survived` and unique id `PassengerId` using single call to Speedml.
# 
# Then return the shape information (#samples, #features) or (#features).

# In[ ]:


sml = Speedml('../input/train.csv', '../input/test.csv', 
              target = 'Survived', uid = 'PassengerId')
sml.shape()


# ## Datasets
# 
# Speedml API exposes pandas methods directly so you can do with speedml what you can do with pandas.

# In[ ]:


sml.train.head()


# **Observations**
# 
# - The dataset contains several text features which need to be converted to numeric for model ready data.
# - Name feature may contain inconsistent non-categorical data. Candidate for feature extraction and dropping.
# - Is Ticket feature categorical? Do Ticket values remain same across multiple samples or passengers?

# In[ ]:


sml.train.describe()


# In[ ]:


sml.train.info()
print('-'*40)
sml.test.info()


# **Observations**
# 
# - Age feature contains null values which may need to be imputed.
# - Cabin feature has a lot of null values
# - Embarked feature has few null values for train dataset.

# ## Correlations
# 
# Plot correlation matrix heatmap for numerical features of the training dataset. Use this plot to understand if certain features are duplicate, are of low importance, or possibly high importance for our model.

# In[ ]:


sml.plot.correlate()


# ## Distributions
# 
# Plot multiple feature distribution histogram plots for all numeric features. This helps understand skew of distribution from normal (horizontal middle) to quickly and relatively identify outliers in the dataset.

# In[ ]:


sml.plot.distribute()


# ## Outliers for categorical features
# 
# We use Violin plots on categorical features to note distribution of values across target variable and existence of any outliers (long thin lines extending out in the plots).

# In[ ]:


sml.plot.ordinal('Parch')


# In[ ]:


sml.plot.ordinal('SibSp')


# ## Outliers for continuous features
# 
# We use scatter plots to determine outliers for continuous features. The further out and spread the upper or lower part of the curve, the more the outliers deviate from normal distribution.

# In[ ]:


sml.plot.continuous('Age')


# In[ ]:


sml.plot.continuous('Fare')


# ## Cross-tabulate features and target
# 
# Following analysis uses simple crosstab method to note how samples are distributed across target variable when classified by a certain feature.

# In[ ]:


sml.plot.crosstab('Survived', 'Pclass')


# In[ ]:


sml.plot.crosstab('Survived', 'Parch')


# In[ ]:


sml.plot.crosstab('Survived', 'SibSp')


# In[ ]:


sml.plot.crosstab('Survived', 'Sex')


# In[ ]:


sml.plot.crosstab('Survived', 'Embarked')

