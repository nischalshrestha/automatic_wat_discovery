#!/usr/bin/env python
# coding: utf-8

# # Automated Feature Engineering and Selection for Titanic Dataset

# ***Liana Napalkova***
# 
# ***3 September 2018***

# # Table of contents
# 1. [Introduction](#introduction)
# 2. [Load data](#load_data)
# 3. [Clean data](#clean_data)
# 4. [Automated feature engineering](#afe)
# 5. ["Curse of dimensionality": Feature reduction and selection](#frs)
# 6. [Training and testing the simple model](#ttm)

# ## 1. Introduction <a name="introduction"></a>

# If you have ever manually created hundreds of features for your ML project (I am sure you did it), then you will be happy to find out how the Python package called "featuretools" can help out with this task. The good news is that this package is very easy to use. It is aimed at automated feature engineering. **Of course human expertise cannot be substituted**, but nevertheless **"featuretools" can automate a large amount of routine work**. For the exploration purpose I will use a well-known Titanic dataset. The achieved resuls will be compared to the results obtained with handcrafted feature engineering and manual hyperparameter optimisation.
# 
# The main takeaways from this notebook are:
# 
# * Firstly, going from 11 total features to 146 using automated feature engineering ("featuretools" package).
# * Secondly, applying the feature reduction and selection methods to select X most relevant features out of 146 features.
# * The accuracy of 0.74162 on the public leaderboard with a basic random forest classifier.
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated.**

# In[ ]:


import pandas as pd
#import autosklearn.classification
import featuretools as ft
from featuretools.primitives import *
from featuretools.variable_types import Numeric
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# ## 2. Load data <a name="load_data"></a>

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
answers = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


print(train_df.columns.values)


# ## 3. Clean data <a name="clean_data"></a>
# 
# First of all, it is necessary to clean the data. Since the main focus of this notebook is to explore "featuretools", we will not re-invent the wheel in data cleaning. Therefore, we will apply the code of feature cleaning taken from one of existing Kernels - [Best Titanic Survival Prediction for Beginners](https://www.kaggle.com/vin1234/best-titanic-survival-prediction-for-beginners), where an interested reader can find a very detailed explanation of each steps of the data cleaning procedure.

# In[ ]:


combine = train_df.append(test_df)

passenger_id=test_df['PassengerId']
#combine.drop(['PassengerId'], axis=1, inplace=True)
combine = combine.drop(['Ticket', 'Cabin'], axis=1)

combine.Fare.fillna(combine.Fare.mean(), inplace=True)

combine['Sex'] = combine.Sex.apply(lambda x: 0 if x == "female" else 1)

for name_string in combine['Name']:
    combine['Title']=combine['Name'].str.extract('([A-Za-z]+)\.',expand=True)
    
#replacing the rare title with more common one.
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
combine.replace({'Title': mapping}, inplace=True)

combine = combine.drop(['Name'], axis=1)

titles=['Mr','Miss','Mrs','Master','Rev','Dr']
for title in titles:
    age_to_impute = combine.groupby('Title')['Age'].median()[titles.index(title)]
    combine.loc[(combine['Age'].isnull()) & (combine['Title'] == title), 'Age'] = age_to_impute
combine.isnull().sum()

freq_port = train_df.Embarked.dropna().mode()[0]
combine['Embarked'] = combine['Embarked'].fillna(freq_port)
    
combine['Embarked'] = combine['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
combine['Title'] = combine['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rev': 4, 'Dr': 5} ).astype(int)
combine.fillna(0, inplace=True)


# In[ ]:


combine.info()


# ## 4. Perform automated feature engineering <a name="afe"></a> 

# Once the data is cleaned, we can proceed to the cake in our party - i.e. automated feature engineering. To work with "featuretools" package, we should specify our dataframes "train_df" and "test_df" as entities of the entity set. The entity is just a table with a uniquely identifying column known as an index. The "featuretools" can automatically infer the variable types (numeric, categorical, datetime) of the columns, but it's a good idea to also pass in specific datatypes to override this behavior.

# In[ ]:


es = ft.EntitySet(id = 'titanic_data')

es = es.entity_from_dataframe(entity_id = 'combine', dataframe = combine.drop(['Survived'], axis=1), 
                              variable_types = 
                              {
                                  'Embarked': ft.variable_types.Categorical,
                                  'Sex': ft.variable_types.Boolean,
                                  'Title': ft.variable_types.Categorical
                              },
                              index = 'PassengerId')

es


# Once the entity set is created, it is possible to generate new features using so called **feature primitives**. A feature primitive is an operation applied to data to create a new feature. Simple calculations can be stacked on top of each other to create complex features. Feature primitives fall into two categories:
# 
# * **Aggregation**: these functions group together child datapoints for each parent and then calculate a statistic such as mean, min, max, or standard deviation. The aggregation works across multiple tables using relationships between tables.
# * **Transformation**: these functions work on one or multiple columns of a single table.
# 
# In our case we do not have different tables linked between each other. However, we can create dummy tables using "normalize_entity" function. What will it give us? Well, this way we will be able to apply both aggregation and transformation functions to generate new features. To create such tables, we will use categorical, boolean and integer variables.

# In[ ]:


es = es.normalize_entity(base_entity_id='combine', new_entity_id='Embarked', index='Embarked')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Sex', index='Sex')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Title', index='Title')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Pclass', index='Pclass')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='Parch', index='Parch')
es = es.normalize_entity(base_entity_id='combine', new_entity_id='SibSp', index='SibSp')
es


# In[ ]:


primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(primitives[primitives['type'] == 'aggregation'].shape[0])


# As we can see, the most of "transformation" functions are applied to datetime or time-dependent variables. In our dataset we do not have such variables. Therefore these functions will not be used.

# In[ ]:


primitives[primitives['type'] == 'transform'].head(primitives[primitives['type'] == 'transform'].shape[0])


# 1. Now we will apply a **deep feature synthesis (dfs)** function that will generate new features by automatically applying suitable aggregations, I selected a depth of 2. Higher depth values will stack more primitives. 

# In[ ]:


features, feature_names = ft.dfs(entityset = es, 
                                 target_entity = 'combine', 
                                 max_depth = 2)


# This is a list of new features. For example, "Title.SUM(combine.Age" means the sum of Age values for each unique value of Title.

# In[ ]:


feature_names


# In[ ]:


len(feature_names)


# In[ ]:


features[features['Age'] == 22][["Title.SUM(combine.Age)","Age","Title"]].head()


# By using "featuretools", we were able to **generate 146 features just in a moment**.
# 
# The "featuretools" is a powerful package that allows saving time to create new features from multiple tables of data. However, it does not completely subsitute the human domain knowledge. Additionally, now we are facing another problem known as the "curse of dimensionality".

# ## 5. "Curse of dimensionality": Feature reduction and selection <a name="frs"></a> 

# To deal with the "curse of dimensionality", it's necessary to apply the feature reduction and selection, which means removing low-value features from the data. But keep in mind that feature selection can hurt the performance of ML models. The tricky thing is that the design of ML models contains an artistic component. It's definitely not the deterministic process with strict rules that should be followed to achieve success. In order to come up with an accurate model, it is necessary to apply, combine and compare dozens of methods. In this notebook, I will not explain all possible approaches to deal with the "curse of dimensionality". I will rather concentrate on the following methods:
# 
#    * Determine collinear features
#    * Detect the most relevant features using linear models penalized with the L1 norm

# ### 5.1 Determine collinear features

# Collinearity means high intercorrelations among independent features. If we maintain such features in the mode, it might be difficult to assess the effect of independent features on target variable. Therefore we will detect these features and delete them, though applying a manual revision before removal.

# In[ ]:


# Threshold for removing correlated variables
threshold = 0.95

# Absolute value correlation matrix
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head(50)


# In[ ]:


# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d features to remove.' % (len(collinear_features)))


# In[ ]:


features_filtered = features.drop(columns = collinear_features)

print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])


# **Be aware, however, that it is not a good idea to remove features only by correlation without understanding the removal process**. Features that have very high correlation (for example, Embarked.SUM(combine.Age) and Embarked.SUM(combine.Fare)) with significant difference between may require additional inve. Therefore manual guidance is necessary. But this topic is outside of the scope of this Kernel.

# ### 5.2 Detect the most relevant features using linear models penalized with the L1 norm

# The next step is to use linear models penalized with the L1 norml. 

# In[ ]:


features_positive = features_filtered.loc[:, features_filtered.ge(0).all()]

train_X = features_positive[:train_df.shape[0]]
train_y = train_df['Survived']

test_X = features_positive[train_df.shape[0]:]


# Since the number of features is smaller than the number of observations in "train_X", the parameter "dual" is equal to False.

# In[ ]:


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_X, train_y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(train_X)
X_selected_df = pd.DataFrame(X_new, columns=[train_X.columns[i] for i in range(len(train_X.columns)) if model.get_support()[i]])
X_selected_df.shape


# In[ ]:


X_selected_df.columns


# ## 6. Training and testing the simple model <a name="ttm"></a> 

# Finally, we will create a basic random forest classifier with 2000 estimators. Please notice that I skip essential steps such as crossvalidation, the analysis of learning curves, etc.

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=2000,oob_score=True)
random_forest.fit(X_selected_df, train_y)


# In[ ]:


X_selected_df.shape


# In[ ]:


Y_pred = random_forest.predict(test_X[X_selected_df.columns])


# In[ ]:


print(Y_pred)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': passenger_id, 'Survived': Y_pred})
my_submission.to_csv('submission.csv', index=False)


# If we use 20 selected features, we get the accuracy of 0.74162 on public leaderboard. Indeed it is not high. However, if we compare it to the accuracy that we would receive with the same random forest classfier using 83 features from "features_positive", then we would get the accuracy of 0.73205.
# 
# The recommended approach is to combine outputs of "featuretools" with the human expert knowledge, and use crossvalidation in order to analyze learning curves and pick up the most efficient model.
