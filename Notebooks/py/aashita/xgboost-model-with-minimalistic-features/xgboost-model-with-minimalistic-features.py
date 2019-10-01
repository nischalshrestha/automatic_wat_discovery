#!/usr/bin/env python
# coding: utf-8

# # Titanic survival prediction 

# First we import the data and the relevant python modules:

# In[ ]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
target = train.Survived.astype('category', ordered=False)
train.drop('Survived', axis=1)

test = pd.read_csv('../input/test.csv')
PassengerId = test.PassengerId


# In[ ]:


train.head()


# # Exploratory Data Analysis

# A much higher percentage of female passengers survived the titanic shipwreck as compared to the male passengers.

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train);


# Similarly the graph below suggests that the ticket class can also be a useful indicator in predicting the survival.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train);


# Now we look at how the gender within the ticket classes affects the chance of survival.

# In[ ]:


sns.barplot(x='Sex', y='Survived', hue = 'Pclass', data=train);


# It seems like for both genders, ticket class play a role in predicting the chance of survival. 

# Next we check the survival rate among both genders within the three ticket classes.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', hue = 'Sex', data=train);


# It seems that though the gender affects the chance of survival for all the three ticket classes, its effect is most pronounced for the second class passenger and least prounced for the third class. This is also reflected by the slopes in the graph below.

# In[ ]:


sns.pointplot(x='Sex', y='Survived', hue = 'Pclass', data=train);


# Within each gender, surviving passengers have a higher median fare than those who did not survive. However, the gap in fares between those who survived and who did not is more pronounced among females. 

# In[ ]:


sns.factorplot(x='Sex', y='Fare', hue='Survived', data=train);


# The higher survival rate for female passengers can be explained by their higher fares to some extent. 

# In[ ]:


sns.boxplot(x='Sex', y='Fare', hue='Survived', data=train);


# Does it mean that when we compare the survival rate for male and female passengers who paid similar fares, the men have higher survival than women? The following two graphs indicates otherwise.

# In[ ]:


sns.stripplot(x='Sex', y='Fare', hue='Survived', data=train);


# In[ ]:


sns.swarmplot(x='Sex', y='Fare', hue='Survived', data=train);


# We observed that though each of the feature - Sex, Class and Fare is closely related to the survival, there is an interesting interplay among them. I have not included the interplay among the other remaining features to keep the kernel length as short as possible, but the kernels in the references mentioned below have explored it in more detail.
# After we train the model, we will check again which features the model considered more important than others in predicting the survival.

# # Feature Engineering

# In[ ]:


train.shape


# The training data consists of 891 passengers which means the sample size is small, so a complex model runs a chance of overfitting to the training set and not predicting as well on the unseen test data. Keeping this in view, it is desirable to be selective about which features to include in our model.

# In[ ]:


train.isnull().sum()


# A lot of values from the Age and Cabin columns are missing. We decide to fill the Age column whereas discard the Cabin column. 

# The title of the passengers can be extracted from their names. The titles have a strong correlation with gender and when age as well as the number of parent-children and sibling-spouses are also accounted, the titles does not seems to add much information and hence can be discarded. However, the titles are useful to fill the missing values in age column since passengers with the same title tend to have similar age.

# In[ ]:


def get_Titles(df):
    df.Name = df.Name.apply(lambda name: re.findall("\s\S+[.]\s", name)[0].strip())
    df = df.rename(columns = {'Name': 'Title'})
    df.Title.replace({'Ms.': 'Miss.', 'Mlle.': 'Miss.', 'Dr.': 'Rare', 'Mme.': 'Mr.', 'Major.': 'Rare', 'Lady.': 'Rare', 'Sir.': 'Rare', 'Col.': 'Rare', 'Capt.': 'Rare', 'Countess.': 'Rare', 'Jonkheer.': 'Rare', 'Dona.': 'Rare', 'Don.': 'Rare', 'Rev.': 'Rare'}, inplace=True)
    return df


# We first group the passengers as per their titles and then fill the missing values for the age using the median age for each group.

# In[ ]:


def fill_Age(df):
    df.Age = df.Age.fillna(df.groupby("Title").Age.transform("median"))
    return df


# We observe that multiple passengers were traveling on the same ticket. 

# In[ ]:


train.Ticket.value_counts()[:10]


# Often the tickets are shared among family members with the same last name. 

# In[ ]:


train[train.Ticket == "CA. 2343"]


# Instead of having seperate features for the number of parent-children and sibling-spouse, we can use the family size as a single feature for accompanying family members. So, the two features - family size and co-passengers on the same ticket are closely related. However, it turns out they are not the same.
# In some cases, passengers that appear to be traveling alone by account of their family size were part of a group traveling on the same ticket. 

# In[ ]:


train[train.Ticket == "1601"]


# In other cases, the family members were not traveling on the same ticket.

# In[ ]:


train.loc[[69, 184]]


# To combine the two, a feature named group size is derived by taking the maximum of the family size and the number co-travelers of a passenger. 

# In[ ]:


def get_Group_size(df):
    Ticket_counts = df.Ticket.value_counts()
    df['Ticket_counts'] = df.Ticket.apply(lambda x: Ticket_counts[x])
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Group_size'] = df[['Family_size', 'Ticket_counts']].max(axis=1)
    return df


# I have noticed that using the feature `Embarked` does not improve my model, so I discarded it as well. The feature for gender is encoded as numerical categories and only five features are kept. 

# In[ ]:


def process_features(df):
    df.Sex = df.Sex.astype('category', ordered=False).cat.codes
    features_to_keep = ['Age', 'Fare', 'Group_size', 'Pclass', 'Sex']
    df = df[features_to_keep]
    return df


# We can either process the training and test set seperately or together. Processing them together would mean that while calculating the group size for passengers in the training set, the ticket information about those in the test set is used and vice versa. Similarly, the missing values for age in the training set can either be imputed using the age of passengers from both the training and test set or simply the training set alone. Processing them together would certainly account for considerable data-leakage from the test set to the training set and the resulting model will likely overfit to the test set. In most scenarios, this is bad news, for example online predictions where the model would generalize poorly for unseen future test data. [Data-leakage can be prevented by using scikit-learn Pipelines](https://www.kaggle.com/dansbecker/data-leakage) which is also especially useful to keep the validation and training sets separate while tuning the hyperparameters. The Titanic survival prediction is a different story however, and later on in the next section, I have taken the liberty to deliberately not follow the standard best practise and explained my reasons below.

# Processing the training and test set seperately:

# In[ ]:


def process_data(df):
    df = df.copy()
    df = get_Titles(df)
    df = fill_Age(df)
    df = get_Group_size(df)
    df = process_features(df)
    return df

X_train, X_test = process_data(train), process_data(test)


# In[ ]:


X_train.head()


# In[ ]:


correlation_matrix = X_train.corr()
correlation_matrix


# The correlation matrix measures the linear dependence of the features and it is desirable to have features that have little or no depedence on each other. Decision tree based algorithms including boosted trees are robust to correlated features, so they are safer to use. However, using correlated features makes prediction less accurate for many machine learning algorithms, for example linear regression. It can remedied by using Principal Component Analysis to obtain uncorrelated features.  It is a good practise to know the correlation before proceeding to build models.

# In[ ]:


plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(correlation_matrix);


# The heatmap suggests the selected features are not overly dependent on each other. 

# # Building the model

# XGBoost has an in-built cross validation function that will help us to decide the number of estimators for the gradient boosting tree. 

# In[ ]:


dtrain = xgb.DMatrix(data=X_train, label=target)
params = {
    "Objective": 'gbtree',
    "eval_metric": 'error',
    #"eta": 0.1
}
cv = xgb.cv(params=params, dtrain=dtrain, num_boost_round=100, nfold=5, seed=41, early_stopping_rounds= 10)
cv.tail(1)


# I have left out the detailed hyperparamter tuning to get the best XGBoost classifier. This minimalistic model can achieve an accuracy of around 0.79 with optimal number of estimators, which is quite good for a single classifier for this dataset. Using multiple XGBoost classifiers with different optimal paramters along with other classifiers in an ensemble has given me an accuracy score upwards of 0.82 which is top 3% in the leaderboard. 
# The ensembling/stacking of dissimilar classifiers is a very effective way to improve performance, but it is a good learning process to try to make the best out of a single classfier.

# Titanic sank only once and the passengers' age and their tickets were known before the tragedy.
# Processing train and test set together in this case, to fill the missing values for Age and deriving the feature Group size, can be seen as using all information available to make predictions and thus striving to make a model that best predict the test set but disregard the performance on unseen examples outside of the test set. This will overfit the model to the test set, which is not the best practise in general, but can be argued to be desirable in this case where the test set is limited, fixed and known beforehand. Unsurpisingly, processing data together in the following way increases accuracy for the test set.

# Processing the training and test set together (optional):

# In[ ]:


# Please comment this cell if you would decide on not processing the training and test set together:
def process_data_together(train, test):
    df = pd.concat([train, test], join='inner', keys=['train', 'test']).copy()
    df = get_Titles(df)
    df = fill_Age(df)
    df = get_Group_size(df)
    df = process_features(df)
    return df.loc['train'], df.loc['test']

X_train, X_test = process_data_together(train, test)


# Irrespective of whether we process the training and test set together, after we decide on the hyperparameters (here `n_estimators=14`), we use the entire training set to retrain the model and then use this model to predict the test cases. 

# In[ ]:


xgbcl = XGBClassifier(n_estimators=14, seed=41)
xgbcl.fit(X_train, target)
print(xgbcl.score(X_train, target))


# # Interpreting the model

# Machine learning algorithms need not be the blackboxes, especially with the availability of various tools today. XGBoost has in-built functions to plot the decision tree as well as the importance of each feature. The heatmap to gauge the correlation of each feature with the predictor variable also adds a different perspective to understand the working of our model. Lastly, partial dependence plots are excellent to understand the relationship between individual features and the predictor variable.

# In[ ]:


# xgb.plot_tree(xgbcl);
xgb.to_graphviz(xgbcl)


# Here, Sex = 1 implies male and Sex = 0 implies female. In case of female passengers, ticket class is a deciding factor early on in the decision tree whereas for males, age plays that role.

# In[ ]:


xgb.plot_importance(xgbcl);


# Interestingly, gender is the least important feature in the model we built which is counterintuitive considering what is largely believed about the tragedy.

# In[ ]:


X_w_target = X_train.copy()
X_w_target['Survived'] = target.astype('int64')
correlation_matrix_w_target = X_w_target.corr()
correlation_matrix_w_target


# Indeed, the gender is the feature that is most highly correlated with survival among all with correlation coefficient of $-0.54$. It is a good demonstration of "Correlation does not imply causation".

# In[ ]:


plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features along with target variable', y=1.05, size=15)
sns.heatmap(correlation_matrix_w_target);


# [Partial dependence plots](https://www.kaggle.com/dansbecker/partial-dependence-plots) are useful to see how continuous variables affect the model's predictions. They show us the relationship of a particular variable with target variable when all other features are kept constant. I am unaware of making `plot_partial_dependence` work on `XGBoost Classifier`, so I am rebuilding the model using its equivalent in `scikit-learn`, that is `GradientBoostingClassifier`. 

# In[ ]:


gbcl = GradientBoostingClassifier(learning_rate=0.2, n_estimators=70)
gbcl.fit(X_train, target)
plot_partial_dependence(gbcl, X=X_train, features=[0, 1, 2], feature_names=['Age', 'Fare', 'Group_size'], grid_resolution=10);


# This suggests that Age is a strong indicator of survival in the range of 0-10 years.

# # Prediction and submission

# In[ ]:


predictions = xgbcl.predict(X_test)
Predictions = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
Predictions.to_csv('submission.csv', index=False)


#  ## References for kernels
#  * https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
#  * https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
#  * https://www.kaggle.com/pliptor/divide-and-conquer-0-82296

# If you learned from the kernel and/or liked it, an upvote would be appreciated!
