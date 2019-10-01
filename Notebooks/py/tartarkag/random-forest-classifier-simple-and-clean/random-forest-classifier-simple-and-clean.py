#!/usr/bin/env python
# coding: utf-8

# <h1>Table of contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Loading-data-and-preparatory-steps" data-toc-modified-id="Loading-data-and-preparatory-steps-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Loading data and preparatory steps</a></span><ul class="toc-item"><li><span><a href="#Variables-description-(from-the-Kaggle-dataset-page)" data-toc-modified-id="Variables-description-(from-the-Kaggle-dataset-page)-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Variables description (from the Kaggle dataset page)</a></span></li></ul></li><li><span><a href="#Exploratory-data-analysis" data-toc-modified-id="Exploratory-data-analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Exploratory data analysis</a></span><ul class="toc-item"><li><span><a href="#Observations" data-toc-modified-id="Observations-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Observations</a></span></li><li><span><a href="#Hypotheses" data-toc-modified-id="Hypotheses-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Hypotheses</a></span></li><li><span><a href="#Observations" data-toc-modified-id="Observations-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Observations</a></span></li></ul></li><li><span><a href="#Data-visualisation" data-toc-modified-id="Data-visualisation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data visualisation</a></span><ul class="toc-item"><li><span><a href="#Age-histograms-by-survival" data-toc-modified-id="Age-histograms-by-survival-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Age histograms by survival</a></span><ul class="toc-item"><li><span><a href="#By-passenger-class" data-toc-modified-id="By-passenger-class-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>By passenger class</a></span></li><li><span><a href="#By-place-of-embarkation" data-toc-modified-id="By-place-of-embarkation-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>By place of embarkation</a></span></li><li><span><a href="#By-gender" data-toc-modified-id="By-gender-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>By gender</a></span></li></ul></li><li><span><a href="#Fare-histogram-by-passenger-class-and-survival" data-toc-modified-id="Fare-histogram-by-passenger-class-and-survival-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Fare histogram by passenger class and survival</a></span></li></ul></li><li><span><a href="#Feature-engineering" data-toc-modified-id="Feature-engineering-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Feature engineering</a></span><ul class="toc-item"><li><span><a href="#New-features:-Family_size-and-Family_bins" data-toc-modified-id="New-features:-Family_size-and-Family_bins-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>New features: Family_size and Family_bins</a></span></li><li><span><a href="#New-feature:-Title" data-toc-modified-id="New-feature:-Title-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>New feature: Title</a></span></li><li><span><a href="#New-feature:-Cabin_type" data-toc-modified-id="New-feature:-Cabin_type-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>New feature: Cabin_type</a></span></li><li><span><a href="#New-feature:-Ticket_count" data-toc-modified-id="New-feature:-Ticket_count-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>New feature: Ticket_count</a></span></li><li><span><a href="#Imputing-null-values" data-toc-modified-id="Imputing-null-values-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Imputing null values</a></span><ul class="toc-item"><li><span><a href="#Embarked-and-Fare" data-toc-modified-id="Embarked-and-Fare-4.5.1"><span class="toc-item-num">4.5.1&nbsp;&nbsp;</span>Embarked and Fare</a></span></li><li><span><a href="#Age" data-toc-modified-id="Age-4.5.2"><span class="toc-item-num">4.5.2&nbsp;&nbsp;</span>Age</a></span></li></ul></li><li><span><a href="#Creating-dummy-features" data-toc-modified-id="Creating-dummy-features-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Creating dummy features</a></span></li></ul></li><li><span><a href="#Cleanup" data-toc-modified-id="Cleanup-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Cleanup</a></span></li><li><span><a href="#Modelling" data-toc-modified-id="Modelling-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modelling</a></span><ul class="toc-item"><li><span><a href="#Hyperparameter-tuning-with-Cross-Validation" data-toc-modified-id="Hyperparameter-tuning-with-Cross-Validation-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Hyperparameter tuning with Cross Validation</a></span></li></ul></li></ul></div>

# # Titanic dataset prediction challenge
# This notebook delineates the process through which we can predict whether a passenger that boarded the Titanic will have survived the journey, based on different predictor variables. This particular notebook is meant to go through the steps that lead to run a random forest classifier in a way that is simple and clear.<br>
# The competition description can be found on [Kaggle](https://www.kaggle.com/c/titanic).

# ## Loading data and preparatory steps

# In[ ]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


print(os.listdir('../input/'))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

data = [train, test]


# ### Variables description (from the Kaggle dataset page)

# __pclass__: A proxy for socio-economic status (SES)<br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# 
# __age__: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# __sibsp__: The dataset defines family relations in this way...<br>
# Sibling = brother, sister, stepbrother, stepsister<br>
# Spouse = husband, wife (mistresses and fiancés were ignored)<br>
# 
# __parch__: The dataset defines family relations in this way...<br>
# Parent = mother, father<br>
# Child = daughter, son, stepdaughter, stepson<br>
# Some children travelled only with a nanny, therefore parch=0 for them.

# ## Exploratory data analysis

# In[ ]:


def data_info():
    print(train.info())
    print('\n')
    print(test.info())


# In[ ]:


data_info()


# In[ ]:


train.head()


# In[ ]:


train.describe(include='all')


# In[ ]:


males = train['Sex'] == 'male'
print(round((males.sum() / len(train.index)), 3))


# ### Observations
# 
# * At least 50% of the passengers travelled in 3rd class and most of them (at least 75%) were 38 or younger, with no parents or children, with either 1 sibling/spouse or none at all.
# * The ticket price varied greately, with an average of 32 and a standard deviation of 49; the majority of passengers paid 31 or less.
# * All names are unique, but they contain titles, which might be useful for a better categorisation.
# * In the training dataset, 64.8% of people are male
# * There are several cabin duplicates, and this might mean that more people shared the same cabin. It also only has 204 non-null values over 891 records, which might mean that a high number of people did not have a cabin.
# * PassengerId is just an incremental number for each entry, and useless for this analysis - it will however be needed for the submission file.
# * At a first glance, the place from which people boarded the ship would not seem to be a good predictor. However, it is possible that different embarking points might reflect differences in socioeconomic status, and therefore, survival chances (as it is already known for the Titanic disaster).
# 
# ### Hypotheses
# * The variables that are most likely to be useful predictors would seem to be sex, age, and passenger class.
# * Number of siblings and number of parents will be useful in engineering a new variable, that is family size.
# * The Name variable in itself is not useful, except that it contains the passenger's title, which we can extract.

# In[ ]:


survived = train['Survived'].copy()
passenger_id = test['PassengerId'].copy()

for df in data:
    df.drop('PassengerId', axis=1, inplace=True)


# We can inspect the differences in survival rates among the various variables by pivoting the data:

# In[ ]:


def pivot_var(var_name):
    return train[[var_name, 'Survived']]            .groupby([var_name])                        .mean()                                     .sort_values(by='Survived',
                     ascending=False)       \
        .round(3)


pivot_var('Pclass')


# In[ ]:


pivot_var('Sex')


# In[ ]:


pivot_var('SibSp')


# In[ ]:


pivot_var('Parch')


# In[ ]:


pivot_var('Embarked')


# ### Observations
# 
# * 1st class passenger had the highest rate of survival, almost half of 2nd class passenger survived, whereas this proportion drops by half for 3rd class passengers.
# * Females had a higher survival rate: roughly 74%
# * Number of siblings/spouses and number of parents/children would not, at a first glance, seem to have a direct correlation with survival.
# * People Embarked from Cherbourg had a 55% survival rate, whereas this was 39% for Queenstown and 34% for Southampton.

# ## Data visualisation

# ### Age histograms by survival

# In[ ]:


sns.set(font_scale=1.5, style='whitegrid')

plot = sns.FacetGrid(train,
                     col='Survived',
                     size=5)

plot.map(plt.hist, 'Age', bins=30);


# Young children had a higher rate of survival, whereas large numbers of 20-30 year-old people had a lower survival rate.

# #### By passenger class

# In[ ]:


def age_hist_by(catvar):
    sns.FacetGrid(train,
                  col='Survived',
                  row=catvar,
                  size=3,
                  aspect=1.5)          \
        .map(plt.hist, 'Age', bins=30)


age_hist_by('Pclass')


# Large numbers of people in 3rd class did not survive, while survival chances for people in 1st class were higher.

# #### By place of embarkation

# In[ ]:


age_hist_by('Embarked')


# Most people embarked from Southampton, and it would appear that of the few people in this sample that represent passengers embarked from Queenstown, even fewer of them survived.<br>
# Let's investigate this variable from a different angle:

# In[ ]:


sns.FacetGrid(train,
              row='Embarked',
              col='Sex',
              hue='Embarked',
              size=2.5,
              aspect=1.5) \
    .map(sns.barplot,
         'Pclass',
         'Survived',
         ci=68, # using SEM for the error bars
         order=train['Pclass'].unique().sort());


# There are clear differences here - Embarked may be a useful predictor and will be included in the model.

# #### By gender

# In[ ]:


age_hist_by('Sex')


# Females had a generally higher rate of survival across all ages, except for a spike in the rate of survival for male children.

# ### Fare histogram by passenger class and survival

# In[ ]:


sns.FacetGrid(train,
              col='Survived',
              row='Pclass',
              size=2.5,
              aspect=1.5) \
    .map(plt.hist, 'Fare', bins=15);


# The majority of passengers who paid less for a ticket were in third class - as one would expect, these two variables have some correlation.

# In[ ]:


colormap = plt.cm.PiYG

sns.set(font_scale=1.4)
plt.figure(figsize=(10, 8))
plt.title('Feature correlation matrix', y=1.02, size=15)

sns.heatmap(train.corr().round(2), square=True,
            linecolor='gray', linewidth=0.1,
            cmap=colormap, annot=True);


# This correlation matrix gives us an idea as to what features may be more important in predicting the 'Survived' outcome variable: _Pclass_ and _Fare_ being the two most directly correlated with _Survival_.<br>
# However, it is important to note that we haven't explored whether these variables are normally distributed, or whether outliers might be skewing the relationship.<br>
# For this reason, we should look at this graph only as an indication and not draw hasty conclusions from this data.

# ## Feature engineering

# Since several python machine learning models do not support categorical variable types, we will use dummy variables instead. In doing this, we need to be aware that we might overfit our model if we create too many features.
# 
# Also, because feature engineering should be performed on both datasets, all transformations will be applied to both the training set and the test set. Specifically, rather than joining them and using all data to perform our transformations (the test data should technically be _unknown_ to us), we will perform our transformations on these dataset separately.

# In[ ]:


train.head()


# ### New features: Family_size and Family_bins

# *SibSp* reflects the number of siblings/spouses, *Parch* gives us the number of parents/children. Adding these together, plus 1 for the passenger itself, will return the size of its family:

# In[ ]:


for df in data:
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1


# We can see the distribution of this new variable we created, and its mean as a vertical dashed line:

# In[ ]:


sns.set(font_scale=1.5, style='whitegrid')
plt.hist(x=train['Family_size'], bins=10, color='C5', edgecolor='black')
plt.axvline(train['Family_size'].mean(), color='black', linestyle='dashed', linewidth=2);


# It would seem sensible to group this data by creating family size bins: passengers who are travelling alone, couples, small families (up to 3), or larger families (more than 3).

# In[ ]:


for df in data:
    df['Family_bins'] = pd.cut(df['Family_size'], bins=[0, 1, 3, df['Family_size'].max()])
    print(df['Family_bins'].unique())


# ### New feature: Title

# Titles can reflect both the sex of a person and their socioeconomic status.<br>
# We can extract them from the *Name* variable by using a regular expression that identifies the first word that ends with a dot.

# In[ ]:


for df in data:
    df['Title'] = df['Name'].str.extract(' (\w+)\.', expand=False)
    df['Title'].value_counts()


# In[ ]:


train['Title'].value_counts()


# Some of the titles occur only once, and therefore would not be useful predictors. We can remap them to something more general:

# In[ ]:


title_categories = {
    'Capt': 'Other',
    'Col': 'Other',
    'Countess' :'Other',
    'Don': 'Other',
    'Jonkheer': 'Other',
    'Lady': 'Mrs',
    'Major': 'Other',
    'Mlle': 'Other',
    'Mme': 'Other',
    'Ms': 'Miss',
    'Sir': 'Mr'
}


# In[ ]:


for df in data:
    df['Title'].replace(title_categories, inplace=True)
    print('{}\n'.format(df['Title'].value_counts()))


# Here we see that in the test dataset we have titles that did not appear in the training dataset - these will go on and form dummy features that will be excluded from the dataset used by the model.

# In[ ]:


sns.set(font_scale=1.4)
sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Title',
            y='Survived',
            data=train,
            ci=68,
            color='C0');


# ### New feature: Cabin_type

# Cabin_count will be a new feature that will show how many passengers share the same cabin:

# In[ ]:


for df in data:
    df['Cabin_count'] = df.groupby(['Cabin'])['Cabin'].transform('count')


# Cabin_type is built on Cabin_count: according to how many passengers share a cabin, we can assign it to a size category.

# In[ ]:


for df in data:
    df['Cabin_type'] = pd.cut(df['Cabin_count'].fillna(-1),
                                   bins=[-2, 0, 1, 2, df['Cabin_count'].max()],
                                   labels=['NA', 'Single', 'Double', 'Multiple'])


# In[ ]:


for df in data:
    print(df['Cabin_type'].unique())


# ### New feature: Ticket_count

# Ticket_count will show how many passengers share the same ticket:

# In[ ]:


# count of tickets that are the same
for df in data:
    df['Ticket_count'] = df.groupby(['Ticket'])['Ticket'].transform('count')
    df['Ticket_count'] = df['Ticket_count'].astype('category')


# ### Imputing null values

# *Embarked*, *Fare*, *Cabin*, and *Age* all contain some null values.<br>
# We will use Cabin to create new features without needing to replace its null values, use a stratification method for *Age*, and because *Embarked* and *Fare* have the smallest number of _nulls_, we will replace these with the most frequent occurrence and with the mean, respectively.

# #### Embarked and Fare

# In[ ]:


for df in data:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)


# In[ ]:


data_info()


# #### Age

# Rather than using the mean to impute missing Age values, we can use a stratification of our dataset to get a more accurate estimate for this variable:

# In[ ]:


for df in data:
    print('{}\n'.format(df.groupby(['Title', 'Pclass'])['Age'].mean().astype('int')))


# In[ ]:


for df in data:
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))


# ### Creating dummy features

# In[ ]:


train.info()


# We will create dummies for the following variables: *Pclass*, *Embarked*, *Title*, *Ticket_count*, *Cabin_type*, *Family_bins*, and *Sex*. Because *Pclass* is an integer, we first need to convert it to a categorical variable.<br>
# The remaining variables, except *Age*, will be dropped in the Cleanup step.

# In[ ]:


train['Pclass'] = train['Pclass'].astype('category')
test['Pclass'] = test['Pclass'].astype('category')


# In[ ]:


dummies = ['Pclass', 'Embarked', 'Title', 'Ticket_count', 'Cabin_type', 'Family_bins', 'Sex']


# In[ ]:


dummy_features = pd.get_dummies(train[dummies])
train = pd.concat([train, dummy_features], axis=1)

dummy_features = pd.get_dummies(test[dummies])
test = pd.concat([test, dummy_features], axis=1)


# ## Cleanup

# In[ ]:


drop_list = ['Cabin_count', 'Cabin', 'Ticket', 'SibSp', 'Parch', 'Name', 'Family_size']
drop_list.extend(dummies)


# In[ ]:


train.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)


# In[ ]:


data_info()


# In[ ]:


len(train.columns) - len(test.columns)


# Some features only exist in one or the other dataset. These will be dropped:

# In[ ]:


train.drop([x for x in train.columns if x not in test.columns], axis=1, inplace=True)
test.drop([x for x in test.columns if x not in train.columns], axis=1, inplace=True)


# Now the training and the testing data sets have the same features:

# In[ ]:


train.info()


# ## Modelling

# We will pick a number at random for our random seed - for reproducibility:

# In[ ]:


RANDOM_SEED = 354135


# And we will assign the correct labels to our training and testing data:

# In[ ]:


x_train = train.copy()
y_train = survived.copy()
x_test  = test.copy()
x_train.shape, y_train.shape, x_test.shape


# For the purposes of this notebook, we will only run a random forest classifier rather than creating multiple models and testing their performance to choose the best one.

# ### Hyperparameter tuning with Cross Validation

# We will start by choosing a range of hyperparameters for our random forest classifier - the best combination will be selected with a 10-fold cross validation using GridSearchCV.

# In[ ]:


random_forest_parameters = {
    'n_jobs': [-1],
    'random_state': [RANDOM_SEED],
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [4, 8, 12, 16],
    'min_samples_split': [3, 5, 7, 12, 16],
    'min_samples_leaf': [1, 3, 5, 7]
}


# Instantiating the random forest classifier with 10-fold cross validation, using all available cores.

# In[ ]:


forest_cv = GridSearchCV(estimator=RandomForestClassifier(),
                         param_grid=random_forest_parameters,
                         cv=10,
                         verbose=0, # I used 4 on my local notebook, but it seems that on Kaggle the output is very long and makes the reading more difficult, so I set this to zero
                         n_jobs=-1)


# Fitting the model to the training data.<br>
# **Note:** this step took roughly 5 minutes on my machine, an Intel i7-6700 @ 3.40 GHz - it might take significantly longer on different machines.

# In[ ]:


forest_cv.fit(x_train, y_train)


# In[ ]:


print('Best cross validation score: {}'.format(forest_cv.best_score_))
print('Optimal parameters: {}'.format(forest_cv.best_params_))


# The CV score is 0.83 - we can expect to obtain a slighlty lower accuracy score on the actual test prediction.

# As this is an ensemble model, there is no decision path we can explore - to obtain that, we should do it for each of the 100 decision trees that have been used in this classifier.<br>
# What we can check are the most important features - we'll need to refit a random forest classifier because GridSearchCV does not yield this data, but we will use the same parameters with the same random seed.<br>
# The features that are most important for a prediction are listed first.

# In[ ]:


random_forest = RandomForestClassifier(**forest_cv.best_params_)
random_forest.fit(x_train, y_train)

importances = pd.DataFrame({'score':random_forest.feature_importances_,
                            'feature':x_train.columns})
importances.sort_values('score', ascending=False)


# We can now create and submit a prediction for the test dataset:

# In[ ]:


y_pred = forest_cv.predict(x_test)


# In[ ]:


submission = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': y_pred.astype('int')
    })
submission.to_csv('submission.csv', index=False)


# ## References

# This notebook has been written after reviewing work carried out in several other notebookes published in the Titanic competition.
