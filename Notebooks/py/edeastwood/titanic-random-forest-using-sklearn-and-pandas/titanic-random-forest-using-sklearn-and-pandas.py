#!/usr/bin/env python
# coding: utf-8

# # Learning Titanic

# In[ ]:


import pandas
import numpy
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib notebook')


# 
# 
# ## load data

# In[ ]:


TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
train = pandas.read_csv(TRAIN_PATH)
test = pandas.read_csv(TEST_PATH)


# ## identify columns that have blank values

# In[ ]:


train.isnull().any()


# In[ ]:


test.isnull().any()


# ### age, embarked, fare and cabin all contain blank values: we'll fill later
# * age could depend on various features, we'll investigate these then use regression to fill in the blanks
# * The majority of passengers embarked from Southampton. We'll just fill the small number of blanks with 'S'.
# * First impressions are that fare doesn't seem particularly useful. It probably can't tell us much more than point of embarkation, class of travel and number of cabins/passengers.
# * Cabin appears too sparsely populated to fill reliably.

# ## derive titles

# In[ ]:


def deriveTitles(s):
    title = re.search('(?:\S )(?P<title>\w*)',s).group('title')
    if title == "Mr": return "adult"
    elif title == "Don": return "gentry"
    elif title == "Dona": return "gentry"
    elif title == "Miss": return "miss" # we don't know whether miss is an adult or a child
    elif title == "Col": return "military"
    elif title == "Rev": return "other"
    elif title == "Lady": return "gentry"
    elif title == "Master": return "child"
    elif title == "Mme": return "adult"
    elif title == "Captain": return "military"
    elif title == "Dr": return "other"
    elif title == "Mrs": return "adult"
    elif title == "Sir": return "gentry"
    elif title == "Jonkheer": return "gentry"
    elif title == "Mlle": return "miss"
    elif title == "Major": return "military"
    elif title == "Ms": return "miss"
    elif title == "the Countess": return "gentry"   
    else: return "other"
    
train['title'] = train.Name.apply(deriveTitles)
test['title'] = test.Name.apply(deriveTitles)

# and encode these new titles for later
le = preprocessing.LabelEncoder()
titles = ['adult', 'gentry', 'miss', 'military', 'other', 'child']
le.fit(titles)
train['encodedTitle'] = le.transform(train['title']).astype('int')
test['encodedTitle'] = le.transform(test['title']).astype('int')


# ## fill embark
# note: embarked only has holes in the training data

# In[ ]:


train.Embarked.fillna(value = 'S', inplace=True)


# ## does this passenger have more than one cabin?
# is this a good indication that they have people to help them?

# In[ ]:


# not expected to add significant value because cabin data is so sparse


# ### Both test and training data sets have missing ages and both have useful insight into how these can be filled

# In[ ]:


combined = pandas.concat([train, test])
# combining train and test casts Survived from int to float because all Survived values in test are blank
combined.ParChCategories = combined.Parch > 2


# ## plot features by age to determine which ones might be useful

# In[ ]:


combined.boxplot(column='Age', by='Pclass')


# # learn the missing ages
# * sex and embarked doesn't seem significant
# * class more-or-less linear: use as feature
# * SibSp: less than 2: could be anything, 2 and 3 gives medium range, >3 low age. Use these groups as categorical features
# * Parch: < 3 could be any age, > 3 and always older. Use these groups as categorical features
# * familySize appears to be a combination of the above two so ignore
# * title has a huge impact, use it as a categorical feature

# ### calculate new features based on number of siblings, spouses, parents and children

# In[ ]:


combined = combined.assign(SibSpGroup1 = combined['SibSp'] < 2)
combined = combined.assign(SibSpGroup2 = combined['SibSp'].between(2, 3, inclusive=True))
combined = combined.assign(SibSpGroup3 = combined['SibSp'] > 2)
combined = combined.assign(ParChGT2 = combined['Parch'] > 2)


# ### split combined into those with an age and those without
# we have just over 1,000 in the training set so let's take 20% for validation
# Note: The fact that the split is random affects the accuracy of the two models below. Both vary between 8 and 9.5. It may be worth investigating further to improve results.

# In[ ]:


age_train, age_validation = train_test_split(combined[combined.Age.notnull()], test_size = 0.2)
age_learn = combined[combined.Age.isnull()]


# ### impute ages using a random forest regressor
# note that scikit learn treats features of type int as categorical
# 
# http://stackoverflow.com/questions/20095187/regression-trees-or-random-forest-regressor-with-categorical-inputs

# #### Random Forest

# In[ ]:


age_rf = RandomForestRegressor()
age_rf.fit(age_train[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']], age_train['Age'])
age_validation = age_validation.assign(rf_age = age_rf.predict(age_validation[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']]))
mean_absolute_error(age_validation['Age'], age_validation['rf_age'], sample_weight=None, multioutput='uniform_average')


# #### Linear Regression
# 1. onehot encode categorical features
# 2. scale features - not required since all features are categorical

# In[ ]:


age_encoder = preprocessing.OneHotEncoder().fit(combined[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']])
age_training_encoded = age_encoder.transform(age_train[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']])
age_validation_encoded = age_encoder.transform(age_validation[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']])
age_model = linear_model.RidgeCV(alphas = [0.1, 0.2, 0.3, 0.4, 0.5])
age_estimator = age_model.fit(age_training_encoded, age_train['Age'])
linear_age_predictions = age_estimator.predict(age_validation_encoded)
mean_absolute_error(age_validation['Age'], linear_age_predictions, sample_weight=None, multioutput='uniform_average')


# The fandom forest model gets slightly better results most of the time (deoending on the random split of data above) so we'll use it.

# In[ ]:


age_learn = age_learn.assign(Age = age_rf.predict(age_learn[['Pclass', 'encodedTitle', 'SibSpGroup1', 'SibSpGroup2', 'SibSpGroup3', 'ParChGT2']]))


# ### fill the combined data set with the imputed ages and then split back into training and test
# need to add an index to each dataframe to enable the join

# In[ ]:


age_learn.set_index('PassengerId', inplace=True, drop=False)
combined.set_index('PassengerId', inplace=True, drop=False)
combined.update(age_learn, join = 'left', overwrite = False)
# careful here... update changes int columns to floats
# https://github.com/pandas-dev/pandas/issues/4094
# this could be problematic later if they're not changed back since
# int features are treated as categorical and floats are not


# ## derived family based features

# In[ ]:


combined = combined.assign(familySize = combined['Parch'] + combined['SibSp'])

def deriveChildren(age, parch):
    if(age < 18): return parch
    else: return 0

combined = combined.assign(children = combined.apply(lambda row: deriveChildren(row['Age'], row['Parch']), axis = 1))
# train['children'] = train.apply(lambda row: deriveChildren(row['Age'], row['Parch']), axis = 1)
# test['children'] = test.apply(lambda row: deriveChildren(row['Age'], row['Parch']), axis = 1)
# I think (but am not certain) the commented code above is functionally equivalent to the preceeding two lines,
# but the commented lines gave settingdwithcopy warnings. I think these were false postives but am not certain.

def deriveParents(age, parch):
    if(age > 17): return parch
    else: return 0
    
combined['parents'] = combined.apply(lambda row: deriveParents(row['Age'], row['Parch']), axis = 1)
    
def deriveResponsibleFor(children, SibSp):
    if(children > 0): return children / (SibSp + 1)
    else: return 0
    
combined['responsibleFor'] = combined.apply(lambda row: deriveResponsibleFor(row['children'], row['SibSp']), axis = 1)
    
def deriveAccompaniedBy(parents, SibSp):
    if(parents > 0): return parents / (SibSp + 1)
    else: return 0
    
combined['accompaniedBy'] = combined.apply(lambda row: deriveAccompaniedBy(row['parents'], row['SibSp']), axis = 1)
    
def unaccompaniedChild(age, parch):
    if((age < 16) & (parch == 0)): return True
    else: return False

combined['unaccompaniedChild'] = combined.apply(lambda row: unaccompaniedChild(row['Age'], row['Parch']), axis = 1)


# ### derive passengers likely location aboard the ship based on cabin number

# In[ ]:


# may not be worth doing given how sparsely populated cabin data is


# ## Random Forest Survival Prediction
# As noted above, scikit learn treats integer features as categorical. Preprocessing has set all integers as floats. These need returning to the correct type so they are handled as expected. Non-numeric values will also need converting.
# 
# Preprocessing of the data frames have left redundant data. The model will use:
# - age (continuous)
# - embarked (categorical)
# - Pclass (continuous) (interesting whether this is actually continuous or whether each class is a category)
# - Sex (categorical)
# - encodedTitle (categorical)
# - SibSpGroups 1 to 3 (categorical)
# - familySize (continuous)
# - children (continuous)
# - parents (continuous)
# - responsibleFor (continuous)
# - accompaniedBy (continuous)
# - unaccompaniedChild (categorical)

# In[ ]:


# drop unused columns
combined = combined.drop(['Name', 'Cabin', 'Fare', 'Parch', 'SibSp', 'Ticket', 'title'], axis=1)
# confirm types
combined.dtypes


# In[ ]:


# label encode string features
categorical_names = {}
categorical_features = ['Embarked', 'Sex']
for feature in categorical_features:
    le = preprocessing.LabelEncoder()
    le.fit(combined[feature])
    combined[feature] = le.transform(combined[feature])
    categorical_names[feature] = le.classes_
    
#combined = combined.assign(encodedTitleInt = combined['encodedTitle'].astype(int, copy=False))
combined['title'] = combined['encodedTitle'].astype(int, copy=False)
combined['class'] = combined['Pclass'].astype(int, copy=False)
combined = combined.drop(['Pclass'], axis=1)
combined = combined.drop(['encodedTitle'], axis=1)

train = combined[combined.PassengerId < 892]
test = combined[combined.PassengerId > 891]
test = test.drop(['Survived'], axis=1)

train['Survived'] = train['Survived'].astype(int, copy=False)
# the warning below is a false positive since the copy input is set to false


# In[ ]:


test.dtypes


# In[ ]:


rf = RandomForestClassifier()
rf.fit(train[['title', 
              'Age', 
              'Embarked', 
              'class', 
              'Sex', 
              'SibSpGroup1', 
              'SibSpGroup2', 
              'SibSpGroup3', 
              'familySize', 
              'children', 
              'parents', 
              'responsibleFor', 
              'accompaniedBy', 
              'unaccompaniedChild']], train['Survived'])

test = test.assign(Survived = rf.predict(test[['title', 
              'Age', 
              'Embarked', 
              'class', 
              'Sex', 
              'SibSpGroup1', 
              'SibSpGroup2', 
              'SibSpGroup3', 
              'familySize', 
              'children', 
              'parents', 
              'responsibleFor', 
              'accompaniedBy', 
              'unaccompaniedChild']]))


# # Output the results to CSV

# In[ ]:


test[['Survived']].to_csv(path_or_buf='~/output.csv')


# In[ ]:




