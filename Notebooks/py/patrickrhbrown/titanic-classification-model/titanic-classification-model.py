#!/usr/bin/env python
# coding: utf-8

# # Stage 1: Exploratory Data Analysis.  
# 
# We are not yet sure of the quality of our data.  We have a training and a test set. Our objective is to use the training set to evaluate our model to the extent that we can represent ground truth as accurately as we can with our test set.  
# 
# To begin, we'll check how clean our data is, and parse as necessary.  To do this, we will explore the data, establish any missing values or aberrations, chart our data to see what relationships we can ascertain with a view to taking our next steps, and establish a few hypotheses.  As we will be fitting a classifier, we will wish to deal with any missing values and to create additional features for levels of each categorical variable.

# ### Data Dictionary
# 
# * passengerid: a unique identifier for each passenger
# * survival: Survival (1 = survived; 2 = died)
# * class: Passenger class (1 = first; 2 = second; 3 = third)
# * name: name of passenger
# * sex: gender of passenger
# * age: age of passenger
# * sibsp: number of siblings/spouses aboard
# * parch: number of parents/children aboard
# * ticket: ticket id
# * fare: passenger fare paid
# * cabin: cabin
# * embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()


# We should establish our baseline model.  In this case, this is how well our model would do if it simply predicted that everyone survived or everyone perished. 

# In[ ]:


print('The mean survival rate across the whole dataset, and therefore our baseline survival only prediction model is approx: ' + str(round(train_df['Survived'].mean(),3) * 100) + '%')


# This also means that if we set a model that predicted everyone perished, we would get accuracy of about 61.6%.  So ideally we would want our model to perform better than this.  But how to start thinking about assembling a model? We can first think about what features we have.  We can think about the popular cultural account of the Titanic. What about looking at gender splits.

# In[ ]:


import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.style.use('fivethirtyeight')

gender_survival = train_df.groupby(['Sex']).mean()
gender_survival['Survived'].plot.bar()


# 
# So a model that predicts solely that women survive would attain an accuracy rating of around 74%.  Again, we'd like to try and do better than this.  Conversely, predicting only men survive would attain an accuracy of around 26%!  What about class? We could use the travel class as a proxy for social class.

# In[ ]:


gender_and_class_survival = train_df.groupby(['Pclass','Sex']).mean()
gender_and_class_survival['Survived'].plot.bar()


# In[ ]:


gender_and_class_df = pd.DataFrame(gender_and_class_survival['Survived'])
gender_and_class_df


# Clearly, women again show strong survival rates but there is a disproportionately high survival rate in first (Pclass == 1) and second class (Pclass == 2) among women.  Pclass also seems to affect survival rates among men with disproportionately high survival rates among men in first class. So what about age?

# In[ ]:


missing_values = pd.DataFrame({'Number of Missing Values': train_df.isnull().sum()}).T
missing_values


# Cabin data is scarce.   A little internet research uncovers that the allocations are in dispute.  A primary source material item recovered from a steward gives some cabin allocations among first class passengers.  There is much speculation about the remainder.  On balance, we'll probably drop the column with cabin data for the purposes of building a simple model on this occasion. 

# It is also clear that we have a large number of missing ages, which won't be helpful when determining the effect of age on survival rates. It is a more soluble problem than the missing cabin data, however.  We will temporarily impute age using the median of the training set for the purposes of our EDA.  We'll also bin the ages to permit a deeper insight as to the effect of age on survival.  We *could* simply fill in the median age of the whole dataset, but given survival rates vary by across age group and within age group across gender, we would be introducing less noise into our data by trying to impute values that are closer to the central tendency of smaller groups of our training sample. As we will need to do the same for both our training and test samples, we'll create a helper function to do that.

# In[ ]:


age_groups = pd.cut(train_df["Age"], bins=12, precision=0, right = False)
group_age = train_df.groupby(age_groups).mean()
group_age['Survived'].plot.bar()


# **Obtaining information from Family Size.**
# 
# Parch and SibSp denote whether or not the passenger was accompanied by close relatives, significant others and so forth.  We need to feed categorical variables where possible to our classifier, and so extracting more value from these two columns which relate to family bonds will be useful for that purpose.  Moreover, there is a statistical reason why we should do so, as the following graph will show. Essentially, again we see that females show better survival rates across categories but we add more to our model by determining the extent of accompanying relatives onboard the Titanic since this makes a difference to survival rates.  Given that we will be running an ensemble model that creates a number of weak classifiers and averages across them, as well as randomising the initial choice of feature (unlike single decision trees that seek to maximise information to the model in descending order with each split), it is good for us to have a number of different features that supply information to our model.

# In[ ]:


#first let's create a new column that sums Parch and SibSp to give a num of additional family members onboard

train_df['family_onboard'] = train_df['Parch'] + train_df['SibSp']
test_df['family_onboard'] = test_df['Parch'] + test_df['SibSp']
train_df.head()


# In[ ]:


#check we've not accidentally introduced any missing values

train_df['family_onboard'].isna().sum()


# Now that we have family sizes, it would be beneficial to turn this into a categorical column that can be used for the purposes of our model. 

# In[ ]:


#next let's establish some further value from the new column by binning the values so that a) we create a categorical column and b) we begin to standardise some family sizes
#we need to do the same for our training and test sets as before.  We'll have to include an upper bin limit to the right to cope with any particularly large families (hence '50' as the rightward bin limit)

train_df['family_onboard'] = pd.cut(train_df.family_onboard, bins = [0,1,2,3,4,5,50], right = False, labels = [1,2,3,4,5,6])
test_df['family_onboard'] = pd.cut(test_df.family_onboard, bins = [0,1,2,3,4,5,50], right = False, labels = [1,2,3,4,5,6])
train_df.head(10)


# In[ ]:


#okay we're good

train_df.isnull().sum()


# In[ ]:


#we plot a bar chart of survival rates for each family size category and across gender of passenger
#we see that single female and females with one other family member showing good survival rates but 
#male passengers and larger family members do not do so well

family_model = train_df.groupby(['family_onboard','Sex']).sum()
family_model['Survived'].plot.bar()


# 

# **The Name Column**
# 
# Now we need to address the name column.

# In[ ]:


#create a helper function to get the titles from the name column where they are currently embedded in a string.  We'll use strip to 
#reduce the string to a list of strings and pick out the string we want

honorifics_train = set()
honorifics_test = set()

for n in train_df['Name']:
    honorifics_train.add(n.split(',')[1].split('.')[0].strip())

for n in test_df['Name']:
    honorifics_test.add(n.split(',')[1].split('.')[0].strip())
        
master_list = honorifics_train | honorifics_test
master_list


# In[ ]:


#we create a title map which creates keys for each title in our existing training and test sets
#and maps to each a smaller set of categories that we will use to create mappings in the next step

title_map = {
    "Capt" : "services",
    "Col" : "services",
    "Don" : "gentry",
    "Dona" : "gentry",
    "Dr" : "profession",
    "Jonkheer" : "gentry",
    "Lady" : "gentry",
    "Major" : "services",
    "Master" : "master",
    "Miss" : "miss",
    "Mlle" : 'miss',
    "Mr" : "mr",
    "Mrs": "mrs",
    "Ms" : "ms",
    "Rev" : "profession",
    "Sir" : "gentry",
    "the Countess" : "gentry"
}



# In[ ]:


#the next bit involves two steps.  First, we repeat the extraction process we ran to pull together a list of titles repeated in the training and test sets
#with a view to replacing the current names field in each observation with just the title.  Second, we will then use our dictionary to map each 
#title to our dictionary of title groups so that we can create a title category column
#given the two steps should ideally happen relatively seamlessly, we'll process them via a helper function

def bin_titles():
    train_df['Honorific'] = train_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    test_df['Honorific'] = test_df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    
    train_df['Honorific'] = train_df.Honorific.map(title_map)
    test_df['Honorific'] = test_df.Honorific.map(title_map)


# In[ ]:


bin_titles()


# In[ ]:


train_df.head(20)


# Now we can check on the extent to which title matters for survival.  Checking on this, it does seem like it does matter, since we can see from the mean that those with the title of ms, mrs, miss and the gentry were most likely to survive.

# In[ ]:


class_model = train_df.groupby(['Honorific']).mean()
class_model_survival = pd.DataFrame(class_model['Survived'])
class_model_survival


# **Extracting Value from Cabins**
# 
# The cabin column as we have seen has around 687 missing values in the training set.  Nevertheless, we can most likely find some use for the information.  Some background research suggests that the cabin data is currently drawn from a list of first class cabin passengers found on the person of a steward, with a number more the subject of speculation.  The cabin codes are a letter followed by a number where present.  The letter seems to correspond, although this may be a matter of conjecture, to the deck on which the cabin is located. 

# In[ ]:


#First, we'll fill any nans with X to denote that we don't know the cabin details for that occupier.  We'll then map the train and the test sets such that we'll reduce each feature
#to the 

train_df['Cabin'].fillna('X', inplace = True)
test_df['Cabin'].fillna('X', inplace = True)
train_df['Cabin'] = train_df['Cabin'].map(lambda x: x[0])
test_df['Cabin'] = test_df['Cabin'].map(lambda x: x[0])
test_df.head(20)

# reached this point.  Need to sort age, fare columns and also rejig model preparation


# Our classifier won't handle missing values well so we will need to replace some of the missing ages we know there to be in the training (and possibly the test) set.  We could impute age by simply adding the median age for the whole dataset by gender, say, but instead, we'll do something a bit more precise.  We can group by passenger class and gender and honorific and impute the median age for each sub-category which will be much finer.  We do this thusly:

# In[ ]:


train_age_groups = train_df.groupby(['Sex', 'Pclass', 'Honorific'])
pd.DataFrame(train_age_groups.Age.median())

test_age_groups = test_df.groupby(['Sex', 'Pclass', 'Honorific'])
pd.DataFrame(test_age_groups.Age.median())

train_df['Age'] = train_age_groups.Age.apply(lambda y: y.fillna(y.median()))
test_df['Age'] = test_age_groups.Age.apply(lambda x: x.fillna(x.median()))

train_df.head()


# In[ ]:


fare_survival = train_df[train_df['Survived'] == 1]['Fare'] 
fare_not_survival = train_df[train_df['Survived'] == 0]['Fare']
fare_comparison = pd.DataFrame({'Mean Fare Paid by Survivors': fare_survival.mean(), 'Median Fare Paid by Survivors': fare_survival.median(), 'Mean Fare Paid by Non-Survivors': fare_not_survival.mean(), 'Median Fare Paid by Non-Survivors': fare_not_survival.median()}, index = ['Values'])
fare_comparison


# Clearly, there is a difference in mean and median between the fare paid by survivors and non-survivors.  The problem is that we would have to use the mean of our training set to create a similar feature in our test set.  At present, that risks data leakage and may be something we come back to in a future model once I've thought about it further. What about point of embarcation?

# In[ ]:


embark_point_survived = train_df[train_df['Survived'] == 1]['Embarked'].value_counts()
embark_point_not_survived = train_df[train_df['Survived'] == 0]['Embarked'].value_counts()
embark_point_combined = pd.DataFrame({'Survival by Embarcation Point': embark_point_survived, 'Non-Survival by Embarcation Point': embark_point_not_survived}, index = ['Q', 'S', 'C'])
embark_point_combined


# It doesn't appear that the embarcation point adds much to the model given that there doesn't appear to be a strong correlation between survival/non-survival and point of embarcation.

# **Prepping our Model**

# Building on our insights and our feature engineering, it's time to prepare our model.

# In[ ]:


#Let's tidy up some features that are currently framed as strings so that they can be dummified in the next steps

train_df['Pclass'] = train_df['Pclass'].astype('object')
test_df['Pclass'] = test_df['Pclass'].astype('object')
train_df['Embarked'] = train_df['Embarked'].astype('object')
test_df['Embarked'] = test_df['Embarked'].astype('object')
train_df['family_onboard'] = train_df['family_onboard'].astype('object')
test_df['family_onboard'] = test_df['family_onboard'].astype('object')
train_df['Sex'] = train_df.Sex.map({'male': 0, 'female': 1})
test_df['Sex'] = test_df.Sex.map({'male': 0, 'female': 1})


# Now we need to create dummy columns for our categorical variables and make our pipeline.

# In[ ]:


#load initial dependencies

from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#create our target

train_y = train_df.Survived

#create our predictors and drop some cols from which we've extracted derived features

train_x = train_df.drop(['Survived','Ticket','Name'], axis = 1)
test_x = test_df.drop(['Ticket','Name'], axis = 1)

#call some descriptive statistics about the data

train_x.dropna()
test_x.dropna()


# In[ ]:


test_x.isnull().sum()


# 

# We need to impute our numeric columns (age in the case of the training set and the test set, but there is also a test observation that contains a missing value for fare so we will impute that value too).

# In[ ]:


train_predictors_numeric = train_x.select_dtypes(exclude = ['object'])
test_predictors_numeric = test_x.select_dtypes(exclude = ['object'])
train_predictors_categorical = train_x.select_dtypes(['object'])
test_predictors_categorical = test_x.select_dtypes(['object'])


# In[ ]:


train_x_one_hot_encoded = pd.get_dummies(train_predictors_categorical)
test_one_hot_encoded = pd.get_dummies(test_predictors_categorical)
coded_train, coded_test = train_x_one_hot_encoded.align(test_one_hot_encoded, join = 'inner', axis = 1)


# In[ ]:


train_x = train_predictors_numeric.merge(coded_train, left_index = True, right_index = True)
train_x.isnull().sum()


# In[ ]:


test_x = test_predictors_numeric.merge(coded_test, left_index = True, right_index = True)
test_x.head()


# Let's fix age.  As we have 177 missing values from our Age column, we have to fill these.  We have opted to go with the median since it will not be affected by outliers as much as the mean.

# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline

#my_pipeline = make_pipeline(Imputer(axis = 1, strategy = 'median'), XGBClassifier(learning_rate = 0.02, n_estimators = 600, objective = 'binary:logistic', silent = True, nthread = 1))

my_pipeline = XGBClassifier(learning_rate = 0.02, n_estimators = 600, objective = 'binary:logistic', silent = True, nthread = 1)


# In[ ]:


#We'll perform a GridSearch to find the best hyperparameters for our model.  
# First, we need to assemble a dictionary of parameters

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

params = {
    'max_depth': [3,4,5],
    'min_child_weight': [1,5,10],
    'gamma': [0.5,1,1.5,2,5],
    'colsample_bytree': [0.6,0.8,1.0],
    'subsample': [0.6,0.8,1.0]
}


# In[ ]:


folds = 3
param_comb = 5

xgb_skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)

random_search = RandomizedSearchCV(my_pipeline, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=xgb_skf.split(train_x,train_y), verbose=3, random_state=42)

random_search.fit(train_x, train_y)


# In[ ]:


print(random_search.best_params_)


# In[ ]:


my_pipeline = make_pipeline(Imputer(axis = 1, strategy = 'median'), XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=2, learning_rate=0.02, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=600,
       n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1))


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(my_pipeline, train_x, train_y, scoring = 'neg_mean_absolute_error')
print(scores)


# In[ ]:


print('Mean across scores: {}'.format(-1 * scores.mean()))


# In[ ]:


my_pipeline.fit(train_x, train_y)
my_predictions = my_pipeline.predict(test_x)


# Make Predictions

# In[ ]:


predicted_survival = my_pipeline.predict(test_x)
test_x['PassengerId'] = test_x['PassengerId'].astype('int64')
my_submission = pd.DataFrame({'PassengerId': test_x.PassengerId, 'Survived': predicted_survival})

my_submission.to_csv('submission.csv', index=False)

