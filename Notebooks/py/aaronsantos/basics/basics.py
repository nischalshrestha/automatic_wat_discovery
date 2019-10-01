#!/usr/bin/env python
# coding: utf-8

# # About This Kernel
# This is my first kernel. I have a programming background and I'm learning machine learning. 
# 
# My goal with this kernel is to:
# * Load the data
# * Clean and fill in missing values
# * Plot and find relationships
# * Build and compare models
# * Make  submission

# # Loading The Data
# Let's start with some basic imports and list the data files.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# The train, test, and sample submission files are all there.
# 
# Let's load the training data set.

# In[2]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[3]:


len(train)


# There are 891 items in the training set.
# 
# Let's load the test set.

# In[4]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[5]:


len(test)


# There are 418 items in the test set.

# # Imputation
# Show the number of missing values in each column
# 

# In[6]:


train.isnull().sum()


# Age, and Embarked look easy enough to fill in. However, cabin is going to be difficult. We'll likely drop it.

# LetStart by creating a functions to plugin into a Pandas pipeline. We'll run both the train and test dataframes through the same pipeline so that they are processed identically.
# 
# See https://chrisalbon.com/python/data_wrangling/pandas_create_pipeline/
# 
# Let's start with a function that fills in age. We'll find the median male and female ages and fill missing values using the apropriate median.
# 

# In[7]:


def fill_median_age(df, sex):
    median_age = df[(df.Sex == sex)]['Age'].median()
    updated = df[(df.Sex == sex)].fillna({'Age': median_age})
    df.update(updated)
    return df
fill_median_age


# Fill in the two missing Embarked values using mode

# In[8]:


def fill_embarked(df):
    embarked = df['Embarked'].mode().iloc[0]
    return df.fillna({'Embarked': embarked})
fill_embarked


# In[9]:


def fill_fare(df):
    fare = df['Fare'].median()
    return df.fillna({'Fare': fare})
fill_fare


# In[10]:


def drop_cabin(df):
    return df.drop(columns=['Cabin'])
drop_cabin


# Pipe together the cleaning functions

# In[11]:


def clean(df):
    return (df
         .pipe(fill_median_age, sex = 'male')
         .pipe(fill_median_age, sex = 'female')
         .pipe(fill_embarked)
         .pipe(fill_fare)
         .pipe(drop_cabin)
    )
clean


# In[12]:


clean_train = clean(train);
clean_train.head()


# In[13]:


clean_train.isnull().sum()


# Do the same for test data

# In[14]:


clean_test = clean(test);
#X_test = clean_test.drop(columns = 'Survived')
#y_test = clean_test['Survived']
clean_test.head()


# In[15]:


clean_test.isnull().sum()


# # Imputation Summary
# We cleaned both the train and test data. First we created cleaning functions and then we created a pipeline. Finally we used that pipeline to assign cleaned dataframes to `clean_test` and `clean_train`. We cleaned
# * Male age by finding the median male age and assigning it to missing male ages
# * Female age by finding the median female age and assigning it to the missing female ages
# * Finding the mode of embarked codes and assigning it to the missing embarked values

# # Getting to know the data
# Next we'll use Seaborn to plot the data. We'll want to find quantitative and caragorial features that correlate with survival.

# In[16]:


sb.heatmap(clean_train.corr(), annot=True, fmt=".2f")


# * Class and fare also stongly correlate with eachother.  High class tickets are simply more expensive.
# * Class and fare both correlate stongly with survival. Higher class passenger have higher survival.
# * Parch and SbSp correlate with eachother.
# * Parch and Fare losely correlate. Do smaller families travel in higher classes?
# 
# We'll want to use either class and fare or both as features when building out model.

# Let's pair plot the quantative variables grouping by class and by sex.

# In[17]:


sb.pairplot(clean_train, hue = 'Pclass')


# In[18]:


sb.pairplot(clean_train, hue = 'Sex')


# Let's highlight some observations.

# In[19]:


sb.countplot(x = 'Survived', hue = 'Pclass', data = clean_train)


# 3rd class passengers have very poor survival rates

# In[20]:


g = sb.FacetGrid(clean_train, hue="Survived", size = 5)
g.map(sb.distplot, "Age")


# Children (Age < 18) have a greater chance of surviving and 30 year olds a much lower chance. It could be useful to synthesize features that correspond to these age ranges.

# In[21]:


g = sb.FacetGrid(clean_train, hue="Survived", size = 5)
g.map(sb.distplot, "Parch")


# When Parch = 0, passengers have a lower rate of survival. Values of 1 and 2 correspond to a higher rate of survival.

# In[22]:


g = sb.FacetGrid(clean_train, hue="Survived", size = 5)
g.map(sb.distplot, "SibSp")


# When SibSp is 1 the survival rate is highest. When SibSp is 1 the survival rates are even. For all other values survival is is lower.

# # Modeling
# Let's start by creting a test/train split. 75% of the data is used to train and 25% is used to test.

# In[23]:


from sklearn.model_selection import train_test_split
X = clean_train.drop(columns = 'Survived')
y = clean_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Import the sklean and sklean_pandas packages we'll need.

# In[24]:


import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper, cross_val_score


# It's important to establish a basetile estimator to compare our results to. This estimator classifies everyone as died. We'll create a mapper for our DataFrame that encodes the Sex column and passes everything else through.

# In[25]:


# Pass values through and binarize the Sex column
mapper = DataFrameMapper([
    ('Pclass', None),
    ('Age', None),
    ('Parch', None),
    ('SibSp', None),
    ('Sex', LabelBinarizer())
])


# In[26]:


pipe = Pipeline([
    ('featurize', mapper),
    ('lm', DummyClassifier(strategy='most_frequent',random_state=0))])
clf = pipe.fit(X_train, y_train)
print("mean accuracy")
print("train: ", clf.score(X_train, y_train))
print("test: ", clf.score(X_test, y_test))


# The DummyClassifier has a mean accuracy of 0.62 on the test data.

# Let's start with a simple linear regression. 

# https://github.com/scikit-learn-contrib/sklearn-pandas
# 
# http://maciejjaskowski.github.io/2016/01/22/pandas-scikit-workflow.html
# 
# https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# 
# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
# 

# In[27]:


from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('featurize', mapper),
    ('lm', LinearRegression())])
clf = pipe.fit(X_train, y_train)
print("train: ", clf.score(X_train, y_train))
print("test: ", clf.score(X_test, y_test))


# A score of 1.0 is perfect while score of 0.0 does no better than expected value of the predicted variable. Our score of 0.40 (R^2 > 0) is fair. Let's do better.

# Reviewing the feature analysis portion let's encode these features
# * Pclass as a one hot encoding
# * Age into one hot encoded ranges. < 18, 18 - 25, 25-35, 35-40, 40-60, >60
# * Parch into one hot encodings 0, 1, 2, 3+
# * SibSp into a binary classification: 1 and everything else
# * Sex as a binary classification

# In[28]:


import sklearn.tree
def class_age(age):
    if age <= 18:
        return 0
    elif age <= 25:
        return 1
    elif age <= 35:
        return 2
    elif age <= 40:
        return 3
    elif age <= 60:
        return 4
    else:
        return 5

def class_parch(parch):
    if parch < 3:
        return parch
    else:
        return 3

def class_sibsp(sibsp):
    if sibsp == 1:
        return 1
    else:
        return 0
    
def lift_to_array(func):
    return lambda X: np.vectorize(func)(X)
featurizer = DataFrameMapper([
        # Scaling features doesn't improve performace in this example, but it's good to get into the habit
        (['Pclass'], StandardScaler()),
        (['Fare'], StandardScaler()),
        (['Age'], Pipeline([
            #('age_scaler', StandardScaler()),
            ('age_func', sklearn.preprocessing.FunctionTransformer(lift_to_array(class_age))),
            ('age_enc', sklearn.preprocessing.OneHotEncoder()),
        ])),
        (['Parch'], Pipeline([
            #('age_scaler', StandardScaler()),
            ('parch_func', sklearn.preprocessing.FunctionTransformer(lift_to_array(class_parch))),
            ('parch_enc', sklearn.preprocessing.OneHotEncoder()),
        ])),
        (['SibSp'], Pipeline([
            #('age_scaler', StandardScaler()),
            ('sibsp_func', sklearn.preprocessing.FunctionTransformer(lift_to_array(class_sibsp))),
            ('sibsp_enc', sklearn.preprocessing.OneHotEncoder()),
        ])),
        ('Sex', LabelBinarizer())
    ])
pipe = Pipeline([
    ('featurize', featurizer),
    ('lm', sklearn.linear_model.LogisticRegression())
])
#np.round(cross_val_score(pipe, X=clean_train.copy(), y=clean_train['Survived'], cv=20, scoring='r2'), 2)
clf = pipe.fit(X_train, y_train)
print("train: ", clf.score(X_train, y_train))
print("test:  ", clf.score(X_test, y_test))


# The feature enginered and logistic regression score of 0.78 (R^2 > 0) is much better than linear regression.
# 
# Let's quantify the model's performance. First, a confusion matrix.

# In[29]:


# Use the model to make predictions
y_predicted = clf.predict(X_test)

import scikitplot as skplt
import matplotlib.pyplot as plt

#skplt.metrics.plot_roc_curve(y_test, y_predicted)
skplt.metrics.plot_confusion_matrix(y_test, y_predicted)
plt.show()


# The false positive and negative rates are roughly equal. Do these represent similar or different kinds of passengers?
# 
# Let's compute general metrics.

# In[30]:


print(sklearn.metrics.classification_report(y_test, y_predicted))


# Finally, let's find the receiver operating characteristic curve for different threshold values.

# In[32]:


y_probas = clf.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()


#  Let's use this estimator to predict the test dataframe and save the results for submission.

# In[33]:


predicted = clf.predict(clean_test)
logistic_submission = pd.DataFrame(
    {'PassengerId': clean_test.PassengerId,
     'Survived': predicted}).astype('int32')
# you could use any filename. We choose submission here
logistic_submission.to_csv('submission.csv', index=False)
logistic_submission.head()


# # Summary
# Let's review what we've done. We performed some exploratory data analysis, decided which features were important, we built three models, and created a submission using a logistic regression model.
# 
# ## Data Analysis
# First we cleaned the data and then explored relationships. Then we found values that might help make predictions. We used the Passanger, Age, Class, Fare, Sex, Parch, and SpSib values to derive features.
# 
# ## Feature Engineering And Modeling
# We built three models. The first model predicted that everyone died. Second, we built a linear regression that didn't perform well. Thid, we did some feature engineering and used a logistic regression that improved on the two prior models. Finally we saved the predictions made by the linear regression model and submitted the results.
# 
# This is a basic kernel that describes the steps of analysis and modeling.
# 
# 

# In[ ]:




