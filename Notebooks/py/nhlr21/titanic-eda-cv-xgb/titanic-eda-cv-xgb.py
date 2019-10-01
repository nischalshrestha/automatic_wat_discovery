#!/usr/bin/env python
# coding: utf-8

# # Titanic : Machine Learning from Disaster

# In this notebook, we are going to create an algorithm which will predict  if a passenger of the titanic survived or not. This algorithm will learn from train dataset and then it will predict if each passenger of the test dataset survived. 
# 
# The plan to do that is the following one :
# 
# 1. Exploration of data
# 2. Features engineering
# 3. Cross validation testing of the model
# 4. Prediction on test data & submission on kaggle

# ## Imports and useful functions

# In[ ]:


import pandas as pd
import matplotlib
import pydot
import re

get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import numpy as np
import sklearn
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC  
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier

from IPython.display import display


# In[ ]:


#path of datasets
path_train = '../input/train.csv'
path_test = '../input/test.csv'

def display_confusion_matrix(sample_test, prediction, score=None):
    cm = metrics.confusion_matrix(sample_test, prediction)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if score:
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15)
    print(metrics.classification_report(sample_test, prediction))
    
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz"""
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
        
def model_tuning(model, param_grid):
    
    gc_cv = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    gc_cv.fit(X_train_sample_train, Y_train_sample_train)
    
    return gc_cv.best_params_, gc_cv.best_score_, gc_cv.cv_results_

def standardize(df):
    
    standardize_df = df.copy()
    
    target = None
    if 'Survived' in standardize_df.columns.tolist():
        target = standardize_df['Survived'] # Separating out the target before standardizing
        standardize_df = standardize_df.drop(['Survived'],  axis=1)

    # Standardizing the features
    scaled_values = StandardScaler().fit_transform(standardize_df.values)
    standardize_df = pd.DataFrame(scaled_values, index=standardize_df.index, columns=standardize_df.columns)
    if target is not None:
        standardize_df = standardize_df.join(target)
    
    return standardize_df


# ## 1. Data exploration

# In[ ]:


#create dataframe for training dataset and print ten first rows as preview
train_df_raw = pd.read_csv(path_train)
train_df_raw.head()


# In[ ]:


# Compute some basical statistics on the dataset
train_df_raw.describe()


# In[ ]:


# Let's plot some histograms to have a previzualisation of some of the data ...
train_df_raw.drop(['PassengerId'], 1).hist(bins=50, figsize=(20,15))
plt.show()


# With this first exploration, we can see that :
# 
# * Only aproximately 35% of passengers survived ...
# * More than the half of passengers are in the lowest class (pclass = 3)
# * Most of the fare tickets are below 50
# * Majority of passengers are alone (sibsp and parch)

# ## 2. Features engineering

# In[ ]:


def preprocess_data(df):
    
    # Replace string data by numeric data
    df['Cabin'].fillna('U0', inplace=True)
    df['Sex'] = df['Sex'].replace('male', 1)
    df['Sex'] = df['Sex'].replace('female', 0)
    df['Embarked'] = df['Embarked'].replace('S', 0)
    df['Embarked'] = df['Embarked'].replace('C', 1)
    df['Embarked'] = df['Embarked'].replace('Q', 2)
    df['Embarked'].fillna(0, inplace=True) # because there is approximately 80% of 0 in embarked column

    # Replace NaN data in age column by mean age of passengers
    mean_age = df['Age'].mean()
    df['Age'].fillna(mean_age, inplace=True)
    
    df['Fare'] = df['Fare'].interpolate()

    # Let's work on 'Name' column : we find the title in the name and extract it on a new column 'Title'
    mapping_title = {title: pos for pos, title in enumerate(set([name.split('.')[0].split(',')[1].strip() for name in df['Name']]))}
    df['Title'] = pd.Series((mapping_title[name2.split('.')[0].split(',')[1].strip()] for name2 in df['Name']), index=df.index)
    df = df.drop(columns=['Name'])

    #Creation of a deck column corresponding to the letter contained in the cabin value
    mapping_deck = {title: pos for pos, title in enumerate(set([cab[:1] for cab in df['Cabin']]))}
    df['Deck'] = pd.Series((mapping_deck[cab2[:1]] for cab2 in df['Cabin']), index=df.index)

    # Modify the cabin column to keep only the cabin number
    cabin_numbers = list()
    for cab3 in df['Cabin']:
        if len(cab3) != 1:
            cabin_numbers.append(int(cab3[1:].strip()) if len(cab3[1:]) <= 3 else int(cab3[len(cab3)-2:].strip()))
        else:
            cabin_numbers.append(0)

    # df['Cabin'] = pd.Series(cabin_numbers, index=df.index)
    df = df.drop(['Cabin'], 1)

    # Modify the ticket column to keep only the ticket number
    ticket_numbers = list()
    for ticket in df['Ticket']:
        try:
            ticket_numbers.append(int(ticket))
        except ValueError:
            splitted = ticket.split(' ')
            if len(splitted) == 1:
                ticket_numbers.append(0)
            else:
                ticket_numbers.append(int(splitted[len(splitted)-1]))

    df['Ticket'] = pd.Series(ticket_numbers, index=df.index)
        
    return df


# ## 3. Cross validation of model on train dataset

# In[ ]:


# Let's divide the train dataset in two datasets to evaluate perfomance of machine learning models used
train_df = train_df_raw.copy()
X_train = train_df.drop(['Survived'], 1)
Y_train = train_df['Survived']

# Split dataset for prediction
X_train_sample_train, X_train_sample_test, Y_train_sample_train, Y_train_sample_test = sklearn.model_selection.train_test_split(X_train,
                                                                                                                                Y_train, 
                                                                                                                                test_size=0.3, 
                                                                                                                                random_state=42)

X_train_sample_train = preprocess_data(X_train_sample_train)
X_train_sample_test = preprocess_data(X_train_sample_test)
X_train_sample_train = standardize(X_train_sample_train)
X_train_sample_test = standardize(X_train_sample_test)

X_train_sample_train.head()


# ### Try several models

# In[ ]:


# Create and train model on train data sample
logisticRegr = LogisticRegression(random_state=42)
logisticRegr.fit(X_train_sample_train, Y_train_sample_train)

# Predict for test data sample
logistic_prediction_train = logisticRegr.predict(X_train_sample_test)

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_train_sample_test, logistic_prediction_train)
display_confusion_matrix(Y_train_sample_test, logistic_prediction_train, score=score)


# In[ ]:


# Create and train model on train data sample
dt = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=8, random_state=42)
dt.fit(X_train_sample_train, Y_train_sample_train)

# Predict for test data sample
dt_prediction_train = dt.predict(X_train_sample_test)

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_train_sample_test, dt_prediction_train)
display_confusion_matrix(Y_train_sample_test, dt_prediction_train, score=score)


# In[ ]:


visualize_tree(dt, X_train_sample_test.columns)
get_ipython().system(u' dot -Tpng dt.dot > dt.png')


# ![title](dt.png)

# In[ ]:


# Create and train model on train data sample
rf = RandomForestClassifier(n_estimators=50, random_state = 42)
rf.fit(X_train_sample_train, Y_train_sample_train)

# Predict for test data sample
rf_prediction_train = rf.predict(X_train_sample_test)

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_train_sample_test, rf_prediction_train)
display_confusion_matrix(Y_train_sample_test, rf_prediction_train, score=score)


# In[ ]:


# Create and train model on train data sample
vot = VotingClassifier([('dt', dt), ('lr', logisticRegr), ('rf', rf)], voting='soft')
vot.fit(X_train_sample_train, Y_train_sample_train)

# Predict for test data sample
vot_prediction_train = vot.predict(X_train_sample_test)

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_train_sample_test, vot_prediction_train)
display_confusion_matrix(Y_train_sample_test, vot_prediction_train, score=score)


# In[ ]:


# First try with a random forest with all default parameters
boost = XGBClassifier(random_state=42) # replace the sckitlearn class here to try other models
boost.fit(X_train_sample_train, Y_train_sample_train)

# Predict for test data sample
boost_prediction_train = boost.predict(X_train_sample_test)

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_train_sample_test, boost_prediction_train)

display_confusion_matrix(Y_train_sample_test, boost_prediction_train, score=score)


# ## 4. Apply on test dataset and submit on kaggle 

# In[ ]:


test_df_raw = pd.read_csv(path_test)
test_df = test_df_raw.copy()
X_test = preprocess_data(test_df)
X_test = standardize(X_test)
X_train = preprocess_data(X_train)
X_train = standardize(X_train)


# In[ ]:


# Create and train model on train data sample
model_test = XGBClassifier(random_state=42) # replace the sckitlearn class here to try other models
model_test.fit(X_train, Y_train)

# Predict for test data sample
model_test_prediction_test = model_test.predict(X_test)

result_df_xgb = test_df_raw.copy()
result_df_xgb['Survived'] = model_test_prediction_test.astype(int)

result_df_xgb.head()
result_df_xgb.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)


# #### *Precision obtained on kaggle : 0.79425 !** 🎉*
