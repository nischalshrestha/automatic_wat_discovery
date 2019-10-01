#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction #
# ## 1.1 Project Summary ##
# 
# The sinking of titanic on its maiden voyage is well known around the world and evidently called for maritime regulations. Titanic left on her maiden voyage on April 15, 1912 and collided with an iceberg killing 1502 out of 2224 passengers and crew. One of the main reasons for such heavy loss of life was that there weren't enough life boats on board. 
# 
# ## 1.2 Objective ##
# 
# As part of this solution we will be exploring how the survival rate has been influenced by various factors like age, gender and passenger class. In the end we will predict the survival of passengers in the test dataset. 

# ## 1.3 Data Dictionary ##
# 
# survival:	Survival (0 = No, 1 = Yes)
# pclass:	    Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
# Sex:	    Sex (male, female)
# Age:	    Age in years	
# sibsp:	    # of siblings / spouses aboard the Titanic	
# parch:	    # of parents / children aboard the Titanic	
# ticket:	    Ticket number	
# fare:	    Passenger fare	
# cabin:	    Cabin number	
# embarked:	Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
# 

# # 2. Exploratory Analysis #
# 
# To perform exploratory analysis, visualization and machine learning techniques on this dataset, we will be using a variety of data science packages. The first step towards implementation is importing the required packages.

# ## 2.1 Loading Data ##
# 
# This section deals with loading necessary packages to perform loading data and exploratory data analysis.
# 
# ### Importing packages ###

# In[ ]:


""" importing required packages """
get_ipython().magic(u'matplotlib inline')

""" packages for data manipulation, munging and merging """
import pandas as pd
import numpy as np

""" packages for visualiztion and exploratory analysis """
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

""" configure visualization """

""" packages for running machine learning algorithms """

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# ### Data loading functions ###

# In[ ]:


""" training and test files """
train_file = '../input/train.csv'
test_file = '../input/test.csv'
    
""" read training data into a pandas dataframe """
def read_training_data(filename):
    """ reading training data using pandas read_csv function """
    titanic_train = pd.read_csv(filename)
    return titanic_train

""" read test data into a pandas dataframe """
def read_test_data(filename):
    titanic_test = pd.read_csv(filename)
    return titanic_test

""" function to combine training and test dataframes """
def combine_dataframes(trainDF, testDF):
    combined = trainDF.append(testDF, ignore_index=True)
    return combined

train_df = read_training_data(train_file)
titanic_test = read_test_data(test_file)


# # 2.2 Exploratory analysis #
# 
# ## Summary statistics ##

# In[ ]:


def training_data_summary(train_df):
        """ print summary statistics for training dataset """
        print(train_df.describe())
        print(train_df.info())
        print(train_df.shape)
        print(train_df.head(5))
        print(train_df.tail(5))

""" output summary statistics on training data """
training_data_summary(train_df)


# Preliminary observations about the training dataset
# 
# 1. the shape of training dataframe is (891,12) i.e 891 rows with 12 columns
# 2. survived, pclass, sex should be of categorical datatypes. We will doing the conversion in the later steps
# 3. values are missing in age, cabin and embarked columns. Although embarked has only 2 missing values
# 
# Some of the steps we need to take to ensure our analysis does not yield incorrect results
# 
# 1. we will use suitable techniques to fill missing values
# 2. we will be doing exploratory analysis on various features by visualizing the data
# 

# From the summary statistics we can observe that
# 
# 1. In case of age, the mean and median (50th percentile) are quite close (not many outliers) and therefore it makes sense to impute missing values using mean
# 2. In case of Fare, the mean and median (50th percentile) have significant difference and therefore we will be using medain to impute missing values
# 
# ## Imputing missing values ##

# In[ ]:


def fill_missing_values(df):
    """ filling missing values in Age column by computing the medain of Age """
    df['Age'] = df.Age.fillna(train_df.Age.median())
    df['Fare'] = df.Fare.fillna(train_df.Fare.median())
    df['Embarked'] = df.Embarked.fillna(method='ffill')
    return df

""" impute missing values """
train_df_filled = fill_missing_values(train_df)


# ## Datatype conversions ##

# In[ ]:


def datatype_conversion(df):
    df['Sex'] = df['Sex'].astype('category')
    df['Pclass'] = df['Pclass'].astype('category')
    df['Embarked'] = df['Embarked'].astype('category')
    return df

""" perform datatype conversions """
titanic_df = datatype_conversion(train_df_filled)

""" converting the outcome variable of the dataset to int type  """
titanic_df["Survived"] = titanic_df['Survived'].astype('int')


# ## Visualizing categorical data ##

# In[ ]:


def visualize_categories(df, **kwargs):
    row = kwargs.get('row',None)
    col = kwargs.get('col',None)
    hue = kwargs.get('hue',None)
    
    target_var = 'Survived'
    category_list = ['Sex','Pclass', 'Embarked']
    hue_var = 'Pclass'
    for cat_var in category_list:
        if cat_var != hue_var:
            plt.figure()
            sns.barplot(x=cat_var, y=target_var, hue=hue_var, data=df, ci=None)
        else:
            plt.figure()
            sns.barplot(x=cat_var, y=target_var, data=df, ci=None)

""" plotting categorial features against surivival rate """
visualize_categories(titanic_df)


# ### Observations on categorical plots ###
# 
# 1. Among the 3 ticket classes, no of females and males survived are highest for ticket class 1. Overall among 3 ticket classes, females survival rate is higher than males
# 2. Among the 3 embarkation points, the highest survival rate is among those embarked from Cherbourg and that too from ticket class 1
# 3. Overall female survival rate is higher than the male survival rate

# ## Visualizing distributions ##

# In[ ]:


def visualize_distribution(df, feature, **kwargs):
    row = kwargs.get('row',None)
    col = kwargs.get('col',None)

    g1 = sns.FacetGrid(df, row = row, col = col, aspect=2)
    g1.map(plt.hist, feature, bins=20, alpha=0.5)
    xlim_max = df[feature].max()
    g1.set(xlim=(0,xlim_max))
    g1.add_legend()

""" distribution plots """
visualize_distribution(titanic_df, feature='Age', col='Survived')
visualize_distribution(titanic_df, feature='Fare', col='Survived')
visualize_distribution(titanic_df, feature='Age', row='Pclass', col='Survived')
visualize_distribution(titanic_df, feature='Age', row='Sex', col='Survived')


# Observations on distribution plots
# 
# 1. Highest human loss is between the age groups 15-30
# 2. Understandably the loss is also highest for ticket class 3 assuming that the no of passengers travelling in that class is higher than the other 2 classes combined together. This could be class most preferred by the working class and those from lower socio economic groups
# 3. Male passengers were far higher than female passengers and this reflects directly on the human loss. Males were highest among those did not survive the day.

# ## Data preparation ##

# In[ ]:


def feature_conversion(df):
    df['Pclass'] = df['Pclass'].astype('int')
    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype(int)
    df['Embarked'] = df['Embarked'].map({'C':0,'Q':1,'S':2}).astype(int)
    
    """ setting numerical age bands based on exploratory analysis observations"""
    df.loc[(df['Age'] <= 20),'Age'] = 0
    df.loc[(df['Age'] > 20) & (df['Age'] <= 28), 'Age'] = 1
    df.loc[(df['Age'] > 28) & (df['Age'] <= 38), 'Age'] = 2
    df.loc[(df['Age'] > 38) & (df['Age'] <= 80), 'Age'] = 3
    df.loc[(df['Age'] > 80), 'Age'] = 4
    
    df.loc[(df['Fare'] <= 7.91),'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454),'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31),'Fare'] = 2
    df.loc[(df['Fare'] > 31),'Fare'] = 3
    
    df['Age'] = df['Age'].astype('int')
    df['Fare'] = df['Fare'].astype('int')
    
    return df

train_df_prep = feature_conversion(titanic_df)


# In[ ]:


def drop_features(df):
    df = df.drop(['Ticket','Cabin','Name'], axis=1)
    return df

train_df_final = drop_features(train_df_prep)

""" perform operations on test dataset """
test_df_filled = fill_missing_values(titanic_test)
test_df_converted = datatype_conversion(test_df_filled)
test_df_prep = feature_conversion(titanic_test)
test_df_final = drop_features(test_df_prep)

combined = combine_dataframes(train_df_final, test_df_final)


# # 3. Modeling and Predicting #
# 
# This particular problem falls into the category of supervised learning as the goal is to predict attribute based on existing features. We will use the following supervised learning algorithms to predict the survival of passengers based on various attributes.
# 
# 1. LogisticRegression
# 2. SVC
# 3. K-Nearest Neighbours
# 4. Naive Bayes
# 5. Decision Trees
# 6. Random Forest
# 
# Before we proceed to modeling and predicting, we have to alter the training and test dataset to suit the input requirements for the algorithms mentioned above. Once we have the training set and external variable to predict, we call the functions to run various supervised learning models. Subsequently when we get the accuracy scores from different models, we plot and see which model score has the highest score and use that model to predict the final outcome. Then we write the PassengerId, Survived (predicted outcome) to a csv file. 
# 
# The model calling steps, combining the accuracy scores and plotting are all performed in the same function *"dataprep_for_modeling"*.
# 
# The inputs to fit() and predict() methods for all the models are passed to functions as
# 
#  1. df_X = Training vector
#  2. df_Y = Target vector relative to X
#  3. test_df_X = Samples

# In[64]:


def perform_logistic_regression(df_X, df_Y, test_df_X):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(df_X, df_Y)
    pred_Y = logistic_regression.predict(test_df_X)
    accuracy = round(logistic_regression.score(df_X, df_Y) * 100,2)
    returnval = {'model':'Logistic Regression','accuracy':accuracy}
    return returnval


# In[65]:


def perform_svc(df_X, df_Y, test_df_X):
    svc_clf = SVC()
    svc_clf.fit(df_X, df_Y)
    pred_Y = svc_clf.predict(test_df_X)
    accuracy = round(svc_clf.score(df_X, df_Y) * 100, 2)
    returnval = {'model':'SVC', 'accuracy':accuracy}
    return returnval


# In[66]:


def perform_linear_svc(df_X, df_Y, test_df_X):
    svc_linear_clf = LinearSVC()
    svc_linear_clf.fit(df_X, df_Y)
    pred_Y = svc_linear_clf.predict(test_df_X)
    accuracy = round(svc_linear_clf.score(df_X, df_Y) * 100, 2)
    returnval = {'model':'LinearSVC', 'accuracy':accuracy}
    return returnval


# In[67]:


def perform_rfc(df_X, df_Y, test_df_X):
    rfc_clf = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None)
    rfc_clf.fit(df_X, df_Y)
    pred_Y = rfc_clf.predict(test_df_X)
    accuracy = round(rfc_clf.score(df_X, df_Y) * 100, 2)
    returnval = {'model':'RandomForestClassifier','accuracy':accuracy}
    return returnval


# In[68]:


def perform_knn(df_X, df_Y, test_df_X):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_X, df_Y)
    pred_Y = knn.predict(test_df_X)
    accuracy = round(knn.score(df_X, df_Y) *100,2)
    returnval = {'model':'KNeighborsClassifier','accuracy':accuracy}
    return returnval


# In[69]:


def perform_gnb(df_X, df_Y, test_df_X):
    gnb = GaussianNB()
    gnb.fit(df_X, df_Y)
    pred_Y = gnb.predict(test_df_X)
    accuracy = round(gnb.score(df_X, df_Y)*100,2)
    returnval = {'model':'GaussianNB','accuracy':accuracy}
    return returnval


# In[70]:


def perform_dtree(df_X, df_Y, test_df_X):
    dtree = DecisionTreeClassifier()
    dtree.fit(df_X, df_Y)
    pred_Y = dtree.predict(test_df_X)
    accuracy = round(dtree.score(df_X, df_Y)*100,2)
    returnval = {'model':'DecisionTreeClassifier','accuracy':accuracy}
    return returnval


# In[ ]:


train_X = train_df_final.drop(['Survived','PassengerId'],axis=1)
train_Y = train_df_final['Survived']
test_X = test_df_final.drop('PassengerId', axis=1).copy()
    
lr_val = perform_logistic_regression(train_X, train_Y, test_X)
svc_val = perform_svc(train_X, train_Y, test_X)
svc_lin_val = perform_linear_svc(train_X, train_Y, test_X)
rfc_val = perform_rfc(train_X, train_Y, test_X)
knn_val = perform_knn(train_X, train_Y, test_X)
gnb_val = perform_gnb(train_X, train_Y, test_X)
dtree_val = perform_dtree(train_X, train_Y, test_X)
    
model_accuracies = pd.DataFrame()
model_accuracies = model_accuracies.append([lr_val,svc_val,svc_lin_val, rfc_val, knn_val, gnb_val, dtree_val])
cols = list(model_accuracies.columns.values)
cols = cols[-1:] + cols[:-1]
model_accuracies = model_accuracies[cols]
model_accuracies = model_accuracies.sort_values(by='accuracy')
print(model_accuracies)
plt.figure()
plt.xticks(rotation=90)
sns.barplot(x='model', y='accuracy', data=model_accuracies)


# # 4. Output #
# 
# The accuracy scores returned from the modeling functions are as listed below.
# 
# * GaussianNB: 75.87 
# * LogisticRegression: 79.91
# * LinearSVC: 79.91
# * SVC: 82.94
# * KNeighborsClassifier: 85.63
# * RandomForestClassifier: 89.00
# * DecisionTreeClassifier: 89.00
# 
# Since RandomForestClassifier and DecisionTreeClassifier has the highest scores, we will use RandomForestClassifier to write the output to CSV file.

# In[ ]:


def write_to_csv(train_X, train_Y, test_df, test_X):
    rfc_clf = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None)
    rfc_clf.fit(train_X, train_Y)
    pred_Y = rfc_clf.predict(test_X)
    pred_Y_list = pred_Y.tolist()
    test_X['Survived'] = pred_Y
    test_X['PassengerId'] = test_df['PassengerId']
    final_df = test_X[['PassengerId','Survived']]
    final_df.to_csv('passenger_survival.csv',sep=',',index=False)
    
write_to_csv(train_X, train_Y, test_df_final, test_X)


# # Summary #
# ## Closing Statement ##
# 
# It is never too late to learn something and try to be good at it. With that in mind and seeking inspiration from fellow kagglers, I decided to step into the world of data science with the titanic machine learning dataset. I am taking it as a challenge and fun exercise and I hope fellow kagglers will look at my work and provide some constructive feedback.
# 
# ## References Used ##
# 
#  1. [Titanic Data Science Solutions][1]
#  2. [Interactive Data Science Tutorial][2]
# 
# [1]: https://www.kaggle.com/startupsci/titanic-data-science-solutions
# [2]: https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial
# 
