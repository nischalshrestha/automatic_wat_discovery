#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib #collection of functions for scientific and publication-ready visualization
import scipy as sp #collection of functions for scientific computing and advance mathematics
import IPython #pretty printing of dataframes in Jupyter notebook
from IPython import display
import sklearn #collection of machine learning algorithms

#misc libraries
import random
import time

#ignore warning
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Load Modelling Algorithms
#We will use the popular scikit-learn library to develop our machine learning algorithms. 
#In sklearn, algorithms are called Estimators and implemented in their own classes. For data visualization, we will use the matplotlib and seaborn library. Below are common classes to load.
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics


#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


#visualization defaults
get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12, 8

#lets first import our data
data_raw = pd.read_csv('../input/train.csv')
data_val  = pd.read_csv('../input/test.csv')

#Make a deep copy, including a copy of the data and the indices. With deep=False neither the indices nor the data are copied.
data1 = data_raw.copy(deep= True)
#in order to clean both datasets at once
data_cleaner = [data1, data_val]

#preview data

data_raw.info()


# In[ ]:


data_raw.sample(10)


# 1. The Survived variable is our outcome or dependent variable. It is a binary nominal datatype of 1 for survived and 0 for did not survive. All other variables are independent variables.
# 2.The PassengerID and Ticket variables are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.
# 3. The Pclass variable is an ordinal datatype for the ticket class, a proxy for socio-economic status (SES), with 1 = upper class, 2 = middle class, and 3 = lower class.
# 4. The Name variable is a nominal datatype. 
# It could be used in feature engineering to derive the SES from titles like doctor or master. 
# Since these variables already exist, we'll make use of it to see if title, like master, makes a difference.
# 5. The Sex and Embarked variables are a nominal datatype. They will be converted to dummy variables for mathematical calculations.
# 6. The Age and Fare variable are continuous quantitative datatypes.
# 7. The SibSp represents number of related siblings/spouse aboard and Parch represents number of related parents/children aboard. Both are discrete quantitative datatypes. This can be used for feature engineering to create a family size and is alone variable.
# 8. The Cabin variable is a nominal datatype that have no impact on the outcome variable.Thus, they will be excluded from analysis.

# In[ ]:



print('Train columns with null values:\n',data1.isnull().sum())
print('*'*20)
print('Test columns with null values:\n', data_val.isnull().sum())


# The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting
# In this stage, we will clean our data by 
# 1) correcting aberrant values and outliers, 
# 2) completing missing information, 
# 3) creating new features for analysis, 
# 4) converting fields to the correct format for calculations and presentation.
# 
# 1. Correcting: Reviewing the data, there does not appear to be any aberrant or non-acceptable data inputs. In addition, we see we may have potential outliers in age and fare. However, since they are reasonable values, we will wait until after we complete our exploratory analysis to determine if we should include or exclude from the dataset. It should be noted, that if they were unreasonable values, for example age = 800 instead of 80, then it's probably a safe decision to fix now. However, we want to use caution when we modify data from its original value, because it may be necessary to create an accurate model.
# 
# 2. Completing: There are null values or missing data in the age, cabin, and embarked field. Missing values can be bad, because some algorithms don't know how-to handle null values and will fail. While others, like decision trees, can handle null values. Thus, it's important to fix before we start modeling, because we will compare and contrast several models. There are two common methods, either delete the record or populate the missing value using a reasonable input. It is not recommended to delete the record, especially a large percentage of records, unless it truly represents an incomplete record. Instead, it's best to impute missing values. A basic methodology for qualitative data is impute using mode. A basic methodology for quantitative data is impute using mean, median, or mean + randomized standard deviation. An intermediate methodology is to use the basic methodology based on specific criteria; like the average age by class or embark port by fare and SES. There are more complex methodologies, however before deploying, it should be compared to the base model to determine if complexity truly adds value. For this dataset, age will be imputed with the median, the cabin attribute will be dropped, and embark will be imputed with mode. Subsequent model iterations may modify this decision to determine if it improves the model’s accuracy.
# 
# 3.Creating: Feature engineering is when we use existing features to create new features to determine if they provide new signals to predict our outcome. For this dataset, we will create a title feature to determine if it played a role in survival.
# 
# 4.Converting: Last, but certainly not least, we'll deal with formatting. There are no date or currency formats, but datatype formats. Our categorical data imported as objects, which makes it difficult for mathematical calculations. For this dataset, we will convert object datatypes to categorical dummy variables.
# 
# 

# In[ ]:


data1.describe(include='all')


# In[ ]:


#COMPLETE or delete missing values in train and test/validation dataset
for dataset in data_cleaner:
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace= True)
    #complete missing Embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the PassengerId, Cabin and Ticket feature to exclude in train dataset
drop_column = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace=True)
    
print('Train columns dropped')

    
    


# In[ ]:




print('Train columns with null values:\n',data1.isnull().sum())
print('*'*20)
print('Test columns with null values:\n', data_val.isnull().sum())


# In[ ]:


drop_column = ['Cabin', 'Ticket']
data_val.drop(drop_column, axis = 1, inplace=True)
print('Test columns with null values:\n', data_val.isnull().sum())


# In[ ]:


###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    #If other family members are present IsAlone will be 0
    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0
    #extracting title from data
    dataset['Title'] = dataset['Name'].str.split(', ', expand = True)[1].str.split(".", expand = True)[0]
     
    #With qcut, the bins will be chosen so that you have the same number of records in each bin 
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    
    #cut will choose the bins to be evenly spaced according to the values themselves and not the frequency of those values

    
    dataset['AgeBin'] = pd.cut(dataset['Age'], 5)
print('Feature Engineering done for FareBin and AgeBin')


# In[ ]:


stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*10)


#preview data again
data1.info()
data_val.info()
data1.sample(10)


# Convert Formats
# We will convert categorical data to dummy variables for mathematical analysis. There are multiple ways to encode categorical variables; we will use the sklearn and pandas functions.
# 
# In this step, we will also define our x (independent/features/explanatory/predictor/etc.) and y (dependent/target/outcome/response/etc.) variables for data modeling.

# In[ ]:


#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset
#code categorical data
label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
#define y variable aka target/outcome
Target = ['Survived']

#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')


#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')

data1_dummy.head()


# In[ ]:


print('Train columns with null values:\n',data1.isnull().sum())
print('*'*20)
print('Test columns with null values:\n', data_val.isnull().sum())


# In[ ]:


data_raw.describe(include = 'all')


# In[ ]:


train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()


# **Perform Exploratory Analysis with Statistics**
# Now that our data is cleaned, we will explore our data with descriptive and graphical statistics to describe and summarize our variables. In this stage, you will find yourself classifying features and determining their correlation with the target variable and each other.

# In[ ]:


for x in data1_x:
    if data1[x].dtype != 'float64':
        print('Survival correlation by ', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())

        
        


# In[ ]:


print(pd.crosstab(data1['Title'], data1[Target[0]]))


# In[ ]:


#graph distribution of quantitative data
plt.figure(figsize=[16, 12])
plt.subplot(231)
plt.boxplot(x = data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(x = data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (years)')

plt.subplot(233)
plt.boxplot(x = data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Familysize')
plt.ylabel('Familysize (Nos)')

plt.subplot(234)
plt.hist(x=[data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']],stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare histogram by survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], stacked = True,color= ['g','b'], label = ['Survived', 'Dead'])
plt.title('Age histogram by Survival')
plt.xlabel('Age (years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], stacked = True, color = ['r','g'], label = ['Survived', 'Dead'], bins = 10)
plt.title('Family size histogram by survival')
plt.xlabel('Family Size')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


fig, saxis = plt.subplots(2, 3, figsize = (16, 12))
sns.barplot(x= 'Embarked', y='Survived', data= data1, ax= saxis[0,0])
sns.barplot(x='Pclass', y='Survived', data=data1, ax=saxis[0,1])
sns.barplot(x='IsAlone', y='Survived', data= data1, ax= saxis[0,2])

sns.pointplot(x='FareBin', y='Survived', data=data1, ax= saxis[1,0])
sns.pointplot(x='AgeBin', y='Survived', data=data1, ax= saxis[1,1])
sns.pointplot(x='FamilySize', y='Survived', data=data1, ax= saxis[1,2])





# In[ ]:


#we know sex mattered in survival, now let's compare sex and Embarked

fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax = qaxis[1])
sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax = qaxis[2])
            



# In[ ]:


fig, qaxis = plt.subplots(1, 3, figsize = (14,12))
sns.distplot(data1["Age"],kde=True, ax =qaxis[0]) #without the kde
sns.distplot(data1['Fare'], kde = True, ax=qaxis[1])
sns.distplot(data1['FamilySize'], kde = True, ax=qaxis[2])


# In[ ]:


def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(16, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(df.corr(), annot= True, cmap=colormap)
    
        
correlation_heatmap(data1)


# **Model Data**
# Data Science is a multi-disciplinary field between mathematics (i.e. statistics, linear algebra, etc.), computer science (i.e. programming languages, computer systems, etc.) and business management (i.e. communication, subject-matter knowledge, etc.). Most data scientist come from one of the three fields, so they tend to lean towards that discipline. However, data science is like a three-legged stool, with no one leg being more important than the other. So, this step will require advanced knowledge in mathematics. But don’t worry, we only need a high-level overview, which we’ll cover in this Kernel. Also, thanks to computer science, a lot of the heavy lifting is done for you. So, problems that once required graduate degrees in mathematics or statistics, now only take a few lines of code. Last, we’ll need some business acumen to think through the problem. After all, like training a sight-seeing dog, it’s learning from us and not the other way around.
# 
# Machine Learning (ML), as the name suggest, is teaching the machine how-to think and not what to think. While this topic and big data has been around for decades, it is becoming more popular than ever because the barrier to entry is lower, for businesses and professionals alike. This is both good and bad. It’s good because these algorithms are now accessible to more people that can solve more problems in the real-world. It’s bad because a lower barrier to entry means, more people will not know the tools they are using and can come to incorrect conclusions. That’s why I focus on teaching you, not just what to do, but why you’re doing it. Previously, I used the analogy of asking someone to hand you a Philip screwdriver, and they hand you a flathead screwdriver or worst a hammer. At best, it shows a complete lack of understanding. At worst, it makes completing the project impossible; or even worst, implements incorrect actionable intelligence. So now that I’ve hammered (no pun intended) my point, I’ll show you what to do and most importantly, WHY you do it.
# 
# First, you must understand, that the purpose of machine learning is to solve human problems. Machine learning can be categorized as: supervised learning, unsupervised learning, and reinforced learning. Supervised learning is where you train the model by presenting it a training dataset that includes the correct answer. Unsupervised learning is where you train the model using a training dataset that does not include the correct answer. And reinforced learning is a hybrid of the previous two, where the model is not given the correct answer immediately, but later after a sequence of events to reinforce learning. We are doing supervised machine learning, because we are training our algorithm by presenting it with a set of features and their corresponding target. We then hope to present it a new subset from the same dataset and have similar results in prediction accuracy.
# 
# There are many machine learning algorithms, however they can be reduced to four categories: classification, regression, clustering, or dimensionality reduction, depending on your target variable and data modeling goals. We'll save clustering and dimension reduction for another day, and focus on classification and regression. We can generalize that a continuous target variable requires a regression algorithm and a discrete target variable requires a classification algorithm. One side note, logistic regression, while it has regression in the name, is really a classification algorithm. Since our problem is predicting if a passenger survived or did not survive, this is a discrete target variable. We will use a classification algorithm from the sklearn library to begin our analysis. We will use cross validation and scoring metrics, discussed in later sections, to rank and compare our algorithms’ performance.
# 
# Machine Learning Selection:
# 
# Sklearn Estimator Overview
# Sklearn Estimator Detail
# Choosing Estimator Mind Map
# Choosing Estimator Cheat Sheet
# Now that we identified our solution as a supervised learning classification algorithm. We can narrow our list of choices.
# 
# Machine Learning Classification Algorithms:
# 
# Ensemble Methods
# Generalized Linear Models (GLM)
# Naive Bayes
# Nearest Neighbors
# Support Vector Machines (SVM)
# Decision Trees
# Discriminant Analysis
# Data Science 101: How to Choose a Machine Learning Algorithm (MLA)
# IMPORTANT: When it comes to data modeling, the beginner’s question is always, "what is the best machine learning algorithm?" To this the beginner must learn, the No Free Lunch Theorem (NFLT) of Machine Learning. In short, NFLT states, there is no super algorithm, that works best in all situations, for all datasets. So the best approach is to try multiple MLAs, tune them, and compare them for your specific scenario. With that being said, some good research has been done to compare algorithms, such as Caruana & Niculescu-Mizil 2006 watch video lecture here of MLA comparisons, Ogutu et al. 2011 done by the NIH for genomic selection, Fernandez-Delgado et al. 2014 comparing 179 classifiers from 17 families, Thoma 2016 sklearn comparison, and there is also a school of thought that says, more data beats a better algorithm.
# 
# So with all this information, where is a beginner to start? I recommend starting with Trees, Bagging, Random Forests, and Boosting. They are basically different implementations of a decision tree, which is the easiest concept to learn and understand. They are also easier to tune, discussed in the next section, than something like SVC. Below, I'll give an overview of how-to run and compare several MLAs, but the rest of this Kernel will focus on learning data modeling via decision trees and its derivatives.
# 

# In[ ]:


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest neighbour
    neighbors.KNeighborsClassifier(),
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability= True),
    svm.LinearSVC(),
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    XGBClassifier()

]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data1[Target]

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
#MLA_predict


# In[ ]:


sns.barplot(x= 'MLA Test Accuracy Mean', y='MLA Name', data = MLA_compare, color='m')
plt.title('MLA Accuracy score')
plt.xlabel('Accuracy score %')
plt.ylabel('Algorithm name')


# In[ ]:



clf = svm.SVC()
clf.fit(data1[data1_x_bin], data1[Target])
pred= clf.predict(data_val[data1_x_bin])
submission = pd.DataFrame({"PassengerId": data_val['PassengerId'], "Survived": pred})
#data_val.columns
submission.to_csv("../working/submission.csv", index=False)

submission.sample(10)

