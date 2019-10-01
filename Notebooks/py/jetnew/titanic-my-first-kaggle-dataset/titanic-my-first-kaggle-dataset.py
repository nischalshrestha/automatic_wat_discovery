#!/usr/bin/env python
# coding: utf-8

# ## To the Reader (You!) [Update: 3/9/2018]
# If you're a beginner in data science/Kaggle/machine learning, I recommend viewing everyone's Titanic notebooks to learn and practise unfamiliar libraries and techniques. If you're an intermediate or an expert, I hope that you can share your valuable experience with me by commenting and giving feedback on areas of improvement to this notebook. All feedback is welcome! :)
# 
# ## Objective
# My objective of doing this competition is to apply everything I've learnt from ML on Coursera in practice. I want to learn Python libraries for numerical analysis like numpy and pandas, libraries for data visualisation like seaborn and matplotlib, and libraries for machine learning like tensorflow, keras and scikit-learn.
# 
# ## My Background
# Prior to this competition, I have completed Machine Learning by Stanford University on Coursera. I only know Matlab implementations of linear/logistic regression, neural networks, support vector machines (SVMs), K-means, principal component analysis (PCA), anomaly detection and collaborative filtering. I also know some machine learning concepts like bias/variance trade-offs, precision/recall and mean normalization/feature scaling.
# See what I learnt here: https://github.com/jetnew/My-Work-in-Machine-Learning/blob/master/ML%20by%20Andrew%20Ng/README.md
# 
# ## Plan (What I Learnt)
# Our plan for the Titanic problem will be as follows:
# 
# 1. Import preprocessing libraries and datasets
# 2. Data visualisation to find correlations and insights
# 3. Data cleaning: Correcting, Completing, Creating, Converting (4C's)
# 4. Choose machine learning model
# 5. Test against cross-validation set and analyse performance metrics (F1 Score)
# 6. Wrap up and submit results

# ## (1/6) Import Preprocessing Libraries and Datasets

# In[ ]:


# Libraries for numerical analysis
import numpy as np # linear algebra and matrix operations
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for data mungling
from sklearn import preprocessing # preprocessing.scale() performs mean normalization and feature scaling across the dataset

# Common Machine Learning Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.impute import SimpleImputer # SimpleImputer fills empty cells with the mean
from sklearn.model_selection import cross_validate # To train with cross validation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Others
import os
import warnings
warnings.filterwarnings('ignore') # Had a lot of future warning messages

# Import datasets
df_train = pd.read_csv('../input/train.csv') # Training set (The data we are given)
df_test = pd.read_csv('../input/test.csv') # Test set (The data we need to test for results)


# ## Visualising Datasets at a Glimpse
# 
# It is important to take a look at the dataset to understand how you can clean the data.
# 
# Note the characteristics of our data: Categorical/Numerical data, Uniqueness/Correlation to Survivability, Range of values across dataset, etc.

# In[ ]:


# Visualise numerical features
display(df_train.describe())
# Visualise categorical features
display(df_train.describe(include=['O']))


# ## (2/6) Data Visualisation and Understanding of Dataset
# 
# Let us make some sense of the correlation between features of the dataset. Take note of how each feature correlates with Survived.
# 
# We note that the features Pclass and Fare have relatively high correlation coefficients (>0.20) as compared to other features. Whilst they are considered having weak linear correlation (<0.50), there is sufficiently significant correlation.
# 
# However, we also note that categorical features, Name, Sex, Ticket, Cabin and Embarked and their correlation to Survived cannot be determined. Hence, there is a need to convert them to numerical values.

# In[ ]:


sns.heatmap(df_train.corr(method='pearson'),annot=True,cmap="YlGnBu")


# ### Feature: Pclass
# 
# Correlation coefficient against Survived: -0.338
# 
# Pclass is negatively correlated to Survived. We decide to include this feature in our model.
# 
# Observation: We notice a downward trend of the survival rate as Pclass increases from 1 to 3, from 0.63 to 0.24.

# In[ ]:


# Graph individual features by survival
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=df_train)


# ### Features: SibSp and Parch
# 
# These features have 0 survival rates for certain values. It may be best to derive a feature or a set of features from these individual features.

# In[ ]:


fig, ax = plt.subplots(1,2)
sns.barplot(x = 'SibSp', y = 'Survived', order=[1,2,3,4,5,6,7], data=df_train, ax=ax[0])
sns.barplot(x = 'Parch', y = 'Survived', order=[1,2,3,4,5,6], data=df_train, ax=ax[1])


# ## (3/6) Data Cleaning
# 
# Data cleaning is greatly beneficial and necessary. The 4 C's of data cleaning are: Correcting, Completing, Creating and Converting.
# 
# 1. **Correcting:** Non-acceptable data inputs and outliers create unnecessary noise.
# 2. **Completing**: Null values or missing data will make some algorithms fail. Either delete the record or impute missing values using mode (qualitative data) or mean, median, mean + randomized standard deviation (quantitative data).
# 3. **Creating**: Use feature engineering to create new features to improve the prediction.
# 4. **Converting**: Categorical data are imported as objects which are difficult for mathematical calculations. Converting to categorical dummy variables will help.

# In[ ]:


y = df_train['Survived']
test_index = df_test['PassengerId']
# For simultaneous data cleaning
combine = [df_train, df_test]


# ## (3/6) Data Cleaning: Extraction of Meaningful Data from Meaningless Data
# **Feature: Name -> Title**
# 
# For the feature Name, we want to extract the passengers' title (Mr, Miss, etc) to use as a feature. We also replace less occuring titles with the label Rare.

# In[ ]:


# For feature: Name
# Convert: Name -> Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # regex

# Convert: rare titles -> Rare, Mlle -> Miss, Ms -> Miss, Mme -> Mrs
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Display the available types in Title feature.
sns.countplot(x = 'Title', order=df_train['Title'].unique(), data=df_train)


# ## (3/6) Data Cleaning: Conversion of Categorical Data to Numerical Data
# 
# ### Features:
# 
# Sex: female -> 1, male -> 0
# 
# Embarked: S -> 0, C -> 1, Q -> 2
# 
# Title: Mr -> 1, Miss -> 2, Mrs -> 3, Master -> 4, Rare -> 5
# 
# We need to convert categorical features to numerical values because our model (MLP Classifier) requires numerical data.

# In[ ]:


# For feature: Sex
# Convert: female -> 1, male -> 0
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# For feature: Embarked
# Fill in missing values with highest frequency port
freq_port = df_train.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
# Convert: S -> 0, C -> 1, Q -> 2
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# For feature: Title
# Convert: Mr -> 1, Miss -> 2, Mrs -> 3, Master -> 4, Rare -> 5
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Display the values in each feature.
print("Values in Sex:", df_train['Sex'].unique())
print("Values in Embarked:", df_train['Embarked'].unique())
print("Values in Title:", sorted(df_train['Title'].unique()))


# **Correlation of Categorical Features with Survived**
# 
# Now, we can identify the categorical features Title, Sex and Embarked and their correlation coefficient against Survived.
# 
# **Correlation Coefficient against Survived**
# * Title: 0.408
# * Sex: 0.543
# * Embarked: 0.107

# In[ ]:


sns.barplot(x=df_train.corr(method='pearson')[['Survived']].index, y=df_train.corr(method='pearson')['Survived'])


# **Feature: Sex**
# 
# Correlation coefficient against Survived: 0.543
# 
# Sex = female has very high survival rate at 74%.

# In[ ]:


df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x = 'Sex', y = 'Survived', order=[1,0], data=df_train)


# **Feature: Fare**
# 
# Correlation coefficient against Survived: 0.257
# 
# Fare has a weak but significant positive correlation with Survived. We can choose to include this feature in our model.
# 
# We observe that although Fare has a sufficiently significant linear correlation against Survived (>0.20), there is too much noise in the data (too many unique values). Hence, we decide to bin Fare into 4 subcategories.

# In[ ]:


# Fill in missing values with median
df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)

# Split Fare into 4 bands in FareBand to find out where the bins are
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


# Convert Ranges of fare prices into bins
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

sns.barplot(x = 'Fare', y = 'Survived',order=[0,1,2,3],data=df_train)


# In[ ]:


# We converted Fare into 4 bins, so FareBand is no longer useful to us, hence we drop it
df_train = df_train.drop(['FareBand'], axis=1)
combine = [df_train, df_test]


# We have improved Fare's correlation with Survived from 0.257 to 0.300.

# In[ ]:


sns.barplot(x=df_train.corr(method='pearson')[['Survived']].index, y=df_train.corr(method='pearson')['Survived'])


# ## (3/6) Data Cleaning: Creating New Features from Current Data
# 
# We also note that features with particularly low correlation coefficients against Survived (Age, SibSp, Parch) needs to be further cleaned to make sense of the data. We decide to do this by creating new features.
# 
# **Correlation Coefficients Against Survived**
# * Age: -0.077
# * SibSp: -0.035
# * Parch: 0.082

# **Imputation (filling of empty cells) of the Dataset**
# 
# Imputation of the dataset can be done by filling empty cells with each feature data's:
# * Mean value
# * Median value
# * Mode value
# * Median - Standard Deviation
# 
# We can also drop rows of data that have missing values but because there are many missing cells in Age, we cannot afford to drop 200+ rows out of 800+ in the entire dataset. We decide to use median to fill missing Age cells.

# In[ ]:


for dataset in combine:
    dataset.loc[(dataset.Age.isnull()), 'Age'] = dataset.Age.median()


# We bin Age into AgeBand so as to reduce noise.

# In[ ]:


df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    
sns.barplot(x = 'Age', y = 'Survived',order=[0,1,2,3,4],data=df_train)


# In[ ]:


df_train = df_train.drop(['AgeBand'], axis=1)
combine = [df_train, df_test]
df_train.head()


# We create a new feature FamilySize that combines SibSp and Parch.

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

sns.barplot(x = 'FamilySize', y = 'Survived', order=[1,2,3,4,5,6,7], data=df_train)


# We create a new feature IsAlone for passengers with FamilySize of 1.

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

sns.barplot(x = 'IsAlone', y = 'Survived',order=[0,1],data=df_train)


# We drop parch, sibsp and family size as correlation coefficient is lower than IsAlone and will create noise.

# In[ ]:


display(df_train.corr(method='pearson')[['Survived']])
sns.barplot(x=df_train.corr(method='pearson')[['Survived']].index, y=df_train.corr(method='pearson')['Survived'])

df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [df_train, df_test]

display(df_train.head())


# We create a new feature Age * Class to increase the number of features.

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

df_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(5)


# We leave out Name, Ticket and Cabin because the syntax brings about no direct insight to the survival rate.

# In[ ]:


df_train = df_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]
display(combine[0].head())
display(combine[1].head())


# ### Note: Mean Normalization and Feature Scaling of Data
# 
# Our dataset's values now do not have vastly different ranges of values as they range only from 1 to 6. These small difference in ranges will not affect much our model's loss convergence to the minimum.
# 
# If our dataset has big differences in ranges of values, we will need to use mean normalization and feature scaling on the training data.
# Mean normalization and feature scaling: X = (X - mean(X)) / std(X).
# Mean normalization adjusts the mean of the values to 0 while feature scaling scales the range of values to -1 to 1.

# ### Note: Importance of a Cross-Validation Set
# 
# A cross-validation set is important to ensure that our model does not train to overfit the training and test data. Because we experiment with different hyperparameters of the model (learning rate, iterations, etc), there is a possibility of the model overfitting to the test data as well. Therefore, we use the training set to train the model, evaluate the model using the cross-validation set, then experiment with hyperparameters before evaluating the model on the test set.
# 
# It is important to distinguish between the cross validation set derived from the training data (20% of train.csv) and the actual test data (test.csv) we are supposed to predict against.
# 
# We split our training data into 2 portions: 80% training set and 20% cross-validation set.

# In[ ]:


# df_train_X is a list of features used for model training.
df_train_X = combine[0].drop(["Survived", "PassengerId"], axis=1)
# train_y is the training output.
train_y = combine[0]["Survived"]
test_test_X = combine[1].drop("PassengerId", axis=1).copy()


# ## (4/6) Choosing a Machine Learning Algorithm
# 
# We take a list of ML algorithms and train them and compare the cross-validation test accuracies.
# 
# Referred from: https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook

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
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = {}

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, df_train_X, train_y, cv  = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    # Fit, predict test input and evaluate using F1 score.
    alg.fit(df_train_X, train_y)
    MLA_compare.loc[row_index, 'F1 Score'] = metrics.f1_score(train_y, alg.predict(df_train_X))
    MLA_predict[MLA_name] = alg.predict(test_test_X)
    
    row_index+=1
    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['F1 Score'], ascending = False, inplace = True)


# ## (5/6) Test Against Cross-Validation Set and Test Set and Analyse Performance Metrics (F1 Score)
# The performance metric we are currently using is accuracy. Accuracy is defined as (no. of correct predictions)/(no. of predictions).
# However, this is not a robust indicator of the model's performance. For a binary classification problem which has 2 classes, 0 or 1, a model that predicts all cases as 1 will easily achieve 50% accuracy.
# 
# In other problems such as identification of cancer cells, there may only be 1% of cases with cancer. A model that assumes all cases as non-cancer will easily achieve 99% accuracy. Hence, we need a more robust metric, such as the F1 score.
# 
# We want to measure the precision and recall, and the corresponding F1 score.
# * Precision = Of all that we predict are true, how many are actually true? (no. of true positives)/(no. of true positives + no. of false positives)
# * Recall = Of all that are really true, how many did we predict are true? (no. of true positives)/(no. true positives + no. of false negatives)
# 
# Let's now see how well our model fits the cross validation set.

# In[ ]:


#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html
sns.barplot(x='F1 Score', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm F1 Score \n')
plt.xlabel('F1 Score (%)')
plt.ylabel('Algorithm')


# ## (6/6) Wrap Up and Submit Results
# Let's now see how well our model fits the test set.
# 
# We predict the results of the test set and prepare it for submission

# In[ ]:


best_model = MLA_compare.loc[MLA_compare['F1 Score'].idxmax()]['MLA Name']
best_model_score = round(MLA_compare.loc[MLA_compare['F1 Score'].idxmax()]['F1 Score'],3)
print("Best model:",best_model)
print("F1 Score:",best_model_score)


# In[ ]:


# predict against test set
predictions = MLA_predict[best_model]
predictions = predictions.ravel() # To reduce ND-array to 1D-array
data_to_submit = pd.DataFrame({
    'PassengerId': test_index,
    'Survived': predictions
})
# output results to results.csv
data_to_submit.to_csv("results.csv", index=False)


# ## Accuracy based on Kaggle's leaderboard: 0.78
