#!/usr/bin/env python
# coding: utf-8

# #Titanic Survival Prediction
# **Introduction**
# 
# RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean on 15th April, 1912. In this accident, more than 1500 out of the estimated total 2224 passengers and crew died, making it one of the most severe peacetime maritime disasters in modern history. This story was also filmed by director Steven Spielberg in to the movie Titanic, which was considered as one of the best movie in the 20th centery. As one of the major reasons of a such loss of life was the lack of the lifeboats, the survival chance was not determined randomly, but mostly the opportunity to acquire a spot on the lifeboats, which is influenced by several parameters. In this study, we estimate how do parameters such as sex, age, social class influence the survival chance.
# 
# **Table of Contents:**
# 
# 1. Import libraries
# 2. Import data
# 3. Data cleaning, feature engineeirng, data visualization
# 4. Establishing the model
# 5. Model Implementation
#    * Logistic regression
#    * linear supporter vector classifier
#    * PerceptronNaive Bayes
#    * Support vector classifier
#    * K-nearest neighbors
#    * Decision tree
#    * Random forest
#    * Gradient boost classifier
# 6. Model ensembling
# 7. Prediction and submit the data
# 
# **1. Import libraries**

# In[ ]:


import sys #access to sysem parameters
import pandas as pd #collection of functions for data processing and alaysis 
pd.set_option('display.max_columns',None)
import numpy as np #foundation package for scientific computing
np.set_printoptions(threshold=np.inf)
import sklearn #collection of machine learning algorithms
import matplotlib.pyplot as plt #collection of functions for visualization
import scipy as sp #collection of functions for scientific computing and advance mathematics

#import IPython
#from Iphython import display # pretty printing of dataframes in Jupyter notebook
#misc libraries
import random
import time
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

###load data modelling libraries
#Common model algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#from xgboost import XGBClassifier
#Common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

###Visualization
import matplotlib as mpl
#import matplotlib.pylot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
#Configure visualization defaults
#%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize']=12,8


# **2. Import data**
# 
# Here, we can see that the data contains features including:
# 
# * PassingerID: which is used to index the data and do not have obvious relationship with other features, data is complete
# * Survived: the classification in this project, data is complete for training data
# * Pclass: ticket class, which is largely determined by the social class, data is complete
# * Sex and Age: which can potentially influence the survival rate as women and kids may have higher priority to get help in ethics. Sex feature is complete while the age feature lacks 177 values in the training data and 86 in the test data.
# * Sibsp and Parch: # of siblins/sponses and parents/children aboard the Titanic, influence to the survival unknown, data is complete
# * Ticket: relationship unkown, data is complete
# * Fare: fare should be determined by the pclass, cabin, and embarked, data is complete in training data while have 1 missing value in the test data
# * Cabin: may provide insight together with pclass, fare, embarked, however, this features have 687 missing values in the training data, so we will not use it directly in the model
# * Embarked: have 2 missing values in the training data, will be filled in later

# In[ ]:


data_raw = pd.read_csv('../input/train.csv')
#891 passengers in the training set
data_val = pd.read_csv('../input/test.csv')
#418 passengers in the test set

#print (train.info())
#create a copy for the train data
#data1 = data_raw.copy(deep = True)
combine = [data_raw, data_val]

data_raw.describe()


# In[ ]:


# Check for the missing values
print('Train columns with null values:\n', data_raw.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)


# **3. Data cleaning, feature engineering, data visualization**
# 
# In this section we are going to do some exploratory data analysis with descriptive and graphical methods. The features of interest fall into the following categories:
# 
# 1. Pclass, Fare, Embarked, Cabin, Ticket: which can be used to fill in missing embarked and fare values
# 2. Sex, Name (with Title), Age, SibSp, Parch: which can be used to fill in missing age values
# 
# **Category 1: Pclass, Fare, Embarked, Cabin, Ticket**
# 
# Missing values: 2 Embarked values, 1 Fare value, 1014 Cabin values
# 

# In[ ]:


###Visualize embarkment, Pclass, fare
plt.figure(figsize=[15,7])

plt.subplot(121)
sns.boxplot(x = 'Embarked', y = 'Fare', data = data_raw)
#Passengers embarked at C tend to pay higher fare
plt.subplot(122)
sns.boxplot(x = 'Pclass', y = 'Fare', data = data_raw)
#Passengers from higher class tend to pay higher fare


# We noticed that several cabin features are missing, we assume that one important reason for the value missing is that they did not survived so there is no record. To prove our hypothesis, we establish a new feature to indicate if the cabin feature is missing and to study their relationship with the survival rate.

# In[ ]:


for dataset in combine:
    dataset['Cabin_null'] =data_raw['Cabin'].isnull()*1
age_survive = sns.barplot(x='Cabin_null', y='Survived', data=data_raw)
age_survive = plt.xlabel('Cabin is Missing')
age_survive = plt.ylabel('Survived')
plt.figure(figsize=[7,7])
plt.show()


# The 2 missing Embarked values are filled with the most frequent value for the Embarked feature. 
# 
# Since Fare feature is very related to the pclass feature, the 1 missing Fare value is filled in with the mean Fare value for the pclass the passenger belongs to.

# In[ ]:


###Complete the missing values
# The missing Embarked data is filed in with the most frequent value
for dataset in combine:
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace = True) # The missing embarked data is filled in with mode
    #dataset['Fare'].fillna(dataset['Fare'].median(),inplace = True) # The missing fare data is filled in with median 
    
# The missing Fare data is filled in with the mean fare value for the pclass the passenger belongs to 
for x in range(len(data_val["Fare"])):
    if pd.isnull(data_val["Fare"][x]):
        pclass = data_val["Pclass"][x] #Pclass = 3
        data_val["Fare"][x] = round(data_raw[data_raw["Pclass"] == pclass]["Fare"].mean(), 4)


# A new feature FareBin is created.
# 
# The first letter of the Cabin is extracted and stored as the Cabin feature.

# In[ ]:


###Feature engineering
#Create the FareBin feature
#Keep the first letter for Cabin data
Fare_Bins=[-1,7.91,14.454,31,10000]
Fare_Labels=['cheap','medium','medium high','high']

for dataset in combine:
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
    dataset['FareBin']=pd.cut(dataset['Fare'], Fare_Bins, labels=Fare_Labels)

data_raw.head()


# The figures showing survival rate ploted as functions of Pclass, Embarked, Cabin, FareBin are shown as below.

# In[ ]:


###Data visualization

plt.figure(figsize=[12,10])

plt.subplot(221)
sns.barplot(x = 'Pclass', y = 'Survived', data = data_raw)
#People with higher socieconomic class had a higher rate of survival

plt.subplot(222)
sns.barplot(x = 'Embarked', y = 'Survived', data = data_raw)
#People embarked at C are more likely to survive

plt.subplot(223)
sns.barplot(x='Cabin',y='Survived',data = data_raw)
#People with a recorded Cabin number are more likely to survive

plt.subplot(224)
sns.barplot(x='FareBin',y='Survived',data = data_raw)
#People who pay a higher fare are more likely to survive


# The FareBin feature is converted into numbers.

# In[ ]:


###Convert the features into numbers
#cleanup_Embarked = {'S':0,'C':1,'Q':2}
cleanup_FareBin ={'cheap':0,'medium':1,'medium high':2,'high':3}
for dataset in combine:
    #dataset['Embarked']=dataset['Embarked'].map(cleanup_Embarked).astype(int)
    dataset['FareBin']=dataset['FareBin'].map(cleanup_FareBin).astype(int)

data_raw.head()


# The prefix part of the ticket feature is extracted and stored as the ticket feature.

# In[ ]:


# For ticket feature we only keep the prefix part
for dataset in combine:
    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit() :
            Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
        else:
            Ticket.append("X")
    dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# **Category 2: Sex, Age, SibSp, Parch**
# 
# Missing values: 177 Age values
# 
# **Feature: Age**
# 
# Firstly, let us have a look at the relationship between the age and the survival rate. It is interesting to discovered that babies of age 0-5 have a higher chance to survive, while at the same time, elderly passengers who are order than 65 have the survival rate as low as only ~10%.

# In[ ]:


age_bins=[0,5,12,18,50,65,120]
age_labels=['baby','kid','teenager','adult','aging','elderly']
for dataset in combine:
    dataset['AgeGroup']=pd.cut(dataset['Age'], age_bins, labels=age_labels)
age_survive = sns.barplot(x='AgeGroup', y='Survived', data=data_raw)
age_survive = plt.xlabel('Age')
age_survive = plt.ylabel('Survived')
plt.figure(figsize=[7,5])
plt.show()


# We noticed that several age features are missing, we assume that one important reason for the value missing is that they did not survived so there is no record. To prove our hypothesis, we establish a new feature to indicate if the age feature is missing and to study their relationship with the survival rate.

# In[ ]:


for dataset in combine:
    dataset['Age_null'] =data_raw['Age'].isnull()*1
age_survive = sns.barplot(x='Age_null', y='Survived', data=data_raw)
age_survive = plt.xlabel('Age is Missing')
age_survive = plt.ylabel('Survived')
plt.show()


# We noticed that in the name, there is title information. As different titles usually refer to different age ranges, the title information can help us to predict the age features. So we extract this informatino from the name and save it in a new feature named 'title'.

# In[ ]:


# Complete missing age Values. we noticed that in the name, there is also title information, this can be helpful to predict the age
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(data_raw['Title'], data_raw['Sex'])


# In[ ]:


# Group different title (especially rare titles) into common groups which are more closely related to the age
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Sir', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Rev', 'Countess', 'Lady', 'Dona'], 'Rare')
#     dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Dona'], 'Rare_F')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
pd.crosstab(data_raw['Title'], data_raw['Sex'])


# According to our common sense, the title should be related to the age, e.g. title Miss should be younger than Mrs. To identify if the title is related to the title, we visualize their relationship.

# In[ ]:


# According to our common sense, the title should be related to the age, e.g. title Miss should be younger than Mrs. To identify if the title is related to the title, visualize their relationship
plt.figure(figsize=[15,7])
plt.subplot(121)
age_title = sns.boxplot(x="Title", y="Age",hue="Survived", data=data_raw)
age_title = plt.xlabel('Title')
age_title = plt.ylabel('Age')
# We assume the age is also related to the pclass so we plot the corresponding diagram to confirm
plt.subplot(122)
age_title = sns.boxplot(x="Pclass", y="Age",hue="Survived", data=data_raw)
age_title = plt.xlabel('PClass')
age_title = plt.ylabel('Age')
plt.show()


# According to the above analysis, the age value is related to both the title and the pclass, as we hypothesized initially. It is also interesting to discovered that in all pclass group, survived people are of younger median age than dead ones. So we assign the median of each [title, pclass] group if the age value is missed.

# In[ ]:


# assign the median of each [title, pclass] group if the age value is missed
def impute_age(dataset):
   for pclass in [1,2,3]:
        for title in ['Master','Miss','Mr','Mrs','Rare']:
            ds=dataset[dataset['Pclass']==pclass]
            ds=ds[ds['Title']==title]
            median=ds['Age'].median()
            dataset.loc[
                (dataset['Age'].isnull())&
                (dataset['Pclass']==pclass)&
                (dataset['Title']==title),
                'Age'
            ]=median
impute_age(data_raw)
impute_age(data_val)


# **Feature SibSp and Parch**
# 
# Both these features represent the number of relatives on the Titanic, so we create a new feature to represent the size of the family for each passengers.

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch'] + 1

# to study the influence of the family size to the survival chance and the distributino of different family size, 
# we use the bubble plot to visualize it. The size of each bubble is count of each family size.
fs_count = data_raw['FamilySize'].value_counts()
fs_prob_survived = data_raw['FamilySize'][data_raw['Survived'] == 1].value_counts()
fs_prob = fs_prob_survived/fs_count
familysize = sns.relplot(fs_prob.index, fs_prob.values, size = fs_count.values, sizes = (100,1000), data=data_raw)
plt.xlabel('FamilySize')
plt.ylabel('Prob of Survived')
plt.show()


# Here we can learn that if you have a family size of 2-4, you will in geneneral have a higher chance to survive. However, it is noted that as passengers with family size 4 or higher are limisted, there might be bias here. 
# 

# **Feature Sex**
# 
# As can be seen below, over 70% of female passengers are survived while only ~20% male are survived. Thank you gentlemen!

# In[ ]:


sns.set_style("white")
sex_sur = sns.barplot(x="Sex", y="Survived", data=data_raw)
sex_sur = plt.xlabel('Sex')
sex_sur = plt.ylabel('Survived')
plt.show()


# In[ ]:


#for dataset in combine:
 #   dataset["Age"] = dataset["Age"].fillna(-0.5)

age_bins=[0,5,12,18,50,65,120]
age_labels=['baby','kid','teenager','adult','aging','elderly']
for dataset in combine:
    dataset['AgeGroup']=pd.cut(dataset['Age'], age_bins, labels=age_labels)

cleanup_agegroup = {'baby':1,'kid':2,'teenager':3,'adult':4,'aging':5,'elderly':6}
cleanup_sex = {'male':0,'female':1}
for dataset in combine:
    dataset['AgeGroup']=dataset['AgeGroup'].map(cleanup_agegroup).astype(int)
    dataset['Sex']=dataset['Sex'].map(cleanup_sex).astype(int)
data_raw.head()


# **4. Establishing the model**
# 
# To establish our model, we drop features including name, passenger ID  as they can hardly provide new insights (we already took out the title features)
# We establish two types of model dealing with the age and fare information: in type 1, we keep the numerical age and fare and drop the agegroup and farebin features. in type 2, we instead use the agegroup and farebin to represent the age and fare.
# as we use agegroup instead, sibsp and parch as we used family size instead, and Fare as we think it is determined by the pclass and embarked features. 

# In[ ]:


# drop_column = ['Name','Age','PassengerId','Fare','Cabin']
# data_raw.drop(drop_column, axis=1, inplace=True)
# data_val.drop(drop_column, axis=1, inplace=True)
# data_val.head()


# In[ ]:


# First we combine the training data and test dat in order to get the same number of dummy features when using one hot encoded
# to deal with the training data and test data together, we firstly separate features and labels in training data into data_raw and Y_train 
Y_train = data_raw["Survived"]
drop_column = ['Survived']
data_raw.drop(drop_column, axis=1, inplace=True)

data_raw_len = len(data_raw)
dataset_comb =  pd.concat(objs=[data_raw, data_val], axis=0).reset_index(drop=True)


# In[ ]:


# in feature group1, we drop ['Name','AgeGroup','PassengerId','FareBin','Ticket','Cabin'] to use numerical age and fare for the model
# in feature group2, instead we drop ['Name','Age','PassengerId','Fare','FamilySize','Cabin']

Feature_Group1 = ['Pclass','Sex','Age','Age_null', 'FamilySize','SibSp', 'Parch','Fare','Cabin_null','Embarked', 'Title', 'IsAlone']
Feature_Group2 = ['Pclass','Sex', 'AgeGroup','Age_null','SibSp', 'Parch', 'Ticket','FareBin','Cabin_null','Cabin','Embarked', 'Title', 'IsAlone']

dataset1 = dataset_comb[Feature_Group1]
dataset2 = dataset_comb[Feature_Group2]
dataset1.head()


# In[ ]:


# As for title and embarked features, 
# each different types of values should be of the same weight, 
# we encode them by one hot encoder to transfer each possible value into a boolean type
from sklearn.preprocessing import OneHotEncoder
one_hot_encoded_dataset1 = pd.get_dummies(dataset1)
one_hot_encoded_dataset2 = pd.get_dummies(dataset2)
one_hot_encoded_dataset1.head()


# We standardardize the age and fare features in type 1 model

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
# scaler_age = MinMaxScaler()
# scaler_fare = MinMaxScaler()

# # print([dataset1['Age']].size)
# scaled_age = scaler_age.fit_transform(dataset1[['Age']])
# scaled_fare = scaler_fare.fit_transform(dataset1[['Fare']])
# dataset1['Age'] = scaled_age
# dataset1['Fare'] = scaled_fare

one_hot_encoded_dataset1[['Age', 'Fare']] = MinMaxScaler().fit_transform(dataset1[['Age', 'Fare']])
#one_hot_encoded_dataset1[['Age']] = MinMaxScaler().fit_transform(dataset1[['Age']])
#one_hot_encoded_dataset1.head()
# dataset2['Fare'] = preprocessing.normalize(dataset1['Fare'])

# preprocessing.normalize(dataset1['Age'])
# dataset2['Fare'] = preprocessing.normalize(dataset1['Fare'])


# In[ ]:


#Y_train = one_hot_encoded_data_raw["Survived"]
#drop_column = ['Survived']
#one_hot_encoded_data_raw.drop(drop_column, axis=1, inplace=True)

X_train = one_hot_encoded_dataset1[:data_raw_len]
X_test = one_hot_encoded_dataset1[data_raw_len:]

X_train.head()


# To establish our model, we split the training data into training set and development set with the ratio 75:25

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_dev, y_train, y_dev = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 1)


# **5. Model implementation**
# 
# We compare the result of following result:
# 
# * Logistic regression
# * linear supporter vector classifier
# * PerceptronNaive Bayes
# * Support vector classifier
# * K-nearest neighbors
# * Decision tree
# * Random forest
# * Gradient boost classifier

# In[ ]:


## Logistic classification
from sklearn import linear_model
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty = 'l2', C = 0.2,random_state = 0)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_dev)
y_train_pred = logreg.predict(x_train)
acc_log = round(accuracy_score(y_pred, y_dev) * 100, 2)
acc_log_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_log, "the training error is", acc_log_train)


# In[ ]:


## Linear supporter vector classifier
from sklearn.svm import SVC, LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_dev)
y_train_pred = linear_svc.predict(x_train)
acc_linear_svc = round(accuracy_score(y_pred,y_dev) * 100, 2)
acc_linear_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_linear_svc, "the training error is", acc_linear_train)


# In[ ]:


## Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()

perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_dev)
y_train_pred = perceptron.predict(x_train)
acc_perceptron = round(accuracy_score(y_pred,y_dev) * 100, 2)
acc_perceptron_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_perceptron, "the training error is", acc_perceptron_train)


# In[ ]:


## Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_dev)
y_train_pred = gaussian.predict(x_train)
acc_gaussian = round(accuracy_score(y_pred, y_dev) * 100, 2)
acc_gaussian_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_gaussian, "the training error is", acc_gaussian_train)


# In[ ]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, min_samples_split=15)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_dev)
y_train_pred = decisiontree.predict(x_train)
acc_dt = round(accuracy_score(y_pred, y_dev) * 100, 2)
acc_dt_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_dt, "the training error is", acc_dt_train)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(max_depth=4, max_features = 5 , min_samples_leaf=2, min_samples_split=5)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_dev) 
y_train_pred = randomforest.predict(x_train)
acc_rf = round(accuracy_score(y_pred, y_dev) * 100, 2)
acc_rf_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_rf, "the training error is", acc_rf_train)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC
svc = SVC(kernel='rbf',gamma=0.001,C=10)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_dev)
y_train_pred = svc.predict(x_train)
acc_svc = round(accuracy_score(y_pred, y_dev) * 100, 2)
acc_svc_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)

# x_dev_copy = x_dev
# x_dev_copy['Label'] = y_dev.values
# x_dev_copy['Pred'] = y_pred
# x_dev_copy[x_dev_copy['Label']!=x_dev_copy['Pred']].head(100)
# # for i in range(0,len(y_dev)):
# # #     print(y_pred[i], y_dev.values[i])
# # #     print(i)
# #     if y_pred[i] != y_dev.values[i]:
# # #         print(y_dev.index[i])
# # #         print(x_dev.loc[680])
# #         errors.append(x_dev.loc[y_dev.index[i]])
# # errors
print("the test error is", acc_svc, "the training error is", acc_svc_train)


# In[ ]:


# K-nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_dev)
y_train_pred = knn.predict(x_train)
acc_knn = round(accuracy_score(y_pred, y_dev) * 100, 2)
acc_knn_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_knn, "the training error is", acc_knn_train)


# In[ ]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#kfold = StratifiedKFold(n_splits=10)

GBC = GradientBoostingClassifier(max_depth=4,min_samples_leaf=100,max_features=0.2)
#gbc_param_grid = {'loss':["deviance"],
 #               'n_estimators':[100,200,300],
  #              'learning_rate':[0.1,0.05,0.01],
   #             'max_depth':[4,8],
    #            'min_samples_leaf':[100,150],
     #           'max_features':[0.3,0.1]
      #          }
#gsGBC = GridSearchCV(GBC,param_grid = gbc_param_grid, cv=kfold, scoring ="accuracy",n_jobs = 4,verbose =1)
GBC.fit(x_train, y_train)
y_pred = GBC.predict(x_dev)
y_train_pred = GBC.predict(x_train)
acc_gbc = round(accuracy_score(y_pred,y_dev)*100,2)
acc_gbc_train = round(accuracy_score(y_train_pred, y_train) * 100, 2)
print("the test error is", acc_gbc, "the training error is", acc_gbc_train)


# **6. Model ensembling**
# 
# We chose a voting classifer to combine the predictions coming from the 4 classifers which have the highest scores.

# In[ ]:


from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators=[('rf', randomforest), ('lr', logreg), ('sv',svc),('gb',GBC)], voting='hard')


# **7. Prediction and submit results**

# In[ ]:


votingC.fit(X_train, Y_train)
Y_pred = votingC.predict(X_test)

data_val1 = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": data_val1["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

