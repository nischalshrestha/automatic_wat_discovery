#!/usr/bin/env python
# coding: utf-8

# I am a begineer in datascience and machine learning. This is my first kernel submitted.
# 
# Taken references from the below kernels for building the model
# https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling   
# https://www.kaggle.com/startupsci/titanic-data-science-solutions  
# https://www.kaggle.com/longyin2/titanic-machine-learning-from-disaster-0-842
# 
# Please review and let me know how can I improve the accuracy. Thanks
# 
# Extracted the title from Name as used it as features as mentioned in the above kernels. The accuracy significantely improved.
# Edited for - Both train and test set are considered for calculating the mean for Age.
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

## Visulization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


## Load the training and test dataset
input_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

data_df = input_df.append(test_df) #Entire dataset

# display the first 5 records of input data set
input_df.head()


# **Implementation** : **Data Exploration**        
# an investigation of the dataset  below will determine how many people survived or not survived from each category and will also tell us the percentage of the people that are survived.
# 
# Total number of passengers in the input data - 891    
# no of passengers survived -- 342       
# no of passenger not survived - 549
# 
# Percentage of people survived - 38%   
# Female passengers survived most.  
# Most passengers in 1st class are survived   

# In[3]:


input_df.describe()


# In[4]:


n_survived = len(input_df[input_df['Survived'] == 1])
not_survived = len(input_df[input_df['Survived'] == 0])
print ("Total number of passengers survived: {}".format(n_survived))
print ("Total number of passengers survived: {}".format(not_survived))


# In[5]:


sns.countplot(x='Survived', hue="Sex", data=input_df)


# In[6]:


sns.countplot(x='Survived', hue='Pclass', data=input_df)


# In[7]:


## Visulizing distributions of Age and Fare
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].hist(input_df['Fare'], bins=20)
#axes[1].hist(input_df['Age'])

input_df['Age'].hist(axes=axes[1], bins=15, density=True)
input_df['Age'].plot(kind='density', color='green')


# **Featureset Exploration**    
# Age - Continuous         
# Pclass - Categorical  ( 1- Upper, 2 - Middle, 3-Lower)       
# PassengerId - Sequence of the passenger ID. Continuous         
# Sex - Categorical ( Male, Female)       
# Name - Unique name of the passenger         
# Sibsp - No of siblings/spouses aboard. Numerical, Continuous          
# Parch - No of parents/Childern aboard. Numerical, Continuous          
# Fare - Continuous             
# Embarked - Categorical (C, Q, S)             
# Cabin - Alphanumeric number             
# Ticket - Unique value             
# Survived - Categorical ( 0 - No, 1 - Yes)               

# **Preparing the Data** -- Data Preprocessing               
# **Missing Values**       
# For this dataset, we can see there are missing values present for Age, Embarked and Cabin. While 20% of the data has missing values for Age, very large proportion of the data has values missed for Cabin. So considering this, the column Cabin can be dropped instead of filling the data.    
# For the Age, the missing values can be replaced with some form of imputation like th mean of the Age.
# For Embarked, only 2 values are missing and can be filled in with the most Embarked category.
# 

# In[8]:


input_df.info()
print ("---------------------------------------------")
test_df.info()


# In[9]:


sns.countplot(input_df['Embarked'])


# Extracting the Title from the Name column

# In[10]:


def Name_Title_Code(x):
    if x == 'Mr.':
        return 1
    if (x == 'Mrs.') or (x=='Ms.') or (x=='Lady.') or (x == 'Mlle.') or (x =='Mme'):
        return 2
    if x == 'Miss':
        return 3
    if x == 'Rev.':
        return 4
    return 5

print (input_df['Name'].head())
print ('--------------------------------------')
input_df['Name_Title'] = input_df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
print (input_df['Name_Title'].head())

input_df['Name_Title'] = input_df['Name_Title'].apply(Name_Title_Code)
print (input_df['Name_Title'].head())

print ('----------Test data.........')
test_df['Name_Title'] = test_df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
print (test_df['Name_Title'].head())

test_df['Name_Title'] = test_df['Name_Title'].apply(Name_Title_Code)
print (test_df['Name_Title'].head())


# Filling the missing data for Age - One way is we can fill in with the  mean age of all the passengers or we can check the correlation of Age with other features like pclass and populate the mean age based on the pclass.
# 
# For Embarked - we can fill in with the Most Embarked station - 'S'

# In[11]:


#mean_age = input_df['Age'].mean()
def impute_age(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        return data_df.groupby('Pclass').median()['Age'][pclass]   
    else:
        return age

def impute_embarked(embarked):
    if pd.isnull(embarked):
        return 'S'
    else:
        return embarked
        
input_df['Age'] = input_df[['Age', 'Pclass']].apply(impute_age, axis=1)
test_df['Age'] = test_df[['Age', 'Pclass']].apply(impute_age, axis=1)
input_df['Embarked'] = input_df['Embarked'].apply(impute_embarked)
test_df['Embarked'] = test_df['Embarked'].apply(impute_embarked)

### filling the missing value for the Fare in the test dataset
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


# In[12]:


## Verifying the data after the missing values are filled in for Age and Embarked
input_df.info()
print ("----------------------------------------------")
test_df.info()


# **Transforming skewed continuous features**
# Age and Fare are two continuous features in the dataset. 

# In[13]:


## seperate out the target
target_df = input_df['Survived']
features_raw = input_df.drop('Survived', axis=1)


# In[14]:


# Log-transform the skewed features
skewed = ['Age', 'Fare']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

## applying the log transformation to the test data
test_df_log_transformed = pd.DataFrame(data=test_df)
test_df_log_transformed[skewed] = test_df[skewed].apply(lambda x: np.log(x+1))

# Visualize the new log distributions
fig, axes = plt.subplots(1,2, figsize=(10,4))
axes[0].hist(features_log_transformed['Fare'], bins=30)
axes[0].set_title("Fare: Feature Distribution")
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("No of records")
axes[1].hist(input_df['Age'], bins=25)
axes[1].set_title("Age: Feature Distribution")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("No of records")

fig.tight_layout()


# In[25]:


## Adding the family feature
features_log_transformed['Family'] = features_log_transformed['SibSp'] +  features_log_transformed['Parch']

test_df_log_transformed['Family'] =  test_df_log_transformed['SibSp'] +  test_df_log_transformed['Parch']

features_log_transformed.head()


# **Normalizing Numerical Features**
# Applying a scaling/normalization ensures that each feature is treated equally when applying the learning algorithm.

# In[26]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#numerical = ['Age', 'Fare', 'SibSp', 'Parch']

numerical = ['Age', 'Fare', 'Family']

features_normalized = pd.DataFrame(data=features_log_transformed)
features_normalized[numerical] = scaler.fit_transform(features_log_transformed[numerical])

print (features_normalized.head(5))

## Normalizing test data set
test_features_normalized = pd.DataFrame(data=test_df_log_transformed)
test_features_normalized[numerical] = scaler.fit_transform(test_df_log_transformed[numerical])


# **Data Preprocssing - Converting the categorical values**    
# Using one hot encoding scheme the categorical values for Sex, Embarked  can be converted to numberical values.

# In[27]:


## one hot coding for the categorical values. SEX, EMBARKED
## Before that lets drop the columns that are not required for the algorithm 
## Cabin, Name, PassengerId, TicketId

features_final = features_normalized.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1)
test_df_final = test_features_normalized.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'], axis=1)

#features_final['Sex'].replace(['male', 'female'], [0,1], inplace=True)
#test_df_final['Sex'].replace(['male', 'female'], [0,1], inplace = True)

input_train_one_hot_encoded = pd.get_dummies(features_final)
print (input_train_one_hot_encoded.head())

### one hot encoding for test dataset
test_df_one_hot_encoded = pd.get_dummies(test_df_final)


# In[17]:


input_train_one_hot_encoded.isnull().any()  ## to check if any column has null or nan values


# **Splitting data into Training and Testing data sets. 
# 80% of the input data will be used for training and 20% for testing.
# 

# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_train_one_hot_encoded, 
                                                    target_df, test_size=0.20, random_state=42)
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print ("Training set has {} samples".format(X_train.shape[0]))
print ("Testing set has {} samples".format(X_test.shape[0]))


# **Model application - Logistic Classificatioin******

# In[29]:


## Logistic classification
from sklearn import linear_model
from sklearn.metrics import accuracy_score, fbeta_score

clf_A = linear_model.LogisticRegression()
clf_A.fit(X_train, y_train)
pred = clf_A.predict(X_test)
accuracy = accuracy_score(pred, y_test)
f_score = fbeta_score(y_test, pred, beta=0.5)

print ("Accuracy score of logistic classification: {}".format(accuracy))
print ("f_score of logistic classification: {}".format(f_score))


# In[30]:


## Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
clf_B = AdaBoostClassifier(random_state = 100)
clf_B.fit(X_train, y_train)
pred_1 = clf_B.predict(X_test)
accuracy_1 = accuracy_score(pred_1, y_test)
f_score_1 = fbeta_score(y_test, pred_1, beta=0.5)

print ("Accuracy score of AdaBoost classification: {}".format(accuracy_1))
print ("f_score of AdaBoost classification: {}".format(f_score_1))


# In[21]:


## Support Vector machine
from sklearn import svm
clf_C = svm.SVC(random_state = 100)
clf_C.fit(X_train, y_train)
pred_2 = clf_C.predict(X_test)
accuracy_2 = accuracy_score(pred_2, y_test)
f_score_2 = fbeta_score(y_test, pred_2, beta=0.5)

print ("Accuracy score of SVM classification: {}".format(accuracy_2))
print ("f_score of SVM classification: {}".format(f_score_2))


# In[31]:


## Implementaion & Model tuning of RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score

clf_r = RandomForestClassifier(random_state = 100)
## Parameters list to fine tune
#parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [10, 20],
 #            'min_samples_split': [10, 15]}
parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [50,100,400,700,1000],
             'min_samples_split': [2, 4, 10,12,16], }
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search on the classifier 
grid_obj = GridSearchCV(clf_r, param_grid=parameters, scoring= scorer)
grid_fit = grid_obj.fit(X_train, y_train)

# best estimator
best_clf_r = grid_fit.best_estimator_
best_predictions = best_clf_r.predict(X_test)

print(grid_fit.best_score_)
print(grid_fit.best_params_) 

##Optimized model
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# In[23]:


## Implementaion - Model tuning of AdaBoost Classifier
'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score

clf = AdaBoostClassifier(random_state = 100)
## Parameters list to fine tune
parameters = {'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5], 'n_estimators': [600, 800,1000]}
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search on the classifier 
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring= scorer)
grid_fit = grid_obj.fit(X_train, y_train)

# best estimator
best_clf = grid_fit.best_estimator_
best_predictions = best_clf.predict(X_test)

##Optimized model
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
'''


# Since accuracy and F-score is high when used the RandomForestClassifier, submitting the predictions on the test data set using this model.

# In[33]:


test_pred = best_clf_r.predict(test_df_one_hot_encoded)

submission = pd.DataFrame({
           "PassengerId": test_df["PassengerId"],
           "Survived": test_pred
           })

submission.to_csv('submission.csv', index=False)


# In[ ]:




