#!/usr/bin/env python
# coding: utf-8

# # **Getting Started**
# In this project, the goal is to predict which passengers survived the tragedy of the sinking of Titanic.
# Let's start by importing the necessary modules and importing data:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.shape


# In[ ]:


test_df.head()


# # **Pre-processing**

# ## **Title Extraction from Passenger Names**

# Each row of data already has a unique identifier (PassengerId). I have preprocessed the Name column to extract the passenger title from the text data in the Name column. The extracted titles can act as another feature in the model

# In[ ]:


#process name column to extract passenger titles
#training data
name_col = train_df['Name']
titles_ls = []
for name in name_col:
    title = name.split(', ')[1].split('. ')[0]
    titles_ls.append(title)
list(set(titles_ls)) #get the unique titles


# It would help the generalizability of the model to have broader categories as far as the different title categories go. So, next, I manually narrowed down the titles to the following:

# In[ ]:


title_mapper = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Dr":         "Officer",
    "Rev":        "Officer",    
    "Jonkheer":   "Upper_Class",
    "Don":        "Upper_Class",
    "Sir" :       "Upper_Class",
    "Lady" :      "Upper_Class",    
    "the Countess":"Upper_Class",
    "Dona":       "Upper_Class",    
    "Mlle":       "Miss",
    "Miss" :      "Miss",
    "Mme":        "Mrs",    
    "Ms":         "Mrs",
    "Mrs" :       "Mrs",    
    "Master" :    "Master",
    "Mr" :        "Mr"    
}


# Add Title column to data:

# In[ ]:


title_col = []
for title in titles_ls:
    title_col.append(title_mapper[title])
train_df['Title'] = title_col  
train_df.head()


# Before we add the Title column to test data, we need to make sure that we will not face any unseen titles (i.e. absent keys in the mapper):

# In[ ]:


#test data
name_col = test_df['Name']
titles_ls = []
for name in name_col:
    title = name.split(', ')[1].split('. ')[0]
    titles_ls.append(title)
list(set(titles_ls)) #get the unique titles


# Okay there are no new titles in the test set. With that being confirmed, let's add the feature to the test set as well

# In[ ]:


title_col = []
for title in titles_ls:
    title_col.append(title_mapper[title])
test_df['Title'] = title_col  
test_df.head()


# Now that the feature Title has been successfully created, we can go ahead and remove the Name column:

# In[ ]:


train_df.drop('Name', axis=1, inplace=True)
train_df.head()


# In[ ]:


test_df.drop('Name', axis=1, inplace=True)
test_df.head()


# In[ ]:


passengerId = test_df['PassengerId']


# In[ ]:


train_df.drop('PassengerId', axis=1, inplace=True)
train_df.head()


# In[ ]:


test_df.drop('PassengerId', axis=1, inplace=True)
test_df.head()


# ## **Handling Missing Values**

# As the quick scanning of the dataset hints, there are columns in both train and test data that contain missing values. Let's look closer to the data for possible missing values and go about handling these missing entries in the data:

# In[ ]:


#print number of missing entries in each column of the training data
train_df.isnull().sum()


# In[ ]:


#detect which columns include NaN's in the training data
nan_col_train = train_df.columns[train_df.isna().any()].tolist() #train data
nan_col_train


# In[ ]:


#print number of missing entries in each column of the test data
test_df.isnull().sum()


# In[ ]:


#detect which columns include NaN's in the test dataset
nan_col_test = test_df.columns[test_df.isna().any()].tolist() #test data
nan_col_test


# As indicated above, each of the training and test dataset has three columns that include at least one NaN value. We can also take a look at the percentage of the missing entries for each of these features:

# In[ ]:


#train data
for col in nan_col_train:
    col_data = train_df[col]
    null_entry_cnt = col_data.isnull().sum()
    #print(null_entry_cnt)    
    total_entry_cnt = len(col_data)
    nul_ratio = float(null_entry_cnt)/float(total_entry_cnt)
    print("For the training set, %.2f percent of data in the column %s is missing entries"           % (100*nul_ratio, col))
# test data
for col in nan_col_test:
    col_data = test_df[col]
    null_entry_cnt = col_data.isnull().sum()
    #print(null_entry_cnt)
    total_entry_cnt = len(col_data)
    nul_ratio = float(null_entry_cnt)/float(total_entry_cnt)
    print("For the test set, %.2f percent of data in the column %s is missing entries"           % (100*nul_ratio, col))


# I have used imputation to handle the missing values:

# In[ ]:


imp_const = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="unk")
imp_const.fit(train_df[['Cabin']])


# In[ ]:


train_df['Cabin'].head(10)


# In[ ]:


train_df['Cabin'] = imp_const.transform(train_df[['Cabin']]).ravel()


# In[ ]:


train_df['Cabin'].head(10)


# For the column "Cabin", since the percentage of missing values is very big, I used the constant imputation method to replace all the NaN's with string "unk". Let's use the fitted imputer to transform the test set as well:

# In[ ]:


test_df['Cabin'] = imp_const.transform(test_df[['Cabin']]).ravel()
test_df['Cabin'].head(10)


# For Age and Fare, use median of the existing values for imputation:

# In[ ]:


imp_median_age = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median_age.fit(train_df[['Age']])


# In[ ]:


train_df['Age'] = imp_median_age.transform(train_df[['Age']]).ravel()


# In[ ]:


test_df['Age'] = imp_median_age.transform(test_df[['Age']]).ravel()


# In[ ]:


imp_median_fare = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median_fare.fit(train_df[['Fare']])


# In[ ]:


test_df['Fare'] = imp_median_fare.transform(test_df[['Fare']]).ravel()


# In[ ]:


imp_frequent = SimpleImputer(strategy='most_frequent')
imp_frequent.fit(train_df[['Embarked']])


# In[ ]:


train_df['Embarked'] = imp_frequent.transform(train_df[['Embarked']]).ravel()


# ## **Handling Categorical Features**

# Now, let's perform one-hot encoding on categorical features. For that, we would first need to detect which column(s) represent categorical variables:

# In[ ]:


# categorical variables in training data
all_col = train_df.columns
num_cols = train_df._get_numeric_data().columns
cat_cols_train = list(set(all_col) - set(num_cols))
# categorical variables in test data
all_col = test_df.columns
num_cols = test_df._get_numeric_data().columns
cat_cols_test = list(set(all_col) - set(num_cols))


# In[ ]:


cat_cols = list(set(cat_cols_train + cat_cols_test))
cat_cols


# In[ ]:


#train = pd.get_dummies(train_df, columns=cat_cols)
#test = pd.get_dummies(test_df, columns=cat_cols)
len_train = train_df.shape[0]
train_test_df = pd.concat([train_df, test_df])
X_train_test_ohe = pd.get_dummies(train_test_df, columns=cat_cols)

# Separate them again into train and test
train_df, test_df = X_train_test_ohe.iloc[:len_train, :], X_train_test_ohe.iloc[len_train:, :]


# Now let's further process train data to extract the labels:

# In[ ]:


y_train_col = train_df.Survived 
y_train = y_train_col.values
x_train_df = train_df.drop(['Survived'], axis=1)
x_train = x_train_df.values


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


# drop the label column in test data
x_test_df = test_df.drop(['Survived'], axis=1)
x_test = x_test_df.values
x_test.shape


# To get more insight into the data, let's take a look at the training labels to get a sense of the relative population of the two classes ("survived" vs. "not-survived")

# In[ ]:


label_map = {}
label_map = {1: "survived", 0: "deceased"} #map label to survival status for visualization purposes
label_map


# In[ ]:


passngr_cnt  = y_train_col.value_counts()
passngr_idx = passngr_cnt.index
plt_idx = [label_map[idx] for idx in passngr_idx]    
plt.figure()
sns.barplot(plt_idx, passngr_cnt.values, alpha=0.8)
plt.title('Passenger survival according to training data')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('passenger survival', fontsize=12)
plt.show()


# ## **Feature Scaling**

# Let's standardize the features before fitting our model:

# In[ ]:


#perform feature standardization
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


# ## **Classification**

# For my classification model, I have used Random Forest classifier from Scikit-learn. I performed hyperparameter tuning via cross validation as below:

# In[ ]:


# Random Forest
clf = RandomForestClassifier(random_state=0)
param_grid = {'n_estimators': [5, 10, 15, 20, 25, 50],
              'max_depth': [2, 5, 7, 10],
              'max_features': [250, 300, 400, 500],
              'min_samples_leaf': [5, 10, 50]
             }
#'min_samples_leaf': [2, 3, 4, 5]
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                        cv=5, scoring ='accuracy')
grid_search.fit(x_train, y_train)

grid_search.best_estimator_


# In[ ]:


preds = [int(pred) for pred in grid_search.predict(x_test)]
result = pd.DataFrame({'PassengerId': passengerId, 'Survived': preds})
# save to csv
result.head(10)


# In[ ]:


fname = 'Titanic_randomforest_7.csv'
result.to_csv(fname,index=False)

