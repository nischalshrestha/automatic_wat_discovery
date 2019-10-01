#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Have a look at the data: 

# In[ ]:


# Dataset to train and evaluate the models
titanic_train = pd.read_csv("../input/train.csv")
# Dataset without target labels used to generate the final result.
titanic_final = pd.read_csv("../input/test.csv")
# Saving passenger IDs from titanic_final. Used to create the submission file.
titatic_final_passenger_ids = titanic_final['PassengerId']
titanic_train.head()


# In[ ]:


titanic_final.head()


#  Look at the datatypes and missing values. 
# 
# 

# In[ ]:


titanic_train.info()


# We have 891 entries. 
# 
# The following data colums have missing values: 'Age', 'Cabin' and 'Embarked'.
# 
# Data Dictionary can be found [here](https://www.kaggle.com/c/titanic/data).
# 
# 

#  Drop atributes that seem uninteresting from the training and final datasets:
# 
# 

# In[ ]:


def drop_from_dataset(datasets):
    for dataset in datasets:
        dataset.drop("Embarked", axis=1, inplace=True)
        dataset.drop("Cabin", axis=1, inplace=True)
        dataset.drop("PassengerId", axis=1, inplace=True)
        dataset.drop("Ticket", axis=1, inplace=True)
        dataset.drop("Name", axis=1, inplace=True)
        
drop_from_dataset([titanic_train, titanic_final])
titanic_train.info()


# Now, all but 'Sex' are numeric values. So let's make it a numeric value too:

# In[ ]:


titanic_train['Sex'],_ =pd.factorize(titanic_train['Sex'])
titanic_final['Sex'],_ =pd.factorize(titanic_final['Sex'])
titanic_train.info()


# Show summary of numerical attributes:

# In[ ]:


titanic_train.describe()


# Lets plot the data.

# In[ ]:


titanic_train.hist(bins=100, figsize=(20,15))
plt.show()


# Lets have a look how the attributes correlate with the target label (Survived). 

# In[ ]:


corr_matrix = titanic_train.corr()
corr_matrix['Survived'].sort_values(ascending=False)


# We have a strong positive correlation of the survival attribute with the sex and fare and a strong negative correlation with the Pclass attribute. Lets plot all correlations.

# In[ ]:


import seaborn as sns

sns.heatmap(corr_matrix, 
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values, linewidths=5.5, annot=True)
plt.show()


# ### Create a Test Set
# We need a test set to evaluate the performance of the machine learning algorithms. As the data set is realatively small, the test set will not be chosen randomly but stratified. We will keep the ratio of the Age. We will build age categories.
# 

# Some Ages are missing. So we have to fill them first using the median.

# In[ ]:


median_age_train = titanic_train['Age'].median()
median_age_final = titanic_final['Age'].median()
titanic_train['Age'].fillna(median_age_train,inplace=True)
titanic_final['Age'].fillna(median_age_final, inplace=True)


# Let's have a look, if there are missing values in the final dataset too:

# In[ ]:


titanic_final.info()


# Fare has one NaN value.  We fill it with the median too.

# In[ ]:


median_fare_final = titanic_train['Fare'].median()
titanic_final['Fare'].fillna(median_fare_final, inplace=True)
titanic_final.info()


# Building age categories by deviding the Age by some value > 1. 
# 17 brings up 5 classes. That seems enough.

# In[ ]:


titanic_train['Age_cat'] = np.ceil(titanic_train["Age"] /17  )
titanic_train['Age_cat'].hist()
plt.show()


# Create the stratified test set:

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in split.split(titanic_train, titanic_train['Age_cat']):
    strat_train_set = titanic_train.loc[train_index]
    strat_test_set = titanic_train.loc[test_index]


# Let's check if the test set has the same distribution by Age as the training set.

# In[ ]:


strat_test_set["Age_cat"].value_counts() / len(strat_test_set)


# In[ ]:


titanic_train["Age_cat"].value_counts() / len(titanic_train)


# We don't need the Age_cat any more. Lets delete it.

# In[ ]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop('Age_cat', axis=1, inplace=True)


# In[ ]:


strat_train_set.head()


# In[ ]:


strat_test_set.head()


# Create the target labels:

# In[ ]:


titanic_train_labels = strat_train_set["Survived"].copy()
strat_train_set = strat_train_set.drop("Survived", axis=1) # drop labels for training set
titanic_test_labels = strat_test_set["Survived"].copy()
strat_test_set = strat_test_set.drop("Survived", axis=1) # drop labels for test set
strat_train_set.head()


# In[ ]:


titanic_final.head()


# Perform one hot encoding of the categorical attributes.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
sex_cat_1hot_train = encoder.fit_transform(strat_train_set['Sex'].values.reshape(-1,1)).toarray()
sex_cat_1hot_test = encoder.fit_transform(strat_test_set['Sex'].values.reshape(-1,1)).toarray()
sex_cat_1hot_final = encoder.fit_transform(titanic_final['Sex'].values.reshape(-1,1)).toarray()

strat_train_set.drop('Sex', axis=1, inplace=True)
strat_test_set.drop('Sex', axis=1, inplace=True)
titanic_final.drop('Sex', axis=1, inplace=True)
strat_train_set.head()

sex_cat_1hot_train


# 1. Create the feature aund label arrays for model training:

# In[ ]:


X_train = strat_train_set.values
X_test = strat_test_set.values
X_final = titanic_final.values
X_train


# In[ ]:


X_final


# Append the one hot encoded entries:

# In[ ]:


X_train = np.c_[X_train, sex_cat_1hot_train]
X_test = np.c_[X_test, sex_cat_1hot_test]
X_final = np.c_[X_final, sex_cat_1hot_final]
y_train = titanic_train_labels.values
y_test = titanic_test_labels.values
X_train


# In[ ]:


X_final


# Perform scaling:

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_final)
X_final = scaler.transform(X_final)
X_final


# ### Train Models
# 
# * SCV
# * KNN
# * Decision Tree
# * Random Forrest
# * Ada Boost
# * GaussianNB

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def classify(classifier):
    y_pred = cross_val_predict(classifier, X_train, y_train, cv=3)
    return accuracy_score(y_train, y_pred)
  
svc_score = classify(SVC())
knn_score = classify(KNeighborsClassifier())
dt_score = classify(DecisionTreeClassifier())
rf_score = classify(RandomForestClassifier())
ada_score = classify(AdaBoostClassifier())
gau_score = classify(GaussianNB())
 


# Print accuracy:

# In[ ]:


models = pd.DataFrame({
    'Model': ['SVC', 'KNN', 'Decision Tree','RandomForestClassifier','AdaBoostClassifier','GaussianNB'],
    'Score': [svc_score, knn_score, dt_score, rf_score, ada_score, gau_score]})
models.sort_values(by='Score', ascending=False)


# SVC performs best most oft the time. Let's fine tune it:

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'C': [0.7, 0.8, 1.0,1.2,2.0],
     'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
]

svc_eval = SVC()
grid_search = GridSearchCV( svc_eval, param_grid, cv = 3, scoring ='precision') 
grid_search.fit( X_train, y_train)
grid_search.best_params_


# In[ ]:


grid_search.best_score_


# Predict the labels for the final dataset:

# In[ ]:


svc_final = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'])
svc_final.fit(X_train, y_train)
y_final = svc_final.predict(X_final)
y_final[:10]


# Create submission file:

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": titatic_final_passenger_ids,
        "Survived": y_final
    })
submission.head()
submission.to_csv('submission2.csv', index=False)

