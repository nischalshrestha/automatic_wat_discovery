#!/usr/bin/env python
# coding: utf-8

# # flyingOllie meets Kaggle's Titanic
# * this is a Python 3 notebook running on a kaggle/python docker image:
#      https://github.com/kaggle/docker-python
# * thanks to tanlikesmath for some insights:
#      https://www.kaggle.com/tanlikesmath/titanic/titanic/notebook

# ## note: observations on kernel behavior
# * you cannot delete a kernel but you can hide a kernel from the public
# * if you you don't see the usual kernel controls, click the 'Edit' button
# * if kernel is 'Stopped' click the 'Restart kernel' button
# * if you want to run some else's kernel you will have to fork

# ## note: observations on Python Notebook
# * use Shift-Return to execute a cell
# * click on 'Styling with Markdown supported' to see how to mark style

# In[ ]:


import pandas as pd # feature engineering
import numpy as np # linear algebra
import re # regular expressions
import matplotlib.pyplot as plt
import csv as csv

from subprocess import check_output # i/o

# from sklearn import cross_validation
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb


# In[ ]:


# input files are here
print(check_output(["ls", "../input"]).decode("utf8"))
# output files (to current directory) go here
print(check_output(["ls", "../working"]).decode("utf8"))


# ## here is the feature engineering to apply to all data

# In[ ]:


def feature_engineer_pandas_df(the_original_pandas_df): # _df means dataframe

    #
    # Original fields:
    # 
    #  0 PassengerId (dropped, unused)
    #  1 Survived (target, only in train data)
    #  2 Pclass
    #  3 Name (dropped, unused)
    #  4 Sex (dropped, see Gender)
    #  5 Age (dropped, see AgeFill)
    #  6 SibSp
    #  7 Parch
    #  8 Ticket (dropped, unused)
    #  9 Fare (dropped, see FareFill)
    # 10 Cabin (dropped, unused)
    # 11 Embarked (dropped, see EmbarkedInteger)
    # 
    # Added fields:
    # 
    # 12 Gender (integer version of Sex)
    # 13 EmbarkedInteger (null: 1, 'S': 0, 'C': 1, 'Q': 2)
    # 14 FareFill (null fares are filled with median for associated Pclass)
    # 15 FareIsNull
    # 16 AgeFill (null ages are filled with median for associated Pclass and Gender)
    # 17 AgeIsNull
    # 18 FamilySize (SibSp + Parch)
    # 19 Age*Class (AgeFill * Pclass)
    #
    
    the_df = the_original_pandas_df
    the_df['Gender'] = the_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    the_df['EmbarkedInteger'] = the_df['Embarked'].fillna('X')
    the_df['EmbarkedInteger'] = the_df['EmbarkedInteger'].map( {'X': 1, 'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    median_fares = np.zeros(3)
    for j in range(0, 3):
        median_fares[j] = the_df[(the_df['Pclass'] == j+1)]['Fare'].dropna().median()
    the_df['FareFill'] = the_df['Fare']
    for j in range(0, 3):
        the_df.loc[ (the_df.Fare.isnull()) & (the_df.Pclass == j+1),                'FareFill'] = median_fares[j]
    the_df['FareIsNull'] = pd.isnull(the_df.Fare).astype(int)
 
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = the_df[(the_df['Gender'] == i) &                               (the_df['Pclass'] == j+1)]['Age'].dropna().median()
    the_df['AgeFill'] = the_df['Age']
    for i in range(0, 2):
        for j in range(0, 3):
            the_df.loc[ (the_df.Age.isnull()) & (the_df.Gender == i) & (the_df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
    the_df['AgeIsNull'] = pd.isnull(the_df.Age).astype(int)
    
    the_df['FamilySize'] = the_df['SibSp'] + the_df['Parch']
    the_df['Age*Class'] = the_df.AgeFill * the_df.Pclass
    the_df = the_df.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1) 
    return the_df


# # read the data and feature-engineer it

# In[ ]:


train_df = pd.read_csv('../input/train.csv', header=0)
train_engineered_df = feature_engineer_pandas_df(train_df) # column 0 is target
test_df = pd.read_csv('../input/test.csv', header=0)
test_engineered_df = feature_engineer_pandas_df(test_df) # no target column


# # create the random forest decision trees

# In[ ]:


forest_classifier = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=10, min_samples_leaf=5)
forest = forest_classifier.fit(train_engineered_df.iloc[:,1:],train_engineered_df.iloc[:,0])


# ## train and create predictions

# In[ ]:


train_actual = train_engineered_df.iloc[:,0]
train_prediction = forest.predict(train_engineered_df.iloc[:,1:])
test_prediction = forest.predict(test_engineered_df)


# ## calc accuracy on training data

# In[ ]:


mask = np.array(train_prediction == train_engineered_df.iloc[:,0], dtype = bool)
accuracy_train = len(train_prediction[mask])/len(train_prediction)
print(accuracy_train)


# ## write test data predictions to file for later submission

# In[ ]:


predictions_file = open("randomforestmodel.csv", "w", newline='')
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])
for row in range(len(test_df)):
    predictions_file_object.writerow([test_df['PassengerId'][row], "%d" % int(test_prediction[row])])
predictions_file.close()
print(check_output(["ls", "../working"]).decode("utf8"))


# In[ ]:


print(test_prediction)


# ## note: how to publish your prediction file
# * 'restart kernel' (button at top of notebook)
# * 'execute all cells' (button at top of notebook)
# * 'Publish' this kernel (button at top of notebook)
# * might have to click on 'view the latest version of the script'
# * tap on the 'Output' tab
# * then click on the 'Submit to Titanic: ...' button
