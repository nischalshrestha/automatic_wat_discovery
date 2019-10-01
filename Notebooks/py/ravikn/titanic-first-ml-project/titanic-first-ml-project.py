#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/train.csv", index_col = 0)


# In[ ]:


df_test = pd.read_csv("../input/test.csv", index_col = 0)


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# In[ ]:


# Drop the columns Name, Ticket, Cabin and Embarked from both train and test dataset. Assumption is 
# that these would not have impacted the survival of the passengers
df.drop(['Name'], 1, inplace=True)
df.drop(['Ticket'], 1, inplace=True)
df.drop(['Cabin'], 1, inplace=True)
df.drop(['Embarked'], 1, inplace=True)
df_test.drop(['Name'], 1, inplace=True)
df_test.drop(['Ticket'], 1, inplace=True)
df_test.drop(['Cabin'], 1, inplace=True)
df_test.drop(['Embarked'], 1, inplace=True)


# In[ ]:


# Convert the Gender into numerics
numsex = {"male":0.0 ,"female" :1.0}
df['Sex'] = df['Sex'].replace(numsex)
df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')
df_test['Sex'] = df_test['Sex'].replace(numsex)
df_test['Sex'] = pd.to_numeric(df_test['Sex'], errors='coerce')


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# In[ ]:


df["FinalSurvived"] = df.Survived # Moving Survived to the last column
df.drop(['Survived'], 1, inplace=True) # Drop the Survived column


# In[ ]:


# Clean up the Train and Test dataset. For many passengers Age is missing. We use the Imputer strategy 
# of taking the median age and fill the missing cells with the median age
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer_test = Imputer(strategy="median")


# In[ ]:


df_num = df
df_num_test = df_test


# In[ ]:


imputer.fit(df_num)
imputer_test.fit(df_num_test)


# In[ ]:


imputer.statistics_


# In[ ]:


imputer_test.statistics_


# In[ ]:


df_num.median().values


# In[ ]:


df_num_test.median().values


# In[ ]:


X = imputer.transform(df_num)


# In[ ]:


Xt = imputer_test.transform(df_num_test)


# In[ ]:


df_tr = pd.DataFrame(X, columns=df_num.columns,
                          index = list(df.index.values))


# In[ ]:


df_tr_test = pd.DataFrame(Xt, columns=df_num_test.columns,
                          index = list(df_test.index.values))


# In[ ]:


#Check for sample rows which have empty cells
sample_incomplete_rows = df[df.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[ ]:


sample_incomplete_rows_test = df_test[df_test.isnull().any(axis=1)].head()
sample_incomplete_rows_test


# In[ ]:


df_tr.loc[sample_incomplete_rows.index.values]


# In[ ]:


df_tr_test.loc[sample_incomplete_rows_test.index.values]


# In[ ]:


#Check if the empty cell is filled up with the median value
df_tr = pd.DataFrame(X, columns=df_num.columns)
df_tr.head(6)


# In[ ]:


df_tr_test = pd.DataFrame(Xt, columns=df_num_test.columns)
df_tr_test.head(11)


# In[ ]:


# Move the training data to an array that can be trained
X_train = df_tr.iloc[:,:-1].values.tolist() 
y_train = df_tr.iloc[:,-1].tolist() # Final Survived column in a target array


# In[ ]:


# Prepare the Test dataset
from csv import reader
def load_csv(filename):
    file = open(filename)
    lines = reader(file)
    dataset = list(lines)
    return dataset
def column(matrix, i):
    return [row[i] for row in matrix]


# In[ ]:


X_test = df_tr_test.iloc[:,:].values.tolist()
X_test[0]


# In[ ]:


filename = '../input/genderclassmodel.csv'
dataset = load_csv(filename)
y_test = column(dataset, 1) # Take the Survived column as the independent variable
y_test.pop(0) #Drop the first row which is just a label
y_test[0]


# In[ ]:


# First model to use is Linear Classification
from sklearn.svm import LinearSVC
from sklearn import metrics


# In[ ]:


svmClassifier = LinearSVC(random_state=42)


# In[ ]:


svmClassifier.fit(X_train,y_train)


# In[ ]:


predicted = svmClassifier.predict(X_test)


# In[ ]:


#Convert the test target "Survived" to array and float data type
y_test_array = np.array(y_test, dtype=float)


# In[ ]:


metrics.accuracy_score(y_test_array, predicted)


# In[ ]:


# Logistic Regression Model
from sklearn import linear_model
logClassifier = linear_model.LogisticRegression(C=1)
logClassifier.fit(X_train, y_train)


# In[ ]:


predicted = logClassifier.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test_array, predicted)


# In[ ]:


#K-Nearest Neighbours Model
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)


# In[ ]:


predicted = neigh.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test_array, predicted)


# In[ ]:


# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, random_state=111)
rf = rf.fit(X_train, y_train)


# In[ ]:


predicted = rf.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test_array, predicted)


# In[ ]:


# Most promising model had been the Logistic Regression model with almost 95% accuracy.
# Also, dropping the columns of Name, Ticket, Cabin and Embarked from both train and test dataset 
# worked out well as they they don't seem to have much effect on the prediction accuracy.

