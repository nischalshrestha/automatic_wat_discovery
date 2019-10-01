#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[3]:



full_data = pd.read_csv("../input/train.csv")


# Print the first few entries of the RMS Titanic data
# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
data.shape


# In[4]:


missing_df = full_data.isnull().sum(axis=0).reset_index()
missing_df


# In[5]:


embarked_stn_count =  data.Embarked.value_counts(dropna=False)
#data.Embarked.unique()
embarked_stn_count 
#unique, counts = np.unique(data.Embarked, return_counts=True)

data.Embarked.value_counts(dropna=False).plot(kind = 'bar')

#plt.bar(left=data.Embarked.unique(), height =  data.Embarked.value_counts(dropna=False) , data = embarked_stn_count,  align='center', alpha=0.5)
#plt.xticks(data.Embarked.unique())
#plt.show()


# In[6]:


#Filling missing values
data['Embarked'] =data['Embarked'].fillna('S')
## Fill Age with median values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())


# In[7]:


#dropping unimportant features
data = data.drop('Cabin', axis = 1)
data = data.drop('Name' , axis = 1)
data = data.drop('Ticket', axis = 1)


# In[8]:


from ggplot import *
ggplot(data,aes(x='Age', color='Sex')) +     geom_density(alpha=1)


# In[9]:


#pd.unique((data['Embarked']))
features_final = pd.get_dummies(data)
encoded = list(features_final.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))
#features_final


# In[10]:


from sklearn.cross_validation import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    outcomes, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))


# In[11]:


from sklearn.ensemble import AdaBoostClassifier

clf =  AdaBoostClassifier(random_state =42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
results = accuracy_score(y_test, predictions)

print (results)


# In[12]:


# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
# TODO: Initialize the classifier
clf = AdaBoostClassifier(random_state =42)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'n_estimators' : [50, 100, 200], 'learning_rate' : [0.1, 1, 10]}

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer =  make_scorer(fbeta_score, beta=1)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print ("Unoptimized model\n------")
print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print ("\nOptimized Model\n------")
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# In[13]:


print ("Parameters for the optimal model: {}".format(best_clf.get_params()))


# In[14]:


test =  pd.read_csv("../input/test.csv")
test  = test.drop('Cabin', axis = 1)
test  = test.drop('Name' , axis = 1)
test = test.drop('Ticket', axis = 1)
#Filling missing values
test['Embarked'] =test['Embarked'].fillna('S')
## Fill Age with median values
test["Age"] = test["Age"].fillna(data["Age"].median())
test["Fare"] = test["Fare"].fillna(data["Fare"].median())
test_features =  pd.get_dummies(test)


# In[15]:


test_features.head(5)


# In[16]:


prediction = best_clf.predict(test_features)


# In[17]:


submission_DF = pd.DataFrame({ 
    "PassengerId" : test["PassengerId"],
    "Survived" : prediction
    })
print(submission_DF.head(5))


# In[18]:


submission_DF.to_csv("submission.csv", index=False)

