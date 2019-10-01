#!/usr/bin/env python
# coding: utf-8

# **Downloading the necessary libraries**
# 
# For the Titanic Classification Problem we will use:
# * Pandas
# * Numpy
# * Scikit-Learn
# * Maplotlib
# * Seaborn
# * OS
# 
# 
# 
# 

# In[23]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Import visualization library, as well as Machine Learning classes
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import warnings

# Any results you write to the current directory are saved as output.

# Import the train data
# Display and review summary statistics 
train_data = pd.read_csv('../input/train.csv', index_col=0)

# Handle missing data by assigning the mean value of the dataset
mean_value_age = round(train_data['Age'].mean())
train_data['Age'].fillna(mean_value_age, inplace=True)


# In[2]:


train_data.head(10)


# In[2]:


# Compare the surviving and non-surviving passengers by the ticket fare and gender
plt.subplots(figsize=(20, 10))

plt.subplot(2,2,1)
plt.title('Surviving passengers \n by gender and ticket price')
sns.swarmplot(x='Sex', y='Fare', hue='Pclass', data=train_data[train_data['Survived']==1])
plt.legend(loc='upper left', title='Passenger Class')

plt.subplot(2,2,2)
plt.title('Non-surviving passengers\n by gender and ticket price')
sns.swarmplot(x='Sex', y='Fare', hue='Pclass', data=train_data[train_data['Survived']==0])
plt.legend(loc='upper left', title='Passenger Class')

plt.subplot(2,2,3)
plt.title('Surviving passengers \n by gender and ticket price')
sns.swarmplot(x='Sex', y='Fare', hue='Embarked', data=train_data[train_data['Survived']==1])
plt.legend(loc='upper left', title='Location Embarked')

plt.subplot(2,2,4)
plt.title('Non-surviving passengers\n by gender and ticket price')
sns.swarmplot(x='Sex', y='Fare', hue='Embarked', data=train_data[train_data['Survived']==0])
plt.legend(loc='upper left', title='Location Embarked')

plt.tight_layout()
plt.show()


# In[34]:


# Continue to use the plots to explore the relationship between features
plt.figure(figsize=(6,4))
plt.title('Breakdown of survival \n by age and embarkation location')
sns.swarmplot(x='Embarked', y='Age', hue='Survived', data=train_data)
plt.legend(loc='upper left', title='Survived vs Fatalities')

plt.figure(figsize=(6,4))
plt.title('Breakdown of survival \n by fare and embarkation location')
sns.swarmplot(x='Embarked', y='Fare', hue='Survived', data=train_data)
plt.legend(loc='upper left', title='Survived vs Fatalities')

plt.figure(figsize=(6,4))
plt.title('Survived vs Fatalities')
sns.countplot(train_data['Survived'])

sns.pairplot(train_data)


# ** Preprocessing the Data by selecting the features that we will use to predict the survival encoding of the passengers in the test set**
# 
# *In the chunk of code below we select continuous variables such as age and ticket price paid as well as discrete variables such as passenger class, number of siblings/spouses on board, number of parents/children on board, gender and embarkation location. *

# In[24]:


# Get dummy variables for sex, pclass, embarked
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

processed_data = pd.get_dummies(data=train_data, columns=['Sex', 'Pclass', 'Embarked'])

# Get the feature columns of interest and assign them to X
X = processed_data[['Age', 'Fare', 'FamilySize', 'SibSp', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = processed_data[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ** Implementing a Basic KNN model**
# 
# *This will allow to get a feel for the model accuracy and will set the stage for further hyper parameter tuning.*
# *We picked 22 n neighbors and the distance method for weights parameter.*
# *The model performed at almost 75.98% accuracy.*

# In[6]:


# Perform a train test split on the preliminary train data
N_NEIGHBORS = 22
knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance')
knn.fit(X_train, y_train.values.ravel())
print('Accuracy of the basic model with', N_NEIGHBORS, 'neighbors is', knn.score(X_test, y_test))


# *The code below plots the accuracy vs. number of n_neighbors, using the uniform method for the weight parameter.*
# *The maximum accuracy is achieved at 6 and 26 neighbors, when the accuracy is 74.3%*

# In[129]:


# Determine the best value for n_neighbors that increases accuracy
neighbors = np.arange(1,40)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train.values.ravel())
    
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    
# Plot the relationship between n_neighbors and accuracy
plt.figure(figsize=(7,5))
plt.title('Training vs Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.xlabel('K Neighbors')
plt.ylabel('Accuracy')
plt.legend()

max_val = test_accuracy.max()
index_number_of_max_val = list(test_accuracy).index(max_val)
print('Highest accuracy is', max_val, ', achieved with ', index_number_of_max_val+1, ' n_neighbors')


# *This code modifies the one above slightly by changing the weights parameter to distance method. We can see that this change paid off, with the new accuracy at 75.98% with 22 neighbors.*

# In[130]:


# Determine the best value for n_neighbors that increases accuracy
neighbors = np.arange(1,40)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train.values.ravel())
    
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
    
# Plot the relationship between n_neighbors and accuracy
plt.figure(figsize=(7,5))
plt.title('Training vs Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.xlabel('K Neighbors')
plt.ylabel('Accuracy')
plt.legend()

max_val = test_accuracy.max()
index_number_of_max_val = list(test_accuracy).index(max_val)
print('Highest accuracy is', max_val, ', achieved with ', index_number_of_max_val+1, ' n_neighbors')


# *We then perform a CV on the Classifier, using 22 n_neighbors. The mean accuracy of 5 fold CV is 71.62%*

# In[133]:


# Perform Cross Validation on the train data
N_NEIGHBORS_CV = 22
cv = cross_val_score(KNeighborsClassifier(n_neighbors=N_NEIGHBORS_CV), X, y.values.ravel(), cv=5)
array = [1,2,3,4,5]
cv_accuracy_scores = pd.DataFrame({'CV Holdout Set': array, 'Accuracy': cv})
cv_accuracy_scores
print('Mean accuracy of Cross Validation with', N_NEIGHBORS_CV, 'neighbors is', cv.mean())


# *Then, we implement a GridSearchCV  that searches for the optimal n_neighbors parameter only. It reveals that 7 is the optimal value for the parameter, with the average accuracy across 5 fold CV at 71.83%*

# In[134]:


param_grid = {'n_neighbors': np.arange(1,50)}
knn_pre = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_pre, param_grid, cv=5)
knn_cv.fit(X,y.values.ravel())
print(knn_cv.best_params_)
print(knn_cv.best_score_)
print(knn_cv.best_estimator_)
warnings.filterwarnings(action='ignore')


# *Finally, we implement a GridSearchCV with both weights and n_neighbors parameters. The search determines that the best parameter for n_neighbors is 8 and the best parameter for weight is distance, with the average accuracy across 5 fold CV at 74.07%*

# In[135]:


weight_list = ['uniform', 'distance']
n_neighbors = list(range(1, 40))
param_grid = {'n_neighbors': n_neighbors, 'weights': weight_list}
knn_e = KNeighborsClassifier()
grid_knn = GridSearchCV(knn_e, param_grid, cv=10, scoring='accuracy')
grid_knn.fit(X,y.values.ravel())
print(grid_knn.best_score_)
print(grid_knn.best_params_)
print(grid_knn.best_estimator_)


# **Basic Logistic Regression Model**
# 
# *This model is a basic model with default hyperparameters, in order for us to see what accuracy to expect.
# *We obtain *

# In[136]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Score of default model is', logreg.score(X_test, y_test))
y_pred = logreg.predict_proba(X_test)[:,1]
print('AUC score of default model is', roc_auc_score(y_test, y_pred))
y_pred = logreg.predict(X_test)


# *Below is the confusion matrix that shows the True Positive, True Negative, False Positive and False Negative values, as well as the classification report that computes Precision, Recall and F1 score based on the confusion matrix*

# In[137]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# *Here is the ROC Curve of the above basic Logistic Regression Model*

# In[138]:


y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure()
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

print('The AUC score of the baseline logit model is', roc_auc_score(y_test, y_pred_prob))


# **Let's tune the hyperparameters of the baseline model.**
# 
# *We need to tune the C and the penalty parameters of the baseline model and see if we can improve on the accuracy.*

# In[139]:


params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2'] }
logit_grid = GridSearchCV(logreg, params, cv=5)
logit_grid.fit(X_train, y_train.values.ravel())


# *Once we have tuned the hyperparameters, let's see how the accuracy and AUC score change as a result. It improved by relatively little, 0.00128, to 87.77%. However, since the accuracy of the base model with C (regularization parameter) equal to 1 was higher, I will simply proceed with that result.*

# In[140]:


print('Score of tuned model is', logit_grid.score(X_test, y_test))
y_pred_grid = logit_grid.predict_proba(X_test)[:,1]
print('AUC score of tuned model is', roc_auc_score(y_test, y_pred_grid))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_grid)

plt.figure()
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

print('The model has improved by', roc_auc_score(y_test, y_pred_grid) - roc_auc_score(y_test, y_pred_prob))

print(logit_grid.best_params_)
print(logit_grid.best_score_)
print(logit_grid.best_estimator_)

y_pred_grid_binary = logreg.predict(X_test)
print(confusion_matrix(y_test, y_pred_grid_binary))
print(classification_report(y_test, y_pred_grid_binary))


# **Basic DecisionTreeClassifier**

# In[25]:


tree = DecisionTreeClassifier()
tree.fit(X_train, y_train.values.ravel())
print('The baseline Decision Tree model accuracy is', tree.score(X_test, y_test))
print(confusion_matrix(y_test, tree.predict(X_test)))
print(classification_report(y_test, tree.predict(X_test)))


# *Based on the graph, the best maximum depths is 9*

# In[26]:


max_depths = np.linspace(1,32,32,endpoint=True)
train_results_tree = []
test_results_tree = []

for max_depth in max_depths:
    tree = DecisionTreeClassifier(max_depth = max_depth)
    tree.fit(X_train, y_train.values.ravel())
    train_pred = tree.predict(X_train)
    train_results_tree.append(roc_auc_score(y_train, train_pred))

    tree = DecisionTreeClassifier(max_depth = max_depth)
    tree.fit(X_train, y_train.values.ravel())
    test_pred = tree.predict(X_test)
    test_results_tree.append(roc_auc_score(y_test, test_pred))

plt.title('Train vs Test Accuracy')    
plt.plot(max_depths, train_results_tree, label='Train')
plt.plot(max_depths, test_results_tree, label='Test')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()


# *The maximum depth parameter that maximized AUC score is 3*

# In[27]:


tree = DecisionTreeClassifier()
max_depths = {'max_depth':np.linspace(1,40,40, endpoint=True)}
tree_grid = GridSearchCV(tree, max_depths, scoring = 'roc_auc', cv=5)
tree_grid.fit(X_train, y_train.values.ravel())

print(tree_grid.best_estimator_)
print(tree_grid.best_params_)
print(tree_grid.best_score_)



# In[28]:


test_data = pd.read_csv('../input/test.csv')


# **RandomForestClassifier, baseline model with default parameters**
# 
# *The model shows the AUC score to be 87.44%*

# In[147]:


forest = RandomForestClassifier()
forest.fit(X_train, y_train.values.ravel())
y_pred_forest = forest.predict_proba(X_test)[:,1]

print('The baseline accuracy is', forest.score(X_test, y_test))
print('The baseline model AUC score is', roc_auc_score(y_test, y_pred_forest))


# *Tuning the n_estimators to 500 improves the model AUC score to over 88.63%*

# In[150]:


forest_tuned = RandomForestClassifier(n_estimators = 500)
forest_tuned.fit(X_train, y_train.values.ravel())
y_pred_forest_tuned = forest_tuned.predict(X_test)
y_pred_forest_tuned_proba = forest_tuned.predict_proba(X_test)[:,1]

print('The tuned model Accuracy score is', forest_tuned.score(X_test, y_test))
print('The tuned model AUC score is', roc_auc_score(y_test, y_pred_forest_tuned_proba))
print(confusion_matrix(y_test, y_pred_forest_tuned))
print(classification_report(y_test, y_pred_forest_tuned))


# **Now that we have trained 4 classification models, we can apply them to predict the Survival Rate on the test.csv file**
# 
# *From the models that we trained, namely KNN, Logistic Regression, Decision Tree Classifier and Random Forest Classifier, we can see that Random Forest method with tuned n_estimators parameter is the one that consistently produces highest AUC score. Therefore we will use RF on the test data and write the results to a submission file.*

# In[29]:


test_data = pd.read_csv('../input/test.csv')
test_data.describe()


# **Preprocessing the data**

# In[30]:


def numeric_missing_value(data, column):
    # Function takes in data in the form of a Pandas DataFrame and column names
    # in list form, iterates through the columns and replaces the NaN values by
    # the mean of the column.
    for i in column:
        try:
            mean_value = round(data[i].mean())
            data[i].fillna(mean_value, inplace = True)
        except:
            raise('Columns need to be numeric type')
            
numeric_missing_value(test_data, ['Age', 'Fare'])


# In[31]:


test_data.head()


# In[32]:


test_data['FamilySize'] = test_data['Parch'] + test_data['SibSp']
# Get dummy variables for sex, pclass, embarked
test_data = pd.get_dummies(data=test_data, columns=['Sex', 'Pclass', 'Embarked'])


# Get the feature columns of interest and assign them to X
X_for_prediction = test_data[['Age', 'Fare', 'FamilySize', 'SibSp', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]


# Get dummy variables for sex, pclass, embarked

# Get the feature columns of interest and assign them to X


# In[33]:


test_data.head()
#X_for_prediction


# In[34]:


y_pred = tree_grid.predict(X_for_prediction)
results = {'PassengerID': test_data['PassengerId'], 'Survived': y_pred}
submission_df = pd.DataFrame(results)

submission_df.to_csv('submission_2.csv')


# In[107]:


# IMPROVEMENTS TO THE MODELS ABOVE

# Focus on the Accuracy score, not AUC score
# Feature Engineering: FamilySize, int_term(PclassFemale), title
# XGBoosting with GridSearchCV
# Get the title of passengers, group them into classes
# For KNN perform scaling for the feature variables that have the biggest range, standartize them
# Produce a holdout data set, perform the classification methods above and test on the holdout set. This will be an 
# indicator on performance on the test.csv data


# In[ ]:




