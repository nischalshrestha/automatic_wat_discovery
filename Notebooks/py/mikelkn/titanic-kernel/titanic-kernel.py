#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as score
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print('The size of the training set: ', train_data.shape)
print('The size of the test set is: ' ,test_data.shape)



# In[ ]:


train_data.head()


# In[ ]:


#CLEANING DATA AND REPLACING MISSING DATA


# In[ ]:


test_data.isnull().sum()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


for x in [train_data, test_data]:
    x['Age'] = x['Age'].fillna (x['Age'].median())
    x['Fare'] = x['Fare'].fillna (x['Fare'].mean())
#lest take a look at the data again
#We will take care of the cabin situation later
test_data.isnull().sum()


# In[ ]:


#lets see what is the percentage of all enbarkes
train_data['Embarked'].value_counts(normalize = True)
#we see that 72 percent of all enkarked were S so we use S
train_data['Embarked'] = train_data['Embarked'].fillna('S')
#Lets take a look at uour data onr more time
train_data.isnull().sum()


# In[ ]:


lb = LabelBinarizer()
for x in [train_data, test_data]:
    x['Sex'] = lb.fit_transform(x['Sex'])


# In[ ]:


input_Embarked = {'S':0, 'Q':1, 'C':2}
train_data['Embarked'] = train_data['Embarked'].map(input_Embarked)
test_data['Embarked'] = test_data['Embarked'].map(input_Embarked)
train_data.tail()


# In[ ]:


train_data = train_data.drop(['Name', 'Ticket','Cabin'], axis = 1)


# In[ ]:


test_data = test_data.drop(['Name', 'Ticket','Cabin'], axis = 1)


# In[ ]:


train_data.dtypes.value_counts()


# In[ ]:


test_data.dtypes.value_counts()


# In[ ]:


train_data['child'] = 0
train_data['child'][train_data['Age'] < 18] = 1
train_data.head()


# In[ ]:


#we do the same for the test data
test_data['child'] = 0
test_data['child'][test_data['Age'] < 18] = 1
test_data.head(5)


# In[ ]:


#  now that my data are all numerical, i can start modeling
#Iwill consider this as a classification problem


# In[ ]:


# I will use comparing the following classifiers 
# Logistic regression, SVC, DECISION TREES, RANDOM FOREST, GRADIENT BOOSTING AND KNN


# In[ ]:


#divide the dtata into train and test datas
X = train_data.drop(['Survived', 'Age', 'PassengerId'], axis = 1)#since i will be suing the childs column instead
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                    y, random_state = 42, test_size = 0.33)

X_train.shape, y_train.shape, X_test.shape, y_test.shape



# In[ ]:


#LOGISTIC REGRESSION
lr_l1 = LogisticRegressionCV(Cs=10, cv =4, penalty='l2',
                             solver = 'liblinear').fit (X_train, y_train)

lr_l1_pred = lr_l1.predict(X_test)
lr_l1_score = lr_l1.score(X_train, y_train)
print("logreg accuracy score = {}".format(lr_l1_score*100))


# In[ ]:


#LinearSVC
linSVC = LinearSVC(penalty='l2', C= 10.0).fit (X_train, y_train)

linSVC_pred = linSVC.predict(X_test)
linSVC_score = linSVC.score(X_train, y_train)
print("linear svc accuracy score = {}".format(linSVC_score*100))


# In[ ]:


#SVC with rbf kernel
rbfSVC = SVC(kernel='rbf', C= 10.0, gamma = 1.0).fit (X_train, y_train)

rbfSVC_pred = rbfSVC.predict(X_test)
rbfSVC_score = rbfSVC.score(X_train, y_train)
print("rbfsvc accuracy score = {}".format(rbfSVC_score*100))


# In[ ]:


#Decision Trees
DTC = DecisionTreeClassifier().fit(X_train, y_train)

DTC_pred = DTC.predict(X_test)
DTC_score = DTC.score(X_train, y_train)
print("Decision tree accuracy score = {}".format(DTC_score*100))


# In[ ]:


#RandomForest
randfr = RandomForestClassifier(n_estimators = 100, max_features= 7).fit(X_train, y_train)

randfr_pred = randfr.predict(X_test)
randfr_score =randfr.score(X_train, y_train)
print("Random forest accuracy score = {}".format(randfr_score*100))


# In[ ]:


#KNN
KNN = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)

KNN_pred = KNN.predict(X_test)
KNN_score = KNN.score(X_train, y_train)
print("Knearestneighbors accuracy score = {}".format(KNN_score*100))


# In[ ]:


#joiing all toether in a table

prediction_models = pd.DataFrame({
                    'Models' : ['Logistic regression', 'LinearSVC', 'rbfSVC', 'decision Trees', 
                                 'Random Forests', 'KnearestNeighbors'],

                    'Model_scores' : [lr_l1_score, linSVC_score, rbfSVC_score, DTC_score,
                                       randfr_score, KNN_score]})

prediction_models.sort_values(by = 'Model_scores', ascending = True)


# In[ ]:


#As we can see above the decion tree has the best score  out of all the model so i will use that to make my final predcition


# In[ ]:


#decison tree tends to overfit so we make some analysis first
#we check the nuber of nodes and the max depth of the tree

DTC.tree_.node_count, DTC.tree_.max_depth


# In[ ]:


##we measure the prediction error on both the training and the test sets

def error_measure (y_true, y_pred, label):
    return pd.Series({'Accuracy' : accuracy_score(y_true, y_pred),
                      'Precision': precision_score(y_true, y_pred),
                      'Recall': recall_score(y_true, y_pred),
                      'F1_score': score(y_true, y_pred),
        
    }, name = label)


# In[ ]:


# We predict for train and test data

y_train_predict = DTC.predict(X_train)
y_test_predict = DTC.predict(X_test)


train_test_error = pd.concat([error_measure(y_train, y_train_predict, 'Train'),
                             error_measure(y_test, y_test_predict, 'Test')], axis = 1)

train_test_error


# In[ ]:


# as we can see the decision predicts very well on the training data but very poorly on the test data
#This is a sign of overfitting. We will remedy this by using grid search cross_validation

#Gridsearchcv will help find the decision tree that perfroms well ont the test data

DT = DecisionTreeClassifier()

param_grid = {'max_depth': range(1, DTC.tree_.max_depth+1),
             'max_features': range(1, len(DTC.feature_importances_)+1)}

GR = GridSearchCV(DT, param_grid = param_grid)

GR = GR.fit(X_train, y_train)

GR.best_params_


# In[ ]:


#Now we determine the numbe of nodes and the depth of the tree

GR.best_estimator_.tree_.node_count, GR.best_estimator_.tree_.max_depth


# In[ ]:


#just a check on random forest
y_train_RF_predict = randfr.predict(X_train)
y_test_RF_predict = randfr.predict(X_test)


train_test_RF_error = pd.concat([error_measure(y_train, y_train_RF_predict, 'Train'),
                             error_measure(y_test, y_test_RF_predict, 'Test')], axis = 1)

train_test_RF_error


# In[ ]:





# In[ ]:


#Waouw see th difference, from (247, 16)to (107, 7)
#NOw we chech the scorings for eachmodels using the gridsearchcv

y_train_GR_predict = GR.predict(X_train)
y_test_GR_predict = GR.predict(X_test)


train_test_GR_error = pd.concat([error_measure(y_train, y_train_GR_predict, 'Train'),
                             error_measure(y_test, y_test_GR_predict, 'Test')], axis = 1)

train_test_GR_error


# In[ ]:


#and we difference between the train and the test data i not large. Meaning this qill probably not overfit the data

# I will use the GRcv to make my prediction

dt = DecisionTreeClassifier()

param_grid = {'max_depth': range(1, GR.best_estimator_.tree_.max_depth+1, 2)}

GRs = GridSearchCV(dt, param_grid = param_grid)


# In[ ]:


prediction_cols = train_data.drop(['Survived', 'Age','PassengerId'], axis = 1)

train_y = train_data.Survived
train_X = prediction_cols

my_model = GRs.fit(train_X, train_y)

pred_cols= test_data.drop(['PassengerId', 'Age'], axis = 1)

prediction = GRs.predict(pred_cols)
prediction.shape


# In[ ]:


#It is always better to use a random forest than a single decision tress so i will go with the random forest

prediction_cols = train_data.drop(['Survived', 'Age','PassengerId'], axis = 1)

train_y = train_data.Survived
train_X = prediction_cols

my_model = randfr.fit(train_X, train_y)

pred_cols= test_data.drop(['PassengerId', 'Age'], axis = 1)

predictionRF = randfr.predict(pred_cols)
predictionRF.shape


# In[ ]:


titanic_submission_classic2 = pd.DataFrame ({"PassengerId": test_data["PassengerId"],
                             "Survived": predictionRF})
titanic_submission_classic2.to_csv('Titanic_Submission_classification_RF.csv', index = False)

titanic_submission_classic2.sample(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




