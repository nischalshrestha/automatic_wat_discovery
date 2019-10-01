#!/usr/bin/env python
# coding: utf-8

# 1.1 Setup and Load Dataset
# -------------------------

# In[ ]:


# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;


# In[ ]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')



# Store our passenger ID for easy access
PassengerId = test['PassengerId']

#Check to see what we have
train.head(3)


# 2.1 Preprocess Data into One Normalized Set
# ----------

# In[ ]:


trainONS = train.copy()
testONS = test.copy()

full_data = [trainONS, testONS]
drop_elementsONS = list()
#preprocessing will normalize and digitize all possible information into a number between 0 and 1
#this should allow use to use the data in any type of model (specifically neural nets)


#Change Pclass to numbers between 0 and 1 
for dataset in full_data:
    dataset['Class1'] = dataset["Pclass"].apply(lambda x: 0 if x != 1 else 1)
    dataset['Class2'] = dataset["Pclass"].apply(lambda x: 0 if x != 2 else 1)
    dataset['Class3'] = dataset["Pclass"].apply(lambda x: 0 if x != 3 else 1)

drop_elementsONS.append('Pclass')






# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

#Must get the title and parse
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', ''], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    dataset['isMrTitle'] = dataset["Title"].apply(lambda x: 0 if x != 'Mr' else 1)
    dataset['isMissTitle'] = dataset["Title"].apply(lambda x: 0 if x != 'Miss' else 1)
    dataset['isMrsTitle'] = dataset["Title"].apply(lambda x: 0 if x != 'Mrs' else 1)
    dataset['isRareTitle'] = dataset["Title"].apply(lambda x: 0 if x != 'Rare' else 1)

drop_elementsONS.append('Title')
drop_elementsONS.append('Name')






#Make gender into numbers 0 == female, 1 == male (not sexist)
for dataset in full_data:
    dataset['Gender'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    
drop_elementsONS.append('Sex')






    
#Find family information
for dataset in full_data:
    #todo Normalize
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    #Maybe try getting if parents or children

    
#Embarked parsing
for dataset in full_data:
    dataset['EmbarkedS'] = dataset["Embarked"].apply(lambda x: 0 if x != 'S' else 1)
    dataset['EmbarkedC'] = dataset["Embarked"].apply(lambda x: 0 if x != 'C' else 1)
    dataset['EmbarkedQ'] = dataset["Embarked"].apply(lambda x: 0 if x != 'Q' else 1)
    
drop_elementsONS.append('Embarked')  




#Find Per Person Fare using the tickets
combinedData = trainONS.copy().append(testONS.copy(), ignore_index=True)
for dataset in full_data:
    dataset['PerPersonFare'] = dataset['Fare']
    for rowIndex in range(0, dataset['PassengerId'].size):
        selectedTicket = dataset['Ticket'][rowIndex]
        numOfTickets = combinedData['Ticket'].value_counts()[selectedTicket]
        dataset.set_value(rowIndex, 'PerPersonFare', dataset['Fare'][rowIndex]/numOfTickets)
        #people per ticket should pretty much be family size
        dataset.set_value(rowIndex, 'PeoplePerTicket', numOfTickets)
        
        if dataset['Fare'][rowIndex] != dataset['Fare'][rowIndex]:
            if dataset['Class1'][rowIndex] == 1:
                dataset.set_value(rowIndex, 'PerPersonFare', 26)
            elif dataset['Class2'][rowIndex] == 1:
                dataset.set_value(rowIndex, 'PerPersonFare', 13)
            elif dataset['Class3'][rowIndex] == 1:
                dataset.set_value(rowIndex, 'PerPersonFare', 7)
            print(dataset['PerPersonFare'][rowIndex])
        #todo Normalize
        
        
drop_elementsONS.append('Fare')        
drop_elementsONS.append('Ticket')  
#drop_elements.append('NumOfTickets')  




#Age editing
for dataset in full_data:
    dataset['EditedAge'] = dataset["Age"].apply(lambda x: 0 if x != x else x)
    dataset['hasAge'] = dataset["Age"].apply(lambda x: 0 if x != x else 1)
    #todo Normalize
    #can also check to see if parch == 2 and People per ticket != 4 if that is case than probably a child

drop_elementsONS.append('Age')  


# Define function to extract cabin number
def get_cabin_num(name):
    if type(name) != float:
        cabin_search = re.search('([0-9]+)', name)
        
        if cabin_search:
            
            return cabin_search.group(0)
    return "0"

def get_cabin_letter(name):
    if type(name) != float:
        cabin_search = re.search('[A-G][0-9]{1,4}', name)
        if cabin_search:
            cabin_search = re.search('[A-G]', cabin_search.group(0))
            if cabin_search:
                return cabin_search.group(0)
    return ""

for dataset in full_data:
    dataset['CabinNum'] = dataset['Cabin'].apply(get_cabin_num)
    dataset['isCabinLetters'] = dataset['Cabin'].apply(get_cabin_letter)
    dataset['isCabinA'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'A' else 1)
    dataset['isCabinB'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'B' else 1)
    dataset['isCabinC'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'C' else 1)
    dataset['isCabinD'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'D' else 1)
    dataset['isCabinE'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'E' else 1)
    dataset['isCabinF'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'F' else 1)
    dataset['isCabinG'] = dataset['isCabinLetters'].apply(lambda x: 0 if x != 'G' else 1)

    
drop_elementsONS.append('Cabin')     
drop_elementsONS.append('isCabinLetters')  

trainONS = trainONS.drop(drop_elementsONS, axis = 1)
testONS = testONS.drop(drop_elementsONS, axis = 1)

testONS.head(500)
#train['perPersonFare'] = train['Fare']/train['Ticket'].value_counts();
#train[['editedFare', 'Fare']].head(500)
#train = train.drop(drop_elements, axis = 1)


# ## 2.2 Get Age Ranges ##

# In[ ]:



trainONSWAR = trainONS.copy()
testONSWAR = trainONS.copy()

full_data = [trainONSWAR, testONSWAR]
drop_elementsONSWAR = list()

for dataset in full_data:
    for rowIndex in range(0, dataset['PassengerId'].size):
        dataset.set_value(rowIndex, 'Age0to4', 0)
        dataset.set_value(rowIndex, 'Age5to10', 0)
        dataset.set_value(rowIndex, 'Age11to13', 0)
        dataset.set_value(rowIndex, 'Age14to18', 0)
        dataset.set_value(rowIndex, 'Age19to22', 0)
        dataset.set_value(rowIndex, 'Age23to29', 0)
        dataset.set_value(rowIndex, 'Age30to42', 0)
        dataset.set_value(rowIndex, 'Age43to55', 0)
        dataset.set_value(rowIndex, 'Age56to65', 0)
        dataset.set_value(rowIndex, 'Age66up', 0)
        if dataset['hasAge'][rowIndex] == 1:
            if dataset['EditedAge'][rowIndex] <= 4:
                dataset.set_value(rowIndex, 'Age0to4', 1)
            elif dataset['EditedAge'][rowIndex] <= 10:
                dataset.set_value(rowIndex, 'Age5to10', 1)
            elif dataset['EditedAge'][rowIndex] <= 13:
                dataset.set_value(rowIndex, 'Age11to13', 1)
            elif dataset['EditedAge'][rowIndex] <= 18:
                dataset.set_value(rowIndex, 'Age14to18', 1)
            elif dataset['EditedAge'][rowIndex] <= 22:
                dataset.set_value(rowIndex, 'Age19to22', 1)
            elif dataset['EditedAge'][rowIndex] <= 29:
                dataset.set_value(rowIndex, 'Age23to29', 1)
            elif dataset['EditedAge'][rowIndex] <= 42:
                dataset.set_value(rowIndex, 'Age30to42', 1)
            elif dataset['EditedAge'][rowIndex] <= 55:
                dataset.set_value(rowIndex, 'Age43to55', 1)
            elif dataset['EditedAge'][rowIndex] <= 65:
                dataset.set_value(rowIndex, 'Age56to65', 1)
            elif dataset['EditedAge'][rowIndex] <= 120:
                dataset.set_value(rowIndex, 'Age66up', 1)
                
drop_elementsONSWAR.append('hasAge')
drop_elementsONSWAR.append('EditedAge')

trainONSWAR = trainONSWAR.drop(drop_elementsONSWAR, axis = 1)
testONSWAR = testONSWAR.drop(drop_elementsONSWAR, axis = 1)

trainONSWAR.head(3)


# ## 2.3 Get No Missing Data Set ##

# In[ ]:


trainNMD = trainONS.copy()
testNMD = trainONS.copy()

drop_elementsNMD = list()


drop_elementsNMD.append('hasAge')
drop_elementsNMD.append('EditedAge')
drop_elementsNMD.append('CabinNum')
drop_elementsNMD.append('isCabinA')
drop_elementsNMD.append('isCabinB')
drop_elementsNMD.append('isCabinC')
drop_elementsNMD.append('isCabinD')
drop_elementsNMD.append('isCabinE')
drop_elementsNMD.append('isCabinF')
drop_elementsNMD.append('isCabinG')
drop_elementsNMD.append('EmbarkedC')
drop_elementsNMD.append('EmbarkedS')
drop_elementsNMD.append('EmbarkedQ')


trainNMD = trainNMD.drop(drop_elementsNMD, axis = 1)
testNMD = testNMD.drop(drop_elementsNMD, axis = 1)

trainNMD.head(3)


# ## 2.4 Get Missing Data Set ##

# In[ ]:





# ## 3.1 Build Models ##

# In[ ]:


# Some useful parameters which will come in handy later on
ntrainONS = trainONS.shape[0]
ntestONS = testONS.shape[0]

ntrainONSWAR = trainONSWAR.shape[0]
ntestONSWAR = testONSWAR.shape[0]

ntrainNMD = trainNMD.shape[0]
ntestNMD = testNMD.shape[0]


SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kfONS = KFold(ntrainONS, n_folds= NFOLDS, random_state=SEED)
kfONSWAR = KFold(ntrainONSWAR, n_folds= NFOLDS, random_state=SEED)
kfNMD = KFold(ntrainNMD, n_folds= NFOLDS, random_state=SEED)


# In[ ]:


# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer


# In[ ]:


def get_oofONS(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrainONS,))
    oof_test = np.zeros((ntestONS,))
    oof_test_skf = np.empty((NFOLDS, ntestONS))

    for i, (train_index, test_index) in enumerate(kfONS):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def get_oofONSWAR(clf, x_train, y_train, x_test):
    oof_train2 = np.zeros((ntrainONSWAR,))
    oof_test2 = np.zeros((ntestONSWAR,))
    oof_test_skf2 = np.empty((NFOLDS, ntestONSWAR))

    for i, (train_index, test_index) in enumerate(kfONSWAR):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train2[test_index] = clf.predict(x_te)
        oof_test_skf2[i, :] = clf.predict(x_test)

    oof_test2[:] = oof_test_skf2.mean(axis=0)
    return oof_train2.reshape(-1, 1), oof_test2.reshape(-1, 1)

def get_oofNMD(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrainNMD,))
    oof_test = np.zeros((ntestNMD,))
    oof_test_skf = np.empty((NFOLDS, ntestNMD))

    for i, (train_index, test_index) in enumerate(kfNMD):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[ ]:


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

rfwar = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
etwar = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
adawar = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gbwar = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svcwar = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

rfnmd = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
etnmd = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
adanmd = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gbnmd = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svcnmd = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[ ]:


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_trainONS = trainONS['Survived'].ravel()
trainONS = trainONS.drop(['Survived'], axis=1)
x_trainONS = trainONS.values # Creates an array of the train data
x_testONS = testONS.values # Creats an array of the test data

y_trainONSWAR = trainONSWAR['Survived'].ravel()
yytrainONSWAR = trainONSWAR.drop(['Survived'], axis=1)
x_trainONSWAR = trainONSWAR.values # Creates an array of the train data
x_testONSWAR = testONSWAR.values # Creats an array of the test data

y_trainNMD = trainNMD['Survived'].ravel()
yytrainNMD = trainNMD.drop(['Survived'], axis=1)
x_trainNMD = trainNMD.values # Creates an array of the train data
x_testNMD = testNMD.values # Creats an array of the test data


# In[ ]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_trainONS, et_oof_testONS = get_oofONS(et, x_trainONS, y_trainONS, x_testONS) # Extra Trees
rf_oof_trainONS, rf_oof_testONS = get_oofONS(rf,x_trainONS, y_trainONS, x_testONS) # Random Forest
ada_oof_trainONS, ada_oof_testONS = get_oofONS(ada, x_trainONS, y_trainONS, x_testONS) # AdaBoost 
gb_oof_trainONS, gb_oof_testONS = get_oofONS(gb,x_trainONS, y_trainONS, x_testONS) # Gradient Boost
svc_oof_trainONS, svc_oof_testONS = get_oofONS(svc,x_trainONS, y_trainONS, x_testONS) # Support Vector Classifier


et_oof_trainONSWAR, et_oof_testONSWAR = get_oofONSWAR(etwar, x_trainONSWAR,
                                                      y_trainONSWAR, x_testONSWAR) # Extra Trees
rf_oof_trainONSWAR, rf_oof_testONSWAR = get_oofONSWAR(rfwar,x_trainONSWAR,
                                                      y_trainONSWAR, x_testONSWAR) # Random Forest
ada_oof_trainONSWAR, ada_oof_testONSWAR = get_oofONSWAR(adawar, x_trainONSWAR,
                                                        y_trainONSWAR, x_testONSWAR) # AdaBoost 
gb_oof_trainONSWAR, gb_oof_testONSWAR = get_oofONSWAR(gbwar,x_trainONSWAR,
                                                      y_trainONSWAR, x_testONSWAR) # Gradient Boost
svc_oof_trainONSWAR, svc_oof_testONSWAR = get_oofONSWAR(svcwar,x_trainONSWAR,
                                                        y_trainONSWAR, x_testONSWAR) # Support Vector Classifier


et_oof_trainNMD, et_oof_testNMD = get_oofNMD(etnmd, x_trainNMD, y_trainNMD, x_testNMD) # Extra Trees
rf_oof_trainNMD, rf_oof_testNMD = get_oofNMD(rfnmd,x_trainNMD, y_trainNMD, x_testNMD) # Random Forest
ada_oof_trainNMD, ada_oof_testNMD = get_oofNMD(adanmd, x_trainNMD, y_trainNMD, x_testNMD) # AdaBoost 
gb_oof_trainNMD, gb_oof_testNMD = get_oofNMD(gbnmd,x_trainNMD, y_trainNMD, x_testNMD) # Gradient Boost
svc_oof_trainNMD, svc_oof_testNMD = get_oofNMD(svcnmd,x_trainNMD, y_trainNMD, x_testNMD) # Support Vector Classifier


print("Training is complete") 


# In[ ]:


rf_featureONS = rf.feature_importances(x_trainONS,y_trainONS)
et_featureONS = et.feature_importances(x_trainONS, y_trainONS)
ada_featureONS = ada.feature_importances(x_trainONS, y_trainONS)
gb_featureONS = gb.feature_importances(x_trainONS, y_trainONS)

rf_featureONSWAR = rf.feature_importances(x_trainONSWAR, y_trainONSWAR)
et_featureONSWAR = et.feature_importances(x_trainONSWAR, y_trainONSWAR)
ada_featureONSWAR = ada.feature_importances(x_trainONSWAR, y_trainONSWAR)
gb_featureONSWAR = gb.feature_importances(x_trainONSWAR, y_trainONSWAR)

rf_featureNMD = rf.feature_importances(x_trainNMD, y_trainNMD)
et_featureNMD = et.feature_importances(x_trainNMD, y_trainNMD)
ada_featureNMD = ada.feature_importances(x_trainNMD, y_trainNMD)
gb_featureNMD = gb.feature_importances(x_trainNMD, y_trainNMD)


# In[ ]:


base_predictions_trainONS = pd.DataFrame( {'RandomForest': rf_oof_trainONS.ravel(),
     'ExtraTrees': et_oof_trainONS.ravel(),
     'AdaBoost': ada_oof_trainONS.ravel(),
      'GradientBoost': gb_oof_trainONS.ravel()
    })
base_predictions_trainONS.head()


# In[ ]:


x_trainONS = np.concatenate(( et_oof_trainONS, rf_oof_trainONS, ada_oof_trainONS,
                          gb_oof_trainONS, svc_oof_trainONS), axis=1)
x_testONS = np.concatenate(( et_oof_testONS, rf_oof_testONS, ada_oof_testONS,
                         gb_oof_testONS, svc_oof_testONS), axis=1)

x_trainONSWAR = np.concatenate(( et_oof_trainONSWAR, rf_oof_trainONSWAR, ada_oof_trainONSWAR,
                          gb_oof_trainONSWAR, svc_oof_trainONSWAR), axis=1)
x_testONSWAR = np.concatenate(( et_oof_testONSWAR, rf_oof_testWAR, ada_oof_testWAR,
                         gb_oof_testWAR, svc_oof_testWAR), axis=1)

x_trainNMD = np.concatenate(( et_oof_trainNMD, rf_oof_trainNMD, ada_oof_trainNMD,
                          gb_oof_trainNMD, svc_oof_trainNMD), axis=1)
x_testNMD = np.concatenate(( et_oof_testNMD, rf_oof_testNMD, ada_oof_testNMD,
                         gb_oof_testNMD, svc_oof_testNMD), axis=1)


# In[ ]:


gbmONS = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_trainONS, y_trainONS)
predictionsONS = gbmONS.predict(x_testONS)

gbmONSWAR = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_trainONSWAR, y_trainONSWAR)
predictionsONSWAR = gbmONSWAR.predict(x_testONSWAR)

gbmNMD = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_trainNMD, y_trainNMD)
predictionsNMD = gbmNMD.predict(x_testNMD)


# In[ ]:


# Generate Submission File 
StackingSubmissionONS = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictionsONS })
StackingSubmissionONS.to_csv("StackingSubmissionONS.csv", index=False)

# Generate Submission File 
StackingSubmissionONSWAR = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictionsONSWAR })
StackingSubmissionONSWAR.to_csv("StackingSubmissionONSWAR.csv", index=False)

# Generate Submission File 
StackingSubmissionNMD = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictionsNMD })
StackingSubmissionNMD.to_csv("StackingSubmissionNMD.csv", index=False)


