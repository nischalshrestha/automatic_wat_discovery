#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


# In[ ]:


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
PassengerId = test['PassengerId']
combine = [train, test]


# In[ ]:


#Method for finding substrings
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan
    


# In[ ]:


#Mappings
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}


# In[ ]:


#Passenger Class is 3, so fill nan row with the mode
fare_mode = test[test['Pclass']==3]['Fare'].mode()
test['Fare'] = test['Fare'].fillna(fare_mode[0])


# In[ ]:


#Find the mode of embarked of passengers with the same class and similar fare
emb_mode = train[(train['Pclass']==1)&(train['Fare']<=85)&(train['Fare']>75)]['Embarked'].mode()
train['Embarked'] = train['Embarked'].fillna(emb_mode[0])


# In[ ]:


for df in combine:
    # Convert the male and female groups to integer form
    df["Sex"][df["Sex"] == "male"] = 0
    df["Sex"][df["Sex"] == "female"] = 1
    
    #Map and Create Title Feature
    df['Title'] = df['Name'].astype(str).map(lambda x: substrings_in_string(x, title_list))
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    #Map and Create Deck feature
    df['Deck'] = df['Cabin'].astype(str).map(lambda x: substrings_in_string(x, cabin_list))
    df["Deck"][df["Deck"] == "A"] = 1
    df["Deck"][df["Deck"] == "B"] = 2
    df["Deck"][df["Deck"] == "C"] = 3
    df["Deck"][df["Deck"] == "D"] = 4
    df["Deck"][df["Deck"] == "E"] = 5
    df["Deck"][df["Deck"] == "F"] = 6
    df["Deck"][df["Deck"] == "G"] = 7
    df["Deck"][df["Deck"] == "T"] = 8
    df["Deck"] = df["Deck"].fillna(0)
    
    #Create Family size, Fare per person, and isAlone features
    df['Family_size'] = df['SibSp']+df['Parch']+1
    
    #Create isAlone feature based off family size
    df['isAlone']=0
    df.loc[df['Family_size']==1, 'isAlone'] = 1
    
    # Convert the Embarked classes to integer form
    df["Embarked"][df["Embarked"] == "S"] = 0
    df["Embarked"][df["Embarked"] == "C"] = 1
    df["Embarked"][df["Embarked"] == "Q"] = 2

    #Impute Age based off random numbers in one standard deviation from the mean
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['Age'][np.isnan(df['Age'])] = age_null_random_list
    
    # Mapping Age and removing child feature
    df.loc[ df['Age'] <= 16, 'Age'] 					= 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']                      = 4


# In[ ]:


#Create target feature set
excl = ['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch']
train = train.drop(excl, axis = 1)
test  = test.drop(excl, axis = 1)


# In[ ]:


corr = train.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=1, annot=True, square=True)
plt.title('feature correlations')


# In[ ]:


#Set parameters for ensembling
ntrain = train.shape[0]
ntest = test.shape[0]
seed = 10
nfolds = 5
kf = KFold(ntrain, n_folds = nfolds, random_state=seed)


# In[ ]:


#Sklearn custom class

class SklearnHandler(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        
    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x,y)
    
    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_


# In[ ]:


#Class to get out-of-fold predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((nfolds, ntest))
    
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr, y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test.reshape(-1, 1)


# In[ ]:


#Create parameters for all classifiers
#Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

#Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':1000,
    'max_depth': 9,
    'min_samples_split': 6,
    'min_samples_leaf': 4,
    'verbose': 0
}

#AdaBoost parameters
ada_params = {
    'n_estimators': 1000,
    'learning_rate' : 0.75
}

#Gradient Boosting parameters
gb_params = {
    'n_estimators': 1000,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

#Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[ ]:


#Create models
rf = SklearnHandler(clf=RandomForestClassifier, seed=seed, params=rf_params)
et = SklearnHandler(clf=ExtraTreesClassifier, seed=seed, params=et_params)
ada = SklearnHandler(clf=AdaBoostClassifier, seed=seed, params=ada_params)
gb = SklearnHandler(clf=GradientBoostingClassifier, seed=seed, params=gb_params)
svc = SklearnHandler(clf=SVC, seed=seed, params=svc_params)


# In[ ]:


#Create arrays for the models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values 


# In[ ]:


#Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


# In[ ]:


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# In[ ]:


cols = train.columns.values
#Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['Random Forest feature importances'])


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['Extra Trees  feature importances'])


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['AdaBoost feature importances'])


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['Gradient Boost feature importances'])


# In[ ]:


#Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['mean'])


# In[ ]:


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()


# In[ ]:


corr = base_predictions_train.astype(float).corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, vmax=1, annot=True, square=True)
plt.title('feature correlations')


# In[ ]:


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


# In[ ]:


gbm = xgb.XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[ ]:


# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

