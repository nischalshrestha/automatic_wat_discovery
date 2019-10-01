#!/usr/bin/env python
# coding: utf-8

# 

# In[148]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold


# In[149]:


#Print you can execute arbitrary python code
train_df = pd.read_csv("../input/train.csv" )
test_df = pd.read_csv("../input/test.csv" )
PassengerId = test_df['PassengerId']
combine = [train_df, test_df]


# In[150]:


train_df = train_df.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Ms')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Ms')
    dataset['Title'] = dataset['Title'].replace('Miss', 'Mrs')

title_mapping = {"Mr": 1, "Ms": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)  

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
    
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset.loc[ dataset['FamilySize'] <= 1, 'FamilySize'] = 0
    dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 'FamilySize'] = 1
    dataset.loc[(dataset['FamilySize'] > 4) & (dataset['FamilySize'] <= 5), 'FamilySize']   = 2
    dataset.loc[ dataset['FamilySize'] > 5, 'FamilySize'] = 3
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass    
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)    
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
combine = [train_df, test_df]     
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Age*Fare']=dataset.Age * dataset.Fare
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]  


# In[151]:


train_df.head()
test_df.head()
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[152]:


corr = train_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, vmax=1, annot=True, square=True)
plt.title('feature correlations')


# In[153]:


#Set parameters for ensembling
ntrain = train_df.shape[0]
ntest = test_df.shape[0]
seed = 10
nfolds = 5
kf = KFold(ntrain, n_folds = nfolds, random_state=seed)


# In[154]:


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


# In[155]:


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


# In[157]:


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


# In[105]:


#Create models
rf = SklearnHandler(clf=RandomForestClassifier, seed=seed, params=rf_params)
et = SklearnHandler(clf=ExtraTreesClassifier, seed=seed, params=et_params)
ada = SklearnHandler(clf=AdaBoostClassifier, seed=seed, params=ada_params)
gb = SklearnHandler(clf=GradientBoostingClassifier, seed=seed, params=gb_params)
svc = SklearnHandler(clf=SVC, seed=seed, params=svc_params)


# In[158]:


#Create arrays for the models
y_train = train_df['Survived'].ravel()
train_df = train_df.drop(['Survived'], axis=1)
x_train = train_df.values
x_test = test_df.values 


# In[159]:


#Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


# In[160]:


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# In[161]:


cols = train_df.columns.values
#Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature,                            
    })


# In[73]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['Random Forest feature importances'])


# In[74]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['Extra Trees  feature importances'])


# In[24]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['AdaBoost feature importances'])


# In[25]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['Gradient Boost feature importances'])


# In[162]:


#Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe


# In[163]:


plt.figure(figsize=(12,8))
sns.barplot(feature_dataframe['features'], feature_dataframe['mean'])


# In[164]:


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
    })
base_predictions_train.head()


# In[165]:


corr = base_predictions_train.astype(float).corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, vmax=1, annot=True, square=True)
plt.title('feature correlations')


# In[166]:


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


# In[81]:


# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

