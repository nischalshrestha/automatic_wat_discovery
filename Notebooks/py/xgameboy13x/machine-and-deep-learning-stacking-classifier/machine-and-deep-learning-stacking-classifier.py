#!/usr/bin/env python
# coding: utf-8

# **Titanic: Using Machine and Deep Learning**
# 
# This is my attempt to create good classifier to predict survival rate of passengers. I was inspired by Anisotropic and his notebook(link bellow). 
# 
# In this notebook I will use scikit-learn and Keras(tensorflow backend) libraries to do this task :) .
# 
# [Introduction to ensembling/stacking in python][1]
# 
# 
#   [1]: https://www.kaggle.com/arthurtok/titanic/introduction-to-ensembling-stacking-in-python

# First step as usual in Python.

# In[ ]:


# Import libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cross_validation import KFold
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM,                             BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier


# In[ ]:


# From here starts the Anisotropic code
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
PassengerId = test['PassengerId']


# In[ ]:


train.head()


# In[ ]:


full_data = [train, test]

# Some features of my own that I have added in
# Gives the length of the name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)
# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train['CategoricalAge'] = pd.cut(train['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] ;


# In[ ]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)


# In[ ]:


train.head()


# In[ ]:


# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 42 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED, shuffle=True)


# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        test = clf.predict(x_test)
        test = test.flatten() #  -> HERE i need to flatten because NN return (nbsamples,1) shape predicts
        oof_test_skf[i, :] = test

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values


# In[ ]:


x_train.shape


# In[ ]:


"""This is main code for building neural network. I had to use Sequential layer. I will explain cell lower.
   In this network I use activation function 'tanh' except last layer which is 'sigmoid'.
   Feel free to experiment with this function."""
def create_NN(shape):
    model = Sequential()

    model.add(Dense(16, init='he_uniform', input_shape=(shape,)))
    model.add(Activation('tanh'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(8, init='he_normal'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    
    opt = Adam()    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[ ]:


# Classifiers
"""You can 'wrap' your neural network to use scikit-learn function on it. 
   To make this work you need to create NN with Sequential layer because it has method 'predict classes'.
   Without this method your classifier net won't work."""

et_clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=500, warm_start=True,
                                max_depth=6, min_samples_leaf=2, random_state=SEED)

dt_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, max_features='sqrt',
                                random_state=SEED)

ada_clf = AdaBoostClassifier(n_estimators=500, learning_rate=0.5, random_state=SEED)

"""If you want to check the accuracy or loss change verbose value to 1 o 2, or 3"""
est_clf = KerasClassifier(build_fn=create_NN, nb_epoch=100, 
                          batch_size=5, verbose=0, shape=x_train.shape[1])

svc_clf = SVC(C=0.5, random_state=SEED)


# In[ ]:


et_oof_train, et_oof_test = get_oof(et_clf, x_train, y_train, x_test)
dt_oof_train, dt_oof_test = get_oof(dt_clf, x_train, y_train, x_test)
ada_oof_train, ada_oof_test = get_oof(ada_clf, x_train, y_train, x_test)
est_oof_train, est_oof_test = get_oof(est_clf, x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc_clf,x_train, y_train, x_test)

print("Finished!!!")


# In[ ]:


est_pred = est_clf.predict(x_test)


# In[ ]:


base_predictions_train = pd.DataFrame( {'DecisionTree': dt_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
     'Neural Network': est_oof_train.ravel(),
     'SVC': svc_oof_train.ravel()
    })
base_predictions_train.head()


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
sns.heatmap(base_predictions_train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)


# In[ ]:


x_train = np.concatenate((et_oof_train, dt_oof_train, ada_oof_train, est_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, dt_oof_test, ada_oof_test, est_oof_test, svc_oof_test), axis=1)


# In[ ]:


x_train.shape[0]


# In[ ]:


"""You can use the same function to build another neural network. 
   You can make fucntion more elastic using keyword arguments in definition.
   After that you can pass kwargs valuse into KerasClassifier"""
estimator = KerasClassifier(build_fn=create_NN, nb_epoch=100, 
                          batch_size=5, verbose=2, shape=x_train.shape[1])


# In[ ]:


estimator.fit(x_train, y_train)


# In[ ]:


net_pred = estimator.predict(x_test)


# In[ ]:


gbm = xgb.XGBClassifier(
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
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[ ]:


net_pred.shape


# In[ ]:


predictions.shape


# In[ ]:


net_pred = net_pred.reshape(net_pred.shape[0],)


# In[ ]:


StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)


# In[ ]:


StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': net_pred})
StackingSubmission.to_csv("NetStackingSubmission.csv", index=False)


# In[ ]:


est_pred = est_pred.flatten()
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': est_pred})
StackingSubmission.to_csv("EstStackingSubmission.csv", index=False)


# **For now one**
# That is it right now. Later(if I will have some time) I will try to get higher accuracy score by doing e.g. feature extraction or dimensionality reduction(hope it will work :| ).

# In[ ]:




