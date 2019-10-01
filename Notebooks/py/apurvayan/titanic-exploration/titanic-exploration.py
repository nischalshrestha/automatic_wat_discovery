#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(5)


# In[ ]:


train.shape


# In[ ]:


train.info()


# ### Check for missing values

# In[ ]:



train.columns[train.isnull().any()].tolist() 


# ### Check for survival average based on passenger class
# 
# ##### Higher the class, more is the chance of servival 

# In[ ]:



print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# ### Check for survival average based on gender
# 
# ##### Survival average for female is more than that of a  male

# In[ ]:



print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())


# * ### Check for survival average based on Embarked

# In[ ]:



print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).count())


# In[ ]:


pd.crosstab(index = train["Survived"],  # Make a crosstab
                              columns="count")


# In[ ]:


full_data = [train, test]


# #### Siblings and Partners

# In[ ]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# #### Check whether the person is travelling alone

# In[ ]:


for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[ ]:


print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# #### Fill missing values in Embarked with mode

# In[ ]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# #### Fill missing values in Age with average age

# In[ ]:


for dataset in full_data:
    avg_age = round(dataset['Age'].mean(),2)
    dataset['Age'][np.isnan(dataset['Age'])] = avg_age
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# #### Fill the missing values in Fare

# In[ ]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# ### Get Titles from Name

# In[ ]:


def get_title(name):
    title_search = name.split(",")[1].split(".")[0].strip()
    if title_search:
        return title_search
    else:
        return ""
    
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[ ]:


print(pd.crosstab(train['Title'], train['Sex']))


# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# ### Data Cleaning

# In[ ]:


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
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

# Feature Selection
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp',                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))






# In[ ]:


train.columns


# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# #### Logistic Regression
# 

# In[ ]:


cols=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
       'IsAlone', 'Title']  
X=train[cols]
Y=train['Survived']


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X, Y)

logreg.score(X, Y)


# #### training 80:20 splitM

# In[ ]:


from sklearn.model_selection import train_test_split
train1, test1 = train_test_split(train, test_size=0.2)


# In[ ]:


X1=train1[cols]
Y1=train1['Survived']

logreg.fit(X1, Y1)
logreg.score(X1, Y1)


# In[ ]:


from sklearn import metrics


X1_test = test1[cols]
Y1_test = test1['Survived']

Y3test_pred = logreg.predict(X1_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X1_test, Y1_test)))


# ### Random forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model =  RandomForestClassifier(random_state=42)


# In[ ]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


from sklearn.grid_search import GridSearchCV
CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
CV_rfc.fit(X1, Y1)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=5, criterion='gini')


# In[ ]:


rfc1.fit(X1, Y1)


# In[ ]:


pred=rfc1.predict(X1_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ",accuracy_score(Y1_test,pred))


# #### Modelling

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, plot_importance 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV


# In[ ]:


clf_rf = RandomForestClassifier()
clf_et = ExtraTreesClassifier()
clf_bc = BaggingClassifier()
clf_ada = AdaBoostClassifier()
clf_dt = DecisionTreeClassifier()
clf_xg = XGBClassifier()
clf_lr = LogisticRegression()
clf_svm = SVC()


# In[ ]:


npX = np.array(X).copy()
npy = np.array(Y).copy()


# In[ ]:


Classifiers = ['RandomForest','ExtraTrees','Bagging','AdaBoost','DecisionTree','XGBoost','LogisticRegression','SVM']
scores = []
models = [clf_rf, clf_et, clf_bc, clf_ada, clf_dt, clf_xg, clf_lr, clf_svm]
for model in models:
    score = cross_val_score(model, npX, npy, scoring = 'accuracy', cv = 10, n_jobs = -1).mean()
    scores.append(score)


# In[ ]:


mode = pd.DataFrame(scores, index = Classifiers, columns = ['score']).sort_values(by = 'score',
             ascending = False)


# In[ ]:


print(mode)


# In[ ]:


parameters_xg = {'max_depth':[3,6,7], 'learning_rate': [0.1,0.2], 'n_estimators': [300,200], 
                 'min_child_weight': [4], 'reg_alpha': [6,0], 'reg_lambda': [1,8],'max_delta_step':[2],
                 'gamma':[0],'seed':[1]}

parameters_svm = {'C':[0.9,0.01],'kernel':['rbf','linear'], 'gamma':[0,0.1,'auto'], 'probability':[True,False],
                  'random_state':[0,7,16],'decision_function_shape':['ovo','ovr'],'degree':[3,4,10]}



# In[ ]:


def grid(model,parameters):
    grid = GridSearchCV(estimator = model, param_grid = parameters, cv = 10, 
                        scoring = 'accuracy')
    grid.fit(npX,npy)
    return grid.best_score_, grid.best_estimator_.get_params()


# In[ ]:


def imp_features(model, model_name, params):
    Model = model(**params)
    Model.fit(npX,npy)
    names = X.columns
    feature = Model.feature_importances_
    important_features = pd.Series(data = feature, index = names,)
    important_features = important_features.sort_values(ascending = True)
    return important_features.plot(kind = 'barh', grid = False,title = model_name)


# In[ ]:


best_score_xg, best_params_xg = grid(clf_xg,parameters_xg)
print(best_score_xg)
imp_features(XGBClassifier, 'XGBoostClassifier', best_params_xg)


# In[ ]:


best_score_svm, best_params_svm = grid(clf_svm, parameters_svm)
print(best_score_svm)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
x = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size = .2)
X_train_reduced = PCA(n_components = 2).fit_transform(X_train)
X_test_reduced  = PCA(n_components=  2).fit_transform(X_test)


# In[ ]:


def boundaries(model, heading, best_params):
    Model = model(**best_params)
    Model.fit(X_train_reduced, y_train)

    X_set, y_set = np.concatenate([X_train_reduced, X_test_reduced], axis = 0), np.concatenate([y_train, y_test], axis = 0)
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    #plt.figure(figsize = [15,16])
    plt.contourf(X1, X2, Model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.5, cmap = ListedColormap(('k', 'blue')))
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    plt.xticks(fontsize = 3)
    plt.yticks(fontsize = 3)


# In[ ]:


fig = plt.figure(figsize=[15,7])


plt.title('XGBClassifier')
boundaries(XGBClassifier,'eXtreme Boosting Classifier', best_params_xg)


# In[ ]:


from mlxtend.plotting import plot_decision_regions
t = np.array(y_train)
t = t.astype(np.integer)
clf_svm = SVC(**best_params_svm)
clf_svm.fit(X_train_reduced,t)
plt.figure(figsize = [15,10])
plot_decision_regions(X_train_reduced, t, clf = clf_svm, hide_spines = False, colors = 'purple,limegreen',
                      markers = ['^','v'])
plt.title('Support Vector Machines')


# In[ ]:


clf_svm = SVC(**best_params_svm)
clf_svm.fit(npX,npy)


# In[ ]:


test.columns


# In[ ]:


test.head()


# In[ ]:


test_df = test[cols]


# In[ ]:


nptest = np.array(test_df)
pred = clf_svm.predict(nptest)


# In[ ]:



predictions = pd.DataFrame(pred, index = test_df.index, columns = ['Survived'])


# In[ ]:


pred_df=pd.DataFrame(test['PassengerId'])
pred_df['Survived']=predictions


# In[ ]:


pred_df.head()


# In[ ]:


pred_df.to_csv('predictions_svm_titanic_final.csv',index=False)


# ### References
# 
# ##### Titanic Data Processing with Python: Jarvis Yang
# ##### mlxtend library: Eike Dehling
# ##### Decision Boundary from scratch: bronson
# ##### Titanic Analysis_Learning to Swim with Python: SarahG
# ##### Titanic Best Working Classfier : by Sina
# 

# In[ ]:




