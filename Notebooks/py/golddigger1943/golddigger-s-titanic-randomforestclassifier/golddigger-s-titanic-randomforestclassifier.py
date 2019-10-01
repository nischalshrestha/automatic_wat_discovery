#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn import preprocessing 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
pd.options.mode.chained_assignment = None
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submit = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


submit.head(5)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


data = train.append(test)
data.reset_index(inplace=True,drop=True)


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


sns.countplot(data['Survived'])


# In[ ]:


sns.countplot(data['Pclass'], hue=data['Survived'])


# In[ ]:


sns.countplot(data['Sex'], hue=data['Survived'])


# In[ ]:


sns.countplot(data['Embarked'], hue=data['Survived'])


# In[ ]:


g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Age', kde=False)


# In[ ]:


g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)


# In[ ]:


g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Parch', kde=False)


# In[ ]:


g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)


# In[ ]:


data['Family_Size'] = data['Parch'] + data['SibSp']
g = sns.FacetGrid(data, col='Survived')
g.map(sns.distplot, 'Family_Size', kde=False)


# In[ ]:


data['Title1'] = data['Name'].str.split(", ", expand=True)[1]
data['Title1'] = data['Title1'].str.split(".", expand=True)[0]
data['Title1'].unique()


# In[ ]:


pd.crosstab(data['Title1'],data['Sex']).T.style.background_gradient(cmap='summer_r')


# In[ ]:


pd.crosstab(data['Title1'],data['Survived']).T.style.background_gradient(cmap='summer_r')


# In[ ]:


data.groupby(['Title1'])['Age'].mean()


# In[ ]:


data.groupby(['Title1','Pclass'])['Age'].mean()


# In[ ]:


data['Title2'] = data['Title1'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
         ['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr','Mrs'])
data['Title2'].unique()


# In[ ]:


data.groupby('Title2')['Age'].mean()


# In[ ]:


data.groupby(['Title2','Pclass'])['Age'].mean()


# In[ ]:


pd.crosstab(data['Title2'],data['Sex']).T.style.background_gradient(cmap='summer_r')


# In[ ]:


pd.crosstab(data['Title2'],data['Survived']).T.style.background_gradient(cmap='summer_r')


# In[ ]:


data.info()


# In[ ]:


data['Ticket_info'] = data['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
data['Ticket_info'].unique()


# In[ ]:


pd.crosstab(data['Ticket_info'],data['Survived']).T.style.background_gradient(cmap='summer_r')


# In[ ]:


data['Embarked'] = data['Embarked'].fillna('S')
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data.info()


# In[ ]:


data["Cabin"] = data['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')
data["Cabin"].unique()


# In[ ]:


sns.countplot(data['Cabin'], hue=data['Survived'])


# In[ ]:


data.describe(include=['O'])


# In[ ]:


data['Sex'] =  data['Sex'].astype('category').cat.codes
data['Embarked'] = data['Embarked'].astype('category').cat.codes
data['Pclass'] = data['Pclass'].astype('category').cat.codes
data['Title1'] = data['Title1'].astype('category').cat.codes
data['Title2'] = data['Title2'].astype('category').cat.codes
data['Cabin'] = data['Cabin'].astype('category').cat.codes
data['Ticket_info'] = data['Ticket_info'].astype('category').cat.codes
data


# In[ ]:


dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|(np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))]
rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])
ageNullValues = rfModel_age.predict(X=dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)
data.info()


# In[ ]:


dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTrain.columns


# In[ ]:


dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTrain


# In[ ]:


rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)


# In[ ]:


pd.concat((pd.DataFrame(dataTrain.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# In[ ]:


features = pd.DataFrame()
features['feature'] = dataTrain.iloc[:, 1:].columns
features['importance'] = rf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(10, 10))


# In[ ]:


#model = SelectFromModel(rf, prefit=True)
#train_reduced = model.transform(dataTrain.iloc[:, 1:])
#train_reduced.shape


# In[ ]:


#test_reduced = model.transform(dataTest)
#test_reduced.shape


# In[ ]:


logreg = LogisticRegression()
svc = SVC()
rf = RandomForestClassifier(n_estimators=100)
gboost = GradientBoostingClassifier()
knc = KNeighborsClassifier(n_neighbors = 3)
gnb = GaussianNB()

models = [logreg,svc,rf,gboost,knc,gnb]

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=3 ,scoring=scoring)
    return np.mean(xval)

for model in models:
    print ('Cross-validation of : {0}', format(model.__class__))
    score = compute_score(clf=model, X=dataTrain.iloc[:, 1:], y=dataTrain.iloc[:, 0], scoring='accuracy')
    print ('CV score = {0}',format(score))
    print('****')


# In[ ]:


# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
# Choose some parameter combinations to try
    parameter_grid = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
              }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


rf_res =  model.predict(dataTest)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)
submit

