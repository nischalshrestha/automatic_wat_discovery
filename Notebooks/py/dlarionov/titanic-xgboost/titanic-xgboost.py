#!/usr/bin/env python
# coding: utf-8

# # Loading data

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import model_selection


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = pd.concat([train, test])

train_size = train.shape[0] # 891
test_size = test.shape[0] # 418

del train
del test

# data[888:894]


# # Feature engineering

# In[3]:


lb = LabelEncoder()


# In[4]:


def correlation_by(feature):
    return data[[feature, 'Survived']].groupby([feature]).agg(['count','mean'])


# In[5]:


data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[6]:


age_ref = data.groupby('Title').Age.mean()
data = data.assign(
    Age = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
)


# In[7]:


data['AgeBand'] = pd.cut(data['Age'], 5, labels=range(5)).astype(int)
correlation_by('AgeBand')


# In[8]:


data['Title'] = data['Title'].replace(['Don', 'Capt', 'Col', 'Major', 'Sir', 'Jonkheer', 'Rev', 'Dr'], 'Honor')
data['Title'] = data['Title'].replace(['Lady', 'Dona', 'Mme', 'Countess'], 'Mrs')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')

data['TitleCode'] = lb.fit_transform(data['Title'])
correlation_by('TitleCode')


# In[9]:


data['SexCode'] = lb.fit_transform(data['Sex'])
correlation_by('SexCode')


# In[10]:


correlation_by('Pclass')


# In[11]:


data['Fare'] = data['Fare'].fillna(data['Fare'].median())


# In[12]:


data['FareBand'] = 0
data.loc[(data.Fare > 0) & (data.Fare <= 7.91), 'FareBand'] = 1
data.loc[(data.Fare > 7.91) & (data.Fare <= 14.454), 'FareBand'] = 2
data.loc[(data.Fare > 14.454) & (data.Fare <= 31), 'FareBand'] = 3
data.loc[data.Fare > 31, 'FareBand'] = 4
correlation_by('FareBand')


# In[13]:


data['EmbarkedCode'] = data['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2})
correlation_by('EmbarkedCode')


# In[14]:


data['Deck'] = data['Cabin'].str.slice(0,1)
data['DeckCode'] = (data['Deck']
                        .map({
                            'C':1, 
                            'E':2, 
                            'G':3,
                            'D':4, 
                            'A':5, 
                            'B':6, 
                            'F':7, 
                            #'T':8 to rare
                        })
                        .fillna(0)
                        .astype(int))
correlation_by('DeckCode')


# In[15]:


data['Room'] = (data['Cabin']
                    .str.slice(1,5).str.extract('([0-9]+)', expand=False)
                    .fillna(0)
                    .astype(int))

data['RoomBand'] = pd.cut(data['Room'], 5, labels=range(5)).astype(int)
correlation_by('RoomBand')


# In[16]:


data['FamilySize'] = (data['SibSp'] + data['Parch']).astype(int)
correlation_by('FamilySize')


# In[17]:


data['FamilySizeBand'] = 0
data.loc[(data.FamilySize > 0) & (data.FamilySize <= 2), 'FamilySizeBand'] = 1
data.loc[(data.FamilySize > 2) & (data.FamilySize <= 4), 'FamilySizeBand'] = 2
data.loc[data.FamilySize > 4, 'FamilySizeBand'] = 3
correlation_by('FamilySizeBand')


# In[18]:


data['IsAlone'] = (data['SibSp'] + data['Parch'] == 0).astype(int)
correlation_by('IsAlone')


# # Feature correlations

# In[19]:


features = data[:train_size][[
    'Survived',
    'Pclass',
    'SexCode',
    'FamilySize',
    'FamilySizeBand',
    'IsAlone',
    'Age',
    'AgeBand',
    'Fare',
    'FareBand',
    'TitleCode',
    'EmbarkedCode',
    'DeckCode',
    'Room',
    'RoomBand']]
features.corr()


# In[20]:


plt.figure(figsize=(16,14))
sns.heatmap(features.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=plt.cm.RdBu, annot=True)


# # Training

# In[ ]:


one_hot_features = [
    #'Pclass',
    #'AgeBand',
    #'FareBand',
    'TitleCode',
    'EmbarkedCode',
    'DeckCode',
]
# data = pd.get_dummies(data, columns = one_hot_features)


# In[21]:


cols = [
    'Pclass',
    'SexCode',
    #'FamilySize',
    #'FamilySizeBand',
    'IsAlone',
    #'Age',
    'AgeBand',
    #'Fare',
    'FareBand',
    'TitleCode',
    'EmbarkedCode',
    'DeckCode',
    'Room',
    #'RoomBand'
]
X_train = data[:train_size][cols]
Y_train = data[:train_size]['Survived'].astype(int)
X_test = data[train_size:][cols]

X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# sclr = StandardScaler()
# X_train = sclr.fit_transform(X_train)
# X_test = sclr.transform(X_test)


# In[22]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)


# In[23]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc.score(X_train, Y_train)


# In[24]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn.score(X_train, Y_train)


# In[25]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
gaussian.score(X_train, Y_train)


# In[26]:


perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
perceptron.score(X_train, Y_train)


# In[27]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
decision_tree.score(X_train, Y_train)


# In[28]:


sgd = SGDClassifier(max_iter=5)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)


# In[29]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
linear_svc.score(X_train, Y_train)


# In[30]:


rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    #'warm_start': True, 
    #'max_features': 0.2,
    #'max_depth': 12,
    #'min_samples_leaf': 2,
    #'max_features' : 'sqrt',
    #'verbose': 0
}

random_forest = RandomForestClassifier(**rf_params)

scores = model_selection.cross_val_score(random_forest, X_train, Y_train, cv=5, scoring='accuracy')
print("Kfold: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))

random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
score = random_forest.score(X_train, Y_train)
print("Out of fold: %0.4f" % score)


# In[33]:


params1 = {
    'n_estimators'      : [50,100],
    'min_samples_split' : [5,10,15,20,25,30],
    'min_samples_leaf'  : [1,3,5],
    'max_depth'         : [5,10,15,20]
}

clf1 = model_selection.GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=params1,
    cv = 5, 
    n_jobs = -1, 
    verbose = 1)
clf1.fit(X_train, Y_train)

print(clf1.best_score_)
print(clf1.best_estimator_)


# In[34]:


params2 = {
    'algorithm': ['auto'], 
    'weights': ['uniform', 'distance'], 
    'leaf_size': [1,5,10,50,100], 
    'n_neighbors': [3,4,5,6,7,8,12,22]
}

clf2 = model_selection.GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=params2,
    cv = 5, 
    n_jobs = -1, 
    verbose = 1)
clf2.fit(X_train, Y_train)

print(clf2.best_score_)
print(clf2.best_estimator_)


# In[36]:


clf = clf2
clf.best_estimator_.fit(X_train, Y_train)
# Y_pred = clf.best_estimator_.predict(X_test)
clf.best_estimator_.score(X_train, Y_train)


# # Submission

# In[ ]:


submission = pd.DataFrame({
    "PassengerId": data[train_size:]["PassengerId"], 
    "Survived": Y_pred 
})
submission.to_csv('submission.csv', index=False)

