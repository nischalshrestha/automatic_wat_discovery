#!/usr/bin/env python
# coding: utf-8

# # Titanic kaggle dataset

# ### Import libs

# In[ ]:


from sklearn import metrics, cross_validation, grid_search, linear_model

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().magic(u'pylab inline')


# ## Import data

# In[ ]:


data = pd.read_csv("../input/train.csv", header = 0, sep = ',')


# ## Inspect data

# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# ## Visualization

# In[ ]:


sns.set(font_scale=1)
pd.options.display.mpl_style = 'default'
data.drop(['PassengerId', 'Survived', 'Pclass'], axis=1).hist(figsize=(10, 7), grid=False)
plt.show()


# In[ ]:


plt.figure()

plt.subplot(221)
data.Pclass.value_counts().plot(kind='bar', figsize=(10, 10))
plt.xlabel("Passenger class")
plt.ylabel("Count")
plt.title("Passenger class distribution")

plt.subplot(222)
data.Embarked.value_counts().plot(kind='bar', figsize=(10, 10))
plt.xlabel("Emabarked")
plt.ylabel("Count")
plt.title("Embarked distribution")
plt.show()


# In[ ]:


plt.figure(1)

plt.subplots(1, 1, figsize=(10, 10))
plt.subplot(221)
sns.barplot(y='Survived', x='Pclass', data=data)
plt.title("Survived by passenger class")

plt.subplot(222)
sns.barplot(y='Survived', x='Embarked', data=data)
plt.title("Survived by Embarked")
plt.show()


# In[ ]:


sns.barplot(y='Survived', x="Sex", data=data)
plt.title("Male/female survived distribution")
plt.ylabel("Survived")
plt.show()


# In[ ]:


plt.figure(1)

plt.subplots(1, 1, figsize=(10, 10))

plt.subplot(221)
ax = data[data.Survived == 1].Age.plot(kind='hist', alpha=0.5)
ax = data[data.Survived == 0].Age.plot(kind='hist', alpha=0.5)
plt.title("Age distribution")
plt.xlabel("Age")
plt.legend(("survived", "not survived"), loc='best')

plt.subplot(222)
data.Age.plot(kind='kde', grid=False)
plt.title("Age distribution")
plt.xlabel("Age")
plt.xlim((0,80))
plt.show()


# In[ ]:


corr = data.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr, square=True)
plt.title("Feature correlations")


# # Preprocessing

# In[ ]:


t_data = data.drop(['Cabin', 'Ticket', 'PassengerId', 'Survived'], axis=1)
t_labels = data['Survived']


# In[ ]:


t_data.head()


# ## Name inspecting/processing

# In[ ]:


t_data['Name_pred'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


pd.crosstab(t_data['Name_pred'], t_data['Sex'])


# In[ ]:


t_data['Name_pred'] = t_data['Name_pred'].replace("Mlle", "Miss")
t_data['Name_pred'] = t_data['Name_pred'].replace("Ms", "Miss")
t_data['Name_pred'] = t_data['Name_pred'].replace("Mme", "Mrs")


# In[ ]:


t_data['Name_pred'] = t_data['Name_pred'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer',                                                  'Lady', 'Major', 'Rev', 'Sir'], 'Other')


# In[ ]:


preds = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}

t_data['Name_pred'] = t_data['Name_pred'].map(preds)


# In[ ]:


t_data = t_data.drop('Name', axis=1)


# In[ ]:


t_data.head()


# ## Some categorical transformations
# (Not really necessary)

# In[ ]:


t_data['Sex'] = t_data['Sex'].apply(lambda x: int(x == 'male'))


# In[ ]:


t_data.Embarked = t_data.Embarked.fillna(value='S')


# In[ ]:


emb = { 'S': 1, 'C': 2, 'Q': 3}


# In[ ]:


t_data.Embarked = t_data.Embarked.map(emb)


# In[ ]:


# zeros as first try
t_data.Age = t_data.Age.fillna(value=0)


# In[ ]:


t_data.head()


# ## Dividing by feature type

# In[ ]:


real_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = list(set(t_data.columns.values.tolist()) - set(real_cols))


# In[ ]:


X_real = t_data[real_cols]
X_cat = t_data[cat_cols]


# ## Categorical features encoding

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer


# In[ ]:


encoder = OneHotEncoder(categorical_features='all', sparse=True, n_values='auto')


# In[ ]:


X_cat.head()


# In[ ]:


X_cat_oh = encoder.fit_transform(X_cat).toarray()


# ## Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()

X_real_scaled = scaler.fit_transform(X_real)


# ## Stacking

# In[ ]:


X = np.hstack((X_real_scaled, X_cat_oh))


# In[ ]:


(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(X, t_labels,
                                                                      test_size=0.3,
                                                                      stratify=t_labels)


# ## First fitting SGDClassifier

# In[ ]:


clf = linear_model.SGDClassifier(class_weight='balanced')


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


print(metrics.roc_auc_score(y_test, clf.predict(X_test)))


# In[ ]:


param_grid = {
    'loss': ['hinge', 'log', 'squared_hinge', 'squared_loss'],
    'penalty': ['l1', 'l2'],
    'n_iter': list(range(3, 10)),
    'alpha': np.linspace(0.0001, 0.01, num=10)
}


# In[ ]:


grid_cv = grid_search.GridSearchCV(clf, param_grid, scoring='accuracy', cv=3)


# In[ ]:


grid_cv.fit(X_train, y_train)


# In[ ]:


print(grid_cv.best_params_)


# In[ ]:


print(metrics.roc_auc_score(y_test, grid_cv.best_estimator_.predict(X_test)))


# ## Decision tree

# In[ ]:


from sklearn import tree


# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=3, class_weight='balanced')


# In[ ]:


clf.get_params().keys()


# In[ ]:


params_grid = {
    'max_depth': list(range(1, 10)),
    'min_samples_leaf': list(range(2, 10))
}
grid_cv = grid_search.GridSearchCV(clf, params_grid, scoring='accuracy', cv=4)


# In[ ]:


grid_cv.fit(X_train, y_train)


# In[ ]:


print(grid_cv.best_params_)


# In[ ]:


print(metrics.roc_auc_score(y_test, grid_cv.best_estimator_.predict_proba(X_test)[:,1]))


# ## RandomForest

# In[ ]:


from sklearn import ensemble


# In[ ]:


rf_clf = ensemble.RandomForestClassifier()


# In[ ]:


rf_clf.get_params().keys()


# In[ ]:


rf_clf.fit(X_train, y_train)


# In[ ]:


print(metrics.roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:,1]))


# In[ ]:


params_grid = {
    'min_samples_leaf': list(range(1, 10)),
    'n_estimators': [10, 50, 100, 250, 500, 1000],
    'max_depth': list(range(1, 10))
}

rand_cv = grid_search.RandomizedSearchCV(rf_clf, params_grid, scoring='accuracy', cv=4, n_iter=40)

rand_cv.fit(X_train, y_train)


# In[ ]:


print(metrics.roc_auc_score(y_test, rand_cv.predict_proba(X_test)[:,1]))


# ## First test

# In[ ]:


test = pd.read_csv("../input/test.csv", header=0, sep=',')


# In[ ]:


test.head()


# In[ ]:


test.isnull().sum()


# In[ ]:


test_data = test.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)


# In[ ]:


test_data['Name_pred'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


test_data['Name_pred'] = test_data['Name_pred'].replace("Mlle", "Miss")
test_data['Name_pred'] = test_data['Name_pred'].replace("Ms", "Miss")
test_data['Name_pred'] = test_data['Name_pred'].replace("Mme", "Mrs")


# In[ ]:


test_data['Name_pred'] = test_data['Name_pred'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer',                                              'Lady', 'Major', 'Rev', 'Sir'], 'Other')


# In[ ]:


preds = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}
test_data['Name_pred'] = test_data['Name_pred'].map(preds)
test_data = test_data.drop('Name', axis=1)


# In[ ]:


test_data.Name_pred = test_data.Name_pred.fillna(value=5)


# In[ ]:


test_data.Name_pred = test_data.Name_pred.apply(int)


# In[ ]:


test_data['Sex'] = test_data['Sex'].apply(lambda x: int(x == 'male'))


# In[ ]:


test_data.Embarked = test_data.Embarked.fillna(value='S')
emb = { 'S': 1, 'C': 2, 'Q': 3}
test_data.Embarked = test_data.Embarked.map(emb)


# In[ ]:


test_data.Age = test_data.Age.fillna(value=0)


# In[ ]:


real_cols = ['Age', 'SibSp', 'Parch', 'Fare']
cat_cols = list(set(test_data.columns.values.tolist()) - set(real_cols))


# In[ ]:


Test_real = test_data[real_cols]
Test_cat = test_data[cat_cols]


# In[ ]:


encoder = OneHotEncoder(categorical_features='all', sparse=True, n_values='auto')
Test_cat_oh = encoder.fit_transform(Test_cat).toarray()


# In[ ]:


Test_real.Fare = Test_real.Fare.fillna(value=0)


# In[ ]:


scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(Test_real)


# In[ ]:


X = np.hstack((Test_real, Test_cat_oh))


# In[ ]:


predict = rand_cv.predict(X)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test.PassengerId,
        "Survived": predict
    })
submission.to_csv("predict.csv", index=False)


# In[ ]:


rand_cv.score(X_train, y_train)


# In[ ]:


print(rand_cv.best_estimator_)


# In[ ]:




