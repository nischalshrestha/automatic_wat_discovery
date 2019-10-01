#!/usr/bin/env python
# coding: utf-8

# <center><h1>Quick Start Guide to Kaggle's <i>Titanic: Machine Learning from Disaster<i></h1></center>

# In[ ]:


import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, 
                              RandomForestClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
sns.set(rc={'figure.figsize': (9, 7)})
np.random.RandomState(seed=42)
print(os.listdir("../input"));


# ## 0. Data Retrieval

# ### Load Datasets

# In[ ]:


# load training and testing datasets using pandas.read_csv() 
train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')

# Print the shapes of both training and testing datasets
print('Training Dataset: %s, Testing Dastaset: %s' %(str(train.shape), str(test.shape)))

# Inspect column dtypes
train.dtypes.reset_index()


# ## 1. Data Preparation

# ### Data Processing & Wrangling, Exploratory Data Analysis

# In[ ]:


# Detect potential outliers in our training dataset using z scores
def outliers_z_score(series, threshold=3):
    outliers = list()
    mean_, std_, threshold = np.mean(series), np.std(series), threshold
    z_scores = [(elem - mean_) / std_ for elem in series]
    return series[np.abs(z_scores) > threshold] 


# In[ ]:


numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
outlier_ind = list()
for num_feature in numeric_features:
    outliers = outliers_z_score(train[num_feature])
    index_outliers = list(outliers.index)
    outlier_ind += index_outliers


# In[ ]:


train = train.drop(list(set(outlier_ind)), axis=0).reset_index(drop=True)  # Drop outliers via indices
train_len = len(train)  # Store len(train) before merging training & testing datasets
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)  # Merge training and testing sets


# In[ ]:


dataset = dataset.fillna(np.nan)  # Fill in empty values with NaN
dataset.isnull().sum()  # See which columns have NaN values


# In[ ]:


train.describe()


# In[ ]:


sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(), 
            annot=True, fmt = ".2f", cmap = "autumn_r", square=True);


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("SibSp", "Survived", data=train, kind="bar")
    g.set_axis_labels("SibSp", "Pr (Survived)")
    g = sns.catplot("SibSp", "Survived", 'Sex', data=train, kind="bar")
    g.set_axis_labels("SibSp", "Pr (Survived)")


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot("Parch", "Survived", data=train, kind="bar")
    g.set_axis_labels("Parch", "Pr (Survived)")
    g = sns.catplot("Parch", "Survived", 'Sex', data=train, kind="bar")
    g.set_axis_labels("Parch", "Pr (Survived)");


# In[ ]:


# Compare number of dead versus survived
survivors = train[train['Survived'] == 1]['Pclass'].value_counts()
dead = train[train['Survived'] == 0]['Pclass'].value_counts()
df_survival_pclass = pd.DataFrame([survivors, dead])
df_survival_pclass.index = ['Dead', 'Survived']
df_survival_pclass.plot(kind='bar', stacked=True, title='Survival Based on by Passenger Class')
train['Dead'] = 1 - train['Survived']
train.groupby('Sex').agg('sum')[['Survived', 'Dead']].plot(kind='bar', stacked=True, color=['g', 'r'], title='Survival Based on by Sex');


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(sns.distplot, "Age");


# In[ ]:


with sns.axes_style(style='ticks'):
    g = sns.catplot(x="Survived", y = "Age", data=train, kind="box")
    g.set_axis_labels("Age", "Pr (Survived)");


# ### Feature Engineering & Scaling, Feature Selection

# In[ ]:


dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
dataset['Embarked'] = dataset['Embarked'].fillna('S')

nan_age_index = list(dataset['Age'][dataset['Age'].isnull()].index)
for i in nan_age_index:
    age_similar = dataset['Age'][(dataset['SibSp'] == dataset.loc[i]['SibSp']) & 
                   (dataset['Parch'] == dataset.loc[i]['Parch']) & 
                   (dataset['Pclass'] == dataset.loc[i]['Pclass'])].median()
    age_median = dataset['Age'].median()
    
    if not np.isnan(age_similar):
         dataset['Age'].loc[i] = age_similar
    else:
        dataset['Age'].loc[i] = age_median


# In[ ]:


encoder_embarked, encoder_sex = LabelEncoder(), LabelEncoder()

encoder_embarked.fit(dataset['Embarked'])
encoder_sex.fit(dataset['Sex'])

dataset['Embarked'] = encoder_embarked.transform(dataset['Embarked'])
dataset['Sex'] = encoder_sex.transform(dataset['Sex'])


# In[ ]:


# Combine 'SibSp' and 'Parch' => 'Relatives'
dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']


# In[ ]:


g = sns.catplot(x='Relatives', y='Survived', data=dataset)
g.set_axis_labels('Relatives', 'Pr (Survived)');


# In[ ]:


dataset['Pclass'] = dataset['Pclass'].astype('category')
dataset = pd.get_dummies(dataset, columns = ['Pclass'], prefix='Pc')


# In[ ]:


scaler_age, scaler_age = StandardScaler(), StandardScaler()
dataset['Age'] = scaler_age.fit_transform(np.array(dataset['Age']).reshape(-1, 1))
dataset['Fare'] = scaler_age.fit_transform(np.array(dataset['Fare']).reshape(-1, 1))


# In[ ]:


features = ['Age', 'Embarked', 'Fare', 'Parch', 'Sex', 
            'SibSp', 'Relatives', 'Pc_1', 'Pc_2', 'Pc_3']


# ## 2. Modeling

# ### Model Selection

# In[ ]:


train, test = dataset[:train_len], dataset[train_len:]
train['Survived'] = train['Survived'].astype(int)
test.drop(labels=['Survived'], axis=1,inplace=True)
train, test, train_survived = train[features], test[features], train['Survived']
X_train, y_train = train.values, train_survived.values
X_test = test.values


# In[ ]:


kfold = StratifiedKFold(n_splits=10)
estimators = [AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.1),
              DecisionTreeClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(),
              KNeighborsClassifier(), LinearDiscriminantAnalysis(), LogisticRegression(),
              MLPClassifier(), RandomForestClassifier(), SVC(), XGBClassifier()]

classifiers = list()
for est in estimators:
    classifiers.append(est)


# In[ ]:


get_ipython().run_cell_magic(u'capture', u'', u"cross_val_results = []\nfor classifier in classifiers :\n    cross_val_results.append(cross_val_score(classifier, X_train, y_train, \n                                             scoring='accuracy', cv=kfold))\n    \ncross_val_means, cross_val_stds = [], []\nfor cv_result in cross_val_results:\n    cross_val_means.append(cv_result.mean())\n    cross_val_stds.append(cv_result.std())\n    \ndf_cv = pd.DataFrame({'CrossValMeans':cross_val_means,\n                      'CrossValErrors': cross_val_stds,\n                      'Algorithms':['AdaBoost', 'DecisionTree', 'ExtraTrees', 'GradientBoosting',\n                                    'KNeighboors', 'LinearDiscriminantAnalysis', 'LogisticRegression',\n                                    'MultipleLayerPerceptron', 'RandomForest', 'SVC', 'XGBClassifier']})")


# In[ ]:


g = sns.barplot("CrossValMeans", "Algorithms", data=df_cv, 
                palette="Set3", orient="h", **{'xerr':cross_val_stds})
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores");


# In[ ]:


selected_models = df_cv.sort_values(by=['CrossValMeans'], ascending=False).reset_index(drop=True)
selected_models


# ### Model Evaluation and Tuning

# In[ ]:


estimator_scores = dict()


# In[ ]:


# AdaBoost Classifier
dec_tree_est = DecisionTreeClassifier()
clf_AdaBoost_DTC = AdaBoostClassifier(dec_tree_est)

parameters_ada_tree = {
    "algorithm": ["SAMME","SAMME.R"],
    "base_estimator__criterion": ["gini", "entropy"],
    "base_estimator__splitter": ["best", "random"],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2],
    "n_estimators": [1, 2]
}

clf_ada_tree = GridSearchCV(clf_AdaBoost_DTC, parameters_ada_tree, cv=kfold, scoring="accuracy", n_jobs=4)
clf_ada_tree.fit(X_train, y_train)

clf_ada_tree_best = clf_ada_tree.best_estimator_
estimator_scores[clf_ada_tree.best_score_] = clf_ada_tree_best
print('Best score: ', clf_ada_tree.best_score_)
print('Best params: ', clf_ada_tree.best_params_)


# In[ ]:


# Extra Trees Classifier
clf_ExtraTrees = ExtraTreesClassifier()

parameters_extra_trees = {
    "bootstrap": [False],
    "criterion": ["gini"],
    "max_features": [1, 3, 10],
#     "min_samples_leaf": np.linspace(0.1, 0.5, 6),
#     "min_samples_split": np.linspace(0.1, 0.5, 6),
#     "n_estimators": np.array(range(50,301,50)),
    "n_estimators": [100],
    "max_depth": [None]
}

clf_extra_trees = GridSearchCV(clf_ExtraTrees, parameters_extra_trees, cv=kfold, scoring="accuracy", n_jobs=4)
clf_extra_trees.fit(X_train, y_train)

clf_extra_trees_best = clf_extra_trees.best_estimator_
estimator_scores[clf_extra_trees.best_score_] = clf_extra_trees_best
print('Best score: ', clf_extra_trees.best_score_)
print('Best params: ', clf_extra_trees.best_params_)


# In[ ]:


# Gradient Boosting Classifier
clf_GradientBoost = GradientBoostingClassifier()

parameters_gradient_boost = {
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "loss": ["deviance"],
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
#     "min_samples_leaf": np.linspace(0.1, 0.5, 6),
#     "min_samples_split": np.linspace(0.1, 0.5, 6),
#     "n_estimators": range(50, 301, 100),
    "n_estimators": [300]
}

clf_gradient_boost = GridSearchCV(clf_GradientBoost, parameters_gradient_boost, cv=kfold, scoring="accuracy", n_jobs=4)
clf_gradient_boost.fit(X_train, y_train)

clf_gradient_boost_best = clf_gradient_boost.best_estimator_
estimator_scores[clf_gradient_boost.best_score_] = clf_gradient_boost_best
print('Best score: ', clf_gradient_boost.best_score_)
print('Best params: ', clf_gradient_boost.best_params_)


# In[ ]:


# Multiple Layer Perceptron Classifier
clf_MLP = MLPClassifier()

parameters_MLP = {
    'alpha': [1e-2, 1e-3, 1e-4],
#     'max_iter': np.arange(1000, 2001, 500),
    'hidden_layer_sizes': [10],
    'solver': ['lbfgs']
}

clf_mlp = GridSearchCV(clf_MLP, parameters_MLP, cv=kfold, scoring="accuracy", n_jobs=4)
clf_mlp.fit(X_train, y_train)

clf_mlp_best = clf_mlp.best_estimator_
estimator_scores[clf_mlp.best_score_] = clf_mlp_best
print('Best score: ', clf_mlp.best_score_)
print('Best params: ', clf_mlp.best_params_)


# In[ ]:


# Support Vector Classifier (SVC)
clf_SVC = SVC(probability=True)

parameters_SVC = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.1, 1], 
    'kernel': ['linear', 'rbf']
}

clf_svc = GridSearchCV(clf_SVC, parameters_SVC, cv=kfold, scoring="accuracy", n_jobs=4)
clf_svc.fit(X_train, y_train)

clf_svc_best = clf_svc.best_estimator_
estimator_scores[clf_svc.best_score_] = clf_svc_best
print('Best score: ', clf_svc.best_score_)
print('Best params: ', clf_svc.best_params_)


# In[ ]:


# XG Boost Classifier
clf_XGBoost = XGBClassifier()

parameters_xg_boost = {
    'gamma': [0.5, 1, 1.5, 2, 5],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [300],
#     "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
#     "max_depth": [3, 5, 8],
#     'min_child_weight': [1, 5, 10],
#     "n_estimators": range(50, 301, 100),
#     'subsample': [0.6, 0.8, 1.0],
}

clf_xg_boost = GridSearchCV(clf_XGBoost, parameters_xg_boost, cv=kfold, scoring="accuracy", n_jobs= 4)
clf_xg_boost.fit(X_train, y_train)

clf_xg_boost_best = clf_xg_boost.best_estimator_
estimator_scores[clf_xg_boost.best_score_] = clf_xg_boost_best
print('Best score: ', clf_xg_boost.best_score_)
print('Best params: ', clf_xg_boost.best_params_)


# In[ ]:


best_score = max(estimator_scores.keys())
best_classifier = estimator_scores[best_score]


# ## 3. Deployment & Prediction

# In[ ]:


y_pred = best_classifier.predict(X_test)
test["Survived"] = y_pred
result = test[["PassengerId", "Survived"]]
result.to_csv('titanic_disaster_pred.csv', index=False)

