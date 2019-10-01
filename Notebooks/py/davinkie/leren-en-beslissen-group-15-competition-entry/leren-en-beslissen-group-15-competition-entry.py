#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1">Introduction</a></span></li><li><span><a href="#1.-Data-Analysis" data-toc-modified-id="1.-Data-Analysis-2">1. Data Analysis</a></span><ul class="toc-item"><li><span><a href="#1.1-Loading-and-Checking-Data" data-toc-modified-id="1.1-Loading-and-Checking-Data-2.1">1.1 Loading and Checking Data</a></span></li><li><span><a href="#1.2-Handling-Missing-Data" data-toc-modified-id="1.2-Handling-Missing-Data-2.2">1.2 Handling Missing Data</a></span></li><li><span><a href="#1.3-Feature-Engineering" data-toc-modified-id="1.3-Feature-Engineering-2.3">1.3 Feature Engineering</a></span></li><li><span><a href="#1.4-Data-Visualisation" data-toc-modified-id="1.4-Data-Visualisation-2.4">1.4 Data Visualisation</a></span></li></ul></li><li><span><a href="#2.-Model-Selection" data-toc-modified-id="2.-Model-Selection-3">2. Model Selection</a></span><ul class="toc-item"><li><span><a href="#2.1-Neighbors" data-toc-modified-id="2.1-Neighbors-3.1">2.1 Neighbors</a></span><ul class="toc-item"><li><span><a href="#2.1.1-Normal-KNN" data-toc-modified-id="2.1.1-Normal-KNN-3.1.1">2.1.1 Normal KNN</a></span></li><li><span><a href="#2.1.2-Weighted-KNN" data-toc-modified-id="2.1.2-Weighted-KNN-3.1.2">2.1.2 Weighted KNN</a></span></li></ul></li><li><span><a href="#2.2-Decision-Trees" data-toc-modified-id="2.2-Decision-Trees-3.2">2.2 Decision Trees</a></span><ul class="toc-item"><li><span><a href="#2.2.1-Discrete-Tree" data-toc-modified-id="2.2.1-Discrete-Tree-3.2.1">2.2.1 Discrete Tree</a></span></li></ul></li><li><span><a href="#2.3-Logistic-Regression" data-toc-modified-id="2.3-Logistic-Regression-3.3">2.3 Logistic Regression</a></span></li><li><span><a href="#2.4-Ensemble-Methods" data-toc-modified-id="2.4-Ensemble-Methods-3.4">2.4 Ensemble Methods</a></span><ul class="toc-item"><li><span><a href="#2.4.1-Extra-Trees-Classifier" data-toc-modified-id="2.4.1-Extra-Trees-Classifier-3.4.1">2.4.1 Extra Trees Classifier</a></span></li><li><span><a href="#2.4.2-Random-Forest" data-toc-modified-id="2.4.2-Random-Forest-3.4.2">2.4.2 Random Forest</a></span></li><li><span><a href="#2.4.3-Boosting" data-toc-modified-id="2.4.3-Boosting-3.4.3">2.4.3 Boosting</a></span><ul class="toc-item"><li><span><a href="#ADA-Boosting" data-toc-modified-id="ADA-Boosting-3.4.3.1">ADA Boosting</a></span></li><li><span><a href="#Gradient-Boosting" data-toc-modified-id="Gradient-Boosting-3.4.3.2">Gradient Boosting</a></span></li><li><span><a href="#Extreme-Gradient-Boosting" data-toc-modified-id="Extreme-Gradient-Boosting-3.4.3.3">Extreme Gradient Boosting</a></span></li></ul></li><li><span><a href="#2.4.4-Voting-Methods" data-toc-modified-id="2.4.4-Voting-Methods-3.4.4">2.4.4 Voting Methods</a></span><ul class="toc-item"><li><span><a href="#Hard-/-Soft-Voting" data-toc-modified-id="Hard-/-Soft-Voting-3.4.4.1">Hard / Soft Voting</a></span></li><li><span><a href="#Hard-/-Soft-voting,-Parameter-Search" data-toc-modified-id="Hard-/-Soft-voting,-Parameter-Search-3.4.4.2">Hard / Soft voting, Parameter Search</a></span></li></ul></li><li><span><a href="#2.4.5-Stacking" data-toc-modified-id="2.4.5-Stacking-3.4.5">2.4.5 Stacking</a></span></li></ul></li></ul></li><li><span><a href="#3.-Evaluation" data-toc-modified-id="3.-Evaluation-4">3. Evaluation</a></span><ul class="toc-item"><li><span><a href="#3.1-Acurracy" data-toc-modified-id="3.1-Acurracy-4.1">3.1 Acurracy</a></span></li><li><span><a href="#3.2-Model-Correlations" data-toc-modified-id="3.2-Model-Correlations-4.2">3.2 Model Correlations</a></span></li><li><span><a href="#3.3-Custom-features-per-classifier" data-toc-modified-id="3.3-Custom-features-per-classifier-4.3">3.3 Custom features per classifier</a></span></li></ul></li></ul></div>

# # Introduction
# 
# This project was done by Group 15 from the Leren en Beslissen Course given during the Bachelor Artficial Intelligence at the University of Amsterdam.
# 
# The team members were: Noa Visser, Laurens Weitkamp, Jim Voorn and Daniël Vink.
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

# Data processing packages
import numpy as np
import pandas as pd

# Plotting/visualisation packages
import seaborn as sns
import graphviz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# sklearn helper functions
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

# sklearn classification models
from sklearn import tree, neighbors, linear_model, neighbors, ensemble
import xgboost as xgb

# Model stacking package
from vecstack import stacking


# Create a custom colormap for pearson coefficient heatmap
colors1 = plt.cm.Blues_r(np.linspace(0., 1, 128))
colors2 = plt.cm.Blues(np.linspace(0, 1, 128))
colors = np.vstack((colors1, colors2))
pearson_colors = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# Notebook settings
sns.set_style("whitegrid")
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")
get_ipython().magic(u'matplotlib inline')
alpha = 0.7

def create_csv(results, filename, test):
    results = pd.DataFrame(results)
    results['PassengerId'] = test['PassengerId']
    results.columns = ['Survived', 'PassengerId']
    results.to_csv(filename, index = False)


# # 1. Data Analysis
# 
# ## 1.1 Loading and Checking Data

# In[ ]:


test  = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")

print(train.shape, test.shape)


# In[ ]:


train.head()


# ## 1.2 Handling Missing Data
# 
# As we can see in the above DataFrame, the Cabin column often has _NaN_'s, missing values. In the case of Cabin this is a major issue as we can see in the cell below, over 77% is missing. The cabins themselves range from A to G, with A being the top level cabins, and this might indicate some correlation with the Pclass and perhaps even with the Ticket column. 

# In[ ]:


pd.isnull(train).sum()


# The total number of data samples we have for training is 891, and we can see that Cabin is missing 687 values in total here. 

# In[ ]:


plt.figure(figsize = (18, 6))
train[train['Survived'] == 1]['Age'].hist(label='Survived', alpha=alpha)
train[train['Survived'] == 0]['Age'].hist(label='Died', alpha=alpha)
plt.title("Age ")
plt.legend();


# ## 1.3 Feature Engineering
# 
# korte uitleg over het aanmaken van nieuwe features, tokenisation van sex, tokenisation van age

# In[ ]:


def feature_engineering(dataset):
    """
    dataset: DataFrame
    """
    
    # First up is some 
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['FarePP'] = (dataset['Fare'] / (dataset['SibSp'] + dataset['Parch'] + 1)).round().astype(int)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = dataset['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    dataset['SexBinary'] = dataset['Sex'].apply(lambda x: 1 if x == 'female' else 0)

    # Knipt de titels uit de namen en voegt deze als kolom toe aan de dataframe
    dataset['Title'] = dataset['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    # Vervangt obscure of buitenlandse titels met overkoepelende titel.
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Sir', 'Major', 'Rev'], 'Otherm')
    dataset['Title'] = dataset['Title'].replace(['Dona', 'Lady', 'the Countess'], 'Otherf')

    # berekent de mediaan per titel
    medians = dataset[['Age', 'Title']].groupby(['Title'], as_index=False).median()
    # vult ontbrekende leeftijden in met de mediaan van de betreffende titel
    dataset.loc[dataset['Title'] == 'Master', 'Age'] = dataset.loc[dataset['Title'] == 'Master', 'Age'].fillna(medians.loc[medians['Title'] == 'Master']['Age'][0])
    dataset.loc[dataset['Title'] == 'Mr', 'Age'] = dataset.loc[dataset['Title'] == 'Mr', 'Age'].fillna(medians.loc[medians['Title'] == 'Mr']['Age'][2])
    dataset.loc[dataset['Title'] == 'Mrs', 'Age'] = dataset.loc[dataset['Title'] == 'Mrs', 'Age'].fillna(medians.loc[medians['Title'] == 'Mrs']['Age'][3])
    dataset.loc[dataset['Title'] == 'Miss', 'Age'] = dataset.loc[dataset['Title'] == 'Miss', 'Age'].fillna(medians.loc[medians['Title'] == 'Miss']['Age'][1])
    dataset.loc[dataset['Title'] == 'Otherm', 'Age'] = dataset.loc[dataset['Title'] == 'Otherm', 'Age'].fillna(medians.loc[medians['Title'] == 'Otherm']['Age'][5])
    dataset.loc[dataset['Title'] == 'Otherf', 'Age'] = dataset.loc[dataset['Title'] == 'Otherf', 'Age'].fillna(medians.loc[medians['Title'] == 'Otherf']['Age'][4])
    
    # discretiseert de leeftijden in leeftijdsgroepen. Aantal kan aangepast worden, returnt ook een lijst met de ranges van 
    # de leeftijdsgroepen.
    dataset['AgeGroup'], agebins = pd.cut(dataset['Age'], 6, labels=range(6), retbins=True)
    dataset['AgeGroup'].head(20), agebins;
    dataset['AgeGroup'] = dataset['AgeGroup'].astype(int) # it keeps the categorical dtype, we want int

    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()

    dataset['DiscreteTitle'] = label.fit_transform(dataset['Title'])

    dataset['Fare'] = dataset['Fare'].apply(lambda x: dataset['Fare'].median() if x < 5 else x)
    
    # Some more discretisations
    dataset['EmbarkedDiscrete'] = label.fit_transform(dataset['Embarked'].apply(lambda x: x if not pd.isnull(x) else 'S'))
    dataset['FareAdjusted'] = dataset['Fare'].apply(lambda x: x if x != 0 else dataset['Fare'].median()) 
    dataset['FareBinned'] = label.fit_transform(pd.qcut(dataset['FareAdjusted'], 4))
    
    dataset['AgeGroup'] = dataset['AgeGroup'].astype(int)
    
    return dataset

train    = feature_engineering(train)
testing  = feature_engineering(test)


# ## 1.4 Data Visualisation
# 
# To analyse our new features, let's create a Pearson Coefficient heatmap. This ranges from -1 to 1 and corresponds to the correlation between each feature. -1 indicates a strong negative correlation, and 1 indicates a strong positive correlation. 0 indicates that there is no correlation at all between features.

# In[ ]:


features = train[['Embarked', 'Survived', 'Pclass', 'IsAlone', 'SexBinary', 'AgeGroup', 'FareBinned', 'DiscreteTitle']].corr()
plt.figure(figsize = (18, 8))
sns.heatmap(features, cmap=pearson_colors, linewidths=1, annot=features, vmin=-1, vmax=1);
plt.yticks(rotation=0)
plt.title("Pearson Coefficient for all numerical features")
plt.show()


# We see that __SexBinary__, __Pclass__, __IsAlone__ and __FareBinned__ have the highest correlations (in that order). What we can also see is that __Pclass__ and __FareBinned__ have a significantly strong correlation between eachother. Likewise, __Pclass__ and __AgeGroup__ have a good correlation too. We can use this information later during feature selection, we might for example choose to use __Pclass__ and drop __FareBinned__.
# 
# 
# Now let's try to figure out some more about the difference between sex and survival rate.

# In[ ]:


plt.figure(figsize = (18, 8))
sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data = train, alpha=alpha)
plt.title("Survival rate of male and female per embarked location")
plt.xlabel("Sex");
plt.ylabel("Survival rate");


# In[ ]:


plt.figure(figsize = (18, 8))
sns.barplot(x = 'AgeGroup', y = 'Survived', hue = 'Sex', data = train, alpha = alpha)
plt.title("Survival rate of male and female in age groups")
plt.xlabel("Age group")
plt.ylabel("Survival rate");


# In[ ]:


plt.figure(figsize = (18, 8))
sns.barplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = train, alpha=alpha)
plt.title("Survival rate of male and female per different Pclass")
plt.xlabel("Pclass")
plt.ylabel("Survival rate");


# In[ ]:


plt.figure(figsize = (18, 8))
sns.barplot(x = 'FamilySize', y = 'Survived', data = train, alpha = alpha);


# # 2. Model Selection
# 
# 
# For the purpose of this classification task, a wide variety of models will be used. We will start of with simpler ones like K-NN and Decision Trees, and then venture into ensemble methods like Random Forest and Voting Classifiers. For each model we will calculate both the result from the default settings and the result from the grid searched settings. Grid search is a method in which a range of parameters will be used to calculate the optimal parameters for a given model on a dataset.
# 
# ```
# Classification Models
# │
# └─── Neighbors
# │     └─ K-NN
# │     └─ Weighted K-NN
# │
# └─── Decision Trees
# │     └─ Discrete Tree
# │
# └─── Logistic Regression
# │
# └─── Ensembles
# │     └─ Extra Trees
# │     └─ Random Forest
# │     └─ Boosting
# │         └─ ADA
# │         └─ Gradient
# │         └─ Extreme Gradient
# │     └─ Voting Classifier
# │         └─ Hard
# │         └─ Soft
# │     └─ Stacking
# 
# ```

# In[ ]:


features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']


X = train[features_used]
y = train['Survived']
test = testing[features_used]

model_scores = {}
cross_vals = 4

#One Hot Encoding. In some cases it might be interesting to see if this actually improves the accuracy
enc = OneHotEncoder()
X_bin = enc.fit_transform(X).toarray().astype(int)
test_bin = enc.fit_transform(test).toarray().astype(int)


# Grid search 
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]


# ## 2.1 Neighbors
# 
# ### 2.1.1 Normal KNN

# In[ ]:


# Default parameter model
knn = neighbors.KNeighborsClassifier()

# Grid search parameter model
knn_params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


knn_grid = GridSearchCV(estimator=neighbors.KNeighborsClassifier(),
                        param_grid=knn_params,
                        cv=cross_vals,
                        scoring='roc_auc')
knn_grid = knn_grid.fit(X, y).best_estimator_

knn_score      = cross_val_score(knn, X, y, scoring='accuracy', cv=cross_vals)
knn_grid_score = cross_val_score(knn_grid, X, y, scoring='accuracy', cv=cross_vals)

print("KNN default:\t{0}\nKNN grid:\t{1}".format(
    knn_score.mean(), knn_grid_score.mean()))
model_scores['KNN'] = knn_score


# ### 2.1.2 Weighted KNN

# In[ ]:


# Default parameter model
knn = neighbors.KNeighborsClassifier(weights='distance')

# Grid search parameter model
knn_w_params = {'weights': ['distance'],
              'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


knn_grid = GridSearchCV(estimator = neighbors.KNeighborsClassifier(),
                                     param_grid = knn_w_params, 
                                     cv = cross_vals, 
                                     scoring = 'roc_auc')
knn_grid = knn_grid.fit(X, y).best_estimator_


knn_score = cross_val_score(knn, X, y, scoring='accuracy', cv = cross_vals)
knn_grid_score = cross_val_score(knn_grid, X, y, scoring='accuracy', cv = cross_vals)

print("KNN weighted default:\t{0}\nKNN weighted grid:\t{1}".format(knn_score.mean(), knn_grid_score.mean()))
model_scores['KNN weighted'] = knn_score


# ## 2.2 Decision Trees
# 
# ### 2.2.1 Discrete Tree

# In[ ]:


# Default parameter model
DTree = tree.DecisionTreeClassifier()

# Grid search parameter model 
DTree_params = {'criterion': grid_criterion,
                'max_depth': grid_max_depth,
                'random_state': grid_seed}
DTree_grid = GridSearchCV(estimator = tree.DecisionTreeClassifier(),
                                     param_grid = DTree_params, 
                                     cv = cross_vals, 
                                     scoring = 'roc_auc')
DTree_grid = DTree_grid.fit(X, y).best_estimator_


DTree_score      = cross_val_score(DTree, X, y, scoring='accuracy', cv=cross_vals)
DTree_grid_score = cross_val_score(DTree_grid, X, y, scoring='accuracy', cv=cross_vals)

print("Discrete Tree default:\t{0}\nDiscrete Tree Grid:\t{1}".format(DTree_score.mean(), DTree_grid_score.mean()))
model_scores['DecisionTree'] = DTree_grid_score


# In[ ]:


# tree visualisation
DTree.fit(X, y)
DTree_out_train = DTree.predict(X)
DTree_out_test = DTree.predict(test)

dot_data = tree.export_graphviz(DTree, 
                     out_file=None, 
                     feature_names=X.keys(),  
                     class_names=['Survived', 'Died'],
                     filled=True, rounded=True,
                     special_characters=True) 


graph = graphviz.Source(dot_data)
#graph.render("Tree") # Save tree
#graph


# In[ ]:


plt.figure(figsize = (18, 7))
sns.barplot(x = X.keys(), y = DTree.feature_importances_, alpha = alpha)
plt.title("Feature importance for Decision Tree classifier");


# ## 2.3 Logistic Regression

# In[ ]:


# Default parameter model
LR = linear_model.LogisticRegression()

# Grid search parameter model
logres_params = {'fit_intercept': grid_bool,
                 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                 'random_state': grid_seed}

LR_grid = GridSearchCV(estimator = linear_model.LogisticRegression(),
                                     param_grid = logres_params, 
                                     cv = cross_vals, 
                                     scoring = 'roc_auc')
LR_grid = LR_grid.fit(X, y).best_estimator_

LR_score = cross_val_score(LR, X, y, scoring='accuracy', cv=cross_vals)
LR_grid_score = cross_val_score(LR_grid, X, y, scoring='accuracy', cv=cross_vals)

print("Logistic Regression default:\t{0}\nLogistic Regression Grid:\t{1}".format(LR_score.mean(), LR_grid_score.mean()))
model_scores['Logistic Regression'] = LR_grid_score


# ## 2.4 Ensemble Methods
# 
# ### 2.4.1 Extra Trees Classifier

# In[ ]:


# Default parameter model 
ETree = ensemble.ExtraTreesClassifier()

# Grid search parameter model
etree_params = {'n_estimators': grid_n_estimator,
                'criterion': grid_criterion,
                'max_depth': grid_max_depth,
                'random_state': grid_seed}

ETree_grid = GridSearchCV(estimator = ensemble.ExtraTreesClassifier(),
                                     param_grid = etree_params, 
                                     cv = cross_vals, 
                                     scoring = 'roc_auc')
ETree_grid = ETree_grid.fit(X, y).best_estimator_

ETree_score = cross_val_score(ETree, X, y, scoring='accuracy', cv=cross_vals)
ETree_grid_score = cross_val_score(ETree_grid, X, y, scoring='accuracy', cv=cross_vals)

print("Extra trees default:\t{0}\nExtra trees grid:\t{1}".format(ETree_score.mean(), ETree_grid_score.mean()))
model_scores['Extra Trees'] = ETree_grid_score


# In[ ]:


plt.figure(figsize = (18, 7))
sns.barplot(x = X.keys(), y = ETree_grid.feature_importances_, alpha = alpha)
plt.title("Feature importance for Extra Trees classifier");


# ### 2.4.2 Random Forest

# In[ ]:


# Default parameter model
RandomForest = ensemble.RandomForestClassifier()

# Grid search parameter model
rf_params = {'n_estimators': grid_n_estimator,
             'criterion': grid_criterion,
             'max_depth': grid_max_depth,
             'oob_score': [True],
             'random_state': grid_seed}

RandomForest_grid = GridSearchCV(ensemble.RandomForestClassifier(), 
                                 param_grid=rf_params,
                                 cv = cross_vals, 
                                 scoring = 'roc_auc')
RandomForest_grid = RandomForest_grid.fit(X, y).best_estimator_

RandomForest_score      = cross_val_score(RandomForest, X, y, scoring='accuracy', cv=cross_vals)
RandomForest_grid_score = cross_val_score(RandomForest_grid, X, y, scoring='accuracy', cv=cross_vals)

print("Random Forest default:\t{0}\nRandom Forest grid:\t{1}".format(RandomForest_score.mean(), RandomForest_grid_score.mean()))
model_scores['Random Forest'] = RandomForest_grid_score


# In[ ]:


plt.figure(figsize = (18, 7))
sns.barplot(x = X.keys(), y = RandomForest_grid.feature_importances_, alpha = alpha)
plt.title("Feature importance for Random Forest classifier");


# ### 2.4.3 Boosting
# 
# #### ADA Boosting

# In[ ]:


# Default parameter model
ADA = ensemble.AdaBoostClassifier()

# Grid search parameter model
ada_params = {'n_estimators': grid_n_estimator,
              'learning_rate': grid_learn,
              'random_state': grid_seed}

ADA_grid = GridSearchCV(ensemble.AdaBoostClassifier(), 
                                 param_grid=ada_params,
                                 cv = cross_vals, 
                                 scoring = 'roc_auc')
ADA_grid = ADA_grid.fit(X, y).best_estimator_

ADA_score      = cross_val_score(ADA, X, y, scoring='accuracy', cv=cross_vals)
ADA_grid_score = cross_val_score(ADA_grid, X, y, scoring='accuracy', cv=cross_vals)


print("ADA Boosting default:\t{0}\nADA Boosting grid:\t{1}".format(ADA_score.mean(), ADA_grid_score.mean()))
model_scores['ADA Boosting'] = ADA_grid_score


# #### Gradient Boosting

# In[ ]:


# Default parameter model
GB =  ensemble.GradientBoostingClassifier()

# Grid search parameter model
gb_params = {'learning_rate': [.05], 
              'n_estimators': [300],
              'max_depth': grid_max_depth,
              'random_state': grid_seed}

GB_grid = GridSearchCV(ensemble.GradientBoostingClassifier(), 
                                 param_grid = gb_params,
                                 cv = cross_vals, 
                                 scoring = 'roc_auc')
GB_grid = GB_grid.fit(X, y).best_estimator_

GB_score      = cross_val_score(GB, X, y, scoring='accuracy', cv=cross_vals)
GB_grid_score = cross_val_score(GB_grid, X, y, scoring='accuracy', cv=cross_vals)


print("Gradient Boosting default:\t{0}\nGradient Boosting grid:\t{1}".format(GB_score.mean(), GB_grid_score.mean()))
model_scores['Gradient Boosting'] = GB_grid_score


# #### Extreme Gradient Boosting

# In[ ]:


xboost = xgb.XGBClassifier()

knn.fit(X, y)
knn_out_train = knn.predict(X)

dtree = tree.DecisionTreeClassifier()
dtree.fit(X, y)
clf_out_train = dtree.predict(X)

LR.fit(X, y)
LR_out_train = LR.predict(X)


GB.fit(X, y)
ABC_out_train = GB.predict(X)

ETree.fit(X, y)
ETree_out_train = ETree.predict(X)

RandomForest.fit(X, y)
RandomForest_out_train = RandomForest.predict(X)


classifier_train_matrix = np.concatenate((knn_out_train.reshape(-1, 1), clf_out_train.reshape(-1, 1), ETree_out_train.reshape(-1, 1), 
                                    RandomForest_out_train.reshape(-1, 1), LR_out_train.reshape(-1, 1), ABC_out_train.reshape(-1, 1)), axis=1)


xgb_params = {'learning_rate': [0.2, 0.4], 
              'n_estimators': [100, 1000],
              'max_depth': [2, 4, 6, 8, 10]}

XGB_grid = GridSearchCV(xgb.XGBClassifier(), 
                                 param_grid = xgb_params,
                                 cv = cross_vals, 
                                 scoring = 'roc_auc')

XGBx_grid = GridSearchCV(xgb.XGBClassifier(), 
                                 param_grid = xgb_params,
                                 cv = cross_vals, 
                                 scoring = 'roc_auc')

XGB_grid = XGB_grid.fit(X,y).best_estimator_

XGBx_grid = XGBx_grid.fit(classifier_train_matrix,y).best_estimator_

XGB_score      = cross_val_score(xboost, X, y, scoring='accuracy', cv=cross_vals)
XGB_grid_score = cross_val_score(XGB_grid, X, y, scoring='accuracy', cv=cross_vals)

XGBx_score     = cross_val_score(xboost, classifier_train_matrix, y, scoring='accuracy', cv=cross_vals)
XGBx_grid_score = cross_val_score(XGB_grid, classifier_train_matrix, y, scoring='accuracy', cv=cross_vals)

print("XGBoost default:\t{0}\nXGBoost grid:\t{1}".format(XGB_score.mean(), XGB_grid_score.mean()))
model_scores['XGBoost'] = XGB_grid_score

print("XGBoost extra default:\t{0}\nXGBoost grid:\t{1}".format(XGBx_score.mean(), XGBx_grid_score.mean()))
model_scores['XGBoost Extra'] = XGBx_grid_score


# ### 2.4.4 Voting Methods

# #### Hard / Soft Voting

# In[ ]:


vote_est = [
    ('ada', ensemble.AdaBoostClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    ('lr',  linear_model.LogisticRegressionCV()),
    ('knn', neighbors.KNeighborsClassifier()),
    ('xgb', xgb.XGBClassifier())
]

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_score = cross_val_score(vote_hard, X, y, cv  = cross_vals)
model_scores['Voting Hard'] = vote_hard_score

vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_score = cross_val_score(vote_soft, X, y, cv  = cross_vals)
model_scores['Voting Soft'] = vote_hard_score

print("Hard vote score: {0}\nSoft vote score: {1}".format(vote_hard_score.mean(), vote_soft_score.mean()))


# #### Hard / Soft voting, Parameter Search

# In[ ]:


vote_est_grid = [
     ('ada', ADA_grid),
     ('etc', ETree_grid),
     ('gbc', GB_grid),
     ('rfc', RandomForest_grid),
     ('lr',  LR_grid),
     ('knn', knn_grid),
     ('xgb', XGB_grid),
     ('xgb extra', XGBx_grid)
]

vote_hard_grid = ensemble.VotingClassifier(estimators = vote_est_grid, voting = 'hard')
vote_hard_grid_score = cross_val_score(vote_hard_grid, X, y, cv  = cross_vals)
model_scores['Voting Hard Grid'] = vote_hard_grid_score

vote_soft_grid = ensemble.VotingClassifier(estimators = vote_est_grid, voting = 'soft')
vote_soft_grid_score = cross_val_score(vote_soft_grid, X, y, cv  = cross_vals)
model_scores['Voting Soft Grid'] = vote_soft_grid_score

print("Hard vote score: {0}\nSoft vote score: {1}".format(vote_hard_grid_score.mean(), vote_soft_grid_score.mean()))


# ### 2.4.5 Stacking

# In[ ]:


# Make a list containing all models
models = [a[1] for a in vote_est]

# 1st layer train
S_train, S_test = stacking(models, X.as_matrix(), y.as_matrix(), test.as_matrix(), 
                           regression = False,
                           n_folds = 2,
                           random_state = 1)

# 2nd layer train using random forest
clf = ensemble.RandomForestClassifier()
clf.fit(S_train, y.as_matrix())


model_scores['Stacking'] = cross_val_score(clf, X, y, cv=cross_vals)

print(model_scores['Stacking'].mean())


# In[ ]:





# # 3. Evaluation
# 
# ## 3.1 Acurracy

# In[ ]:


plt.figure(figsize = (18, 8))

scores = {key : val.mean() for key, val in model_scores.items()}
algos = sorted(scores, key = scores.get, reverse=True)
algos_scores = [scores[x] for x in algos]

stds   = [model_scores[x].std() for x in algos]

sns.barplot(y = algos, x = algos_scores, color='Blue', alpha=alpha, xerr=stds)
plt.xlim(0, 0.9)
plt.show()
scores.keys()


# ## 3.2 Model Correlations

# In[ ]:


# k_out_train
# clf_out_train
# ETree_out_train
# RandomForest_out_train
# LR_out_train
# ABC_out_train

train_output_matrix = np.array([y, knn_out_train, clf_out_train, ETree_out_train, RandomForest_out_train, LR_out_train, ABC_out_train])
train_output_corr = np.corrcoef(train_output_matrix)

labels = ['Training', 'KNN', 'DecisionTree', 'ExtraTrees', 'RandomForest', 'Log Regression', 'GradientBoosting']

plt.figure(figsize = (18, 8))
sns.heatmap(train_output_corr, cmap=pearson_colors, annot=train_output_corr, xticklabels=labels, yticklabels=labels, linewidths=1, vmin=-1, vmax=1);
plt.yticks(rotation=0)
plt.show()


# ## 3.3 Custom features per classifier
# 
# This section is about checking each different features for each individual model. The overal accuracies have been printed, but it is noticeable that they do not improve the previous score. Previously we used backward selection of features, starting with all and removing one by one untill the accuracy improves significantly.

# In[ ]:


features_used = ['Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FareBinned']
X = train[features_used]
test = testing[features_used]

knn = neighbors.KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(X, y)
k_out= knn.predict(test)
xk_out = knn.predict(X)

print(cross_val_score(knn, X, y, scoring='accuracy', cv=7).mean())

############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
knn.fit(X, y)

print(cross_val_score(knn, X, y, scoring='accuracy', cv=7).mean())


# In[ ]:


features_used =  ['Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize']
X = train[features_used]
test = testing[features_used]

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
clf_out= clf.predict(test)
xclf_out = clf.predict(X)

print(cross_val_score(clf, X, y, scoring='accuracy', cv=5).mean())

############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
clf.fit(X, y)

print(cross_val_score(clf, X, y, scoring='accuracy', cv=5).mean())


# In[ ]:


features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]

LR = linear_model.LogisticRegression()
LR.fit(X, y)
LR_out = LR.predict(test)
xLR_out = LR.predict(X)

print(cross_val_score(LR, X, y, scoring='accuracy', cv=5).mean())

#############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
LR.fit(X,y)

print(cross_val_score(LR, X, y, scoring='accuracy', cv=5).mean())


# In[ ]:


features_used = ['SexBinary', 'AgeGroup', 'FamilySize']
X = train[features_used]
test = testing[features_used]

RandomForest = ensemble.RandomForestClassifier()
RandomForest.fit(X, y)
R_out = RandomForest.predict(test)
xR_out = RandomForest.predict(X)

print(cross_val_score(RandomForest, X, y, scoring='accuracy', cv=5).mean())

#############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
RandomForest.fit(X, y)

print(cross_val_score(RandomForest, X, y, scoring='accuracy', cv=5).mean())


# In[ ]:


features_used = ['Pclass', 'SexBinary', 'DiscreteTitle', 'FamilySize']
X = train[features_used]
test = testing[features_used]

xboost = xgb.XGBClassifier(max_depth = 2, learning_rate = 0.1, n_estimators = 150)
xboost.fit(X, y)
x_out = xboost.predict(test)
xx_out = xboost.predict(X)

print(cross_val_score(xboost, X, y, cv = 5).mean())

#############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
xboost.fit(X, y)

print(cross_val_score(xboost, X, y, cv = 5).mean())


# In[ ]:


features_used = ['Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize']
X = train[features_used]
test = testing[features_used]

ETree = ensemble.ExtraTreesClassifier(n_estimators=7)
ETree.fit(X, y)
E_out = ETree.predict(test)
xE_out = ETree.predict(X)

print(cross_val_score(ETree, X, y, scoring='accuracy', cv=5).mean())

#############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
ETree.fit(X, y)

print(cross_val_score(ETree, X, y, scoring='accuracy', cv=5).mean())


# In[ ]:


features_used = ['SexBinary', 'AgeGroup','FamilySize']
X = train[features_used]
test = testing[features_used]

ABC = ensemble.GradientBoostingClassifier()
ABC.fit(X, y)
ABC_out = ABC.predict(test)
xABC_out = ABC.predict(X)

print(cross_val_score(ABC, X, y, scoring='accuracy', cv=5).mean())

#############################################################################################

features_used = ['EmbarkedDiscrete', 'Pclass', 'SexBinary', 'AgeGroup', 'DiscreteTitle', 'FamilySize', 'FareBinned']
X = train[features_used]
test = testing[features_used]
ABC.fit(X, y)

print(cross_val_score(ABC, X, y, scoring='accuracy', cv=5).mean())


# In[ ]:


x_out_matrix = np.concatenate((xk_out.reshape(-1, 1), xclf_out.reshape(-1, 1), xE_out.reshape(-1, 1), 
                                    xR_out.reshape(-1, 1), xLR_out.reshape(-1, 1), xABC_out.reshape(-1, 1)), axis=1)


out_matrix = np.concatenate((k_out.reshape(-1, 1), clf_out.reshape(-1, 1), E_out.reshape(-1, 1), 
                                    R_out.reshape(-1, 1), LR_out.reshape(-1, 1), ABC_out.reshape(-1, 1)), axis=1)

XGB_out = xgb.XGBClassifier()
XGB_out.fit(x_out_matrix, y)
print(cross_val_score(XGB_out, x_out_matrix, y, cv = 5).mean())

# create_csv(XGB_out.predict(out_matrix), 'XGB_customfeat_results.csv', testing)
# ==> 0.77511 Kaggle


# In[ ]:


hardvote_y = np.zeros(len(out_matrix), dtype = 'int')
hardvote_y

for i in range(len(out_matrix)):
    hardvote_y[i] = np.bincount(out_matrix[i]).argmax()

# create_csv(hardvote_y, 'HardVote_customfeat_results.csv', testing)
# ==> 0.78947 Kaggle


# Stacking and Random Forest granted the highest Kaggle scores, both coming in at 0.80382.
# 
# By the way, we checked both the training and the test set and there were no Jack or Rose aboard the Titanic. We feel incredibly let down by James Cameron. 
