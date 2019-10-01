#!/usr/bin/env python
# coding: utf-8

# # Credits
# Many thanks got to the creators of the super-helpful introductory notebooks, where I took a lot of inspiration and actually copied some parts, as they were too good not to use:
# 
# Manav Sehgal https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# Helge Bjorland https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial
# 
# Jeff Delaney https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish
# 
# Anisotropic https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

# In[ ]:


import os
import itertools

import copy
from tempfile import mkdtemp
from shutil import rmtree

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core import display as ICD

get_ipython().magic(u'matplotlib inline')

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# Modelling Helpers
from sklearn.externals.joblib import Memory
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV, SelectKBest, chi2, RFE, SelectPercentile, f_classif, mutual_info_classif, SelectFromModel
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer 


# In[ ]:


import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore')


# In[ ]:


import time
start = time.time()


# In[ ]:


def draw_correlation_map(df):
    correlation = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
            correlation, 
            cmap = cmap,
            square=True, 
            cbar_kws={ 'shrink' : .9 }, 
            ax=ax, 
            annot = True, 
            annot_kws = { 'fontsize' : 8 }
    )


# # Load Data

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "."]).decode("utf8"))
print(check_output(["ls", ".."]).decode("utf8"))


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

combined_df = train_df.append(test_df)

del train_df , test_df


# # Inspect Data before Transformation

# ## Dataframe info
# The field **age** contains ~20% NULL values. It has to be analyzed, whether the age can be imputed by either a constant value or a good guess. 
# 
# **Cabin** only is filled for ~30% of the rows. As there is not enough information it might make no sense to fill up the values.

# In[ ]:


combined_df.info()


# ## Describe Numeric Values

# In[ ]:


combined_df.describe()


# ## Describe Non-Numeric Values
# **Name** is mostly unique for every person, maybe the extraction of titles might be helpful.
# 
# As said before **Cabin** only is filled for ~30% of the rows and might be useless.
# 
# The **ticket** indicates whether a group of passengers has booked together and thus might be family.

# In[ ]:


combined_df.describe(include=['O'])


# ## A heatmap of correlations of columns with the Survival status
# 
# The paid **fare** seems quite a good indicator for survival.
# 
# The **PClass** is an anti-correlation (highest is lowest class) is also a good indicator, but the **Pclass** heavily correlates with the **fare** and might be redundant. 

# In[ ]:


draw_correlation_map(combined_df)


# ## Correlation of Age and Survival
# 
# For **age** one can see there are quite a lot of infants and most passengers are between 18 and 35.
# 
# A lot of infants and old people survived, but a lot of the 18-35 range did not.

# In[ ]:


g = sns.FacetGrid(combined_df)
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


g = sns.FacetGrid(combined_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ## Correlation of Fare and PClass
# Maybe there is a direct correlation Fare and PClass and thus the chance of survival.

# In[ ]:


g = sns.FacetGrid(combined_df, col='Pclass', hue='Survived')
g.map(plt.hist, 'Fare', bins=20, alpha=.7)
g.add_legend()

g = sns.FacetGrid(combined_df[(combined_df.Fare < 400)], col='Pclass', hue='Survived')
g.map(plt.hist, 'Fare', bins=20, alpha=.7)
g.fig.suptitle("Filter highest fare (500)")
g.add_legend()

g = sns.FacetGrid(combined_df[(combined_df.Fare < 400) & (combined_df.Fare > 25)], col='Pclass', hue='Survived')
g.map(plt.hist, 'Fare', bins=20, alpha=.7)
g.fig.suptitle("Filter highest frequent fare and highest fare")
g.add_legend()


# # Easy Transformations

# ## Sex - Transform

# In[ ]:


sex = pd.DataFrame()
sex['Sex'] = combined_df.Sex.map( {'female': 1, 'male': 0} ).astype(int)
sns.barplot(x="Sex", y="Survived", data=combined_df)


# ## Fare - Fill and categorization
# Fill only 1 cell.

# In[ ]:


logical_fare_bins = (-1, 0, 8, 15, 31, 1000)


# In[ ]:


fare = pd.DataFrame()
fare['Survived'] = combined_df.Survived
fare['Pclass'] = combined_df.Pclass
fare['Fare'] = pd.qcut(combined_df.Fare.fillna( combined_df.Fare.mean() ), 5, labels=False)
fare[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)
sns.barplot(x="Fare", y="Survived", data=fare)


# In[ ]:


g = sns.FacetGrid(fare, col='Pclass', hue='Survived')
g.map(plt.hist, 'Fare', bins=5, alpha=.7)
g.add_legend()


# ## Embarked - categorization

# In[ ]:


sns.barplot(x="Embarked", y="Survived", data=combined_df)
embarked = pd.get_dummies( combined_df.Embarked , prefix='Embarked' )       


# ## PClass - categorization

# In[ ]:


sns.barplot(x="Pclass", y="Survived", data=combined_df)
pclass = pd.get_dummies( combined_df.Pclass , prefix='Pclass' )


# # Inspect Transformations

# ## Age - Inspect fill up and categorization
# 
# As we have already seen that there is some correlation between age and the chance of survival, let's try and categorize it.
# 
# Also we will have to fill the gaps, with meaningful values.

# In[ ]:


logical_age_bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
logical_age_group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young_Adult', 'Adult', 'Senior']


# In[ ]:


# Age_Categorized_Mean
age = pd.DataFrame()
age['Age'] = combined_df.Age
age['Survived'] = combined_df.Survived
age['Sex'] = sex['Sex']
age['Pclass'] = combined_df['Pclass']


age['Age_Categorized_Mean'] = pd.cut(age.Age.fillna( age.Age.mean() ), logical_age_bins, labels=logical_age_group_names)
age_mean = age[['Age_Categorized_Mean', 'Survived']].groupby(['Age_Categorized_Mean'], as_index=False).mean().sort_values(by='Age_Categorized_Mean', ascending=True)
age_mean = age_mean.rename(columns={'Age_Categorized_Mean': 'Age_Categorized'})
age_mean['fill_up_type'] = 'mean'

# Age_Categorized_Unknowns
age['Age_Categorized_Unknowns'] = pd.cut(age.Age.fillna( -0.5 ), logical_age_bins, labels=logical_age_group_names)
age_unknown = age[['Age_Categorized_Unknowns', 'Survived']].groupby(['Age_Categorized_Unknowns'], as_index=False).mean().sort_values(by='Age_Categorized_Unknowns', ascending=True)
age_unknown = age_unknown.rename(columns={'Age_Categorized_Unknowns': 'Age_Categorized'})
age_unknown['fill_up_type'] = 'unknowns'

# Age_Categorized_Guess
age['Age_Guess'] = combined_df.Age
guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = age[(age['Sex'] == i) & (age['Pclass'] == j+1)]['Age'].dropna()
        guess_ages[i,j] = int( guess_df.median()/0.5 + 0.5 ) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        age.loc[ (age.Age_Guess.isnull()) & (age.Sex == i) & (age.Pclass == j+1), 'Age_Guess'] = guess_ages[i,j]

age['Age_Categorized_Guess'] = pd.cut(age['Age_Guess'], logical_age_bins, labels=logical_age_group_names)
age_guess = age[['Age_Categorized_Guess', 'Survived']].groupby(['Age_Categorized_Guess'], as_index=False).mean().sort_values(by='Age_Categorized_Guess', ascending=True)
age_guess = age_guess.rename(columns={'Age_Categorized_Guess': 'Age_Categorized'})
age_guess['fill_up_type'] = 'guess'

age_all = age_mean.append(age_unknown).append(age_guess)
sns.barplot(x="Age_Categorized", y="Survived", hue='fill_up_type', data=age_all)
age = pd.get_dummies( age.Age_Categorized_Guess , prefix='Age_Categorized_Guess' )


# # Transform Data

# ##  Name - Extraction of title and clean up

# In[ ]:


titles = combined_df[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip()).unique()
for title in titles:
    print(title, sep=', ', end=', ')


# In[ ]:


title = pd.DataFrame()

title[ 'Title' ] = combined_df[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title['Survived'] = combined_df.Survived
title['Sex'] = sex.Sex


# title
# There are several insights from using the **title** as a feature, e.g. the survival rate of Master is higher than a normal Mr. 

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 7), ncols=3)

sns.barplot(x="Sex", y="Survived", data=title, ax=ax[0])
sns.barplot(x="Title", y="Survived", data=title, ax=ax[1])
sns.barplot(x="Title", y="Survived", hue='Sex', data=title, ax=ax[2])

title = pd.get_dummies( title.Title, prefix='Title')


# ##  Cabin - Extraction of title and clean up

# In[ ]:


cabin = pd.DataFrame()
cabin['Survived'] = combined_df.Survived
cabin['Pclass'] = combined_df.Pclass
cabin['Cabin'] = combined_df.Cabin.fillna( 'U' )
cabin['Cabin'] = cabin['Cabin'].map( lambda c : c[0] )

fig, ax = plt.subplots(figsize=(15, 7), ncols=3)
sns.barplot(x="Cabin", y="Survived", data=cabin, ax=ax[0])
sns.barplot(x="Cabin", y="Pclass", data=cabin, ax=ax[1])
sns.barplot(x="Pclass", y="Survived", data=cabin, ax=ax[2])

cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )


# ##  Ticket - Extraction of title and clean up

# In[ ]:


def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()
ticket['Ticket'] = combined_df[ 'Ticket' ].map( cleanTicket )
ticket['Survived'] = combined_df.Survived
ticket['Pclass'] = combined_df.Pclass

sns.barplot(x="Ticket", y="Survived", data=ticket)

ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )


# In[ ]:


family = pd.DataFrame()
family[ 'FamilySize' ] = combined_df[ 'Parch' ] + combined_df[ 'SibSp' ] + 1

family['Survived'] = combined_df.Survived

fig, ax = plt.subplots(figsize=(15, 7))
sns.barplot(x="FamilySize", y="Survived", data=family, ax=ax)
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )


# ## Concat transformed data into new dataframe

# In[ ]:


transformed_df = pd.DataFrame()

transformed_df['Survived'] = combined_df.Survived
transformed_df['Sex'] = sex.Sex
transformed_df['Fare'] = fare.Fare
transformed_df['Cabin_U'] = cabin.Cabin_U
transformed_df['Cabin_C'] = cabin.Cabin_C
transformed_df['Cabin_E'] = cabin.Cabin_E
transformed_df['Cabin_G'] = cabin.Cabin_G
transformed_df['Cabin_D'] = cabin.Cabin_D
transformed_df['Cabin_A'] = cabin.Cabin_A
transformed_df['Cabin_B'] = cabin.Cabin_B
transformed_df['Cabin_F'] = cabin.Cabin_F
transformed_df['Cabin_T'] = cabin.Cabin_T
transformed_df['Embarked_S'] = embarked.Embarked_S
transformed_df['Embarked_C'] = embarked.Embarked_C
transformed_df['Embarked_Q'] = embarked.Embarked_Q
transformed_df['Pclass_1'] = pclass.Pclass_1
transformed_df['Pclass_2'] = pclass.Pclass_2
transformed_df['Pclass_3'] = pclass.Pclass_3
transformed_df['Age_Categorized_Baby'] = age.Age_Categorized_Guess_Baby    
transformed_df['Age_Categorized_Child'] = age.Age_Categorized_Guess_Child   
transformed_df['Age_Categorized_Teenager'] = age.Age_Categorized_Guess_Teenager    
transformed_df['Age_Categorized_Student'] = age.Age_Categorized_Guess_Student     
transformed_df['Age_Categorized_Young_Adult'] = age.Age_Categorized_Guess_Young_Adult
transformed_df['Age_Categorized_Adult'] = age.Age_Categorized_Guess_Adult   
transformed_df['Age_Categorized_Senior'] = age.Age_Categorized_Guess_Senior
transformed_df['Title_Mr'] = title.Title_Mr
transformed_df['Title_Mrs'] = title.Title_Mrs
transformed_df['Title_Miss'] = title.Title_Miss
transformed_df['Title_Master'] = title.Title_Master
transformed_df['Title_Royalty'] = title.Title_Royalty
transformed_df['Title_Officer'] = title.Title_Officer
transformed_df['Family_Single'] = family.Family_Single
transformed_df['Family_Small'] = family.Family_Small
transformed_df['Family_Large'] = family.Family_Large



# # Inspect Data

# In[ ]:


transformed_df.head()


# In[ ]:


transformed_df.info()


# In[ ]:


transformed_df.describe()


# # Dataset Split into Train and Test

# In[ ]:


train_X = transformed_df[ 0:891 ]
train_y = train_X.Survived
train_X = train_X.drop([
    'Survived',
], axis=1)

train_X, evaluate_X, train_y, evaluate_y = train_test_split(train_X, train_y, random_state=0)

test_X = transformed_df[ 891: ]
test_X = test_X.drop(['Survived'], axis=1)

print(train_X.shape, train_y.shape, evaluate_X.shape, evaluate_y.shape, test_X.shape)


# # Models

# In[ ]:


fits = {}
models = {}
scores = {}
evaluation = {}
feature_selections = {}

cache_dirs = []


# In[ ]:


n_jobs = -1
scorings = ["recall", "precision", "accuracy", "f1"]
scoring_refit = "recall"

grid_search_parameters = {
    'scoring': scorings, 
    'n_jobs': n_jobs, 
    'verbose': 1, 
    'refit': scoring_refit,
    'return_train_score': True,
    'error_score': 0
}


# In[ ]:


def print_features(reduce_dim):
    print('print_features')
    if type(reduce_dim).__name__ == 'SelectKBest':
        feature_scores = ['%.2f' % elem for elem in reduce_dim.scores_ ]
    else:
        feature_scores = np.zeros(len(train_X.columns))
    #feature_scores_pvalues = ['%.3f' % elem for elem in  reduce_dim.pvalues_ ]

    #features_selected_tuple=[(train_X.columns[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in reduce_dim.get_support(indices=True)]
    features_selected_tuple=[(train_X.columns[i], feature_scores[i]) for i in reduce_dim.get_support(indices=True)]
    features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

    print(features_selected_tuple)
    return features_selected_tuple


# In[ ]:


def print_rfecv_features(reduce_dim):
    feature_scores = ['%.2f' % elem for elem in reduce_dim.grid_scores_ ]
    #feature_scores_pvalues = ['%.3f' % elem for elem in  reduce_dim.pvalues_ ]

    #features_selected_tuple=[(train_X.columns[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in reduce_dim.get_support(indices=True)]
    features_selected_tuple=[(train_X.columns[i+1], feature_scores[i]) for i in reduce_dim.get_support(indices=True)]
    features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

    print(features_selected_tuple)
    return features_selected_tuple


# In[ ]:


def evaluate_model(model, cv_X, cv_y):
    return cross_validate(
        clone(model), 
        cv_X, 
        cv_y, 
        scoring=scorings, 
        n_jobs=n_jobs, 
        return_train_score=True
    )


# In[ ]:


def plot_rfecv(model):
    print("Optimal number of features : %d" % model.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(model.grid_scores_) + 1), model.grid_scores_)
    plt.show()


# In[ ]:


def show_group_evaluation(evaluation, model_group, limit=5, max_per_group=False):
    evaluation_stats = []
    for model in evaluation:
        if model.startswith(model_group):
            evaluation_stat = {}
            evaluation_stat['group'] = model.split('-')[0]
            evaluation_stat['type'] = model    
            for scoring in scorings:
                for sample in ['test', 'train']:#['train', 'test']:
                    evaluation_stat[sample + '_' + scoring + '_mean'] = np.array(evaluation[model][sample + '_' + scoring]).mean()
                    evaluation_stat[sample + '_' + scoring + '_std'] = np.array(evaluation[model][sample + '_' + scoring]).std()
            evaluation_stats.append(evaluation_stat)
    evaluation_stats = pd.DataFrame(evaluation_stats)
    for scoring in scorings:
        evaluation_stats_sorted = evaluation_stats.sort_values(['test' + '_' + scoring + '_mean', 'test' + '_' + scoring + '_std'], ascending=[False, True])
        if max_per_group:
            evaluation_stats_sorted = evaluation_stats_sorted.groupby('group', as_index=False).first()
            evaluation_stats_sorted = evaluation_stats_sorted.sort_values(['test' + '_' + scoring + '_mean', 'test' + '_' + scoring + '_std'], ascending=[False, True])
        ICD.display(evaluation_stats_sorted[['group', 'type', 'test' + '_' + scoring + '_mean', 'test' + '_' + scoring + '_std', 'train' + '_' + scoring + '_mean', 'train' + '_' + scoring + '_std']].head(limit))


# In[ ]:


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def show_search(search):
    columns = set([param for param in search.cv_results_['params'][0]])
    
    data = pd.DataFrame(search.cv_results_)
    data_spliced = data[['param_%s' % column for column in columns] + ['mean_test_accuracy', 'std_test_accuracy', 'mean_train_accuracy', 'std_train_accuracy', 'mean_test_precision', 'std_test_precision', 'mean_test_recall', 'std_test_recall']]
    data_spliced.columns = [column.replace('classify__', '').replace('reduce_dim__', '') for column in columns] + ['acc', '+/-acc', 'tr-acc', 'tr+/-acc', 'prec', '+/-prec', 'rec', '+/-rec']
    data_sorted = data_spliced.sort_values(['rec', '+/-rec'], ascending=[False, True])
    data_rounded = np.round(data_sorted, 4)
    print(data_rounded.head(10))


# In[ ]:


def show_scores(scores, model_name, limit=5):
    tmp = []
    for model in scores:
        if model.startswith(model_name):
            tmp.append([model, scores[model][0], scores[model][1]])
    tmp = pd.DataFrame(tmp)
    tmp.columns = ['model_name', 'train', 'test']
    tmp_sorted = tmp.sort_values(['test'], ascending=[False])
    ICD.display(tmp_sorted.head(limit))


# In[ ]:


def clean_cache_dirs(cache_dirs):
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            rmtree(cache_dir)
    cache_dirs = []


# In[ ]:


def build_pipelline(base_model, pipeline, param_grid, reduce_dim_param_grid, grid_search_parameters, cache_dirs):
    param_grid = [{'classify__'+key: value for (key, value) in inner_param_grid.items()} for inner_param_grid in param_grid]
    if type(param_grid) == list:
        for i in range(len(param_grid)):
            for key, value in reduce_dim_param_grid.items():
                param_grid[i][key] = value
    if type(param_grid) == dict:
        for key, value in reduce_dim_param_grid.items():
            param_grid[key] = value
    cachedir = mkdtemp()
    cache_dirs.append(cachedir)
    memory = Memory(cachedir=cachedir, verbose=0)
    pipe = Pipeline(pipeline, memory=memory)
    gridsearch = GridSearchCV(
        pipe, 
        param_grid = param_grid, 
        **grid_search_parameters
    )
    return gridsearch

def build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs):
    
    reduce_dims = [
        None,
        SelectKBest(chi2),
        SelectPercentile(chi2),
        SelectFromModel(LinearSVC(penalty="l1", dual=False))
    ]
    
    scalers = [
        None,
        MinMaxScaler(),
        RobustScaler(quantile_range=(25, 75)),
        Normalizer()
    ]
    
    experiments = []
    
    for reduce_dim in reduce_dims:
        for scaler in scalers:
            experiment_name = model_name
            pipeline = []            
            if reduce_dim != None:
                pipeline.append(('reduce_dim', reduce_dim))
                experiment_name += type(reduce_dim).__name__
            if scaler != None:
                pipeline.append(('scaler', scaler))
                experiment_name += type(scaler).__name__
            pipeline.append(('classify', clone(base_model)))
            experiment = build_pipelline(
                base_model, 
                pipeline , 
                param_grid, 
                {}, 
                grid_search_parameters, 
                cache_dirs
            )
            experiments.append((experiment_name, experiment))
    
    #if hasattr(base_model, 'coef_') or hasattr(base_model, 'feature_importances_'):
    #    rfecv = RFECV(estimator=clone(base_model), step=1, cv=StratifiedKFold(2), scoring='accuracy')
    #    experiments.append((model_name + '-rfecv', rfecv))
    
    return experiments


# In[ ]:


def run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs):
    for (model_name, model) in experiments:
        print('Fit', model_name)
        model.fit(train_X , train_y)

        print('model', type(model).__name__)
        if type(model).__name__ == 'GridSearchCV':
            best_model = model.best_estimator_
            print(type(best_model))
            show_search(model)
            if type(best_model).__name__ == 'Pipeline':
                if 'reduce_dim' in best_model.named_steps:
                    reduce_dim = best_model.named_steps['reduce_dim']
                    selected_features = print_features(reduce_dim)
                    feature_selections[model_name] = [
                        selected_features,
                        model.best_score_
                    ]
        elif type(model).__name__ == 'RFECV':
            plot_rfecv(model)
            best_model = model
            selected_features = print_rfecv_features(best_model)
            feature_selections[model_name] = [
                selected_features,
                model.best_score_
            ]
        else:
            best_model = model
        
        print('best_model', type(best_model).__name__)

        models[model_name] = {}
        models[model_name][scoring_refit] = best_model

        evaluation[model_name] = evaluate_model(best_model, evaluate_X, evaluate_y)
        scores[model_name] = (best_model.score(train_X, train_y), best_model.score(evaluate_X, evaluate_y))
    clean_cache_dirs(cache_dirs)


# ## Model Parameter Search

# ### RandomForestClassifier

# In[ ]:


#reduced_X = 
#reduce_dim = models['RandomForestClassifier'][scoring_refit].named_steps['reduce_dim']
#reduced_X = pd.DataFrame(reduce_dim.transform(train_X), columns=train_X.columns[reduce_dim.get_support()])
#best_model = models['RandomForestClassifier'][scoring_refit].named_steps['classify']

model_name = 'RandomForestClassifier'
base_model = RandomForestClassifier(random_state=0)

param_grid = [
    {
        "max_features" : ['sqrt', 'log2'],
        "max_depth": [None, 1, 5, 10, 100],
        "n_estimators" :[10, 100, 1000],
    }
]

experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


show_group_evaluation(evaluation, model_name)


# In[ ]:


show_scores(scores, model_name)


# 
# ### C-Support Vector Classification

# In[ ]:


model_name = 'SVC'
base_model = SVC(probability=True, random_state=0)

param_grid = [{
    'kernel': ['rbf', 'linear'], 
    'gamma': ['auto', 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013],
    'C': [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.5, 0.9, 1, 1.5, 2 , 3, 4]
}]

#experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
#run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


#show_group_evaluation(evaluation, model_name)


# In[ ]:


#show_scores(scores, model_name)


# ### GradientBoostingClassifier

# In[ ]:


model_name = 'GradientBoostingClassifier'
base_model =  GradientBoostingClassifier(random_state=0, loss='deviance')

param_grid = [{
    'loss' : ['deviance', 'exponential'],
    'n_estimators' : [50, 100, 150],
    'learning_rate': [0.05, 0.075, 0.1],
    'max_depth': [2, 3, 5, 6],
    'min_samples_leaf': [5, 10],
    'max_features' : [None, 'auto', 'log2']
}]

experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


show_group_evaluation(evaluation, model_name)


# In[ ]:


show_scores(scores, model_name)


# ### KNeighborsClassifier

# In[ ]:


model_name = 'KNeighborsClassifier'
base_model =  KNeighborsClassifier()

param_grid = [{
    'n_neighbors' : [3, 5, 7, 9],
    'leaf_size': [10, 30, 50, 75],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}]

experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


show_group_evaluation(evaluation, model_name)


# In[ ]:


show_scores(scores, model_name)


# ### GaussianNB

# In[ ]:


model_name = 'GaussianNB'
base_model =  GaussianNB()

param_grid = [{}]

experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


show_group_evaluation(evaluation, model_name)


# In[ ]:


show_scores(scores, model_name)


# ### LogisticRegression

# In[ ]:


model_name = 'LogisticRegression'
base_model =  LogisticRegression(random_state=0)

param_grid = [{
    'C':[1.0, 2.0, 3.0],
    'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter':[50, 100, 500, 1000]
}]

experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


show_group_evaluation(evaluation, model_name)


# In[ ]:


show_scores(scores, model_name)


# ### Perceptron 

# In[ ]:


model_name = 'Perceptron'
base_model =  Perceptron(random_state=0)

param_grid = [{
    'penalty' : ['l2', 'l1', 'elasticnet'],
    'alpha' : [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.003, 0.004],
    'max_iter':[5, 10, 15, 25, 30]
}]

#experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
#run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


#show_group_evaluation(evaluation, model_name)


# In[ ]:


#show_scores(scores, model_name)


# ### LinearSVC

# In[ ]:


model_name = 'LinearSVC'
base_model =  LinearSVC(random_state=0, dual=False)

param_grid = [{
    'penalty' : ['l2', 'l1'],
    'C' : [0.01, 0.025, 0.05, 0.75, 0.1]
}]

#experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
#run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


#show_group_evaluation(evaluation, model_name)


# In[ ]:


#show_scores(scores, model_name)


# ### SGDClassifier

# In[ ]:


model_name = 'SGDClassifier'
base_model =  SGDClassifier(random_state=0)

param_grid = [{
    'max_iter' : [5, 6, 7, 8],
    'loss': ['hinge', 'log', 'modified_huber', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
}]

#experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
#run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


#show_group_evaluation(evaluation, model_name)


# In[ ]:


#show_scores(scores, model_name)


# ### DecisionTreeClassifier

# In[ ]:


model_name = 'DecisionTreeClassifier'
base_model =  DecisionTreeClassifier(random_state=0)

param_grid = [{
    "max_depth": [7, 8, 9, 10, 11, 12],
    "max_features": [10, 11, 12, 13, 14],
    "min_samples_split": [6, 7, 8, 9, 10, 11, 12],
    "min_samples_leaf": [5, 6, 7, 8, 9, 10, 11]
}]

#experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
#run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


#show_group_evaluation(evaluation, model_name)


# In[ ]:


#show_scores(scores, model_name)


# ### ExtraTreesClassifier

# In[ ]:


model_name = 'ExtraTreesClassifier'
base_model =  ExtraTreesClassifier(random_state=0, bootstrap=False, criterion='gini')

param_grid = [{
    "max_depth": [None, 10, 100],
    "max_features": [None, 'sqrt', 'log2'],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [10],
    "n_estimators" :[10, 100, 1000]
}]

experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


show_group_evaluation(evaluation, model_name)


# In[ ]:


show_scores(scores, model_name)


# ### AdaBoostClassifier

# In[ ]:


model_name = 'AdaBoostClassifier'
base_model =  AdaBoostClassifier(random_state=0, base_estimator = None)

param_grid = [{
    #"base_estimator": []
    "n_estimators" : [10, 50, 100, 500],
    "algorithm" : ["SAMME","SAMME.R"],
    "learning_rate":  [0.01, 0.1, 1.0]
}]

#experiments = build_experiments(model_name, base_model, param_grid, grid_search_parameters, cache_dirs)
#run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, models, evaluation, feature_selections, scores, cache_dirs)


# In[ ]:


#show_group_evaluation(evaluation, model_name)


# In[ ]:


#show_scores(scores, model_name)


# ## Evaluation of Models

# ### Scores

# In[ ]:


show_group_evaluation(evaluation, '', limit=10)


# In[ ]:


show_group_evaluation(evaluation, '', limit=10, max_per_group=True)


# In[ ]:


show_scores(scores, '', limit=20)


# ### Correlation of Predictions

# In[ ]:


test_Survied = {}
for model_name in models:
    test_Survied[model_name] = pd.Series(models[model_name][scoring_refit].predict(test_X), name=model_name)

ensemble_results = pd.concat(test_Survied,axis=1)

fig, ax = plt.subplots(figsize=(15,15))
g= sns.heatmap(ensemble_results.corr(),annot=True, ax=ax)


# # Ensemble Learning

# In[ ]:


ensemble_scores = {}
ensemble_evaluations = {}
ensemble_models = {}
ensemble_feature_selections = {}


# ## VotingClassifier

# In[ ]:


good_estimators = [
    'KNeighborsClassifierSelectFromModelRobustScaler',
    'ExtraTreesClassifierNormalizer',
    'GradientBoostingClassifierSelectFromModelNormalizer',
    'LogisticRegressionRobustScaler',
    'RandomForestClassifierSelectFromModelRobustScaler'
]
good_estimator_tuples = []
for estimator_name in good_estimators:
    good_estimator_tuples.append((estimator_name, clone(models[estimator_name][scoring_refit])))


# In[ ]:


estimators_permutations = list(itertools.combinations(good_estimator_tuples, 2))
estimators_permutations.extend(list(itertools.combinations(good_estimator_tuples, 3)))
estimators_permutations.extend(list(itertools.combinations(good_estimator_tuples, 4)))
estimators_permutations.extend(list(itertools.combinations(good_estimator_tuples, 5)))


# In[ ]:


last_length = 2
for estimators_permutation in estimators_permutations:
    name = ",".join(["".join([c for c in item[0] if c.isupper()]) for item in estimators_permutation])
    print(name, sep="- ", end=" ")
    model_vc = VotingClassifier(estimators = estimators_permutation, voting="soft")
    experiments = [    
        (name, model_vc),
    ]
    run_experiments(train_X , train_y, evaluate_X, evaluate_y, experiments, ensemble_models, ensemble_evaluations, ensemble_feature_selections, ensemble_scores, cache_dirs)
    if len(estimators_permutation) > last_length:
        print(last_length)
    last_length = len(estimators_permutation)


# ## Evaluation of Ensemble Learning

# In[ ]:


show_group_evaluation(ensemble_evaluations, '', limit=10)


# In[ ]:


show_scores(ensemble_scores, '', limit=10)


# # Deployment

# In[ ]:


ensemble_models['KNCSFMRS,ETCN,GBCSFMN,RFCSFMRS'][scoring_refit]


# In[ ]:


test_Y = ensemble_models['KNCSFMRS,ETCN,GBCSFMN,RFCSFMRS'][scoring_refit].predict( test_X )
passenger_id = combined_df[891:].PassengerId
prediction = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
prediction.Survived = prediction.Survived.astype(int)
prediction.shape
print(prediction.head())
prediction.to_csv( 'titanic_prediction.csv' , index = False )


# In[ ]:


output = pd.read_csv('titanic_prediction.csv')
output.head()


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "."]).decode("utf8"))
print(check_output(["ls", ".."]).decode("utf8"))


# In[ ]:


end = time.time()
print("Analysis took %0.2f seconds to train"%(end - start))

