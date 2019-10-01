#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import sklearn
import scipy as sp

import random
import sys
import time

import IPython
from IPython import display

import warnings
warnings.filterwarnings('ignore')
print('-'*25)

import os
print(os.listdir("../input"))
print('-'*25)


# In[2]:


# common model algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from xgboost import XGBRegressor

#common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#configure visualization defaults
get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[3]:


df = pd.read_csv('../input/featureengineering-titanic-procedure/fe_rfrage_scaled_data.csv')
train_raw = pd.read_csv('../input/titanic/train.csv')
print(df.info())


# In[4]:


feature_columns = 'Parch|Pclass|SibSp|Family_Survival|Sex_Code|Embarked_.*|Title_.*|Cabin_.*|Age_scaled|Fare_scaled'
df_data_x = df.filter(regex = feature_columns)
df_data_y = df['Survived']

df_train_x = df_data_x.iloc[:891, :]  # 前891个数据是训练集
df_train_y = df_data_y.iloc[:891]

df_test_x = df_data_x.iloc[891:, :]
df_test_output = df.iloc[891:, :][['PassengerId','Survived']]


# # 4. 模型融合  
# 机器学习的套路是：  
# ·先选择一个基础模型，进行训练和预测，最快建立起一个pipeline。  
# ·在此基础上用交叉验证和GridSearch对模型调参，查看模型的表现。  
# ·用模型融合进行多个模型的组合，用投票的方式（或其他）来预测结果。  
# 一般来说，模型融合得到的结果会比单个模型的要好。  

# ## 4.1 用默认参数算数个模型的CV，从中选取较好的进行进一步的调参+learningcurves+融合

# In[5]:


cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

# MLA_predict = df_train_y
MLA_predict = {}

row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validatiel_selection.
    cv_results = model_selection.cross_validate(alg, df_train_x, df_train_y, cv  = cv_split,return_train_score=True)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    
    #save MLA predictions - see section 6 for usage
    alg.fit(df_train_x, df_train_y)
    MLA_predict[MLA_name] = alg.predict(df_train_x)
    
    row_index+=1
    

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

MLA_compare


# * 选择ExtraTreesClassifier以上的模型进行进一步的调参+融合

# ## 4.2 hyperparameter tunning for best models for ensembling

# In[6]:


vote_est = [
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    ('lr', linear_model.LogisticRegressionCV()),
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    ('knn', neighbors.KNeighborsClassifier()),
    ('svc', svm.SVC(probability=True)),
    ('xgb', XGBClassifier())
]

grid_n_estimator = [10, 50, 100, 300, 500]
grid_ratio = [.5, .8, 1.0]
grid_learn = [.001, .005, .01, .05, .1]
grid_max_depth = [2, 4, 6, 8, 10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
    # AdaBoostClassifier
    {
        'n_estimators':grid_n_estimator,
        'learning_rate':grid_learn,
        'random_state':grid_seed
    },
    # BaggingClassifier
    {
        'n_estimators':grid_n_estimator,
        'max_samples':grid_ratio,
        'random_state':grid_seed
    },
    # ExtraTreesClassifier
    {
        'n_estimators':grid_n_estimator,
        'criterion':grid_criterion,
        'max_depth':grid_max_depth,
        'random_state':grid_seed
    },
    # GradientBoostingClassifier
    {
        'learning_rate':grid_learn,
        'n_estimators':grid_n_estimator,
        'max_depth':grid_max_depth,
        'random_state':grid_seed,

    },
    # RandomForestClassifier
    {
        'n_estimators':grid_n_estimator,
        'criterion':grid_criterion,
        'max_depth':grid_max_depth,
        'oob_score':[True],
        'random_state':grid_seed
    },
    # GaussianProcessClassifier
    {
        'max_iter_predict':grid_n_estimator,
        'random_state':grid_seed
    },
    # LogisticRegressionCV
    {
        'fit_intercept':grid_bool,  # default: True
        'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'random_state':grid_seed
    },
    # BernoulliNB
    {
        'alpha':grid_ratio,
    },
    # GaussianNB
    {},
    # KNeighborsClassifier
    {
        'n_neighbors':range(6, 25),
        'weights':['uniform', 'distance'],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    # SVC
    {
        'C':[1, 2, 3, 4, 5],
        'gamma':grid_ratio,
        'decision_function_shape':['ovo', 'ovr'],
        'probability':[True],
        'random_state':grid_seed
    },
    # XGBClassifier
    {
        'learning_rate':grid_learn,
        'max_depth':[1, 2, 4, 6, 8, 10],
        'n_estimators':grid_n_estimator,
        'seed':grid_seed
    }
]


# tune hyperparameter  
# 可以更改ensemble的模型试试

# In[7]:


start_total = time.perf_counter()
N = 0
for clf, param in zip (vote_est, grid_param):  
    start = time.perf_counter()     
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0) 
    if 'n_estimators' not in param.keys():
        print(clf[1].__class__.__name__, 'GridSearchCV')
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'accuracy')
        best_search.fit(df_train_x, df_train_y)
        best_param = best_search.best_params_
    else:
        print(clf[1].__class__.__name__, 'RandomizedSearchCV')
        best_search2 = model_selection.RandomizedSearchCV(estimator = clf[1], param_distributions = param, cv = cv_split, scoring = 'accuracy')
        best_search2.fit(df_train_x, df_train_y)
        best_param = best_search2.best_params_
    run = time.perf_counter() - start

    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 

run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))


# ## 4.3训练模型，并且查看融合模型CV值，可供前面各种改进的最终参考

# In[8]:


grid_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, df_train_x, df_train_y, cv = cv_split, scoring = 'accuracy')
# grid_hard.fit(df_train_x, df_train_y)

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

grid_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, df_train_x, df_train_y, cv = cv_split, scoring = 'accuracy')
# grid_soft.fit(df_train_x, df_train_y)

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}". format(grid_soft_cv['test_score'].std()*100*3))


# ## 4.4 画learning curves  
# 对模型融合选择作出参考  
# 对过拟合数据：  
# · 做feature selection  
# · 提供更多数据

# In[9]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X = df_train_x
y = df_train_y
cv_lc = model_selection.ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

for alg in vote_est:
    plot_learning_curve(alg[1], alg[0], X, y, ylim=(0.6, 1.01), cv=cv_lc)
    plt.show()


# ## 4.5 看看这几个模型之间的区分度，判断是否适合ensemble

# In[10]:


test_survived = {}
for alg in vote_est:
    alg[1].fit(df_train_x,df_train_y)
    test_survived[alg[0]] = pd.Series(alg[1].predict(df_test_x))
ensemble_results = pd.concat(test_survived, axis=1)
g=sns.heatmap(ensemble_results.corr(),annot=True)


# ## 4.6 模型最终预测结果

# # 6. 提交结果

# In[12]:


grid_hard.fit(df_train_x, df_train_y)
df_test_output['Survived'] = grid_hard.predict(df_test_x)
df_test_output['Survived'] = df_test_output['Survived'].astype(int)
df_test_output.to_csv('model_titanic_procedure.csv', index = False)


# In[ ]:




