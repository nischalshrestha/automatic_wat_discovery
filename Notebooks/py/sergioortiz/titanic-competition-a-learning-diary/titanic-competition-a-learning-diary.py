#!/usr/bin/env python
# coding: utf-8

# # Titanic Competition: A learning diary
# 
# Hi there! This is Sergio and this is my first competition and practical exercise after taking <a href="https://bit.ly/1IXp8Lg">Andrew NG's ML course</a> and having a look at <a href="https://oreil.ly/2nzmN8L">Aurélien Géron's book</a>.  
# 
# Why am I writing a learning diary?  
# First and most important, as a reflection exercise to strengthen my learning. So, let me raise a caution note on the contents as I am not a ML expert but a beginner in this field. In addition to this, I would also like to connect with other people to share and contrast my views. Finally, after many years of practice, I also find this a 'humble exercise' of giving back to the community and supporting other learners.
# 
# Great! Let's start with the learning adventure...
# <hr/>
# 
# ## Table of contents
# * <a href='#learning_strategy'>Defining a learning strategy</a>
# * <a href='#script_kernels'>Script kernels</a>
# * <a href='#data_load'>Loading data</a>
# * <a href='#data_exploration'>Exploring data</a>
# * <a href='#feature_eng'>Feature engineering</a>
# * <a href='#model_selection'>Playing with different models</a>
# * <a href='#model_performance'>Evaluating model performance</a>
# * <a href='#model_tuning'>Fine-tuning model parameters</a>
# * <a href='#save_results'>Saving results</a>
# * <a href='#conclusions'>Conclusions</a>
# <hr/>
# 

# <a id="learning_strategy"></a>
# ## Defining a learning strategy
# From my experience, the most effective learning takes place when you combine theory, practice and reflection - see <a href="https://bit.ly/2AkSCc8" >Kolb's learning cycle</a> <br/>
# Let's see what activities I will take on each of these dimensions:
# * Theory: Andrew NG course  provides a basic theorical basis for ML. Books, popular Kaggle kernels, and other internet resources will be useful to expand my knowledge both on learning algorithms and also in libraries and frameworks.
# * Practice: apply acquired knowledge to develop the best possible solution for the competition.
# * Reflection: writing this diary will force me to synthesise information and reflect on actions done.
# 
# Iterative development will be used to put in practice the strategy and to find the appropriate balance between theory and practice.

# <a id="script_kernels"></a>
# ## Script kernels
# 
# The following kernels contain complete solution scripts for the competition:
# * <a href="https://www.kaggle.com/sergioortiz/titanic-competition-script-1">Script #1: 0.79904</a>
# 
# In the next sections I will be presenting and commenting extracts from these solution scripts.

# <a id="data_load"></a>
# ## Loading data
# Let's start by loading data - both training and test sets.

# In[ ]:


import os
import pandas as pd

input_io_dir="../input/titanic/"

original_train_data=pd.read_csv(input_io_dir+"train.csv")
original_test_data=pd.read_csv(input_io_dir+"test.csv")
print('original_train_data',original_train_data.shape)
print('original_test_data',original_test_data.shape)


# <a id="data_exploration"></a>
# ## Exploring data
# What I think are the main benefits of exploring data?
# * Classify available features - e.g. depending on its type data may need different processing
# * Identify the need to adapt/correct data types, data values outliers, etc...
# * Identify missing data and take decisions (drop/complete)
# * Identify trends and correlations
# * Identify opportunities to create new features from existing
# * Identify potential barriers to model training - e.g. uneven distribution of information between training and test sets
# 
# The following kernels illustrate the different data exploration processes taken:
# * <a href="https://www.kaggle.com/sergioortiz/titanic-competition-data-exploration-1">Iteration #1: initial exploration</a>  
# * <a href="">Iteration #2: further investigation (work in progress)</a>

# <a id="feature_eng"></a>
# ## Feature Engineering
# The objectives for this stage are:
# * Adjusting data types
# * Dealing with missing data
# * Scaling and encoding data
# * Add new features
# * Drop less relevant  features
# 
# The dataset preparation will consist of these steps:
# * Load training and test data
# * Extract labels from training data (Survived column)
# * Extract PassengerId from test data - required for the output generation step
# * Extend the feature set
# * Exclude features we don't like
# * Process data differently depending on its nature
#   * Numeric
#   * Categorical
# * Combine processed data and prepare output data sets
#   * Training data
#   * Training labels
#   * Test data
# 
# The following kernels illustrate the different feature engineering efforts taken:
# * <a href="https://www.kaggle.com/sergioortiz/titanic-competition-feature-engineering-1">Iteration #1: initial feature engineering</a>  
# 
# These kernels are designed to provide output datasets so that results can be used in the following sections.

# <a id="model_selection"></a>
# ## Playing with different models
# Depending on the type of problem,  it is wise to initially evaluate how different learning models apply to your specific data set and circumstances.
# The Titanic competition is basically a binary classification problem. Consequently, these machine learning models are good candidates:
# * DecisionTrees
# * RandomForests
# * XGBoostClassifier
# * GradientBoostingClassifier
# * AdaBoostClassifier
# * LogisticRegression
# * Ensemble of different models
# * Neural networks - to be addressed in next iterations! 
# 
# The following code provides a general framework for model evaluation:

# In[ ]:


from sklearn import ensemble, model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score,train_test_split
from xgboost import XGBClassifier

input_io_dir='../input/titanic-competition-feature-engineering-1/'
# Compare different models
def PrepareDataSets():
    passengerId=pd.read_csv(input_io_dir+"passengerId.csv",header=None)
    train_features=pd.read_csv(input_io_dir+"train_features.csv",header=0)
    train_labels=pd.read_csv(input_io_dir+"train_labels.csv",header=None)
    test_features=pd.read_csv(input_io_dir+"test_features.csv",header=0)
    print('PrepareDataSets: passengerId loaded(%d)'% len(passengerId))
    print('PrepareDataSets: train_features loaded(%d)'% len(train_features))
    print('PrepareDataSets: train_labels loaded(%d)'% len(train_labels))
    print('PrepareDataSets: test_features loaded(%d)'% len(test_features))
    return passengerId,train_features,train_labels, test_features
    
def ModelSelection(clf_list,name_list,train_features,train_labels,scoring='accuracy'):
    best_score=0
    for clf, name in zip(clf_list,name_list) :
        scores = model_selection.cross_val_score(clf, train_features.values.astype(float), train_labels.values.ravel().astype(float), cv=10, scoring=scoring)  
        print("ModelSelection: Scoring  %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
        reference_score=scores.mean()+scores.std()
        if (reference_score>best_score):
            best_clf=name
            best_score=reference_score
            learning_model=clf
    print("ModelSelection: Best model - "+best_clf)
    return learning_model

def ConfigureLearningModelsForBinaryClassification():
    xgb_clf = XGBClassifier(n_estimators=100,max_depth=40, random_state=42)
    dt_clf = DecisionTreeClassifier(random_state=42)
    rf_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    et_clf = ensemble.ExtraTreesClassifier(n_estimators=100, random_state=42)
    gb_clf = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=42)
    ada_clf = ensemble.AdaBoostClassifier(n_estimators=100, random_state=42)
    svm_clf = svm.LinearSVC(C=0.1,random_state=42)
    lg_clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=400,random_state=42)
    e_clf = ensemble.VotingClassifier(estimators=[('xgb', xgb_clf), ('dt', dt_clf),('rf',rf_clf), ('et',et_clf), ('gbc',gb_clf), ('ada',ada_clf), ('svm',svm_clf), ('lg',lg_clf)])
    clf_list = [xgb_clf, dt_clf, rf_clf, et_clf, gb_clf, ada_clf, svm_clf,lg_clf,e_clf]
    name_list = ['XGBoost', 'Decision Trees','Random Forest', 'Extra Trees', 'Gradient Boosted', 'AdaBoost', 'Support Vector Machine', 'LogisticRegression','Ensemble']
    return clf_list,name_list

passengerId,train_features,train_labels, test_features=PrepareDataSets()
clf_list,name_list=ConfigureLearningModelsForBinaryClassification()
learning_model=ModelSelection(clf_list,name_list,train_features,train_labels)


# Good...notice though that models have a very similar performance.  
# However, a single indicator may not be rich enough to understand model performance.

# <a id="model_performance"></a>
# ## Evaluating model performance
# <a href="http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve">Learning curves</a> are a valuable instrument to identify model performance. Particularly, to know whether the learning model is underfitting or overfitting.  
# Let's enrich our perspective on performance analysing the curves for each model.

# In[ ]:


import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Draw learning curve
def plot_learning_curve(ax,learning_model, title, X, y, ylim=None, cv=None, random_state=42,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    ax.set_title(title)
    if ylim is not None:
        ax.ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(learning_model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    ax.legend(loc="best")

def DrawLearningCurves(clf_list,name_list,train_features,train_labels,scoring='accuracy',cols=1,figsize=(20,20)):
    rows=len(clf_list)
    i=1
    f = plt.figure(figsize=figsize)
    for clf, name in zip(clf_list,name_list) :
        ax=f.add_subplot(rows,cols,i)
        plot_learning_curve(ax,clf,name,train_features.values.astype(float), train_labels.values.ravel().astype(float),cv=10)
        i=i+1

passengerId,train_features,train_labels, test_features=PrepareDataSets()
clf_list,name_list=ConfigureLearningModelsForBinaryClassification()
DrawLearningCurves(clf_list,name_list,train_features,train_labels,figsize=(16,60))


# * **Good accuracy but overfitting**  
# These models provide a very good and stable accuracy on the training set but do not generalise well. This is can noticed in the existence of a gap between the training and cross-validation scores.  
# The training score is related with the potential performance of the model while the cross-validation scores defines how well it generalises, that is, how it performs with samples different from these in the training set.  
# Common measures against overfitting are to increase training examples (not feasible), reduce model complexity (not nice as we would reduce our potential) or apply regularisation (preferred option).  
# Let's comment on each particular model:  
#   * Xgboost: despite not having the highest accuracy on the training set (0.885), it's on the top regarding cross-validation performance (0.785 - 0.825 - 0.870).
#   * Decision Trees: great accuracy (0.90) - significant performance gap when using cross-validation set (0.780 - 0.810 - 0.845).
#   * Random Forest:similar accuracy with DecisionTrees (0.90),  generalise better but not as good as Xgboost (0.785 - 0.820 - 0.855).
#   * Extra Trees: similar accuracy with DecisionTrees (0.90),  generalise better but not as good as Xgboost (0.780 - 0.815 - 0.855).
# * **Not as good accuracy but overfitting**  
#   The following models are similar to the former but present lower accuracy scores.
#   * Gradient boosted: potential accuracy scores not as good as previous models (0.855). Cross-validation scores behind training but close (0.780 - 0.820 - 0.855)
#   * Ensemble: good accuracy but behind previous models (0.875). Cross-validation scores not as good but and overfitting (0.785 - 0.825 - 0.870)
# * **Not as good accuracy with signs of underfitting**
#   * AdaBoost: potential accuracy scores not good but stable (0.825). Cross-validation scores (0.785 - 0.820 - 0.845) in similar average values - typical of  underfitting scenarios.
#   * Support vector machine: potential accuracy scores not good and not as stable as other models (0.825 - 0.830 - 0.835). Cross-validation scores in similar average values with pronounced differences in performance (0.800 - 0.830 - 0.860))
#   * Logistic regression: training set accuracy scores not good and slightly unstable (0.830 - 0.835 - 0.840). Cross-validation scores are among the highest and in similar average values as training scores (0.800 - 0.835 - 0.865)
#   
#  In conclusion, logistic regression presents a great performance in the cross-validation set but reduced potential as its training scores are lower than top performing models and more unstable.  
#  Therefore, we will take Xgboost as it presents similar cross-validation performance but higher accuracy on the training scores. Considering we will not be able to increase the training set, we will try to regularise the model to exchange some variance for more accuracy on the test set.
#   

# <a id="model_tuning"></a>
# ## Fine-tuning model parameters
# Next, we are going to set up the hyperparameters for the selected model - Xgboost.  
# Relevant sources of information for this are the parameter tuning section in <a href="https://bit.ly/2Ry4Rrx"> Xgboost official website</a> and any of the existing <a href="https://bit.ly/2cIU6lv">optimisation blogs</a> covering this learning model.  
# Let's build some code to support this process...

# In[ ]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fine tune a model given a param_grid
def FineTuneLearningModel(learning_model, param_grid, train_features,train_labels,scoring='accuracy'):
    grid_search = GridSearchCV(learning_model, param_grid, scoring,cv=10)
    grid_search.fit(train_features.values.astype(float),train_labels.values.ravel().astype(float))
    cvres = grid_search.cv_results_
    for mean_score,std_score, params in zip(cvres["mean_test_score"], cvres["std_test_score"],cvres["params"]):
        print('FineTuneLearningModel:',mean_score,'+-',std_score, params)
    print('FineTuneLearningModel: Best params - '+str(grid_search.best_params_))
    return grid_search.best_estimator_
# Let's run a iterative process in which we
passengerId,train_features,train_labels, test_features=PrepareDataSets()
learning_model = XGBClassifier(objective='binary:logistic')
print('FineTuneLearningModel: round 1 - booster type')
print('---------------------------------------------')
param_grid = [
    {'booster':['gbtree','gblinear'],'n_estimators': [10,30,100],'learning_rate':[0.1]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 2 - complexity params')
print('--------------------------------------------------')
param_grid = [
    {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2),'gamma':[i/10.0 for i in range(0,5)]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 3 - robustness params')
print('--------------------------------------------------')
param_grid = [
    { 'subsample':[1e-5,1e-2,0.1,0.2,0.5,0.8,1], 'colsample_bytree':[1e-5,1e-2,0.1,0.2,0.5,0.8,1]}
]     
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 4 - regularisation')
print('-----------------------------------------------')
param_grid = [
    { 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10,50],'reg_lambda':[0.1,0.5, 1, 2,5,10,50]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))
print('FineTuneLearningModel: round 5 - reduce learning rate as it prevents overfitting')
print('--------------------------------------------------------------------------------')
param_grid = [
    { 'learning_rate': [1e-5,0.01],'n_estimators': [10,100,200,500,1000]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('Current model params:'+str(learning_model))


# During this process it is important to review that scores are consistently improving. If not, check the  model params on previous stage to understand why this happened.  
# Now, we'll have a look at learning curves for the final model...

# In[ ]:


passengerId,train_features,train_labels, test_features=PrepareDataSets()
clf_list=[]
clf_list.append(learning_model)
name_list=['XgBoost']
DrawLearningCurves(clf_list,name_list,train_features,train_labels,figsize=(20,12))


# Let's compare current values with early scores evaluation.
# * Training scores
#   * Model selection stage: 0.885
#   * Fine tuning stage: 0.840-0.845
# * Cross-validation scores
#   * Model selection stage: (0.785 - 0.825 - 0.870)
#   * Fine tuning stage: (0.800 - 0.835 - 0.875)
#  
#   Great...our learning model is fine-tuned and, as planned, we exchanged some accuracy on the training scores to improve our cross-validation performance.  
# 
# Let's now fit the model with the full training set and create predictions...

# In[ ]:


# Train and generate predictions
def TrainModelAndGeneratePredictionsOnTestSet(learning_model,train_features,train_labels,test_features, threshold=-1):
    learning_model.fit(train_features.values.astype(float),train_labels.values.ravel().astype(float))
    if threshold==-1:
        predictions = learning_model.predict(test_features.values.astype(float))
    else:
        if hasattr(learning_model,"decision_function"):
            y_scores=learning_model.decision_function(test_features.values.astype(float))
        else:
            y_proba=learning_model.predict_proba(test_features.values.astype(float))
            y_scores=y_proba[:,1]
        predictions=(y_scores>threshold).astype(float)
    pred=pd.Series(predictions)
    # Ensure no floats go out
    return pred.apply(lambda x: 1 if x>0 else 0)
predictions=TrainModelAndGeneratePredictionsOnTestSet(learning_model,train_features,train_labels,test_features)
print('TrainModelAndGeneratePredictionsOnTestSet: predictions ready')


# Let's have a look at relative relevance of features for the learning model...

# In[ ]:


print('TrainModelAndGeneratePredictionsOnTestSet:Feature importances',sorted(zip(learning_model.feature_importances_,train_features.columns), reverse=True))


# <a id="save_results"></a>
# ## Saving results
# Finally, we will dump results into a file so that we can submit predictions into the platform.

# In[ ]:


def GenerateOutputFile(passengerId,predictions):
    output = pd.DataFrame({ 'PassengerId': passengerId,
                            'Survived': predictions })
    output.to_csv("output.csv", index=False)

passengerId = original_test_data['PassengerId']
GenerateOutputFile(passengerId,predictions)


# <a id="conclusions"></a>
# ## Conclusions
# That's it! We have an initial submission we can post directly into the competition.  
# I hope you enjoyed this learning journey and do not hesitate to comment and share any thoughts you may have!
# 
# Best regards,
# 
# 
# 
