#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning
# ### Logistic Regression - Titanic

# Layout of this notebook
# -------------------------------------------------------------------------------------
# Step 1 - Frame the problem and look at the big picture<br><br>
# Step 2 - Setup<br>
#         2.1 - Common imports <br>
#         2.2 - Define standard function used in this notebook<br><br>
# Step 3 - Get the data<br>
#         3.1 - Combine train and test data <br><br>
# Step 4 - Explore and process data<br>
#         4.1 - Check basic info and stats<br>
#         4.2 - Fill missing values ['Embarked', 'Fare']<br>
# 		4.3 - Rename categorical data<br>
# 		4.4 - Modify Cabin data<br>
# 		4.5 - Create columns for no. of pass. on same ticket and individual Fare<br>
# 		4.6 - Create a new column of family size<br>
# 		4.7 - Modify 'SibSp' and 'Parch' data<br>
# 		4.8 - Create 'Title' Column<br>
# 		4.9 - Create 'Length of Name' Column<br>
# 		4.10 - Create prefix of 'Ticket' as new Column<br>
# 		4.11 - Process data<br>
#         4.12 - Find and delete rows with outlier data<br>
# 		4.13 - Create a new column of Fare Group <br>
#         4.14 - Analyze missing age data in detail<br>
#         4.15 - Fill missing age data<br>
#         4.16 - Create a new column of Age Category<br>
#         4.17 - Visualize numerical features<br>
#         4.18 - Fix skewness and normalize<br>
#         4.19 - Analyze survival data - Visualizing outcome across independent variables<br><br>
# Step 5 - Prepare the data for Machine Learning algorithms<br>
#         5.1 - Process further and create train, target and test dataset <br>
#         5.2 - Visualize how Machine Learning models works on classification with just 2 numerical features<br><br>
# Step 6 - Select and train a model <br>
#         6.1 - Visualize Machine Learning models<br>
# 		6.2 - Comparing Classification Machine Learning models with all independent features.<br>
#         6.3 - Extensive GridSearch with valuable parameters / score in a dataframe<br>
#         6.4 - Find best estimators<br>
#         6.5 - Plot learning curves<br>
#         6.6 - Feature Importance<br><br>
# Step 7 - Fine-tune your model<br>
# 		7.1 - Create voting classifier<br><br>
# Step 8 - Predict and present your solution<br><br>
# Step 9 - Final words!

#  #### **Step 1 - Frame the problem and look at the big picture**

# 1.1  - Define the objective in business terms.<br>
# ans  - Complete the analysis of what sorts of people were likely to survive in the unfortunate event of sinking of Titanic. <br>In particular, it is asked to apply the tools of machine learning to predict which passengers survived the tragedy.<br>
# <br>1.2  - How should you frame this problem? (supervised/unsupervised, online/offline, etc.)?<br>
# ans  - This is supervised learning task because we know output for a set of passengers' data.<br>
#        This is also logistic regression (clsiification) task, since you are asked to predict a binary outcome. <br>
# <br>1.3  - How should performance be measured?<br>
# ans  - Metric - Your score is the percentage of passengers you correctly predict. <br>This is known simply as "accuracy”. I will use other performance measures as well, such as K-fold cross validation, <br>f1_score (combination of precison and recall) etc.

#  #### **Step 2 - Setup**

# **Common imports**

# In[ ]:


# Start time for script
import time
start = time.time()

# pandas / numpy etc
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.special import boxcox1p

# To plot pretty figures
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().magic(u'matplotlib inline')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import seaborn as sns
sns.set_style('dark', {'axes.facecolor' : 'lightgray'})

# for seaborn issue:
import warnings
warnings.filterwarnings('ignore')

# machine learning [Classification]
from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV)
from sklearn.preprocessing import (StandardScaler)
from sklearn.metrics import (accuracy_score, f1_score, log_loss, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier

kfold = StratifiedKFold(n_splits=5)
rand_st =42


# **Define standard function used in this notebook**

# **Function 1 - outliers_iqr [Function to find and delete ouliers]**

# In[ ]:


# Function to find and delete ouliers. [I have to run few steps outside function to be on safer side]
# It uses multiples of IQR (Inter Quartile Range) to detect outliers in specified columns

def outliers_iqr(df, columns_for_outliers, iqr_factor):
    for column_name in columns_for_outliers:
        if not 'Outlier' in df.columns:
            df['Outlier'] = 0
        q_75, q_25 = np.nanpercentile(df[column_name], [75 ,25])
        iqr = q_75 - q_25
        minm = q_25 - (iqr*iqr_factor)
        maxm = q_75 + (iqr*iqr_factor)        
        df['Outlier'] = np.where(df[column_name] > maxm, 1, np.where(df[column_name] < minm, 1, df['Outlier']))
    df['Outlier'] = np.where(df.Survived.notnull(), df['Outlier'], 0) # extra step to make sure only train data rows are deleted
    total_rows_del = df.Outlier.sum()
    print('Total ', total_rows_del, ' rows with outliers from comb_data can be deleted')


# **Function 2 - plot_class_models_two_num_features [visualize classification Machine Learning models with two numerical features]**

# In[ ]:


# Create function to visualize how different Machine Learning models look, by using just 2 numerical features.
# [It is not possible to plot if more than 2 features are used] [use 'X_num' and 'y_train']

# Define set of classifiers
clf_dict = {"clf_Log_reg" : LogisticRegression(random_state=rand_st),
            "clf_Lin_SVC" : SVC(kernel="linear", C=0.1, cache_size=5000, probability=True, random_state=rand_st), 
            "clf_Poly_SVC" : SVC(kernel="poly", degree=2, random_state=rand_st), 
            "clf_Ker_SVC" : SVC(kernel="rbf", C=50, cache_size=5000, gamma=0.001, probability=True, random_state=rand_st), 
            "clf_KNN" : KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
            metric_params=None, n_jobs=-1, n_neighbors=13, p=2, weights='uniform'), 
            "clf_GNB" : GaussianNB(),
            "clf_MLP" : MLPClassifier(alpha=0.0001, learning_rate_init=0.05, shuffle=True, random_state=rand_st),
            "clf_Dec_tr" : DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=1, min_samples_split=2, 
            splitter='best', random_state=rand_st),
            "clf_Gauss" : GaussianProcessClassifier(random_state=rand_st),
            "clf_RF" : RandomForestClassifier(criterion='gini', max_depth=6, n_estimators = 350, n_jobs=-1, 
            random_state=rand_st),
            "clf_AdaBoost" : AdaBoostClassifier(algorithm='SAMME.R', base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8,
            max_features=12, min_samples_leaf=1, min_samples_split=2, random_state=42, splitter='best'), learning_rate=0.2, n_estimators=2, random_state=rand_st),
            "clf_GrBoost" : GradientBoostingClassifier(learning_rate=0.1, loss='deviance', max_depth=4, n_estimators=1500, 
            max_features=12, min_samples_leaf=100, min_samples_split=200, subsample=0.8, random_state=rand_st),
            "clf_ExTree" : ExtraTreesClassifier(criterion='gini', max_depth=4, min_samples_leaf=2, min_samples_split=2,
            n_estimators=200, n_jobs=-1, random_state=rand_st),
            "clf_XGBoost" : XGBClassifier(colsample_bytree=0.6, gamma=0, learning_rate=0.05, max_depth=5, n_estimators=3000,
            n_jobs=-1, reg_alpha=0.01, subsample=0.8, random_state=rand_st)}

# Create a function to plot different classification models
# Define fixed inputs
h = .02  # step size in the mesh

# Define function
def plot_class_models_two_num_features(X, y, clf_dict, title):
    import warnings
    warnings.filterwarnings('ignore')
    figure = plt.figure(figsize=(27, 10))
    
    # preprocess dataset, split into training and test part
    dataset = (X, y)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(dataset), len(clf_dict) + 1, 1)
    ax.set_title(title)
    
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i=2
    
    # iterate over classifiers
    for clf_name, clf in clf_dict.items():
        ax = plt.subplot(len(dataset), len(clf_dict) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(clf_name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right', verticalalignment = 'top', color='black',                 bbox=dict(facecolor='yellow', alpha=0.5))
        i += 1
    
    plt.tight_layout()
    plt.show()


# **Function 3 - clf_cross_val_score_and_metrics [Function to evaluate various classification models]**

# In[ ]:


# Function to evaluate various classification models [Metrics and cross_val-score]
# Cross validate model with Kfold stratified cross val

def clf_cross_val_score_and_metrics(X, y, clf_dict, CVS_scoring, CVS_CV):
    # Train and Validation set split by model_selection
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=rand_st)
    metric_cols = ['clf_name', 'Score', 'Accu_Preds', 'F1_Score', 'Log_Loss', 'CVS_Best', 'CVS_Mean', 'CVS_SD']
    clf_metrics = pd.DataFrame(columns = metric_cols)
    metric_dict = []
    
    # iterate over classifiers   
    for clf_name, clf in clf_dict.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        Score = "{:.3f}".format(clf.score(X_val, y_val))
        Accu_Preds = accuracy_score(y_val, y_pred, normalize=False)
        F1_Score = "{:.3f}".format(f1_score(y_val, y_pred))
        Log_Loss = "{:.3f}".format(log_loss(y_val, y_pred))
        
        CVS_values = cross_val_score(estimator = clf, X = X, y = y, scoring = CVS_scoring, cv = CVS_CV, n_jobs=-1)
        CVS_Best = "{:.3f}".format(CVS_values.max())
        CVS_Mean = "{:.3f}".format(CVS_values.mean())
        CVS_SD = "{:.3f}".format(CVS_values.std())
        
        metric_values = [clf_name, Score, Accu_Preds, F1_Score, Log_Loss, CVS_Best, CVS_Mean, CVS_SD]        
        metric_dict.append(dict(zip(metric_cols, metric_values)))
        
    clf_metrics = clf_metrics.append(metric_dict)
    # Change to float data type
    for column_name in clf_metrics.drop('clf_name', axis=1).columns:
        clf_metrics[column_name] = clf_metrics[column_name].astype('float')
    clf_metrics.sort_values('CVS_Mean', ascending=False, na_position='last', inplace=True)
    print(clf_metrics)
    
    clf_bp = sns.barplot(x='CVS_Mean', y='clf_name', data = clf_metrics, palette='inferno',orient = "h",**{'xerr':clf_metrics.CVS_SD})
    clf_bp.set_xlabel("Mean Accuracy")
    clf_bp.set_ylabel("Classification Models")
    clf_bp.set_title("Cross Validation Scores")


# **Function 4 - clf_GridSearchCV_results [Function to conduct extensive GridSearch and return valuable parameters / score in a dataframe]**

# In[ ]:


# Define estimator and parameters grid for Grid Search CV
# Fitting Neural Net to the Training set

clf_GP_gs = GaussianProcessClassifier(random_state=rand_st)
clf_GP_pg = [{'n_restarts_optimizer': [0], 
              'warm_start': [True], 
              'max_iter_predict': [200]}]

clf_RF_gs = RandomForestClassifier(random_state=rand_st)
clf_RF_pg = [{'max_depth': [7, 8, 9], 
              'max_features': ['auto', 12],  
              'criterion': ['gini'], 
              'n_estimators': [200],
              "min_samples_split": [10],
              "min_samples_leaf": [3]}]

clf_MLP_gs = MLPClassifier(random_state=rand_st)
clf_MLP_pg = [{'activation': ['relu'], 
               'solver': ['adam'], 
               'learning_rate': ['adaptive'], 
               'max_iter': [30],
               'alpha': [0.01], 
               'shuffle': [True, False], 
               'learning_rate_init': [0.01]}]

clf_Dec_tr = DecisionTreeClassifier(random_state=rand_st)
clf_AdaBoost_gs = AdaBoostClassifier(clf_Dec_tr, random_state=rand_st)
clf_AdaBoost_pg = {"base_estimator__criterion" : ["entropy"],
              "base_estimator__splitter" :   ["best"],
              "algorithm" : ["SAMME.R"],
              "n_estimators" :[500],
              "learning_rate":  [0.1]}

clf_Ex_tr_gs = ExtraTreesClassifier(random_state=rand_st)
clf_Ex_tr_pg = {"max_depth": [None, 8],
              "max_features": ['auto', 10],
              "min_samples_split": [5],
              "min_samples_leaf": [3],
              "n_estimators" :[200],
              "criterion": ["gini"]}

clf_XGB_gs = XGBClassifier(random_state=rand_st)
clf_XGB_pg = {'learning_rate': [0.01], 
              'max_depth': [5],
              'subsample': [0.8],
              'colsample_bytree': [0.6],
              'n_estimators': [3000], 
              'reg_alpha': [0.05]}

clf_GB_gs = GradientBoostingClassifier(random_state=rand_st)
clf_GB_pg = {'min_samples_split' : [100],
              'n_estimators' : [3000],
              'learning_rate': [0.1],
              'max_depth': [4],
              'subsample': [0.8],
              'min_samples_leaf': [100],
              'max_features': ['auto', 10]}

clf_SVC_gs = SVC(random_state=rand_st)
clf_SVC_pg = [{'C': [1], 
               'kernel': ['linear'],
               'gamma': [0.5]}]

clf_models_gs = [clf_GP_gs, clf_RF_gs, clf_MLP_gs, clf_AdaBoost_gs, clf_Ex_tr_gs, clf_XGB_gs, clf_GB_gs, clf_SVC_gs]
clf_models_gs_name = ['clf_GP_gs', 'clf_RF_gs', 'clf_MLP_gs', 'clf_AdaBoost_gs', 'clf_Ex_tr_gs', 'clf_XGB_gs', 'clf_GB_gs', 'clf_SVC_gs']
clf_params_gs = [clf_GP_pg, clf_RF_pg, clf_MLP_pg, clf_AdaBoost_pg, clf_Ex_tr_pg, clf_XGB_pg, clf_GB_pg, clf_SVC_pg]
gs_metric_cols = ['clf_name', 'Best_Score', 'Mean_Train_Score', 'Mean_Test_Score', 'Mean_Test_SD', 'Best_Estimator', 'Best_Params']
gs_metrics = pd.DataFrame(columns = gs_metric_cols)

# Define function to conduct extensive GridSearch and return valuable parameters / score in a dataframe
def clf_GridSearchCV_results(gs_metrics, X_train, y_train, GS_scoring, GS_CV):
    
    gs_metric_dict = []
    # iterate over classifiers and param grids 
    for clf_gs_name, clf_gs, params_gs in zip(clf_models_gs_name, clf_models_gs, clf_params_gs):
        clf_gs = GridSearchCV(clf_gs,param_grid = params_gs, cv=GS_CV, scoring=GS_scoring, n_jobs= -1, verbose = 1)
        clf_gs.fit(X_train,y_train)
        
        clf_name = clf_gs
        Best_Score = clf_gs.best_score_
        Mean_Train_Score = np.mean(clf_gs.cv_results_['mean_train_score'])
        Mean_Test_Score = np.mean(clf_gs.cv_results_['mean_test_score'])
        Mean_Test_SD = np.mean(clf_gs.cv_results_['std_test_score'])
        Best_Estimator = clf_gs.best_estimator_
        Best_Params = clf_gs.best_params_
        
        gs_metric_values = [clf_gs_name, Best_Score, Mean_Train_Score, Mean_Test_Score, Mean_Test_SD, Best_Estimator, Best_Params]        
        gs_metric_dict.append(dict(zip(gs_metric_cols, gs_metric_values)))
        
    gs_metrics = gs_metrics.append(gs_metric_dict)
    return gs_metrics


# **Function 5 - plot_learning_curve [Function to plot learning curves]** 

# In[ ]:


# Define function to plot learning curves
def plot_learning_curve(estimator, title, X_train, y_train, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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


#  #### **Step 3 - Get the data**

# In[ ]:


# Manual method
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Combine dataset to create one dataframe for exploration and pre-processing purpose
comb_data = pd.concat([train_data, test_data])

# Mark 'train' and 'test' dataset 
comb_data['DataType'] = np.where(comb_data[['Survived']].isnull().all(1), 'test', 'train')

comb_data.head()


# *For exploration and pre-processing, I combined data which can be later split in train and test set.*

#  #### **Step 4 - Explore and process data**

# **Check basic info and stats**

# In[ ]:


# Check basic stats [Numerical features]
comb_data.describe().transpose()


# *It seems Fare, Parch and SibSp features have outliers*

# In[ ]:


# Check basic stats [Categorical features]
comb_data.describe(include=['object', 'category']).transpose()


# In[ ]:


# Check basic info
print("-----------------------Train Data-----------------------------")
comb_data[comb_data.DataType == 'train'].info()
print("-----------------------Test Data-----------------------------")
comb_data[comb_data.DataType == 'test'].info()
print("-----------------------Combined Data-----------------------------")
comb_data.info()


# *Notice that many categorical attributes are treated as numerical or object data type. There are missing values too.*

# In[ ]:


# Check null values in each column
print(comb_data.isnull().sum())


# *Inferences:*<br>
# *1. 'Age' and 'Fare' are two numerical variables.<br>
# *2. 'Cabin' column has 77.4% missing values. This column can be transformed to indicate 'No Cabin'(null values) and 'Deck' level.<br>
# *3. 'Embarked' column has only 2 missing values, so it can be easily filled with the most frequent category.<br>
# *4. 'Fare' column has only 1 missing values. It can be filled with median value.<br>
# *5. 'Age' column, which might be important for modeling, has 20% missing values. <br> We will handle it separately with some exploration*<br>
# *6. It looks like 'Parch' and 'SibSp' columns have subcategories which can be grouped into one.*

# **Fill missing values ['Embarked', 'Fare']**

# In[ ]:


# Fill with median values [only columns in which missing values are very few].
for column_name in ['Embarked', 'Fare']:
    comb_data[column_name].fillna(comb_data[column_name].value_counts().index[0], inplace=True)


# **Rename categorical data**

# In[ ]:


# Change categorical values to more meaningful values [For visulaization only, I will convert to numerical type before running ML models]
comb_data['Pclass'] = np.where(comb_data['Pclass']==1, 'UpperClass', np.where(comb_data['Pclass']==2, 'MiddleClass', 'LowerClass'))
comb_data['Embarked'].replace(['C','Q', 'S'],['Cherbourg','Queenstown', 'Southampton'], inplace=True)
comb_data['Sex'] = np.where(comb_data['Sex']=='male', 'Male', 'Female')
comb_data['Survived'].replace([0,1],['No','Yes'], inplace=True)


# **Modify Cabin data**

# In[ ]:


# Recreate 'Cabin' with first character which represents deck and 'N' for null values.
comb_data['Cabin'] = np.where(comb_data[['Cabin']].isnull().all(1), 'N', comb_data.Cabin.str[0])
comb_data['Cabin'].value_counts().sort_values(ascending=False)


# **Create columns for no. of pass. on same ticket and individual Fare**

# In[ ]:


# There are many passengers travelling on same ticket and 'Fare' is total fare for all passengers on the same ticket.
comb_data[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).count().sort_values(by='Fare', ascending=False).head()


# In[ ]:


# Create 'PassCountTicket' column to show no. of passengers on same ticket
# Let us explore 11 passengers travelling on same ticket
comb_data['PassCountTicket'] = comb_data['Ticket'].map(comb_data['Ticket'].value_counts())
comb_data[comb_data.Ticket=='CA. 2343']


# In[ ]:


# Create 'IndFare' column by dividing 'Fare' by no. of passengers on same ticket
comb_data['IndFare'] = comb_data.Fare / comb_data.PassCountTicket

# Check 'IndFare' data
comb_data[comb_data.Ticket=='CA. 2343']


# In[ ]:


# Check how many passengers are in each unique 'PassCountTicket' value
comb_data['PassCountTicket'].value_counts().sort_values(ascending=False)


# **Create a new column of family size**

# In[ ]:


# A person with zero 'SibSp' and 'Parch' is travelling alone
comb_data['FamSize'] = comb_data['SibSp'] + comb_data['Parch'] + 1
print(comb_data['FamSize'].value_counts().sort_values(ascending=False))

# Visulize 'FamSize' data across 'Fare' and 'Survived'
v0 = sns.violinplot(data=comb_data[comb_data.DataType=='train'], x='FamSize', y='Fare', hue='Survived', scale='count', split=True, inner="stick")
v0.set_title('Survival across Family Size & Age', fontsize = 15)
plt.show()


# In[ ]:


# Create 'Single', 'Small' and 'Large' Category
comb_data['FamSize'] = np.where(comb_data['FamSize']<2, 'Single', np.where(comb_data['FamSize']<5, 'Small', 'Large'))
comb_data['FamSize'] = comb_data['FamSize'].astype('category')


# **Modify 'SibSp' and 'Parch' data**

# In[ ]:


# Check the count of unique nos. in 'SibSp' and 'Parch' columns
print(comb_data['Parch'].value_counts().sort_values())
print(comb_data['SibSp'].value_counts().sort_values())


# In[ ]:


# Reduce the number of categories in Parch and SibSp as more than 4 are insignificant.
comb_data['Parch'].replace([5, 6, 9],[4, 4, 4], inplace=True)
comb_data['SibSp'].replace([5, 8],[4, 4], inplace=True)


# **Create 'Title' Column**

# In[ ]:


# Create Title columns from 'Name'
comb_data['Title'] = comb_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print(comb_data['Title'].value_counts().sort_values(ascending=False))


# In[ ]:


# Clean up Title column categories
comb_data['Title'] = comb_data['Title'].replace('Mlle', 'Miss')
comb_data['Title'] = comb_data['Title'].replace('Ms', 'Miss')
comb_data['Title'] = comb_data['Title'].replace('Mme', 'Mrs')

comb_data['Title'] = comb_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
print(comb_data['Title'].value_counts().sort_values(ascending=False))


# **Create 'Length of Name' Column**

# In[ ]:


# Create feture for length of name 
# The .apply method generates a new series
comb_data['NameLength'] = comb_data['Name'].apply(lambda x: len(x))


# **Create prefix of 'Ticket' as new Column**

# In[ ]:


# Create new column from 'Ticket' by extracting the ticket prefix. When there is no prefix it returns "N". 

TicketTrim = []
for i in list(comb_data.Ticket):
    if not i.isdigit() :
        TicketTrim.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        TicketTrim.append("N")
        
        
comb_data["TicketTrim"] = TicketTrim
comb_data["TicketTrim"].value_counts().sort_values(ascending=False).head(10)


# **Process data**

# In[ ]:


# Drop columns not needed further
comb_data = comb_data.drop(labels = ['Name', 'Ticket', 'PassCountTicket'],axis = 1)
comb_data.head()


# In[ ]:


# Check data types of columns
comb_data.dtypes


# In[ ]:


# Change PassengerId data type so that it does not show up in plots. (change back to integer before applying ML models)
comb_data["PassengerId"] = comb_data["PassengerId"].astype(str)
# Change categorical columns to category data type
for column_name in ['Cabin', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamSize', 'Title', 'TicketTrim']:
    comb_data[column_name] = comb_data[column_name].astype('category')   
comb_data.dtypes


# **Find and delete rows with outlier data**

# In[ ]:


# Drop rows with extreme outlier data [iqr_factor=10, normally this is equal to 1.5]
# List columns for outliers processing
columns_for_outliers = ['Age', 'Fare', 'NameLength']

# Run function
outliers_iqr(comb_data, columns_for_outliers, 10)

# Delete rows with outlier
comb_data = comb_data[comb_data.Outlier != 1]

# Drop temp. column
comb_data = comb_data.drop(['Outlier'], axis=1)


# In[ ]:


# Check remaining no. of rows
comb_data.shape


# In[ ]:


# Let us check stats once again.
print("----------Stats of numerical columns---------------")
print(comb_data.describe().transpose())
print("----------Stats of categorical columns---------------")
print(comb_data.describe(include=['category']).transpose())
print("-------------Count of null values------------------")
print(comb_data.isnull().sum())
print("-------Count of values in each category------------")
for column_name in comb_data.select_dtypes(include=['category']).columns:
    print(comb_data[column_name].value_counts().sort_values(ascending=False))


# **Create a new column of Fare Group**

# In[ ]:


comb_data['FareGroup'] = np.where(comb_data['Fare']<7.73, 'Tier1', np.where(comb_data['Fare']<10.5, 'Tier2', np.where(comb_data['Fare']<52.5, 'Tier3', 'Tier4')))
comb_data['FareGroup'] = comb_data['FareGroup'].astype('category')


# **I will analyze missing age data in detail to check if missing data is totally random or skewed.<br>
# The goal is to find features which have most no. of missing data and have most variation in age.,<br>
# Then we will use Random Forest Regressor to fill missing age data**

# In[ ]:


# Create a new column to mark missing fares
comb_data['AgeData'] = np.where(comb_data[['Age']].isnull().all(1), 'No', 'Yes')


# In[ ]:


# Find columns which have proportionally more missing age data
f, axes = plt.subplots(3,4, figsize = (28, 21), sharey=True)
for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):
    row = i // 4
    col = i % 4
    ax_curr = axes[row, col]    
    ax1 = sns.barplot(x=col_name, y='Fare', data=comb_data[comb_data.AgeData == 'Yes'], color='blue', alpha = 0.5,    estimator=lambda col_name: len(col_name) / len(comb_data[comb_data.AgeData == 'Yes']) * 100, ax = ax_curr)
    ax2 = sns.barplot(x=col_name, y='Fare', data=comb_data[comb_data.AgeData == 'No'], color='orange', alpha = 0.5,    estimator=lambda col_name: len(col_name) / len(comb_data[comb_data.AgeData == 'No']) * 100, ax = ax_curr)
    ax1.set_ylabel('Percentage')
plt.show()


# *What we can infer is that missing age data is fairly random. 
# <br>However, it seems to be more in case of Queenstown, LowerClass, Survived=0, SibSP=0, Single family size, Cabin = 'N' (null values) and Parch=0.*

# In[ ]:


# Find columns which have more variatation in age [Categorical features]
f, axes = plt.subplots(3,4, figsize = (28, 21), sharey=True)
for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):
    row = i // 4
    col = i % 4
    ax_curr = axes[row, col]    
    sns.barplot(x=col_name, y='Age', data=comb_data[comb_data.AgeData == 'Yes'], ax = ax_curr)
plt.show()


# In[ ]:


# Find if a column can be explained by other column, i.e. highly dependent (correlated)
# Drawing correlation matrix - Standard Pearson coefficients
# Compute the correlation matrix
corr_mat = comb_data[comb_data.AgeData == 'Yes'].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.8, center=0, square=True, annot=True, linecolor='black', linewidths=0, cbar_kws={"shrink": .4}, fmt='.2f')
plt.show()


# *We conclude that 'Fare', 'Sex', 'Embarked', 'Namelength', 'FareGroup' etc. do not explain variation in Age.<br>
# Therefore, we can exclude these columns when computing missing Age data*

# **Fill missing age data**

# In[ ]:


# Predict missing values in age using Random Forest
AgeData = comb_data[['Age', 'Parch', 'SibSp', 'TicketTrim', 'Title','Pclass','FamSize', 'Cabin']]

# Transform categorical features to dummy variables

cat_col_names = AgeData.select_dtypes(include=['category']).columns
AgeData = pd.get_dummies(AgeData, columns=cat_col_names, prefix=cat_col_names)

# Split sets into train and test
train_Age  = AgeData.loc[(AgeData.Age.notnull())]
test_Age = AgeData.loc[(AgeData.Age.isnull())]

# Create target and feature set
X_train_Age = train_Age.values[:, 1::]
y_train_Age = train_Age.values[:, 0]

X_test_Age = test_Age.values[:, 1::]

# Create and fit a model
regr = RandomForestRegressor(max_depth = 8, n_estimators=2000, n_jobs=-1)
regr.fit(X_train_Age, y_train_Age)

# Use the fitted model to predict the missing values
Age_pred = regr.predict(X_test_Age)

# Assign those predictions to the full data set
comb_data.loc[(comb_data.Age.isnull()), 'Age'] = Age_pred

# Check null values in each column
print(comb_data.isnull().sum())


# In[ ]:


# Check how missing age data distribution looks like after imputation
f, axes = plt.subplots(3,4, figsize = (28, 21), sharey=True)
for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):
    row = i // 4
    col = i % 4
    ax_curr = axes[row, col]    
    sns.barplot(x=col_name, y='Age', data=comb_data[comb_data.AgeData == 'No'], ax = ax_curr)
plt.show()


# *Distribution of missing Age after computation looks similar. So, we are good.*

# **Create a new column of Age Category**

# In[ ]:


comb_data['AgeCat'] = np.where(comb_data['Age']<9, 'Child', np.where(comb_data['Age']<20, 'Young', np.where(comb_data['Age']<60, 'Adult', 'Senior')))
comb_data['AgeCat'] = comb_data['AgeCat'].astype('category')


# **Visualize numerical features**

# In[ ]:


# A quick way to get a feel of numerical data is to plot histograms for numerical variables
comb_data.hist(bins=80, figsize=(27,6))
plt.show()


# **Fix skewness and normalize**

# In[ ]:


# Check the skewness of all numerical features
num_cols = comb_data.select_dtypes(include=['float', 'int64']).columns
skewed_cols = comb_data[num_cols].apply(lambda x: ss.skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_cols})

skewness = skewness[abs(skewness) > 0.75]
skewness = skewness.dropna()
print(skewness)
skewed_cols = skewness.index
print("There are {} skewed [skewness > 0.75] numerical features in comb_data to fix".format(skewness.shape[0]))


# In[ ]:


# Fix skewness

# Use boxcox1p method
'''lam = 0.5
comb_data['Fare'] = boxcox1p(comb_data['Fare'], lam)
comb_data['Fare'] = boxcox1p(comb_data['NameLength'], lam)'''

# Use log1p method
comb_data[['Fare']] = np.log1p(comb_data[['Fare']])
comb_data[['NameLength']] = np.log1p(comb_data[['NameLength']])
comb_data[['IndFare']] = np.log1p(comb_data[['IndFare']])


# *log1p method gave better result than boxcox1p method. I learned that method to deal with skewness will affect accuracy of the model.*

# In[ ]:


# Find if a column can be explained by other column, i.e. highly dependent (correlated)
# Drawing correlation matrix - Standard Pearson coefficients
# Compute the correlation matrix
corr_mat = comb_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_mat, mask=mask, cmap=cmap, vmax=.8, center=0, square=True, annot=True, linecolor='black', linewidths=0, cbar_kws={"shrink": .4}, fmt='.2f')
plt.show()


# In[ ]:


# Visualizing numerical data along x and y axis
comb_data.plot(kind = "scatter", x = "Fare", y = "Age", figsize=(24, 6), color = 'green')
plt.show()


# In[ ]:


# Now look at the stats once again.
print("-------------Basic info of columns-----------------")
print(comb_data.info())
print("----------Stats of numerical columns---------------")
print(comb_data.describe().transpose())
print("-------------Count of null values------------------")
print(comb_data.isnull().sum())
print("-------Count of values in each category------------")
for column_name in comb_data.select_dtypes(include=['category']).columns:
    print(comb_data[column_name].value_counts())


# **Analyze survival data - Visualizing outcome across independent variables**

# In[ ]:


# Create a subset that has survival outcome of all passengers
surv = comb_data[comb_data.DataType == 'train']
surv.head()


# In[ ]:


print("------------------------Count & %age----------   ----------------")
print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"      %(len(surv[surv.Survived == 'Yes']), 1.*len(surv[surv.Survived == 'Yes'])        /len(surv)*100.0,len(surv[surv.Survived == 'No']),        1.*len(surv[surv.Survived == 'No'])/len(surv)*100.0,        len(surv)))
print("------------------------Mean Age-------------------------------------")
print("Mean age survivors: %.1f, Mean age non-survivers: %.1f"      %(np.mean(surv[surv.Survived == 'Yes'].Age), np.mean(surv[surv.Survived == 'No'].Age)))
print("------------------------Median Fare-------------------------------------")
print("Median Fare survivors: %.1f, Median Fare non-survivers: %.1f"      %(np.median(surv[surv.Survived == 'Yes'].Fare), np.median(surv[surv.Survived == 'No'].Fare)))


# In[ ]:


# Visulaizing survival data against numerical columns
# Create violin plot to compare against numerical columns.

f, axes = plt.subplots(ncols=4, figsize = (24, 6))
v1 = sns.violinplot(data = surv, x = 'Survived', y = 'Fare', ax = axes[0])
v1.set_title('Survived vs. Fare', fontsize = 12)

v2 = sns.violinplot(data = surv, x = 'Survived', y = 'IndFare', ax = axes[1])
v2.set_title('Survived vs. IndFare', fontsize = 12)

v3 = sns.violinplot(data = surv, x = 'Survived', y = 'NameLength', ax = axes[2])
v3.set_title('Survived vs. NameLength', fontsize = 12)

v4 = sns.violinplot(data = surv, x = 'Survived', y = 'Age', ax = axes[3])
v4.set_title('Survived vs. Age', fontsize = 12)
plt.show()


# *Age data does not look different in dead or alive group.*

# In[ ]:


# Visulaizing survival data against numerical columns
# Create distribution plot to compare against numerical columns.

f, axes = plt.subplots(ncols=4, figsize = (28, 6))
d1 = sns.distplot(surv[surv.Survived == 'Yes']['Age'].dropna().values, color='Green', ax = axes[0], label = 'Survived')
d2 = sns.distplot(surv[surv.Survived == 'No']['Age'].dropna().values, color='Red', ax = axes[0], label = 'Not Survived')
d1.set_title('Survived vs. Age', fontsize = 12)
d1.legend(loc='best')
d1.set(xlabel="Age", ylabel="No. of Passengers")

d3 = sns.distplot(surv[surv.Survived == 'Yes']['Fare'].dropna().values, color='Green', ax = axes[1], label = 'Survived')
d4 = sns.distplot(surv[surv.Survived == 'No']['Fare'].dropna().values, color='Red', ax = axes[1], label = 'Not Survived')
d3.set_title('Survived vs. Fare', fontsize = 12)
d3.legend(loc='best')
d3.set(xlabel="Fare", ylabel="No. of Passengers")

d5 = sns.distplot(surv[surv.Survived == 'Yes']['IndFare'].dropna().values, color='Green', ax = axes[2], label = 'Survived')
d6 = sns.distplot(surv[surv.Survived == 'No']['IndFare'].dropna().values, color='Red', ax = axes[2], label = 'Not Survived')
d5.set_title('Survived vs. Ind. Fare', fontsize = 12)
d5.legend(loc='best')
d5.set(xlabel="IndFare", ylabel="No. of Passengers")
plt.show()

d7 = sns.distplot(surv[surv.Survived == 'Yes']['NameLength'].dropna().values, color='Green', ax = axes[3], label = 'Survived')
d8 = sns.distplot(surv[surv.Survived == 'No']['NameLength'].dropna().values, color='Red', ax = axes[3], label = 'Not Survived')
d7.set_title('Survived vs. Name Length', fontsize = 12)
d7.legend(loc='best')
d7.set(xlabel="NameLength", ylabel="No. of Passengers")
plt.show()


# *Above plots show more detail in variation.*

# In[ ]:


# Caluculate assciation between 2 columns - Cramer's V score [Categorical Features]
# Change Survived data type so that it does not mess up calculation below. (change back to integer before applying ML models)
comb_data["Survived"] = comb_data["Survived"].astype(str)
for i in comb_data.select_dtypes(include=['category']).columns:
    col_1 = i
    for j in comb_data.select_dtypes(include=['category']).columns:
        col_2 = j
        if col_1 == col_2:
            break
        confusion_matrix = pd.crosstab(comb_data[col_1], comb_data[col_2])
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        Cramer_V = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
        if Cramer_V > 0.5:
            print("The Cramer's V score bettween ", col_1, " and ", col_2, " is : ", (Cramer_V))
        result = Cramer_V


# *These values shows correlation among categorical features.<br>
# Clearly Title  and  Sex  is very highly correlated. We have to pick one to avoid Multicollinearity.*

# In[ ]:


# Visulaizing survival data across categorical columns, using Fare numerical column on Y-axis.
f, axes = plt.subplots(4,3, figsize = (28, 21), sharey=True)
for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):
    row = i // 3
    col = i % 3
    ax_curr = axes[row, col]    
    sns.violinplot(data=surv, x=col_name, y='Fare', hue='Survived', ax = ax_curr)
plt.show()


# In[ ]:


# Creating bar plots. 
# I will convert Survived values to number type so that it can be used for bar plots.
surv['Survived'].replace(['No','Yes'],[0,1], inplace=True)

f, axes = plt.subplots(3,4, figsize = (28, 16), sharey=True)
for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):
    row = i // 4
    col = i % 4
    ax_curr = axes[row, col]
    sns.barplot(x=col_name, y='Survived', data=surv, ax = ax_curr)
plt.show()


# In[ ]:


# Find columns which have proportionally more death counts
f, axes = plt.subplots(3,4, figsize = (28, 21), sharey=True)
for i, col_name in enumerate(comb_data.select_dtypes(include=['category']).columns):
    row = i // 4
    col = i % 4
    ax_curr = axes[row, col]    
    ax1 = sns.barplot(x=col_name, y='Fare', data=surv[surv.Survived == 0], color='orange', alpha = 0.5, estimator=lambda col_name: len(col_name) / len(surv[surv.Survived == 0]) * 100, ax = ax_curr)
    ax2 = sns.barplot(x=col_name, y='Fare', data=surv[surv.Survived == 1], color='blue', alpha = 0.5, estimator=lambda col_name: len(col_name) / len(surv[surv.Survived == 1]) * 100, ax = ax_curr)
    ax1.set_ylabel('Percentage')
plt.show()


# In[ ]:


# In last set of charts, I will narrow down data by applying filters and combination of categories to visualize where exactly most no. of dead passengers count is.
# Factorplots are good to see counts across categorical columns, as shown below.

f1 = sns.factorplot(x='FamSize', data=surv, hue='Survived', kind='count', col='Sex', size=6)
plt.show()


# In[ ]:


# Filtered passengers count data across categories
f2 = sns.factorplot(x='Embarked', data=surv, hue='Survived', kind='count', col='Sex', size=6)
plt.show()


# In[ ]:


# Filtered passengers count data across categories
f8 = sns.factorplot(x='FamSize', data=surv[(surv['Title'] == 'Mr') & (surv['Cabin'] == 'N')], hue='Survived', kind='count', col='Embarked', size=6)

plt.show()


# *After extensive visualization of underlying data, I am ready to go on to next step.*

#  #### **Step 5 - Prepare the data for Machine Learning algorithms**

# **Process further and create train, target and test dataset**

# In[ ]:


# Check combined data
comb_data.head()


# In[ ]:


# Save processed comb_data [for sanity check before splitting train and test data]
'''comb_data.to_csv('comb_data_Titanic.csv')'''

# Check data types
comb_data.dtypes


# In[ ]:


# Convert binary categorical columns to integer 0/1
comb_data['Survived'].replace(['No','Yes'],[0,1], inplace=True)

comb_data['Sex'].replace(['Male','Female'],[0,1], inplace=True)
comb_data["Sex"] = comb_data["Sex"].astype(int)

# Change back PassengerId to integer before applying ML models
comb_data["PassengerId"] = comb_data["PassengerId"].astype(int)

# Drop columns to avoid overfitting
# comb_data = comb_data.drop(labels = ["Age", "Sex", "Parch", "SibSp", "IndFare", FareGroup", "AgeCat", "IndFare", "AgeData"],axis = 1)
# comb_data = comb_data.drop(labels = ["Sex", "FareGroup", "AgeCat", "IndFare", "AgeData"],axis = 1)
comb_data = comb_data.drop(labels = ["Sex", "SibSp", "FareGroup", "AgeCat", "AgeData"],axis = 1)

# Transform categorical features in to dummy variables
comb_data["DataType"] = comb_data["DataType"].astype(str) # to exclude from dummy function

# Get the list of category columns
cat_col_names = comb_data.select_dtypes(include=['category']).columns

comb_data = pd.get_dummies(comb_data, columns=cat_col_names, prefix=cat_col_names)
comb_data.head()


# *I tried multiple combination of features and kept the one set which gave me best accuracy.*

# In[ ]:


# Create train and test subset from survival column
print(comb_data.shape)
train = comb_data[comb_data.DataType == 'train']
print(train.shape)

test = comb_data[comb_data.DataType == 'test']
print(test.shape)

train_id = train["PassengerId"]
test_id = test["PassengerId"]

train["Survived"] = train["Survived"].astype(int)
y_train = train["Survived"]
print(y_train.shape)
print(y_train.head())

X_train = train.drop(labels = ["Survived", "PassengerId", "DataType"],axis = 1)
print(X_train.shape)
print(X_train.head())

X_test = test.drop(labels = ["Survived", "PassengerId", "DataType"],axis = 1)
print(X_test.shape)
print(X_test.head())

# Make sure there is no null values in train and test data and also no. of categories in categorical values are equal
print("-------------Null values in Train set------------------")
print(X_train.isnull().values.any())
    
print("-------------Null values in Test set------------------")
print(X_test.isnull().values.any())


# In[ ]:


# Check number of columns and name of columns match between X_train and X_test
print(X_train.shape)
print(X_test.shape)
print(set(X_train.columns) == set(X_test.columns))
print('--------columns present in X_train but not in X_test-------')
missing_col_tt = [i for i in list(X_train) if i not in list(X_test)]
print(missing_col_tt)
print('--------columns present in X_test but not in X_train-------')
missing_col_tr = [i for i in list(X_test) if i not in list(X_train)]
print(missing_col_tr)

# Drop these columns and test again
X_train = X_train.drop(missing_col_tt, axis=1)
X_test = X_test.drop(missing_col_tr, axis=1)

print(X_train.shape)
print(X_test.shape)
print(set(X_train.columns) == set(X_test.columns))
print('--------columns present in X_train but not in X_test-------')
missing_col_tt = [i for i in list(X_train) if i not in list(X_test)]
print(missing_col_tt)
print('--------columns present in X_test but not in X_train-------')
missing_col_tr = [i for i in list(X_test) if i not in list(X_train)]
print(missing_col_tr)


# *I am not sure if above step was necessary. I assume order and no. columns in train and test data should be same.*

# #### **Step 6 - Select and train a model** 

# **Visualize how Machine Learning models works on classification with just 2 numerical features**

# In[ ]:


# Extract numerical columns from train dataseet
X_num = X_train.iloc[:, [0, 1]]
print(X_num.head())


# *'X_num' and 'y_train' will be used to visualize Machine Learning classification models with two numerical independent features*

# In[ ]:


# Run function to plot graphs
plt_start = time.time()
plot_class_models_two_num_features(X = X_num, y = y_train, clf_dict = {k: clf_dict[k] for k in clf_dict.keys() & {'clf_Log_reg', 'clf_KNN', 'clf_RF', 'clf_ExTree', 'clf_XGBoost', 'clf_MLP'}}, title = 'Age & Fare')
plt_end = time.time()


# *As expected, using numerical columns only (fare and age) do not give good accuracy.<br>
# However, nice way to see how algorithms work on classification.*

# **Comparing Classification Machine Learning models with all independent features. [Use 'X_train' and 'y_train']**

# In[ ]:


# Run function to evaluate various classification models [Metrics and cross_val-score]
clf_cross_val_score_and_metrics(X=X_train, y=y_train, clf_dict=clf_dict, CVS_scoring = "accuracy", CVS_CV=kfold)


# **Extensive GridSearch with valuable parameters / score in a dataframe**

# In[ ]:


# Get GridSearch results in a dataframe
gs_start = time.time()
gs_metrics = clf_GridSearchCV_results(gs_metrics, X_train=X_train, y_train=y_train, GS_scoring = "accuracy", GS_CV=kfold)
gs_end = time.time()


# In[ ]:


# Check GridSearch metric data
'''gs_metrics.to_csv('Titanic_GS_Result.csv')'''
gs_metrics


# **Find best estimators**

# In[ ]:


# Extract best estimators
Best_Estimator_RF = gs_metrics.iloc[1, 5]
Best_Estimator_XGB = gs_metrics.iloc[5, 5]
Best_Estimator_MLP = gs_metrics.iloc[2, 5]
Best_Estimator_ExT = gs_metrics.iloc[4, 5]
Best_Estimator_GB = gs_metrics.iloc[6, 5]
Best_Estimator_SVC = gs_metrics.iloc[7, 5]

# AdaBoost for feature importance plot
Best_Estimator_AdaB = gs_metrics.iloc[3, 5]


# **Plot learning curves**

# In[ ]:


# Run function to plot learning curves for top 4 models
plot_learning_curve(Best_Estimator_RF,"RF mearning curves",X_train,y_train,cv=kfold)
plot_learning_curve(Best_Estimator_ExT,"ExtraTrees learning curves",X_train,y_train,cv=kfold)
plot_learning_curve(Best_Estimator_XGB,"XGB learning curves",X_train,y_train,cv=kfold)
plot_learning_curve(Best_Estimator_MLP,"MLP learning curves",X_train,y_train,cv=kfold)


# **Feature Importance**

# In[ ]:


nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(25,20))

names_clf = [("ExtraTrees",Best_Estimator_ExT),("RandomForest",Best_Estimator_RF),("XGBoosting",Best_Estimator_XGB), ("AdaBoosting", Best_Estimator_AdaB)]

nclf = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_clf[nclf][0]
        clf = names_clf[nclf][1]
        # Plot feature importance
        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)[::-1][:32]
        pos = feature_importance[sorted_idx][:32]
        
        g = sns.barplot(y=X_train.columns[sorted_idx][:32],x = pos, orient='h', palette='inferno', ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclf += 1


# #### **Step 7 - Fine-tune your model**

# In[ ]:


vote_clf = VotingClassifier(estimators=[('rfc', Best_Estimator_RF),
('ext', Best_Estimator_ExT), ('xgb',Best_Estimator_XGB)], voting='soft', weights=[1, 1, 1], n_jobs=-1)


# #### **Step 8 - Predict and present your solution**

# In[ ]:


# Fitting / Predicting using Voting Classifier
# vote_clf.fit(X_train, y_train)
# y_pred = vote_clf.predict(X_test)

# Fitting / Predicting using Random Forest classifier
# rfc = RandomForestClassifier(max_depth=8, n_estimators = 500, n_jobs=-1, random_state=rand_st)
# rfc.fit(X_train, y_train)
# y_pred = rfc.predict(X_test)

# Fitting / Predicting using Extra Tree classifier
Best_Estimator_ExT.fit(X_train, y_train)
y_pred = Best_Estimator_ExT.predict(X_test)
print(y_pred)


# *Extra tree gave me better result than voting classifier or Random Forest Classifier*

# In[ ]:


# Combine PassengerId and prediction
Titanic_prediction = np.vstack((test_id, y_pred))


# In[ ]:


# Create output file
np.savetxt('Titanic_Kaggle_Result.csv', np.transpose(Titanic_prediction), delimiter=',', fmt="%s")


# #### **Step 9 - Final words!**
# 1-  My first attempt (version 1) was without looking into any solution online.
# 2-  This version is modified version after looking at works of others.
      e.g. - 'Titanic Top 4% with ensemble modeling' kernel by Yassine Ghouzam, PhD
# 3-  I underestimated feature engineering before.
# 4- Got slighlty better result of 0.80332 [Top 8%]
# 5- I am more interested in approach and doing all the steps correctly along with necessary checks.
# 6- Let me know if I missed anything critical
# In[ ]:


end = time.time()
print('Time taken to plot ML models : ' + str("{:.2f}".format((plt_end - plt_start)/60)) + ' minutes')
print('Time taken to perform Grid Search : ' + str("{:.2f}".format((gs_end - gs_start)/60)) + ' minutes')
print('Total running time of the script : ' + str("{:.2f}".format((end - start)/60)) + ' minutes')

