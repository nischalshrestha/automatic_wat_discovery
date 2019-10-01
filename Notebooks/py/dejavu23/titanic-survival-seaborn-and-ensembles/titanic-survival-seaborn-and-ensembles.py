#!/usr/bin/env python
# coding: utf-8

# 
# 

# <img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg" width="520">

# # Titanic Survival: Seaborn and Ensembles
# **My second Titanic kernel**
# 
# **[Part 0: Imports, Functions](#Part-0:-Imports,-Functions)** 
# 
# **[Part 1: Exploratory Data Analysis](#Part-1:-Exploratory-Data-Analysis)** 
# 
# * Seaborn [heatmaps](#Seaborn-heatmaps) : missing data in df_train and df_test
# * Seaborn [countplots](#Seaborn-Countplots) : Number of (Non-)Survivors as function of features
# * Seaborn [distplots](#Seaborn-Distplots) : Distribution of Age and Fare as function of Pclass, Sex and Survived  
# * [Bar and Box plots](#Bar-and-Box-plots) for categorical features : Pclass and Embarked
# * Seaborn [violin and swarm plots](#Swarm-and-Violin-plots) : Survivals as function of Age, Pclass and Sex
# 
# **[Part 2: Data Wrangling and Feature Engineering](#Part-2:-Data-Wrangling-and-Feature-Engineering)**  
# 
# * [Feature Engineering](#Feature-Engineering): include new features to improve the performance of the classifiers and to fill missing values:  
# Family size, Alone, Name length, Title
# * [Data Wrangling](#Data-Wrangling): fill NaN, convert categorical to numerical, [Standard Scaler](#Standard-Scaler), create X, y and X_test for Part 3
# 
# 
# **[Part 3: Optimization of Classifier parameters, Boosting, Voting and Stacking](#Part-3:-Optimization-of-Classifier-parameters,-Boosting,-Voting-and-Stacking)**  
# 
# * Review: [k fold cross validation](#Review:-k-fold-cross-validation) for SVC and Random Forest: 
#  * SVC, features not scaled 
#  * SVC, features scaled 
#  * RFC, features not scaled 
# * Hyperparameter tuning with GridSearchCV and RandomizedSearchCV for:  
#  * SVC 
#  * KNN
#  * Decision Tree
#  * Random Forest
# 
# * study Ensemble models like Boosting, Stacking and Voting:
#  * Gradient Boost Decicison Tree - [GBDT](#Gradient-Boost-Decicison-Tree-GBDT)
#  * eXtreme Gradient Boosting - [XGBoost](#eXtreme-Gradient-Boosting---XGBoost)   
#  * Adaptive Boosting - [AdaBoost](#Ada-Boost)
#  * Voting: [VotingClassifier 1](#First-Voting), [VotingClassifier 2](#Second-Voting)  
#  * Stacking : [StackingClassifier](#StackingClassifier)  
# * Compare Classifier performance based on the validation score : [comparison plot](#Comparison-plot-for-best-models)
# * Correlation of prediction results : [correlation matrix](#Correlation-of-prediction-results)

# **soon to follow:**
# 
# SVM boost
# 
# CatBoost
# 
# Extra Tree
# 
# Light Gradient Boosting [LGBM]
# 
# Bagging
# 
# 
# **References**
# 
# While this notebook contains some work work based on my ideas, it is also a collection of approaches  
# and techniques from these kaggle notebooks:

# # Part 0: Imports, Functions

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore")
#warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
#warnings.filterwarnings(action='once')

from sklearn.utils.testing import ignore_warnings

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# **some useful functions**

# In[ ]:


def get_best_score(model):
    
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
    
    return model.best_score_


def plot_feature_importances(model, columns):
    nr_f = 10
    imp = pd.Series(data = model.best_estimator_.feature_importances_, index=columns).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    plt.title("Feature importance")
    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')


# **loading the data**

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# # Part 1: Exploratory Data Analysis

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# Looks like the feature Cabin has lots of missing data, also some data for Age and Embarked is missing.  
# Lets plot the seaborn heatmap of the isnull matrix for the train and test data

# ### Seaborn heatmaps  
# missing data in df_train and df_test

# In[ ]:


fig, ax = plt.subplots(figsize=(9,5))
sns.heatmap(df_train.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(9,5))
sns.heatmap(df_test.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()


# ### Seaborn Countplots  
# for all categorical columns

# In[ ]:


cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']


# In[ ]:


nr_rows = 2
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sns.countplot(df_train[cols[i]], hue=df_train["Survived"], ax=ax)
        ax.set_title(cols[i])
        ax.legend() 
        
plt.tight_layout()   


# Of the 891 passengers in df_test, less than 350 survive.  
# Much more women survive than men.  
# Also, the chance to survive is much higher in Pclass 1 and 2 than in Class 3.  
# Survival rate for passengers travelling with SibSp or Parch is higher than for those travelling alone.  
# Passengers embarked in C and Q are more likely to survie than those embarked in S.

# ### Seaborn Distplots 
# **Distribution of Age as function of Pclass, Sex and Survived**

# In[ ]:


bins = np.arange(0, 80, 5)
g = sns.FacetGrid(df_train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()  


# Best chances to survive for male passengers was in Pclass 1 or being below 5 years old.  
# Lowest survival rate for female passengers was in Pclass 3 and being older than 40.  
# Most passengers were male, in Pclass 3 and between 15-35 years old.

# **Disribution of Fare as function of Pclass, Sex and Survived**

# In[ ]:


df_train['Fare'].max()


# In[ ]:


bins = np.arange(0, 550, 50)
g = sns.FacetGrid(df_train, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()  


# ### Bar and Box plots

# Default mode for seaborn barplots is to plot the mean value for the category.  
# Also, the standard deviation is indicated.

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=df_train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass")
plt.show()


# In[ ]:


sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df_train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass and Sex")
plt.show()


# Highest survival rate (>0.9) for women in Pclass 1 or 2.  
# Lowest survival rate (<0.2) for men in Pclass 3.

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=df_train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Embarked Port")
plt.show()


# Passengers embarked in "S" had the lowest survival rate, those embarked in "C" the highest.

# **Boxplot**

# In[ ]:


sns.boxplot(x='Embarked', y='Age', data=df_train)
plt.title("Age distribution as function of Embarked Port")
plt.show()


# In[ ]:


sns.boxplot(x='Embarked', y='Fare', data=df_train)
plt.title("Fare distribution as function of Embarked Port")
plt.show()


# Mean fare for Passengers embarked in "C" was higher.

# ### Swarm and Violin plots
# Although the following swarm and violin plots show the same data like the countplots or distplots before,  
# they can reveal ceratin details that disappear in other plots. However, it takes more time to study these plots in detail.

# In[ ]:


cm_surv = ["darkgrey" , "lightgreen"]


# In[ ]:


fig, ax = plt.subplots(figsize=(13,7))
sns.swarmplot(x='Pclass', y='Age', hue='Survived', split=True, data=df_train , palette=cm_surv, size=7, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()


# Here, the high survival rate for kids in Pclass 2 is easily observed.  
# Also, it becomes more obvious that for passengers older than 40 the best chance to survive is in Pclass 1,  
# and smallest chance in Pclass 3   

# In[ ]:


fig, ax = plt.subplots(figsize=(13,7))
sns.violinplot(x="Pclass", y="Age", hue='Survived', data=df_train, split=True, bw=0.05 , palette=cm_surv, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()


# This violinplot shows exactly the same info like the swarmplot before.

# In[ ]:


g = sns.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=df_train, kind="swarm", split=True, palette=cm_surv, size=7, aspect=.9, s=7)


# In[ ]:


g = sns.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=df_train, kind="violin", split=True, bw=0.05, palette=cm_surv, size=7, aspect=.9, s=7)


# # Part 2: Data Wrangling and Feature Engineering

# ## Feature Engineering
# **New Features: 'FamilySize'  ,  'Alone' , 'NameLen' and 'Title'**

# In[ ]:





# In[ ]:


for df in [df_train, df_test] :
    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    df['Alone']=0
    df.loc[(df.FamilySize==0),'Alone'] = 1
    
    df['NameLen'] = df.Name.apply(lambda x : len(x)) 
    df['NameLenBin']=np.nan
    for i in range(20,0,-1):
        df.loc[ df['NameLen'] <= i*5, 'NameLenBin'] = i
    
    
    df['Title']=0
    df['Title']=df.Name.str.extract(r'([A-Za-z]+)\.') #lets extract the Salutations
    df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:





# ### New Feature: NameLenBin

# In[ ]:


print(df_train[['NameLen' , 'NameLenBin']].head(10))


# In[ ]:


grps_namelenbin_survrate = df_train.groupby(['NameLenBin'])['Survived'].mean().to_frame()
grps_namelenbin_survrate


# In[ ]:


plt.subplots(figsize=(10,6))
sns.barplot(x='NameLenBin' , y='Survived' , data = df_train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of NameLenBin")
plt.show()


# **Looks like there is very strong correlation of Survival rate and Name length**

# In[ ]:


fig, ax = plt.subplots(figsize=(9,7))
sns.violinplot(x="NameLenBin", y="Pclass", data=df_train, hue='Survived', split=True, 
               orient="h", bw=0.2 , palette=cm_surv, ax=ax)
plt.show()


# **Chance to survive increases with length of name for all Passenger classes**

# In[ ]:


g = sns.factorplot(x="NameLenBin", y="Survived", col="Sex", data=df_train, kind="bar", size=5, aspect=1.2)


# **Increase of survival rate with length of name most important for male passengers**

# In[ ]:





# In[ ]:





# 

# ### New Feature: Title

# In[ ]:


grps_title_survrate = df_train.groupby(['Title'])['Survived'].mean().to_frame()
grps_title_survrate


# In[ ]:


plt.subplots(figsize=(10,6))
sns.barplot(x='Title' , y='Survived' , data = df_train)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Title")
plt.show()


# In[ ]:





# ### New Feature: Family size

# In[ ]:


pd.crosstab(df_train.FamilySize,df_train.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')


# ## Data Wrangling

# **Fill NaN with mean or mode**

# In[ ]:


for df in [df_train, df_test]:

    # Title
    df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])

    # Age: use Title to fill missing values
    df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()
    df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()
    df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()
    df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()
    df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()
    df = df.drop('Name', axis=1)




# In[ ]:


# Embarked
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode().iloc[0])
df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode().iloc[0])

# Fare
df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())


# ### Bining for Age and Fare, convert Title to numerical

# In[ ]:


for df in [df_train, df_test]:
    
    df['Age_bin']=np.nan
    for i in range(8,0,-1):
        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i
        
    df['Fare_bin']=np.nan
    for i in range(12,0,-1):
        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i        
    
    # convert Title to numerical
    df['Title'] = df['Title'].map( {'Other':0, 'Mr': 1, 'Master':2, 'Miss': 3, 'Mrs': 4 } )
    # fill na with maximum frequency mode
    df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])
    df['Title'] = df['Title'].astype(int)        


# In[ ]:





# In[ ]:


df_train_ml = df_train.copy()
df_test_ml = df_test.copy()

passenger_id = df_test_ml['PassengerId']


# **double-check for missing values**

# In[ ]:


df_train_ml.info()


# In[ ]:


df_test_ml.info()


# **convert categorical to numerical : get_dummies**

# In[ ]:


df_train_ml = pd.get_dummies(df_train_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_test_ml = pd.get_dummies(df_test_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

df_train_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age', 'Fare_bin'],axis=1,inplace=True)
df_test_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age', 'Fare_bin'],axis=1,inplace=True)

#df_train_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)
#df_test_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)


# In[ ]:


df_train_ml.dropna(inplace=True)


# In[ ]:


for df in [df_train_ml, df_test_ml]:
    df.drop(['NameLen'], axis=1, inplace=True)

    df.drop(['SibSp'], axis=1, inplace=True)
    df.drop(['Parch'], axis=1, inplace=True)
    df.drop(['Alone'], axis=1, inplace=True)


# In[ ]:





# In[ ]:


df_train_ml.head()


# In[ ]:


df_test_ml.fillna(df_test_ml.mean(), inplace=True)
df_test_ml.head()


# ### Standard Scaler

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for df_train_ml
scaler.fit(df_train_ml.drop(['Survived'],axis=1))
scaled_features = scaler.transform(df_train_ml.drop(['Survived'],axis=1))
df_train_ml_sc = pd.DataFrame(scaled_features) # columns=df_train_ml.columns[1::])

# for df_test_ml
df_test_ml.fillna(df_test_ml.mean(), inplace=True)
#scaler.fit(df_test_ml)
scaled_features = scaler.transform(df_test_ml)
df_test_ml_sc = pd.DataFrame(scaled_features) # , columns=df_test_ml.columns)


# In[ ]:


df_train_ml_sc.head()


# In[ ]:


df_test_ml_sc.head()


# In[ ]:


df_train_ml.head()


# In[ ]:





# In[ ]:


X = df_train_ml.drop('Survived', axis=1)
y = df_train_ml['Survived']
X_test = df_test_ml

X_sc = df_train_ml_sc
y_sc = df_train_ml['Survived']
X_test_sc = df_test_ml_sc


# # Part 3: Optimization of Classifier parameters, Boosting, Voting and Stacking

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree


from sklearn.metrics import accuracy_score


# ### Review: k fold cross validation  
# just a short review of this technique that we already studied in the first kernel

# In[ ]:


from sklearn.model_selection import cross_val_score


# ### SVC, features not scaled  

# In[ ]:


svc = SVC(gamma = 0.01, C = 100)
scores_svc = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
print(scores_svc)
print(scores_svc.mean())


# ### SVC, features scaled  

# In[ ]:


svc = SVC(gamma = 0.01, C = 100)
scores_svc_sc = cross_val_score(svc, X_sc, y_sc, cv=10, scoring='accuracy')
print(scores_svc_sc)
print(scores_svc_sc.mean())


# ### RFC, features not scaled  

# In[ ]:


rfc = RandomForestClassifier(max_depth=5, max_features=6)
scores_rfc = cross_val_score(rfc, X, y, cv=10, scoring='accuracy')
print(scores_rfc)
print(scores_rfc.mean())


# 

# ## Hyperparameter tuning with RandomizedSearchCV and GridSearchCV

# **RandomizedSearchCV  and GridSearchCV apply k fold cross validation on a chosen set of parameters**
# **and then find the parameters that give the best performance.**  
# For GridSearchCV, all possible combinations of the specified parameter values are tried out, resulting in a parameter grid.  
# For RandomizedSearchCV, a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform


# **In the following we apply GridSearchCV and RandomizedSearchCV for these Classification models:**  
# **KNN, Decision Tree, Random Forest, SVC**

# #### SVC : RandomizedSearchCV

# In[ ]:


model = SVC()
param_grid = {'C':uniform(0.1, 5000), 'gamma':uniform(0.0001, 1) }
rand_SVC = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100)
rand_SVC.fit(X_sc,y_sc)
score_rand_SVC = get_best_score(rand_SVC)


# In[ ]:





# **SVC : GridSearchCV**

# In[ ]:


param_grid = {'C': [0.1,10, 100, 1000,5000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
svc_grid = GridSearchCV(SVC(), param_grid, cv=10, refit=True, verbose=1)
svc_grid.fit(X_sc,y_sc)
sc_svc = get_best_score(svc_grid)


# In[ ]:





# In[ ]:


pred_all_svc = svc_grid.predict(X_test_sc)

sub_svc = pd.DataFrame()
sub_svc['PassengerId'] = df_test['PassengerId']
sub_svc['Survived'] = pred_all_svc
sub_svc.to_csv('svc.csv',index=False)


# In[ ]:





# In[ ]:





# **KNN**

# In[ ]:


knn = KNeighborsClassifier()
leaf_range = list(range(3, 15, 2))
k_range = list(range(1, 15, 2))
weight_options = ['uniform', 'distance']
param_grid = dict(leaf_size=leaf_range, n_neighbors=k_range, weights=weight_options)
print(param_grid)

knn_grid = GridSearchCV(knn, param_grid, cv=10, verbose=1, scoring='accuracy')
knn_grid.fit(X, y)

score_knn = get_best_score(knn_grid)


# In[ ]:





# **Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

param_grid = {'min_samples_split': [2,4,7,10]}
dtree_grid = GridSearchCV(dtree, param_grid, cv=10, refit=True, verbose=1)
dtree_grid.fit(X_sc,y_sc)

print(dtree_grid.best_score_)
print(dtree_grid.best_params_)
print(dtree_grid.best_estimator_)


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

param_grid = {'max_depth': [3, 5, 6, 7, 8], 'max_features': [2,4,6,7,8,9],  
              'min_samples_split': [3, 5, 6, 7]}

rf_grid = GridSearchCV(rfc, param_grid, cv=10, refit=True, verbose=1)
rf_grid.fit(X_sc,y_sc)
sc_rf = get_best_score(rf_grid)


# In[ ]:


plot_feature_importances(rf_grid, X.columns)


# In[ ]:


pred_all_rf = rf_grid.predict(X_test_sc)

sub_rf = pd.DataFrame()
sub_rf['PassengerId'] = df_test['PassengerId']
sub_rf['Survived'] = pred_all_rf
sub_rf.to_csv('rf.csv',index=False)


# In[ ]:





# In[ ]:





# ### Gradient Boost Decicison Tree GBDT 
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()

param_grid = {'n_estimators': [30, 50, 100], 
              'min_samples_split': [3, 4, 5, 6, 7, 8],
              'max_depth': [3, 4, 5, 6]}
gbdt_grid = GridSearchCV(gbdt, param_grid, cv=10, refit=True, verbose=1)
gbdt_grid.fit(X_sc,y_sc)
sc_gbdt = get_best_score(gbdt_grid)


# In[ ]:


plot_feature_importances(gbdt_grid, X.columns)


# In[ ]:


pred_all_gbdt = gbdt_grid.predict(X_test_sc)

sub_gbdt = pd.DataFrame()
sub_gbdt['PassengerId'] = df_test['PassengerId']
sub_gbdt['Survived'] = pred_all_gbdt
#sub_gbdt.to_csv('gbdt.csv',index=False)


# ### eXtreme Gradient Boosting - XGBoost

# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.5, 1, 2], 'learning_rate': [0.05,0.1, 0.2, 0.3]}

with ignore_warnings(category=DeprecationWarning):
    xgb_grid = GridSearchCV(xgb, param_grid, cv=10, refit=True, verbose=1)
    xgb_grid.fit(X_sc,y_sc)
    sc_xgb = get_best_score(xgb_grid)


# In[ ]:


plot_feature_importances(xgb_grid, X.columns)


# In[ ]:


with ignore_warnings(category=DeprecationWarning):
    pred_all_xgb = xgb_grid.predict(X_test_sc)

sub_xgb = pd.DataFrame()
sub_xgb['PassengerId'] = df_test['PassengerId']
sub_xgb['Survived'] = pred_all_xgb
sub_xgb.to_csv('xgb.csv',index=False)


# ### Ada Boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()

param_grid = {'n_estimators': [30, 50, 100], 'learning_rate': [0.1, 1]}
ada_grid = GridSearchCV(ada, param_grid, cv=10, refit=True, verbose=1)
ada_grid.fit(X_sc,y_sc)
sc_ada = get_best_score(ada_grid)

pred_all_ada = ada_grid.predict(X_test_sc)


# In[ ]:





# In[ ]:





# ### VotingClassifier

# In[ ]:


from sklearn.ensemble import VotingClassifier


# ### First Voting  
# for the first voting ensemble I use three simple models (LR, RF, GNB)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}

with ignore_warnings(category=DeprecationWarning):
    votingclf_grid = GridSearchCV(estimator=eclf, param_grid=params, cv=10)
    votingclf_grid.fit(X_sc,y_sc)
    sc_vot1 = get_best_score(votingclf_grid)


# ### Second Voting
# 
# for the 2nd voting ensemble I use the three models (together with the optimal parameters found by GridSearchCV)  
# that had the best test score based on the cross validations above 

# In[ ]:


clf4 = GradientBoostingClassifier()
clf5 = SVC()
clf6 = RandomForestClassifier()

eclf_2 = VotingClassifier(estimators=[('gbdt', clf4), 
                                      ('svc', clf5), 
                                      ('rf', clf6)], voting='soft')

params = {'gbdt__n_estimators': [50], 'gbdt__min_samples_split': [3],
          'svc__C': [10, 100] , 'svc__gamma': [0.1,0.01,0.001] , 'svc__kernel': ['rbf'] , 'svc__probability': [True],  
          'rf__max_depth': [7], 'rf__max_features': [2,3], 'rf__min_samples_split': [3] } 

with ignore_warnings(category=DeprecationWarning):
    votingclf_grid_2 = GridSearchCV(estimator=eclf_2, param_grid=params, cv=10)
    votingclf_grid_2.fit(X_sc,y_sc)
    sc_vot2 = get_best_score(votingclf_grid_2)


# In[ ]:


with ignore_warnings(category=DeprecationWarning):    
    pred_all_vot2 = votingclf_grid_2.predict(X_test_sc)

sub_vot2 = pd.DataFrame()
sub_vot2['PassengerId'] = df_test['PassengerId']
sub_vot2['Survived'] = pred_all_vot2
sub_vot2.to_csv('vot2.csv',index=False)


# In[ ]:





# ### StackingClassifier

# In[ ]:


from mlxtend.classifier import StackingClassifier


# In[ ]:


# Initializing models
clf1 = xgb_grid.best_estimator_
clf2 = gbdt_grid.best_estimator_
clf3 = rf_grid.best_estimator_
clf4 = svc_grid.best_estimator_

lr = LogisticRegression()
st_clf = StackingClassifier(classifiers=[clf1, clf1, clf2, clf3, clf4], meta_classifier=lr)

params = {'meta-logisticregression__C': [0.1,1.0,5.0,10.0] ,
          #'use_probas': [True] ,
          #'average_probas': [True] ,
          'use_features_in_secondary' : [True, False]
         }

with ignore_warnings(category=DeprecationWarning):
    st_clf_grid = GridSearchCV(estimator=st_clf, param_grid=params, cv=5, refit=True)
    st_clf_grid.fit(X_sc, y_sc)
    sc_st_clf = get_best_score(st_clf_grid)


# In[ ]:





# ### Comparison plot for best models

# In[ ]:


list_scores = [sc_rf, sc_svc, sc_gbdt, sc_xgb, sc_ada, sc_vot2, sc_st_clf]
list_classifiers = ['RF','SVC','GBDT','XGB','ADA','VOT2', 'STACK']


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10,5)
sns.barplot(x=list_classifiers, y=list_scores, ax=ax)
plt.ylabel('Score: Accuracy')
plt.show()


# ### Correlation of prediction results

# In[ ]:


predictions = {'RF': pred_all_rf, 'SVC': pred_all_svc, 'GBDT': pred_all_gbdt, 'XGB': pred_all_xgb, 'ADA': pred_all_ada, 'VOT2': pred_all_vot2}
df_predictions = pd.DataFrame(data=predictions) 
df_predictions.corr()


# 

# In[ ]:





# In[ ]:





# **This is my second notebook for the Titanic classification competition.**
# 
# If you are new to Machine Learning, have a look at  **[my first Titanic notebook](https://www.kaggle.com/dejavu23/titanic-survival-for-beginners-eda-to-ml)** where  I studied the  
# basics of EDA with Pandas and Matplotlib and how to do Classification with the scikit-learn library.  

# In[ ]:





# In[ ]:




