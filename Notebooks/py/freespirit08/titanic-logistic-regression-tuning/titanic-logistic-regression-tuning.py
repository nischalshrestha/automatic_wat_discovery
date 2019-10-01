#!/usr/bin/env python
# coding: utf-8

# # Titanic - Logistic Regression Hyperparameter Optimization
# 
# **Author:** Arindam Chatterjee  
# **Start Date:** 23rd July, 2018
# 
# **Purpose:** The objective of this notebook is to understand how the hyperparameters of an algorithm affect the predictive performance & how tuning them can help us optimize our goal. The goal itself can be simply accuracy or model build time. 
# 
# Since, this is my first try at Kaggle, I chose the simplest yet most tried out dataset of Titanic & a rather simplistic algorithm in Logistic Regression. I will continue to add new things in the notebook like more algorithms & visualizations whenever I get the time from my office work.

# ## Import Libaries

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

path="../input" #For Kaggle
#path="input"
print(os.listdir(path))


# ## Read Files

# In[ ]:


train = pd.read_csv(path+"/train.csv")
test = pd.read_csv(path+"/test.csv")
train.head()


# In[ ]:


train.info()
print('-'*50)
test.info()


# In[ ]:


# Drop PassengerId, Ticket as they are basically Row_identifier(unique ID) type columns
# Drop Cabin since it has 77%,78% missing data in train,test sets respectively making imputation infeasible
train.drop(columns=['PassengerId','Ticket','Cabin'], axis=1, inplace = True)
test.drop(columns=['Ticket','Cabin'], axis=1, inplace = True)
train.info()


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,15))
sns.countplot(x="Survived", hue="Pclass", data=train, ax=axes[0][0])
sns.countplot(x="Survived", hue="Sex", data=train, ax=axes[0][1])
sns.countplot(x="Survived", hue="SibSp", data=train, ax=axes[1][0])
sns.countplot(x="Survived", hue="Parch", data=train, ax=axes[1][1])
sns.countplot(x="Survived", hue="Embarked", data=train, ax=axes[2][0])


# ## Feature Engineering
# 
# 1. Add feature:  
#     a) FamilySize = Parch + SibSp  
#     b) Title = apply regex ([A-Za-z]+)\. on Name 
# 2. Impute Missing values:  
#     a) Fare  
#     b) Age  
#     c) Embarked
# 3. Bin feature  
#     a) Fare  
#     b) Age 
# 4. One-hot encode categorical features  
#     Sex, Embarked, Title, Fare_bin, Age_bin
# 5. Drop Unnecessary features  
#     Name, Age, Fare  
# 6. Typecast & Reduce Feature size from int64 to np.uint8  
#     Survived, Pclass, SibSp, Parch, FamilySize  
# 
# ##### Note that Double space(  ) works as newline in Jupyter Markdown .

# ### 1. Add Features

# In[ ]:


#1. a) Add family_size
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


#1. b) Add Title
import re
def getTitle(name):
    title = re.search('([A-Za-z]+)\.',name)
    if title:
        return title.group(1)
    return ""

train['Title'] = train['Name'].apply(getTitle)
test['Title'] = test['Name'].apply(getTitle)
pd.crosstab(train['Title'], train['Survived'])


# In[ ]:


pd.crosstab(test['Title'], test['Sex'])


# In[ ]:


#Bucket the Titles into appropriate groups
train['Title']=train['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir','Dona'],'Rare')
train['Title']=train['Title'].replace('Ms','Miss')
train['Title']=train['Title'].replace(['Mlle','Mme','Lady','Countess'],'Mrs')
pd.crosstab(train['Title'], train['Survived'])


# In[ ]:


test['Title']=test['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir','Dona'],'Rare')
test['Title']=test['Title'].replace('Ms','Miss')
test['Title']=test['Title'].replace(['Mlle','Mme','Lady','Countess'],'Mrs')
pd.crosstab(test['Title'], test['Sex'])


# ### 2. Impute Missing Values  
# 
# Q) Replace missing values :Mean or Median ?  
# A) https://www.quora.com/What-is-more-accurate-the-median-or-mean  

# In[ ]:


# 2a) Fare: Only test set has missing values
test['Fare'].fillna(test['Fare'].median(), inplace = True)

# 2b) Age
train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)

# 2c) Embarked
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True) #replace with the 1st Mode


# ### 3. Bin feature

# In[ ]:


#3.a) Bin feature Fare into groups
plt.figure(figsize=(50,200))
#fig, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(50,70))
#sns.countplot(x="Fare", data=train[train['Survived']==0], ax=axes2[0])
#sns.countplot(x="Fare", data=train[train['Survived']==1], ax=axes2[0])

#sns.kdeplot(x="Fare", data=train[train['Survived']==0], ax=axes2[0])
#sns.kdeplot(train['Fare'])
#sns.countplot(train['Fare'])
sns.countplot(y="Fare", hue="Survived", data=train)
#sns.distplot(train['Fare'])
#sns.countplot(y="Fare", data=train)
#pd.crosstab(train['Fare'], train['Survived'])


# In[ ]:


train['Fare_bin'] = pd.cut(train['Fare'],bins=[0,7.125,15.1,30,60,120,1000],
                           labels=['very_low_fare', 'low_fare', 'medium_fare', 
                                   'moderate_fare', 'high_fare', 'very_high_fare'])
test['Fare_bin'] = pd.cut(train['Fare'],bins=[0,7.125,15.1,30,60,120,1000],
                           labels=['very_low_fare', 'low_fare', 'medium_fare', 
                                   'moderate_fare', 'high_fare', 'very_high_fare'])
pd.crosstab(train['Fare_bin'], train['Survived'])


# In[ ]:


#3.b) Bin Age
plt.figure(figsize=(50,100))
sns.countplot(y="Age", hue="Survived", data=train)


# In[ ]:


#0,12,20,40,60,120
train['Age_bin'] = pd.cut(train['Age'], bins=[0,12,20,40,60,120], 
                          labels=['Child','Teenage','Adult','MiddleAge','ElderAge'])
test['Age_bin'] = pd.cut(test['Age'], bins=[0,12,20,40,60,120], 
                          labels=['Child','Teenage','Adult','MiddleAge','ElderAge'])
pd.crosstab(train['Age_bin'], train['Survived'])


# ### 4. One-hot encode categorical features  
# Sex, Embarked, Title, Fare_bin, Age_bin

# In[ ]:


train = pd.get_dummies(train, columns = ["Sex","Embarked","Title","Fare_bin", "Age_bin"], prefix_sep='=', 
                             prefix=["Sex","Embarked","Title","Fare_bin", "Age_bin"])
test = pd.get_dummies(test, columns = ["Sex","Embarked","Title","Fare_bin", "Age_bin"], prefix_sep='=', 
                             prefix=["Sex","Embarked","Title","Fare_bin", "Age_bin"])
train.head()


# ### 5. Drop Unnecessary features  
# Name, Age, Fare

# In[ ]:


train.drop(columns=['Name','Age','Fare'], axis=1, inplace = True)
test.drop(columns=['Name','Age','Fare'], axis=1, inplace = True)


# ### 6. Typecast & Reduce Feature size from int64 to np.uint8
# Survived, Pclass, SibSp, Parch, FamilySize

# In[ ]:


cols = ['Pclass','SibSp','Parch','FamilySize']
train[cols] = train[cols].astype(np.uint8)
test[cols] = test[cols].astype(np.uint8)
train['Survived'] = train['Survived'].astype(np.uint8)


# In[ ]:


# Need to save this after all pre-processing done for submission
test_Pids = test['PassengerId']
test.drop(columns=['PassengerId'], axis=1, inplace = True)


# In[ ]:


train.info()


# #### Create Train & Validation Sets

# In[ ]:


X = train.drop(columns='Survived')
Y = train['Survived']

skf = StratifiedKFold(n_splits=4,random_state=1)
for train_index, test_index in skf.split(X, Y):
    X_tr, X_val = X.iloc[train_index], X.iloc[test_index]
    Y_tr, Y_val = Y.iloc[train_index], Y.iloc[test_index]
    print('Train & Validation sets built.')
    break


# ## Model Build & Optimizations  
# 
# The objective of this step is to visually understand how the different hyperparamaters & random_state affect
# the outcome of the model ie: its predictive accuracy.   
# We will initially go through each of them individually even though for most model algorithms, the hyperparameters
# are not independant of each other. The test set used will be constant thrughout the process.
# 
# The below data descriptions are taken from the sklearn documentation in:  
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
# 
# 1. C : float, default: 1.0 : Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# 2. tol : float, default: 1e-4 : Tolerance for stopping criteria. This tells the algorithm to stop searching for a minimum (or maximum) once some tolerance is achieved, i.e. once it is close enough. 
# 3. solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default: ‘liblinear’ Algorithm to use in the optimization problem.
# 4. max_iter : int, default: 100 : Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.

# In[ ]:


#Define generalized function for Scoring current instance of model & data
'''
returns ::  a)acc: accuracy as computed on Validation set
            b)exec_time: model build/fit time
'''
def evaluate(X_tr, Y_tr, X_val, Y_val, params):
    model = LogisticRegression()
    #We should use set_params to pass parameters to model object.
    #This has the advantage over using setattr in that it allows Scikit learn to perform some validation checks on the parameters.
    model.set_params(**params)
    
    start=time()
    model.fit(X_tr,Y_tr)
    exec_time = time() - start
    
    Y_pred = model.predict(X_val)
    acc = accuracy_score(Y_val,Y_pred) * 100.0
    return acc, exec_time


# ### 1. Optimizing C

# In[ ]:


C=0.001
iterations = 500
results = np.zeros((iterations, 5))

for i in range(0,iterations):    
    model_params = {'C':C,'random_state':1}
    acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
    acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
    results[i] = i+1, C, acc_tr, acc_val, time_val
    C+=0.005

res_df = pd.DataFrame(  data=results[0:,0:], 
                        index=results[0:,0],
                        columns=['Sl','C','Train_acc','Val_acc','Build_time'])
res_df['Sl'] = res_df['Sl'].astype(np.uint16)
res_df.head()


# In[ ]:


#Find value of C & Train_acc at which Valiation set acuracy is highest.
res_df[res_df['Val_acc'] == res_df['Val_acc'].max()]


# In[ ]:


plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Train & Validation Set Accuracy w.r.t to Regularization parameter C')
plt.grid(True)
plt.plot(res_df['C'], res_df['Train_acc'] , 'r*-') # plotting t, a separately 
plt.plot(res_df['C'], res_df['Val_acc'] , 'b.') # plotting t, a separately


# C is the inverse of regularization strength. Hence, smaller values increase regularization & can result in underfitting while as we increase C, model becomes more prone to overfitting.  
# Thus we find that, for the train data, model accuracy increases as C value increases. BUT, with increase in C,
# the accuracy on validation data starts to increase initially before peaking at (0.211, 83.03571429). After that,
# the effect of overfitting causes the accuracy to decrease. Hence the optimal value of C is 0.211

# ### 2. Optimzing tol (Tolerance for Stopping Criteria)  
# 
# tol = Tolerance for stopping criteria. This tells the algorithm to stop searching for a minimum (or maximum) once some tolerance is achieved, i.e. once it is close enough. tol will change depending on the objective function being minimized/maximized and the algorithm they use to find the minimum, and thus will depend on the model we are fitting.
# 
# For the newton-cg and lbfgs solvers, the iteration will stop when ``max{|g_i | i = 1, ..., n} <= tol`` where ``g_i`` is the i-th component of the gradient.

# In[ ]:


#tol=1e-6
tol=1e-10
#iterations = 50
iterations = 37
results = np.zeros((iterations, 5))

for i in range(0,iterations):    
    model_params = {'tol':tol,'random_state':1}
    acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
    acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
    results[i] = i+1, tol, acc_tr, acc_val, time_val
    #tol*=5
    tol*=2
    #print(tol)

res_df_tol = pd.DataFrame(  data=results[0:,0:], 
                        index=results[0:,0],
                        columns=['Sl','tol','Train_acc','Val_acc','Build_time'])
res_df_tol['Sl'] = res_df['Sl'].astype(np.uint16)
res_df_tol.head()


# In[ ]:


plt.figure(figsize=(20,5))
plt.xlabel('tol')
plt.ylabel('Accuracy')
plt.title('Train & Validation Set Accuracy w.r.t to Tolerance tol')
plt.grid(True)
plt.plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'r*')
plt.plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'y-')
plt.plot(res_df_tol['tol'], res_df_tol['Val_acc'] , 'b.')
#plt.plot(res_df_tol['tol'], res_df_tol['Build_time'] , 'g')


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))

ax[0].set(xlabel='tol', ylabel='Accuracy')
ax[0].set_title('Train & Validation Set Accuracy w.r.t to Tolerance tol')
ax[0].grid(True)
ax[0].plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'r*')
ax[0].plot(res_df_tol['tol'], res_df_tol['Train_acc'] , 'y')
ax[0].plot(res_df_tol['tol'], res_df_tol['Val_acc'] , 'b.')

ax[1].set(xlabel='tol', ylabel='Model Build Time')
ax[1].set_title('Model Build Time w.r.t to Tolerance tol')
ax[1].grid(True)
ax[1].plot(res_df_tol['tol'], res_df_tol['Build_time'] , 'r*')
ax[1].plot(res_df_tol['tol'], res_df_tol['Build_time'] , 'y')


# Tolerance tol for Logistic regression represents when convergence is achieved for its gradience values in an iteration. Hence, to converge for lower values of tol, more number of iterations are needed at the expense of greater model build time. This is illustrated in Plot-2. 
# 
# But, as we see in Plot-1, for a fast simple algorithm like Logistic Regression working on small sets of data, accuracy also increases for smaller values of tol. We see that, accuracy remains more or less constant between tol values of 0 & 0.33 for both sets. The dafult value of 1e-4 is sufficient for this case as further lowering it increases the buil time wihout any appreciable increase in accuracy. On the other hand, accuracy falls of pretty sharply once tol bcomes more than 1.8. Beyond, tol of ~3.4, accuracy again stabilizes in the low ~61%. 
# 
# By running for greater values of tol upto tol=20, it was found to remain constant. Again, the tol graph will depend on many other factors like nature of the data & the internal solver, which we will see below.

# ### 3. Optimizing the internal solver : 
# 
# ** From sklearn documentation, **  
# solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, 
# default: ‘liblinear’ Algorithm to use in the optimization problem.
# 
# a) For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
# 
# b) For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# 
# c) ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
# 
# Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
# 
# New in version 0.17: Stochastic Average Gradient descent solver.
# 
# New in version 0.19: SAGA solver.
# 
# **LIBLINEAR – A Library for Large Linear Classification**  
# http://www.csie.ntu.edu.tw/~cjlin/liblinear/
# 
# **SAG – Mark Schmidt, Nicolas Le Roux, and Francis Bach**  
# Minimizing Finite Sums with the Stochastic Average Gradient  
# https://hal.inria.fr/hal-00860051/document
# 
# **SAGA – Defazio, A., Bach F. & Lacoste-Julien S. (2014).**  
# SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives.  
# https://arxiv.org/abs/1407.0202
# 
# **Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent**
# methods for logistic regression and maximum entropy models. Machine Learning 85(1-2):41-75.  
# http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
# 

# #### 3.1 Variation of tol wrt to the Solver

# In[ ]:


# 3.1 Variation of tol wrt to the Solver

tol=1e-10
iterations = 37

# There are 5 solvers. For each, we need to see their accuracy on train & validation sets plus their build time.
# Additionaly, first two columns are Sl & tol. Hence, a total of (5*3) + 2 = 17 columns reqd.
results = np.zeros((iterations, 17))
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

for i in range(0,iterations):    
    model_params = {'tol':tol,'random_state':1}
    results[i][0:2] = i+1, tol
    
    j = 2 #internal counter for iterating over each of the solver's results values
    for solver in solver_list:
        model_params.update({'solver': solver})
        acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
        acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
        results[i][j:j+3] = acc_tr, acc_val, time_val
        j+=3
        
    tol*=2

columns = ['Sl','tol']
for solver in solver_list:
    columns.append('Train_acc_'+solver)
    columns.append('Val_acc_'+solver)
    columns.append('Build_time_'+solver)

res_df_solver_tol = pd.DataFrame( data=results[0:,0:], 
                                  index=results[0:,0],
                                  columns=columns)
res_df_solver_tol['Sl'] = res_df_solver_tol['Sl'].astype(np.uint16)
res_df_solver_tol.head()


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

ax[0].set(xlabel='tol', ylabel='Accuracy')
ax[0].set_title('Variation in Training Data Accuracy w.r.t to Tolerance tol for different Solvers')
ax[0].grid(True)

colour_list = ['r','g','b','c','m']
for i in range(0,5):
    ax[0].plot(res_df_solver_tol['tol'],
               res_df_solver_tol['Train_acc_'+solver_list[i]],
               colour_list[i]+'.-', label=solver_list[i])
ax[0].legend()

ax[1].set(xlabel='tol', ylabel='Accuracy')
ax[1].set_title('Variation in Validation Data Accuracy w.r.t to Tolerance tol for different Solvers')
ax[1].grid(True)
for i in range(0,5):
    ax[1].plot(res_df_solver_tol['tol'],
               res_df_solver_tol['Val_acc_'+solver_list[i]] ,
               colour_list[i]+'.-', label=solver_list[i])    
ax[1].legend()
    
ax[2].set(xlabel='tol', ylabel='Build_Time')
ax[2].set_title('Variation in Model Build Time w.r.t to Tolerance tol for different Solvers')
ax[2].grid(True)
for i in range(0,5):
    ax[2].plot(res_df_solver_tol['tol'],
               res_df_solver_tol['Build_time_'+solver_list[i]] ,
               colour_list[i]+'.-', label=solver_list[i])  
ax[2].legend()


# **Conclusion:** From the above plots, we can see that the default solver 'liblinear' loses accuracy rapidly with increaseing value of Tolerance(tol). This is countered by having the fastest build speed. For small values of tol, there is no practical difference in accuracy between the solvers. But as tol increases, the other solvers are able to maintain their accuracy with lbfgs & newton-cg remaining almost constant.

# #### 3.2 Variation of C wrt to the Solver

# In[ ]:


# 3.2 Variation of C wrt to the Solver

C=0.001
iterations = 500

# There are 5 solvers. For each, we need to see their accuracy on train & validation sets plus their build time.
# Additionaly, first two columns are Sl & C. Hence, a total of (5*3) + 2 = 17 columns reqd.
results = np.zeros((iterations, 17))
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

for i in range(0,iterations):    
    model_params = {'C':C,'random_state':1}
    results[i][0:2] = i+1, C
    
    j = 2 #internal counter for iterating over each of the solver's results values
    for solver in solver_list:
        model_params.update({'solver': solver})
        acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
        acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
        results[i][j:j+3] = acc_tr, acc_val, time_val
        j+=3
        
    C+=0.005

columns = ['Sl','C']
for solver in solver_list:
    columns.append('Train_acc_'+solver)
    columns.append('Val_acc_'+solver)
    columns.append('Build_time_'+solver)

res_df_solver_C = pd.DataFrame( data=results[0:,0:], 
                                  index=results[0:,0],
                                  columns=columns)
res_df_solver_C['Sl'] = res_df_solver_C['Sl'].astype(np.uint16)
res_df_solver_C.head()


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,30))

ax[0].set(xlabel='C', ylabel='Accuracy')
ax[0].set_title('Variation in Training Data Accuracy w.r.t to Inverse Regularization parameter C for different Solvers')
ax[0].grid(True)

colour_list = ['r','g','b','c','m']
for i in range(0,5):
    ax[0].plot(res_df_solver_C['C'],
               res_df_solver_C['Train_acc_'+solver_list[i]],
               colour_list[i]+'-', label=solver_list[i])
ax[0].legend()

ax[1].set(xlabel='C', ylabel='Accuracy')
ax[1].set_title('Variation in Validation Data Accuracy w.r.t to Inverse Regularization parameter C for different Solvers')
ax[1].grid(True)
for i in range(0,5):
    ax[1].plot(res_df_solver_C['C'],
               res_df_solver_C['Val_acc_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])    
ax[1].legend()
    
ax[2].set(xlabel='C', ylabel='Build_Time')
ax[2].set_title('Variation in Model Build Time w.r.t to Inverse Regularization parameter C for different Solvers')
ax[2].grid(True)
for i in range(0,5):
    ax[2].plot(res_df_solver_C['C'],
               res_df_solver_C['Build_time_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])  
ax[2].legend()


# **Conclusion:** As Regularization increases, the solvers have more or less similar performance in terms of accuracy with the liblinear giving the highest accuracy of all for a value of C which is the point where the model is neither under-fitted nor over-fitted. The region surrounding the liblinear peak shows the highest accuracy values for the other solvers too but without a peak. But given the size of the data & the number of iterations fixed at default 100(sag & saga did not converge with the default value), it cannot be concluded which solver is the best by accuracy. But, time wise, we see liblinear is the fastest while stochastic gradient methods(sag/saga) being the slowest.

# ### 4. Optimizing the max_iterations for a solver 
# 
# max_iter : int, default: 100 : Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
# 

# In[ ]:


max_iter=5
iterations = 40

# There are 3 solvers. For each, we need to see their accuracy on train & validation sets plus their build time.
# Additionaly, first two columns are Sl & C. Hence, a total of (3*3) + 2 = 11 columns reqd.
results = np.zeros((iterations, 11))
solver_list = ['newton-cg', 'lbfgs', 'sag']

for i in range(0,iterations):    
    model_params = {'max_iter':max_iter,'random_state':1}
    results[i][0:2] = i+1, max_iter
    
    j = 2 #internal counter for iterating over each of the solver's results values
    for solver in solver_list:
        model_params.update({'solver': solver})
        acc_val,time_val = evaluate(X_tr, Y_tr, X_val, Y_val, model_params)
        acc_tr,time_tr = evaluate(X_tr, Y_tr, X_tr, Y_tr, model_params)
        results[i][j:j+3] = acc_tr, acc_val, time_val
        j+=3
        
    max_iter += 5

columns = ['Sl','max_iter']
for solver in solver_list:
    columns.append('Train_acc_'+solver)
    columns.append('Val_acc_'+solver)
    columns.append('Build_time_'+solver)

res_df_solver_max_iter = pd.DataFrame( data=results[0:,0:], 
                                  index=results[0:,0],
                                  columns=columns)
res_df_solver_max_iter['Sl'] = res_df_solver_max_iter['Sl'].astype(np.uint16)
res_df_solver_max_iter.head()


# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,30))

ax[0].set(xlabel='max_iter', ylabel='Accuracy')
ax[0].set_title('Variation in Training Data Accuracy w.r.t to Max Iterations for different Solvers')
ax[0].grid(True)

colour_list = ['r*-','gv-','bs-']
for i in range(0,3):
    ax[0].plot(res_df_solver_max_iter['max_iter'],
               res_df_solver_max_iter['Train_acc_'+solver_list[i]],
               colour_list[i]+'-', label=solver_list[i])
ax[0].legend()

ax[1].set(xlabel='max_iter', ylabel='Accuracy')
ax[1].set_title('Variation in Validation Data Accuracy w.r.t to Max Iterations for different Solvers')
ax[1].grid(True)
for i in range(0,3):
    ax[1].plot(res_df_solver_max_iter['max_iter'],
               res_df_solver_max_iter['Val_acc_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])    
ax[1].legend()
    
ax[2].set(xlabel='max_iter', ylabel='Build_Time')
ax[2].set_title('Variation in Model Build Time w.r.t to Max Iterations for different Solvers')
ax[2].grid(True)
for i in range(0,3):
    ax[2].plot(res_df_solver_max_iter['max_iter'],
               res_df_solver_max_iter['Build_time_'+solver_list[i]] ,
               colour_list[i]+'-', label=solver_list[i])  
ax[2].legend()


# **Conclusion:** Solver newton-cg appears as the least susceptible to accuracy differences with variation in the maximum number of iterations allowed for convergence, for both train & validation data. Its build time also fluctuates the least & almost follows a parralel line to X-axis. After a certain value of max_iter, the other 2 solvers also have constant accuracy curves. But, sag build time increases consistently with time without any accuracy increase.

# In[ ]:


# Final values for Log Reg:
model_params = {'C':0.211, 'tol':1e-6, 'solver':'liblinear', 'random_state':1}
model = LogisticRegression()
model.set_params(**model_params)
model.fit(X_tr,Y_tr)
Y_pred = model.predict(X_val)
acc = accuracy_score(Y_val,Y_pred) * 100.0
print('Final Accuracy = {}%'.format(acc))
results_logreg = model.predict(test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_Pids,
        "Survived": results_logreg})


# In[ ]:


submission.to_csv("try_1_logreg.csv", index=False)


# In[ ]:




