#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Title : Kaggle : Titanic-Ensembling
@author: Ajay Dabas
""" 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import sklearn
from sklearn import ensemble, metrics, model_selection
import xgboost as xgb


# In[9]:


# Load in the train and test datasets
train_data = pd.read_csv('../input/train.csv')
test_data  = pd.read_csv('../input/test.csv')
PassengerId = test_data["PassengerId"]


# **Feature Engineering**

# In[10]:


full_data = [train_data, test_data]

for dataset in full_data:
    # Create new feature FamilySize as a combination of SibSp and Parch
    dataset['FamilySize']  =    dataset['SibSp'] + dataset['Parch'] + 1
    # Remove all Null values in the Fare column
    dataset['Fare']        =    dataset['Fare'].fillna(0)
    # Mapping Sex
    dataset['Sex']         =    dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 8.1, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 8.1) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    # Fill Age null values randomly
    #age_avg                =    dataset['Age'].mean()
    #age_std                =    dataset['Age'].std()
    #age_null_count         =    dataset['Age'].isnull().sum()
    #age_null_random_list   =    np.random.randint(age_avg - 3*age_std, age_avg + 3age_std, size=age_null_count)
    #dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    #dataset['Age']         =    dataset['Age'].astype(int)
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
    dataset['Age']        =    dataset['Age'].fillna(1)
    
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title']   =   dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    dataset['Title']   =   dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title']   =   dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title']   =   dataset['Title'].replace('Ms', 'Miss')
    dataset['Title']   =   dataset['Title'].replace('Mme', 'Mrs')
    # Mapping titles
    dataset['Title']   =   dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title']   =   dataset['Title'].fillna(0)


# **Feature Selection**

# In[11]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',"Parch","Embarked","Sex","Fare"]
train_data    = train_data.drop(drop_elements, axis = 1)
test_data     = test_data.drop(drop_elements, axis = 1)


# In[12]:


train_data.head()


# **Correlation of Features**

# In[13]:


plt.figure(figsize=(14,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.corr(),vmin=0,vmax=1,linewidths=0.5,cmap="Blues",annot=True)


# We see most of the features are not strongly correlated so each feature provides unique and valuable information.

# **Ensembling models**

# In[ ]:


y_train    = train_data['Survived'].ravel()
train_data = train_data.drop(['Survived'], axis=1)
X_train    = train_data.values # Creates an array of the train data
X_test     = test_data.values # Creats an array of the test data


# In[ ]:


# Defining metric for evaluation
accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)

# Random Forest Classifier
params = {'n_estimators':[80,100,120], 'max_depth':[None,1,2]} 
randomforest_best = model_selection.GridSearchCV(ensemble.RandomForestClassifier(criterion="entropy",random_state = 42,min_samples_leaf=2), param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs=-1)
randomforest_best.fit(X_train, y_train)
y_train_pred_rf   = randomforest_best.predict(X_train)
y_test_pred_rf    = randomforest_best.predict(X_test)

# ExtraTrees Classifier
params = {'n_estimators':[80,100,120], 'max_depth':[None,1,2]} 
extratrees_best = model_selection.GridSearchCV(ensemble.ExtraTreesClassifier(random_state=42,min_samples_leaf=2), param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs=-1)
extratrees_best.fit(X_train, y_train)
y_train_pred_et   = extratrees_best.predict(X_train)
y_test_pred_et    = extratrees_best.predict(X_test)

# AdaBoost Classifier
params = {'n_estimators':[180,200,220], "learning_rate":[0.09,0.1,0.2]} 
adaboost_best = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(random_state=42), param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs=-1)
adaboost_best.fit(X_train, y_train)
y_train_pred_ada   = adaboost_best.predict(X_train)
y_test_pred_ada    = adaboost_best.predict(X_test)

# Gradient Boosting Classifier
params = {'n_estimators':[280,300,320], "learning_rate":[0.009,0.01,0.02],'max_depth':[None,1,2,3], "min_samples_leaf":[1,2]} 
gradboost_best = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(random_state=42), param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs=-1)
gradboost_best.fit(X_train, y_train)
y_train_pred_gb   = gradboost_best.predict(X_train)
y_test_pred_gb    = gradboost_best.predict(X_test)

# SVC Classifier
#params = {"C":[0.01,0.02]} 
#svc_best = model_selection.GridSearchCV(svm.SVC(kernel="linear",random_state=42), param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs=-1)
#svc_best.fit(X_train, y_train)
#y_train_pred_svc   = svc_best.predict(X_train)
#y_test_pred_svc    = svc_best.predict(X_test)


# In[ ]:


print("Score of RandomForest Classifier  :",randomforest_best.score(X_train,y_train))
print("Score of ExtraTrees Classifier    :",extratrees_best.score(X_train,y_train))
print("Score of AdaBoost Classifier      :",adaboost_best.score(X_train,y_train))
print("Score of GradientBoost Classifier :",gradboost_best.score(X_train,y_train))
#print("Score of SVC Classifier           :",svc_best.score(X_train,y_train))


# In[ ]:


print("Best Parameters of RandomForest Classifier  :",randomforest_best.best_params_)
print("Best Parameters of ExtraTrees Classifier    :",extratrees_best.best_params_)
print("Best Parameters of AdaBoost Classifier      :",adaboost_best.best_params_)
print("Best Parameters of GradientBoost Classifier :",gradboost_best.best_params_)
#print("Best Parameters of SVC Classifier           :",svc_best.best_params_)


# In[ ]:


X_train_ensembled = pd.DataFrame( {'RandomForest': y_train_pred_rf,'ExtraTrees': y_train_pred_et,
                                        'AdaBoost': y_train_pred_ada,'GradientBoost': y_train_pred_gb})#,'SVC': y_train_pred_svc})


# In[ ]:


X_test_ensembled = pd.DataFrame( {'RandomForest': y_test_pred_rf,'ExtraTrees': y_test_pred_et,
                                        'AdaBoost': y_test_pred_ada,'GradientBoost': y_test_pred_gb})#,'SVC': y_test_pred_svc})


# In[ ]:


gbm = xgb.XGBClassifier().fit(X_train_ensembled.values, y_train)


# In[ ]:


# Generate Submission File 
predictions = gbm.predict(X_test_ensembled.values)
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,'Survived': predictions })
StackingSubmission.to_csv("Submission.csv", index=False)

