#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split   #for split the data
from sklearn.model_selection import KFold              #for K-fold cross validation
from sklearn.model_selection import cross_val_score    #score evaluation
from sklearn.model_selection import cross_val_predict  #prediction
from sklearn.model_selection import GridSearchCV       #for hyperparameter tuning
from sklearn.model_selection import learning_curve
from sklearn.metrics         import accuracy_score     #for accuracy_score
from sklearn.metrics         import confusion_matrix   #for confusion matrix
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import ExtraTreesClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.ensemble        import VotingClassifier


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test  = pd.read_csv("../input/test.csv")


# In[ ]:


df_train.head(5)


# In[ ]:


df_test.head(5)


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


df_train.info()


# In[ ]:


df_test.info() #Since this is the test set, the Survivors class is missing


# In[ ]:


print("Training shape: \n", df_train.shape)
print("Test shape: \n", df_test.shape)


# In[ ]:


def missingData(dataset):
    """Check missing data """
    total = dataset.isnull().sum().sort_values(ascending = False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()*100).sort_values(ascending = False)
    md = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    md = md[md["Percent"] > 0]
    return md


# In[ ]:


missingData(df_train)


# In[ ]:


missingData(df_test)


# In[ ]:


#Drop the Cabin Feature because is full of missing data (>75% missing)
df_train.drop("Cabin", axis=1, inplace = True)
df_test.drop("Cabin", axis=1, inplace = True)


# In[ ]:


#Fill the Age Feature with median
df_train["Age"].fillna(df_train["Age"].median(), inplace = True)
df_test["Age"].fillna(df_test["Age"].median(),  inplace = True)


# In[ ]:


#Fill the Fare Feature in the test set with median
df_test["Fare"].fillna(df_test['Fare'].median(), inplace = True)


# In[ ]:


#Fill the Embarked Feature in the train setwith 0
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])


# In[ ]:


print("Check the NaN value in train data")
print(df_train.isnull().sum())
print("---"*30)
print("Check the NaN value in test data")
print(df_test.isnull().sum())
print("---"*30)


# ***Feature Extraction***

# In[ ]:


all_data = [df_train, df_test]


# In[ ]:


# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #+1 because it indicates the person of the i-th row


# In[ ]:


# Create bin for age features
for dataset in all_data:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,60,120], labels=['Children','Teenager','Adult','Elder'])


# In[ ]:


#Create a Title feature
for dataset in all_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


# Create a copy of dataset 
dfTrain = df_train.copy()
dfTest  = df_test.copy()


# In[ ]:


traindf = pd.get_dummies(dfTrain, columns = ["Pclass","Title","Sex","Age_bin","Embarked"],
                             prefix=["Pclass","Title","Sex","Age_type","Em_type"])

testdf = pd.get_dummies(dfTest, columns = ["Pclass","Title","Sex","Age_bin","Embarked"],
                             prefix=["Pclass","Title","Sex","Age_type","Em_type"])


# In[ ]:


allData = [traindf, testdf]


# In[ ]:


traindf


# In[ ]:


for dataset in allData:
    drop_column = ["Age","Fare","Name","Ticket","SibSp","Parch"]
    dataset.drop(drop_column, axis=1, inplace = True)

traindf.drop(["PassengerId"], axis=1, inplace = True)


# In[ ]:


traindf


# In[ ]:


def valueCounts(dataset, features):
    """Display the features value counts """
    for feature in features:
        vc = dataset[feature].value_counts()
        print(vc)


# In[ ]:


columns_train = list(traindf)
valueCounts(traindf, columns_train)


# In[ ]:


columns_test = list(testdf)
valueCounts(testdf, columns_test)


# In[ ]:


plt.figure(figsize =(20, 14))
sns.heatmap(traindf.corr(),annot=True,cmap='coolwarm',linewidths=0.2)
plt.title('Pearson Correlation of features', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:


#Split into features and class
array = traindf.values #convert into array the train set

features = array[:,1:].astype(float)
targeted = array[:,0].astype(float)


# In[ ]:


#Set the costant parameters for model training

seed      = 7
v_size    = 0.33
num_folds = 10
scoring   = 'accuracy'


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(features,targeted,test_size=v_size,random_state=seed)

print("X_train shape: \n", X_train.shape)
print("X_test shape: \n", X_test.shape)
print("y_train shape: \n", y_train.shape)
print("y_test shape: \n", y_test.shape)


# **Spot-Check Algorithms**

# In[ ]:


def algoSpotCheck(models, X_train, y_train, num_folds, scoring, seed):
    """Makes a spot-check of the models"""
    results = []
    names   = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        print("*"*60)
    return names, results


def boxplotCompare(names, results, title):
    """Generate a boxplot of the models"""
    fig = plt.figure(figsize=(20, 14)) 
    ax = fig.add_subplot(111)
    sns.set(style = 'darkgrid')
    sns.boxplot(data=results)
    ax.set_xticklabels(names) 
    plt.title('Comparison between Algorithms', size = 40, color='k')
    plt.xlabel('Percentage',size = 20,color="k")
    plt.ylabel('Algorithm',size = 20,color="k")
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Spot-Check Algorithms

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('SVM', SVC()),     
         ]

names,results = algoSpotCheck(models,X_train,y_train,num_folds,scoring,seed)
boxplotCompare(names, results, 'Comparison_beetween_Algorithms')


# In[ ]:


#Hyperparameter Tuning Function

def algoGridTune(model, param_grid, X_train, y_train, num_folds, scoring, seed):
    """Makes the hyperparameter tuning of the chosen model"""
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    best_estimator = grid_result.best_estimator_
    print("BestScore: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("*"*60)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Score: %f (%f) with: %r" % (mean, stdev, param))
        print("*"*60)
    return best_estimator


# In[ ]:


svc_clf = SVC()
svc_param_grid = [{"kernel": ["rbf"], 
                   "gamma": [10 ,1, 0.1, 1e-2, 1e-3],
                   "C": [0.1, 1, 10, 100]},
                  {"kernel": ["linear"], "C": [0.1,1,10,100]}
                 ]   

best_SVC = algoGridTune(svc_clf, svc_param_grid, X_train, y_train, num_folds, scoring, seed)


# **Spot-Check Ensembles**

# In[ ]:


# Spot-Check Algorithms

models = [('RFC', RandomForestClassifier()),
          ('ETC', ExtraTreesClassifier()),
          ('ABC', AdaBoostClassifier()),
          ('GBC', GradientBoostingClassifier())
           ]

names,results = algoSpotCheck(models,X_train,y_train,num_folds,scoring,seed)
boxplotCompare(names, results, 'Comparison_beetween_Ensembles')


# In[ ]:


# Adaboost
DTC = DecisionTreeClassifier()
ABC_clf = AdaBoostClassifier(DTC, random_state=7)

ABC_param_grid = {"base_estimator__criterion" : ["gini"],
                  "base_estimator__splitter" :   ["best"],
                  "algorithm" : ["SAMME"],
                  "n_estimators" :[50, 100, 200, 300, 400],
                  "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

best_ABC = algoGridTune(ABC_clf, ABC_param_grid, X_train, y_train, num_folds, scoring, seed)


# In[ ]:


#ExtraTrees 
ETC_clf = ExtraTreesClassifier()

ETC_param_grid = {"max_depth": [None],
              "max_features": ['sqrt'],
              "min_samples_split": [2, 3, 5],
              "min_samples_leaf": [1, 3, 5],
              "bootstrap": [True],
              "n_estimators" :[100, 200, 300],
              "criterion": ["gini"]}

best_ETC = algoGridTune(ETC_clf, ETC_param_grid, X_train, y_train, num_folds, scoring, seed)


# In[ ]:


#Gradient Boosting Classifier
GBC_clf = GradientBoostingClassifier()
GBC_param_grid = {'loss' : ["deviance"],
                  'n_estimators' : [100],
                  'learning_rate': [0.1, 0.01],
                  'max_depth': [3, 5],
                  'min_samples_leaf': [1, 5],
                  'min_samples_split': [2, 6],
                  'max_features': ['sqrt'] 
                 }

best_GBC = algoGridTune(GBC_clf, GBC_param_grid, X_train, y_train, num_folds, scoring, seed)


# In[ ]:


RFC_clf = RandomForestClassifier()
RFC_param_grid = {"max_depth": [None],
                 "max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 5, 10],
                 "min_samples_leaf": [1, 3, 5, 10],
                 "bootstrap": [False],
                 "n_estimators" :[100,300],
                 "criterion": ["gini"]}


best_RFC = algoGridTune(RFC_clf, RFC_param_grid, X_train, y_train, num_folds, scoring, seed)


# In[ ]:


def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure(figsize=(15,10))
    sns.set(style = 'darkgrid')
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

    return plt


# In[ ]:


kfold = KFold(n_splits=num_folds, random_state=seed)

plotLearningCurve(best_SVC,"SupportVectorClf learning curves",X_train,y_train,cv=kfold)


# In[ ]:


plotLearningCurve(best_RFC,"RandomForest learning curves",X_train,y_train,cv=kfold)


# In[ ]:


plotLearningCurve(best_GBC,"GradientBoosting learning curves",X_train,y_train,cv=kfold)


# In[ ]:


plotLearningCurve(best_ABC,"AdaBoost learning curves",X_train,y_train,cv=kfold)


# In[ ]:


plotLearningCurve(best_ETC,"ExtraTrees learning curves",X_train,y_train,cv=kfold)


# **Ensemble's features importance**

# In[ ]:


cols = list(traindf.drop("Survived", axis=1))

sns.set_style('darkgrid')
f,ax = plt.subplots(4,1,figsize=(18,18))

pd.Series(best_RFC.feature_importances_,cols).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0],cmap='RdYlGn')
ax[0].set_title('Feature Importance in Random Forests')

pd.Series(best_ABC.feature_importances_,cols).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1],cmap='RdYlGn')
ax[1].set_title('Feature Importance in AdaBoost')

pd.Series(best_GBC.feature_importances_,cols).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[2],cmap='RdYlGn')
ax[2].set_title('Feature Importance in Gradient Boosting')

pd.Series(best_ETC.feature_importances_,cols).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[3],cmap='RdYlGn')
ax[3].set_title('Feature Importance in ExtraTrees')

plt.show()


# In[ ]:


passengerIds = testdf["PassengerId"].copy()
testdf.drop(["PassengerId"], axis=1, inplace = True)


# In[ ]:


test = testdf.values


# In[ ]:


test_Survived_SVC = pd.Series(best_SVC.predict(test), name="SVC")
test_Survived_RFC = pd.Series(best_RFC.predict(test), name="RFC")
test_Survived_GBC = pd.Series(best_GBC.predict(test), name="GBC")
test_Survived_ETC = pd.Series(best_ETC.predict(test), name="ETC")
test_Survived_ABC = pd.Series(best_ABC.predict(test), name="ABC")

# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_SVC,
                              test_Survived_RFC, 
                              test_Survived_GBC,
                              test_Survived_ETC,
                              test_Survived_ABC
                             ],axis=1)


# In[ ]:


plt.figure(figsize =(20, 14))
sns.heatmap(ensemble_results.corr(),annot=True,cmap='coolwarm',linewidths=0.2)
plt.title('Correlation beetween Models', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# **Combining models with VotingClassifier**

# In[ ]:


votingC = VotingClassifier(estimators=[('svc', best_SVC),
                                       ('rfc', best_RFC),
                                       ('gbc', best_GBC),
                                       ('etc', best_ETC),
                                       ('abc', best_ABC),
                                        ], 
                           voting='hard', n_jobs=-1)

votingC = votingC.fit(X_train, y_train)


# In[ ]:


votingC


# In[ ]:


predictions = votingC.predict(test)
test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([passengerIds,test_Survived],axis=1)

results.to_csv("results.csv",index=False)


# In[ ]:




