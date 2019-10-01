#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[ ]:


#讀取文件
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.info()
#檢查資料中的遺失值數目，以此來考慮如何處理


# In[ ]:


test_df.info()
#檢查資料中的遺失值數目，以此來考慮如何處理


# In[ ]:


test_id = test_df['PassengerId']


# In[ ]:


dataset = pd.concat(objs=[train_df,test_df],axis=0).reset_index(drop=True)


# In[ ]:


dataset.fillna(np.nan)


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


warnings.filterwarnings(action="ignore")
survived = dataset[dataset['Survived']==1]
nonsur = dataset[dataset['Survived'] == 0]
plt.figure(figsize=(12,10))
plt.subplot(331)
sns.distplot(survived['Age'].dropna().values,
             bins=range(0,81,1), kde=False, color='b')
sns.distplot(nonsur['Age'].dropna().values,
              bins=range(0,81,1), kde =False, color='r')
plt.subplot(332)
sns.barplot('Sex', 'Survived', data=dataset)
plt.subplot(333)
sns.barplot('Pclass', 'Survived', data=dataset)
plt.subplot(334)
sns.barplot('Embarked', 'Survived', data=dataset)
plt.subplot(335)
sns.barplot('SibSp', 'Survived', data=dataset)
plt.subplot(336)
sns.barplot('Parch', 'Survived', data=dataset)
plt.subplot(337)
sns.distplot(np.log(survived['Fare'].dropna().values+1), kde=False, color='b',axlabel='Fare')
sns.distplot(np.log(nonsur['Fare'].dropna().values+1), kde=False, color='r')
 


# * 針對年齡來看年齡在10歲以下的存活率明顯較高，似乎沒有離群值。
# * 針對性別來看，很明顯的女性擁有較高的存活率。
# * 針對船艙等級來看，等級越高，則生存率也越高。
# * 針對上船地點來看，來自C的人的生存率也明顯較高
# * 針對船票價格來看，付出越高的價格的人存活率也明顯較高

# In[ ]:


#針對age變量進行分析
g = sns.FacetGrid(train_df,col='Survived')
g = g.map(sns.distplot,'Age')


# In[ ]:


g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 0) & (train_df['Survived'].notnull())] , color='blue', shade=True)
g = sns.kdeplot(train_df['Age'][(train_df['Survived'] == 1) & (train_df['Survived'].notnull())], color='red', shade=True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# *  可以觀察到在約10歲以下的小朋友的存活率明顯較高的現象。

# In[ ]:


g = sns.factorplot(y='Age',x='Sex', data=dataset,kind='box')
g = sns.factorplot(y='Age',x='Sex',hue='Pclass',data=dataset,kind='box')
#觀察age變量是否會受其他變量的影響，結果證明有此影響(Pclass)


# * 為了填補Age的遺失值，決定利用人們對他的稱呼，來決定其恰當的年齡。

# In[ ]:


#創建新特徵--頭銜
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)


# In[ ]:


dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


dataset.groupby('Title')['Age'].describe()


# In[ ]:


dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Mrs":2, "Mr":3, "Rare":4})
dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==0), 'Age']=6
dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==1), 'Age']=22
dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==2), 'Age']=37
dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==3), 'Age']=33
dataset.loc[(dataset.Age.isnull()) & (dataset['Title']==4), 'Age']=46


# In[ ]:


#檢查Age的遺失值是否填補完畢
dataset.isnull().sum()


# In[ ]:


#檢查Embarkde遺失值的情況
print(dataset[dataset['Embarked'].isnull()])


# In[ ]:


#針對Embarked變量作分析
plt.figure(figsize=(20,15))
plt.subplot(221)
sns.countplot('Embarked', data=dataset)
plt.subplot(222)
sns.countplot('Embarked', hue='Pclass', data=dataset)
plt.subplot(223)
sns.countplot('Embarked', hue='Sex',data=dataset)
plt.subplot(224)
sns.countplot('Embarked', hue='Survived',data=dataset)


# * 關於Embarked總體分析，來自S的人佔了絕大多數，但必須考慮到遺失值的那二個對象的屬性為，女性、Pclass為1，且都生存，考慮到上述的情況下，將Embarked 指定為 S的決定應較為可行。

# In[ ]:


#針對Embarked填補遺失值
dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


print(dataset[dataset['Fare'].isnull()])


# In[ ]:


#針對Fare填補遺失值
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'][dataset['Pclass']==3].mean())


# In[ ]:


dataset.info()


# 
# * 特徵工程

# In[ ]:


#創建新特徵 -- Age_cat
#將年齡從continous values 改成 categorical values
dataset['Age_cat']=0
dataset.loc[dataset['Age']<=10,'Age_cat']=0
dataset.loc[(dataset['Age']>10)&(dataset['Age']<=35),'Age_cat']=1
dataset.loc[(dataset['Age']>35)&(dataset['Age']<=50),'Age_cat']=2
dataset.loc[(dataset['Age']>50)&(dataset['Age']<=65),'Age_cat']=3
dataset.loc[dataset['Age']>64,'Age_cat']=4


# In[ ]:


sns.factorplot(x='Age_cat', y='Survived', data=dataset)


# In[ ]:


#創建新特徵---家族數目
dataset["Familysize"] = dataset["SibSp"] + dataset["Parch"] 


# In[ ]:


g = sns.factorplot(x="Familysize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")


# In[ ]:


dataset['Familysize_cat'] = pd.cut(dataset['Familysize'], 4)
dataset.groupby(dataset['Familysize_cat'])['Survived'].describe()


# In[ ]:


#將家族數目的值改為類別
dataset['Familysize_cat'] = 0
dataset.loc[(dataset['Familysize'] > 0.1) & (dataset['Familysize'] <= 2.5), 'Familysize_cat']  = 0
dataset.loc[(dataset['Familysize'] > 2.5) & (dataset['Familysize'] <= 5.0), 'Familysize_cat'] = 1
dataset.loc[(dataset['Familysize'] > 5.0) & (dataset['Familysize'] <= 7.5), 'Familysize_cat']   = 2
dataset.loc[ dataset['Familysize'] > 7.5, 'Familysize_cat']  = 3
dataset['Familysize_cat'] = dataset['Familysize_cat'].astype(int)


# In[ ]:


g = sns.factorplot(x='Familysize_cat',y='Survived',data=dataset,kind='bar')
g.set_ylabels('Survival Probability')


# In[ ]:


dataset['IsAlone'] = np.where(dataset['Familysize'] ==0 ,1,0)


# In[ ]:


sns.factorplot(x='IsAlone', y='Survived', data=dataset)


# In[ ]:


dataset['Fare_R'] = pd.qcut(dataset['Fare'],4)
dataset.groupby(dataset['Fare_R'])['Survived'].describe()


# In[ ]:


dataset['Fare_cat'] = 0
dataset.loc[(dataset['Fare'] <= 7.896), 'Fare_cat']  = 0
dataset.loc[(dataset['Fare'] > 7.896) & (dataset['Fare'] <= 14.454), 'Fare_cat'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.275), 'Fare_cat']   = 2
dataset.loc[ dataset['Fare'] > 31.275, 'Fare_cat']  = 3
dataset['Fare_cat'] = dataset['Fare_cat'].astype(int)


# In[ ]:


sns.factorplot(x='Fare_cat',y='Survived',data=dataset)


# 針對Ticket變量，可能假設為若乘客為團體的話，船票的編號應該會是一致的，接下來考察團體出遊

# In[ ]:


dataset['Shared_ticket'] = np.where(dataset.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
sns.barplot('Shared_ticket', 'Survived', data=dataset)
#可以發現團體出遊的話，存活率的確比較高
print(dataset.groupby('Ticket'))


# * 關於Cabin變量，由於遺失值很多，希望能從非遺失值當中觀察出一些可供我們使用的變量

# In[ ]:


dataset['Cabin_known'] = dataset['Cabin'].isnull()==False
g = sns.barplot('Cabin_known', 'Survived', data=dataset)


# In[ ]:


g = sns.factorplot('Cabin_known', 'Survived',hue='Sex',data=dataset)
g = sns.factorplot('Cabin_known', 'Survived',hue='Pclass',data=dataset)


# In[ ]:


dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})


# In[ ]:


dataset = pd.get_dummies(dataset, columns = ['Title'])
dataset = pd.get_dummies(dataset, columns = ['Embarked'],prefix='Em')


# In[ ]:


dataset.info()


# In[ ]:


#拋棄不需要的特徵
drop_element =['Fare_R','Age','Fare','Cabin', 'PassengerId', 'Familysize', 'Parch', 'SibSp','Ticket','Name']
dataset = dataset.drop(drop_element, axis=1)


# In[ ]:


dataset.info()


# In[ ]:


#分解數據集
train_len = len(train_df)
train = dataset[:train_len]
test = dataset[train_len:]
test.drop('Survived', axis=1, inplace=True)


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


#建立訓練數據集
train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop("Survived",axis = 1)


# In[ ]:


#利用各種演算法建模
kfold =  StratifiedKFold(n_splits=10)


# In[ ]:


random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[ ]:


#LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()

LDA.fit(X_train,Y_train)
#score
LDA_score = LDA.score(X_train,Y_train)
print(LDA_score)


# In[ ]:


#LogisticRegression
LR = LogisticRegression()
lr_param_grid = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}

gsLR = GridSearchCV(LR, param_grid=lr_param_grid,cv=kfold, scoring='accuracy' ,verbose=1)
gsLR.fit(X_train,Y_train)
LR_best = gsLR.best_estimator_
#score
gsLR_best_score = LR_best.score(X_train,Y_train)
print(gsLR_best_score)


# In[ ]:


#模型超參數的調整---GradientBoosting

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[ ]:


#模型超參數的調整-- SVC classifier
scaler=StandardScaler()
X_scaled=scaler.fit(X_train).transform(X_train)
test_X_scaled=scaler.fit(test).transform(test)

SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

gsSVMC.fit(X_scaled,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# In[ ]:


#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[ ]:


RFC = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[ ]:


# Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_


# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-',color='g', label='Cross_validation score')
    
    plt.legend(loc='best')
    return plt

g = plot_learning_curve(gsGBC.best_estimator_,'GradientBoosting learning curve',X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsLR.best_estimator_,'Logistic Regression curves',X_train,Y_train,cv=kfold)
g = plot_learning_curve(LDA,'LinearDiscriminantAnalysis curves',X_train,Y_train,cv=kfold)


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(221)
indices = np.argsort(GBC_best.feature_importances_)[::-1]
g = sns.barplot(y=X_train.columns, x=GBC_best.feature_importances_[indices])
g.set_xlabel('Relative importance', fontsize=14)
g.set_ylabel('Features', fontsize=14)
g.tick_params(labelsize=9)
g.set_title('GBC feature importance')
plt.subplot(222)
indices = np.argsort(ada_best.feature_importances_)[::-1]
g = sns.barplot(y=X_train.columns, x=ada_best.feature_importances_[indices])
g.set_xlabel('Relative importance', fontsize=14)
g.set_ylabel('Features', fontsize=14)
g.tick_params(labelsize=9)
g.set_title('ADA feature importance')
plt.subplot(223)
indices = np.argsort(ExtC_best.feature_importances_)[::-1]
g = sns.barplot(y=X_train.columns, x=ExtC_best.feature_importances_[indices])
g.set_xlabel('Relative importance', fontsize=14)
g.set_ylabel('Features', fontsize=14)
g.tick_params(labelsize=9)
g.set_title('EXT feature importance')
plt.subplot(224)
indices = np.argsort(RFC_best.feature_importances_)[::-1]
g = sns.barplot(y=X_train.columns, x=RFC_best.feature_importances_[indices])
g.set_xlabel('Relative importance', fontsize=14)
g.set_ylabel('Features', fontsize=14)
g.tick_params(labelsize=9)
g.set_title('RFC feature importance')


# In[ ]:


test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")
test_Survived_LR = pd.Series(LR_best.predict(test), name='LR')
test_Survived_LDA = pd.Series(LDA.predict(test), name='LDA')

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC,test_Survived_LR,test_Survived_LDA],axis=1)
g= sns.heatmap(ensemble_results.corr(),annot=True)


# In[ ]:


#Ensemble
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('exc', ExtC_best),
('svc', SVMC_best),('ada',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# In[ ]:


votingC_score = votingC.score(X_train, Y_train)
print(votingC_score)


# In[ ]:


#製作提交文檔
predictions = votingC.predict(test)
Submission = pd.DataFrame({ 'PassengerId': test_id ,
                            'Survived': predictions })
Submission.to_csv("Submission1002ver2.csv", index=False)



# In[ ]:




