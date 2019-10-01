#!/usr/bin/env python
# coding: utf-8

# # 0. ライブラリのインポート

# In[ ]:


import pandas as pd # collection of functions for data processing and analysis modeled after R dataframes with SQL like features
import numpy as np  # foundational package for scientific computing
import re           # Regular expression operations
import matplotlib.pyplot as plt # Collection of functions for scientific and publication-ready visualization
get_ipython().magic(u'matplotlib inline')
import plotly.offline as py     # Open source library for composing, editing, and sharing interactive data visualization 
from matplotlib import pyplot
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from collections import Counter

# Machine learning libraries
#import xgboost as xgb  # Implementation of gradient boosted decision trees designed for speed and performance that is dominative competitive machine learning
import seaborn as sns  # Visualization library based on matplotlib, provides interface for drawing attractive statistical graphics

import sklearn         # Collection of machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# # 1. データ簡易チェック

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


PassengerID = test.PassengerId


# In[ ]:


train.head(3)


# In[ ]:


train.info()
print('_'*40)
test.info()


# ## ⇒ データが一部欠けている

# In[ ]:


test.describe()


# # 2. データ分析

# ## 2.1 性別と生存率の関係

# In[ ]:


sns.barplot(x="Sex", y="Survived",data=train);


# In[ ]:


full_data=[test,train]
for dataset in full_data:
    dataset["Sex"]=dataset["Sex"].map({"female":0,"male":1}).astype(int)


# ## ⇒ 性別は重要な因子。モデルに入れよう。

# ## 2.2 料金と生存率の関係

# In[ ]:


sns.distplot(train["Fare"])


# In[ ]:


for dataset in full_data:
    dataset["Fare"] = dataset["Fare"].apply(lambda x: np.log(x) if x>0 else 0)


# In[ ]:


sns.distplot(train["Fare"][train["Survived"]==0],bins=10,label="not survived",color="red")
sns.distplot(train["Fare"][train["Survived"]==1],bins=10,label="survived",color="blue")
plt.legend()


# In[ ]:


sns.barplot(x="Pclass",y="Survived",data=train)


# In[ ]:


train["Pclass"].isnull().sum()
test["Pclass"].isnull().sum()


# ## ⇒　客室等級は重要な因子。モデルに入れよう。

# ## 2.3 年齢と生存率の関係

# In[ ]:


sns.distplot(train[train["Age"].notnull()&(train["Survived"]==0)]["Age"],label="not survived",color="blue")
sns.distplot(train["Age"][train["Age"].notnull()&(train["Survived"]==1)],label="survived",color="red")
plt.legend()


# In[ ]:


for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
# Qcut is a quantile based discretization function to autimatically create categories (not used here)
# dataset['Age'] = pd.qcut(dataset['Age'], 6, labels=False)
# Using categories as defined above
    dataset.loc[ dataset['Age'] <= 14, 'Age'] 						          = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 30), 'Age']        = 5
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age']        = 1
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age']        = 3
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age']        = 2
    dataset.loc[ dataset['Age'] > 60, 'Age'] 							      = 4
train['Age'].value_counts()


# ## ⇒ 年齢を6つに分けて因子として考慮する。
# ## ⇒ 性別、客室等級、年齢を分析し、整理完了。

# # 3. 訓練データの作成

# ## 3.1 余計な因子を削除

# In[ ]:


test.head()


# In[ ]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Cabin','Fare', 'Embarked']

train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)


# ## 3.2 データ整形

# In[ ]:


# X_train (all features for training purpose but excluding Survived),
# Y_train (survival result of X-Train) and test are our 3 main datasets for the next sections
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]

X_test = test.copy()
#std_scaler = StandardScaler()
#X_train = std_scaler.fit_transform(X_train)
#X_test = std_scaler.transform(X_test)


# # 4. モデル作成

# ## 4.1 Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# ## 4.2. Support Vector Machines (supervised)

# In[ ]:


svc=SVC()
svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ## 4.3. k-Nearest Neighbors algorithm (k-NN)

# In[ ]:


knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=10, p=2, 
                           weights='uniform')
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Preparing data for Submission 1
test_Survived = pd.Series(knn_predictions, name="Survived")
Submission1 = pd.concat([PassengerID,test_Survived],axis=1)
acc_knn


# ## 4.4. Naive Bayes classifier

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ## 4.5. Perceptron

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# ## 4.6. Linear SVC

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# ## 4.7. Stochastic Gradient Descent (sgd)

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# ## 4.8. Decision tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
#Y_pred = decision_tree.predict(test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ## 4.9. Random Forests

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest_predictions = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


objects = ('Logistic Regression', 'SVC', 'KNN', 'Gaussian', 'Perceptron', 'linear SVC', 'SGD', 'Decision Tree', 'Random Forest')
x_pos = np.arange(len(objects))
accuracies1 = [acc_log, acc_svc, acc_knn, acc_gaussian, acc_perceptron, acc_linear_svc, acc_sgd, acc_decision_tree, acc_random_forest]
    
plt.bar(x_pos, accuracies1, align='center', alpha=0.5, color='r')
plt.xticks(x_pos, objects, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Classifier Outcome')
plt.show()


# ## 4.10Cross Validation
# ##         （未知のデータに対するモデルの性能を評価）

# In[ ]:


# Cross validate model with Kfold stratified cross validation
from sklearn.model_selection import StratifiedKFold

#テストデータを10個に分ける
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 
random_state = 2

classifiers = []
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(SVC(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(GaussianNB())
classifiers.append(Perceptron(random_state=random_state))
classifiers.append(LinearSVC(random_state=random_state))
classifiers.append(SGDClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state = random_state))
classifiers.append(RandomForestClassifier(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":['Logistic Regression',  'KNN', 'Gaussian',
    'Perceptron', 'linear SVC', 'SGD', 'Decision Tree','SVMC', 'Random Forest']})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# ##  ⇒　ランダムフォレストを使おう！

# ## 4.11 GridSearchCV
# ##  （ランダムフォレストのハイパーパラメータをチューニング）

# In[ ]:


# Random Forest
rf_param_grid = {"max_depth": [None],
              #"max_features": [1, 3, 7],
              #"min_samples_split": [2, 3, 7],
              "min_samples_leaf": [1, 3, 7],
              "bootstrap": [False],
              "n_estimators" :[300,600],
              "criterion": ["gini"]}
gsrandom_forest = GridSearchCV(random_forest,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsrandom_forest.fit(X_train,Y_train)
# Best score
random_forest_best = gsrandom_forest.best_estimator_
gsrandom_forest.best_score_


# # 5. 提出ファイルの作成

# In[ ]:


test_predict=random_forest_best.predict(test)
submission = pd.concat([PassengerID,pd.DataFrame({"Survived":test_predict})],axis=1)
submission.to_csv("predict_result.csv",index=False)

