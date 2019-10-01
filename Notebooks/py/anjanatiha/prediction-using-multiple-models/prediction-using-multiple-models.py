#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.


# In[ ]:


#feature_cols = ['Pclass','Sex','Age','SibSp', 'Parch','Fare','Cabin','Embarked']
feature_cols = ['Pclass','Sex','Age','SibSp', 'Parch','Fare']
#feature_cols = ['Pclass','Sex','Age','Fare']
#feature_cols = ['Pclass','Sex','Age']
#feature_cols = ['Pclass','Sex']

target_col = 'Survived'

features = train[feature_cols]
target = train[target_col]


features.describe()
target.describe()

# set(features['Fare'])


# In[ ]:


from sklearn import preprocessing
features = features.fillna(0)
features['Age']=preprocessing.normalize([features['Age']], norm='l2').flatten()
features['Fare'] = preprocessing.normalize([features['Fare']], norm='l2').flatten()


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(features['Sex'])
features['Sex']=le.transform(features['Sex'])


# In[ ]:


X = features.values
Y = target


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from sklearn import metrics

    
def model_evaluation(X, Y, splitter, model, report, details):
    accuracy = 0
    f1 = 0
    precision = 0
    recall = 0
    i=0
    if report:
        print("*"*50, " START ", "*"*50)
        print("Spliter Description:")
        print(splitter)
        print("-"*100, "\n")
        print("Model Description:")
        print(model)
        print("-"*100,"\n")
    
    for train_index, test_index in splitter.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # model fitting
        model.fit(X_train, y_train)
        
        # prediction
        predict = model.predict(X_test)

        # evaluation scores
        accuracy_temp = metrics.accuracy_score(y_test, predict)
        precision_temp = metrics.precision_score(y_test, predict, average="micro")
        recall_temp = metrics.recall_score(y_test, predict, average="micro")
        f1_temp = metrics.f1_score(y_test, predict, average="micro")
        hamming_loss = metrics.hamming_loss(y_test, predict)
        
#         precision, recall, thresholds = metrics.precision_recall_curve(y_test, predict)
#         average_precision_score = metrics.average_precision_score(y_test, predict, average="micro")
#         fbeta_score = metrics.fbeta_score(y_test, predict)
#         roc_auc_score = metrics.roc_auc_score(y_test, predict, average="micro")
        
    
        accuracy += accuracy_temp
        precision+=precision_temp
        recall+=recall_temp
        f1+=f1_temp
        
        if details:
            print("*"*25,  " ITERATION - ", i+1, "*"*25)
            #print("TRAIN:", train_index, "TEST:", test_index)
            print("Accuracy Score: ", accuracy_temp)
            print("Precision Score: ", precision_temp)
            print("Recall Score: ", recall_temp)
            print("F1 Score: ", f1_temp)
            print("Hamming Loss: ", hamming_loss)
            print("-"*35)
            print(metrics.classification_report(y_test, predict))
            print("-"*35)
            print("confusion Matrix:\n\n", metrics.confusion_matrix(y_test, predict))
            print("-"*35)
            print("\n")
        
        i+=1
    split_num = splitter.get_n_splits()
    
    accuracy = accuracy/split_num
    precision = precision/split_num
    recall = recall/split_num
    f1 = f1/split_num
    
    if report:
        print("*"*50, " Average For", i+1, " Folds", "*"*50)
        print("\n")
        print("Average Accuracy Score: ", accuracy)
        print("Average pPrecision Score: ", precision)
        print("Average Recall Score: ", recall)
        print("Average F1 Score:", f1)
        print("\n")
        print("*"*50, " END ", "*"*50)
    
    
    
    return accuracy, precision, recall, f1


# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, RandomForestClassifier, VotingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB 
from sklearn.neighbors import KDTree, KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.svm import LinearSVC, NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)


classifiers = {
    "AdaBoostClassifier": AdaBoostClassifier(),
    "BernoulliNB": BernoulliNB(),
#     "BernoulliRBM": BernoulliRBM(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "GaussianMixture": GaussianMixture(),
    "GaussianNB": GaussianNB(),
    "GaussianProcessClassifier": GaussianProcessClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
#     "KDTree": KDTree(),
    "KNeighborsClassifier": KNeighborsClassifier(3),
    "LogisticRegression": LogisticRegression(),
    "LinearSVC": LinearSVC(),
    "MLPClassifier": MLPClassifier(),
    "MultinomialNB": MultinomialNB(),
#     "NearestNeighbors": NearestNeighbors(),
    "NuSVC": NuSVC(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC Linear": SVC(kernel="linear", C=0.025),
    "SVC": SVC(),
    "SVC Gamma": SVC(gamma=2, C=1)
#     VotingClassifier: VotingClassifier(),
}
    
    
splitter = sss
report = None
details = 1


evaluation = {}

for name in classifiers:
    evaluation_temp = []
    accuracy, precision, recall, f1 = model_evaluation(X, Y, splitter, classifiers[name], report=None, details=None)
    evaluation_temp.append(accuracy)
    evaluation_temp.append(precision)
    evaluation_temp.append(recall)
    evaluation_temp.append(f1)
    evaluation[name] = evaluation_temp
    

rows_list = []
for name in evaluation:
    rows_list.append([name]+evaluation[name])
                           
evaluation_pd = pd.DataFrame(rows_list, columns=['model', 'accuracy', 'precision', 'recall', 'f1']) 
evaluation_pd


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


figure(num=None, figsize=(14, 6), dpi=250)

labels= ['accuracy', 'precision', 'recall', 'f1']
ax = plt.subplot(111)

for n in range(0,4):
    plt.plot([name for name in evaluation],[evaluation[name][n] for name in evaluation], label = labels[n])

leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
plt.xticks(rotation=45)
# leg.get_frame().set_alpha(0.5)
plt.legend()
ax.tick_params(labelsize='large', width=5)
ax.grid(True, linestyle='-.')

plt.tight_layout()
plt.xlabel('x label')
plt.ylabel('y label')

plt.title("TITLE")
plt.show()




