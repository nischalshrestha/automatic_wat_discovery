#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# * __Team Name:__ UMN-STAT-5302
# * __Team Members :__ Anick Saha, Gerrit Vreeman, Karthik Unnikrishnan, Manish Rai, Sai Kumar Kayala. 

# ## Abstract
# 
# ### What is it that we're trying to solve?
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, _some groups of people were more likely to survive than others, such as women, children, and the upper-class._
# 
# In this task, we complete the analysis of what sorts of people were likely to survive by applying the tools of machine learning to predict which passengers survived the tragedy.
# 
# ### How did we approach the problem? 
# 
# There were the main stages in how we solved this problem:
# 
# 1. Data Summarization. 
# 2. Data Preprocessing.
# 3. Feature Engineering.
# 4. Try out algorithms that we felt would be a good fit to this problem. 
# 5. Compare and try to improve accuracies by repeating Step 3 and Step 4 to find the model with the best accuracy.
# 
# Based on our analysis, the best accuracy was achieved by using the Random Forest Classifier.
# 
# ### Data Summary and EDA:
# 
# ##### Intro:
# 
# We performed extensive data analysis to get a better idea about the data. We looked at the summaries of the data at various levels like - Class level, Sex level, Class-Sex level, and the NamePrefix (which is a derived variable from the Name) level. 
# 
# Some of the observations that we saw are:
# 
# 1. Survival in the female is higher than male.
# 2. Survival rate for class1 is higher than survival rate in class2 which in turn is higher than the survival rate in class3.
# 3. We also notice that in general the age of people in class1 is greater than the age of people in class2 which is greater than the age of people in class3.
# 
# We also looked at interaction of variables for survivals and non-survivals and made some interesting plots in R.
# 
# [![text](https://raw.githubusercontent.com/anicksaha/blob/master/stat5302/Titanic_Charts.png)](https://github.com/anicksaha/blob/blob/master/stat5302/Titanic_Charts.png)
# 
# ##### What preprocessing did we do?
# 
# 1. We checked for NULL values in each column and noticed that there are a number of null values Cabin, Age and Fare.
# 2. In order to imute the null vaules in Fare, we use linear interpolation since the number of null vaules is very small.
# 3. For imputing age, we took the mean of the age column. 
# 4. Due to the large number of null values in the cabin column, we decide to drop it.
# 5. We also drop the ticket column because it does not seem to have any relation with survival
# 
# ##### What sort of Feaure Engineering were done?
# 1. We created the feature FamilySize to denote the number of people travelling together by adding SibSp and Parch.
# 2. The feature IsAlone was created to denote if a passenger is travelling alone or not
# 3. The feature Title is extracted from the Name attribute and the rare titles were mapped to more general ones. After this, the name column is dropped.
# 4. For the remaining feateures we performed label encoding for the categoriacal variables
# 
# 

# ### The various models that we tried:
# 
# We tried the following models using the same proprocessed data:
# 1. Logistic regression
# 2. SVM
# 3. AdaBoost Classifier
# 4. XGBoost Classifier
# 5. ExtraTrees Classifier
# 6. GradientBoosting Classifier
# 7. GaussianNB Classifier
# 8. Random Forest Classifer
# 
# ##### How did each model fair?
# * For each of the models, we calculated the accuracy in terms of the training dataset. 
# * Amongst all the models, RandomForest classifier and ExtraTrees classifier gave the greterst accuracy valus. 
# * Of these two, the Random Forest Classifier performed better on the Test dataset (Kaggle submission score) and hence generalized better!

# ### Random Forest and ExtraTrees
# 
# One common issue with all machine learning algorithms is Overfitting. For Decision Trees, it means growing too large tree (with strong bias, small variation) so it loses its  ability  to  generalize  the  data  and  to  predict  the  output.  In  order  to  deal  with overfitting,  we  can grow  several  decision  trees  and  take  the  average  of  their predictions.
# 
# * The SciKit Learn library provides to such algorithms - Random Forest and ExtraTrees.
# 
# * In Random Forest, we grow N decision trees based on randomly selected subset of the data and randomly selected M fields. 
# * In ExtraTrees, in addition to randomness of subsets of the data and of field, splits of nodes are chosen randomly.
# 
# References: 
# * [ExtraTreesClassifier | scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
# * [RandomForestClassifier | scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# ### The model that won out heart - Random Forest!
# 
# Random Forest is a supervised learning algorithm. The forest it builds, is an ensemble of Decision Trees, most of the time trained with the “bagging” method. 
# 
# To say it in simple words: Random forest builds multiple decision trees and merges the rules together to get a more accurate Decision Tree. 
# 
# One of the big problems in machine learning is overfitting, but most of the time this won’t happen that easy to a random forest classifier. That’s because if there are enough trees in the forest, the classifier won’t overfit the model. The complex models we used like SVM don't work well with this data because of the low number of training samples available while a random forest being a very simple yet powerful model did a better job and turned out to be a better fit for the low number of traning samples.
# 
# 
# ### Conclusion:
# 
# * As  a  result  of  our  work,  we  gained  valuable experience  of  building  prediction  systems and achieved our best score
# on Kaggle: 0.81818
# * In Kaggle leaderboard, it corresponds to 528(Top 6%) out of 10,000 participants.
# 
# [![text](https://raw.githubusercontent.com/anicksaha/blob/master/stat5302/Titanic_rank.png)](https://raw.githubusercontent.com/anicksaha/blob/master/stat5302/Titanic_rank.png)

# ### Future work: 
# 
# * We can try different __imputation methods__ like - Groupwise means, medians, etc.
# * We can try __ensembling different models__ and make a more robust model that generalizes better.
# *  We can try to think of a way to use 'ticket_number' and other dropped columns. 
# 

# ### References (Documentations, Libraries and Kernels)
# 
# * [scikit-learn](https://scikit-learn.org/stable/index.html)
# * [Feature extraction for familysize, age]( https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# * [Title feature creation]( https://www.kaggle.com/kabure/titanic-eda-keras-nn-pipelines)

# # Code:

# In[ ]:


### Import necessary Libraries and Data
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import model_selection
from sklearn import ensemble,linear_model,tree,svm,naive_bayes

sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data_train=pd.read_csv("../input/train.csv")
data_test=pd.read_csv("../input/test.csv")


# In[ ]:


data_train.info()
print('-'*25)
data_test.info()


# Checking for null values

# In[ ]:


print(data_train.isnull().sum())
print('*'*25)
print(data_test.isnull().sum())


# * We have null values in Age,Cabin and Embarked for training data and Age Fare and Cabin have null values in test data.
# 

# In[ ]:


data_cpy=data_train.copy(deep=True)
data=[data_cpy,data_test]
min_num=5


# #### Feature Engineering | Good Refererences:
# 
# * Continuous variable bins - qcut vs cut -  [What is the difference between pandas-qcut and pandas-cut?](https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut)
# * [pandas : pandas.qcut](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html)

# In[ ]:


### START: Feature Engineering for train and test/validation dataset
for dataset in data:
    #Discrete variables
    dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Embarked']=dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
    dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].interpolate(method='linear'))
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1
    
    
    # Fare Bins/Buckets using qcut or frequency 
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    
### END: Feature Engineering

data_cpy['Title']=data_cpy['Name'].str.split(', ',expand=True)[1].str.split('.',expand=True)[0]
data_test['Title']=data_test['Name'].str.split(', ',expand=True)[1].str.split('.',expand=True)[0]
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty",
    "Dona" : "Mrs"
}
data_cpy['Title']=data_cpy.Title.map(Title_Dictionary)
data_test['Title']=data_test.Title.map(Title_Dictionary)


# In[ ]:


drop_col=['Cabin','Name','Ticket']
data_cpy.drop(columns=drop_col,axis=1,inplace=True)
data_test.drop(columns=drop_col,axis=1,inplace=True)


# In[ ]:


data_cpy.head()


# In[ ]:


# Plotting the correlation heatmap.
plt.figure(figsize=(8,5))
sns.heatmap(data_cpy.corr(),annot=True)
plt.show()


# * From the above heatplot, we can observe that there is a high correlation b/w survived and family size, fare and p-class, etc. (darker shades) 
# 
# 

# In[ ]:


# We will use seaborn graphics for multi-variable comparison: 
# https://seaborn.pydata.org/api.html

#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data_cpy, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data_cpy, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data_cpy, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data_cpy, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data_cpy, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data_cpy, ax = saxis[1,2])


# ### Converting
# 
# * Below are the steps taken for label encoding. 
# * We are changing variables: age and fare as factors instead of using them as continuous variables in our model, since the models that we used works better with discrete values than continuous variables.

# In[ ]:


lbl=LabelEncoder()
for d in data:
    d['Sex']=lbl.fit_transform(d['Sex'])
    d['Embarked']=lbl.fit_transform(d['Embarked'])
    d['Title']=lbl.fit_transform(d['Title'])
    d['Age']=pd.qcut(d['Age'].astype(int),4)
    d['Fare']=pd.qcut(d['Fare'].astype(int),4)
    d['Fare']=lbl.fit_transform(d['Fare'])
    d['Age']=lbl.fit_transform(d['Age'])
data_cpy=pd.get_dummies(data_cpy,columns=['Sex','Embarked','Title','Fare','Age'])
data_test=pd.get_dummies(data_test,columns=['Sex','Embarked','Title','Fare','Age'])



# In[ ]:


data_cpy.head()


# In[ ]:


train_col=['Pclass','Sex_0','Sex_1','Age_0','Age_1','Age_2','Age_3','FamilySize','IsAlone','Fare_0','Fare_1','Fare_2','Fare_3','Embarked_0','Embarked_1','Embarked_2','Title_0','Title_1','Title_2','Title_3','Title_4']
target=['Survived']
X_train=data_cpy[train_col].copy(deep=True)
Y_train=data_cpy[target]
print(X_train.dtypes)
X_train.head()
data_test1=data_test[train_col].copy(deep=True)


# ### Different models tried:

# ### ExtraTrees Classifier: 
# 
# An extra-trees classifier class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
# 
# Reference: [scikit-learn | ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
# 

# In[ ]:


parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
clf=ensemble.ExtraTreesClassifier()
clf.fit(X_train,Y_train)
train_score=clf.score(X_train,Y_train)
train_score


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.05)
ada.fit(X_train,Y_train)
train_score=ada.score(X_train,Y_train)
train_score


# In[ ]:


import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
xgboost.fit(X_train,Y_train)
train_score=xgboost.score(X_train,Y_train)
train_score


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
grad.fit(X_train,Y_train)
train_score=grad.score(X_train,Y_train)
train_score


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(X_train,Y_train)
train_score=model.score(X_train,Y_train)
train_score


# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft')
ensemble_lin_rbf.fit(X_train,Y_train)
train_score=ensemble_lin_rbf.score(X_train,Y_train)
train_score


# In[ ]:


logis=linear_model.LogisticRegressionCV()
logis.fit(X_train,Y_train)
train_score=logis.score(X_train,Y_train)
train_score


# In[ ]:


from sklearn.model_selection import StratifiedKFold
model1 = RandomForestClassifier()
model1.fit(X_train,Y_train)
train_score=model1.score(X_train,Y_train)
train_score


# In[ ]:


y_test=model1.predict(data_test1)


# In[ ]:


y_test


# ### Plotting importance of features

# In[ ]:


important_features=pd.Series(model1.feature_importances_, index=train_col)
important_features.plot(kind='barh',figsize=(12,15))


# * The above plot displays the importance of each features according to the random tree classifier, with Pclass, Sex, FamilySize being the most important

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic_submission.csv', index=False)

