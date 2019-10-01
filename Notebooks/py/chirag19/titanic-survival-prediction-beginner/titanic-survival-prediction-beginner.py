#!/usr/bin/env python
# coding: utf-8

# This kernel is based on the learnings of others kernels and offcourse my own intuitions and methods too. This notebook will be helpful for beginners.
# 
# This is my first kernel so any kind of suggestion or appreaciation is heartly welcomed.

# In[ ]:


import os
print(os.listdir("../input/"))


# ### Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[ ]:


train = pd.read_csv("../input/train.csv")
train['label'] = 'train'
test = pd.read_csv("../input/test.csv")
test['label'] = 'test'
test_passengerId = test.PassengerId  #Save test passengerId. It will be required at the end
df = train.append(test)
df.sample(2)


# ## EDA

# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# So we have to handle missing values of age, cabin, embarked and fare. Survived has missing values of test set.

# In[ ]:


df.describe(include = 'all')


# ### Handling Missing Values

# #### Embarked

# In[ ]:


#Fill missing value
df['Embarked'].fillna('S', inplace = True)    #top value with freq 914


# #### Fare

# In[ ]:


df[df.Fare.isnull()]


# In[ ]:


df.corr().Fare


# Looks like Pclass can help to fill the missing value.

# In[ ]:


print(df[df.Pclass == 1].Fare.quantile([0.25, 0.50, 0.75]))
print(df[df.Pclass == 2].Fare.quantile([0.25, 0.50, 0.75]))
print(df[df.Pclass == 3].Fare.quantile([0.25, 0.50, 0.75]))


# Yup! Values differ totally according to Pclass. Let's look it through visualization.

# In[ ]:


sns.factorplot(x = 'Pclass', y = 'Fare', data = df)


# In[ ]:


df['Fare'].fillna(df[df.Pclass == 3].Fare.median(), inplace = True)   #Fare is dependent on Pclass


# #### Age

# In[ ]:


print("Age column has", df.Age.isnull().sum(), "missing values out of", len(df), ". Missing value percentage =", df.Age.isnull().sum()/len(df)*100)


# Its high! Thus any value derived statistically (mean or median) based on only Age column can mislead the dataset for the classifier. We will fill them based on the relations with other variables.

# In[ ]:


df.corr().Age


# In[ ]:


df.pivot_table(values = 'Age', index = 'Pclass').Age.plot.bar()


# In[ ]:


df.pivot_table(values = 'Age', index = ['Pclass', 'SibSp'], aggfunc = 'median').Age.plot.bar()


# A basic trend can be found from the graph. Thus, we are on right path!

# In[ ]:


df.pivot_table(values = 'Age', index = ['Pclass', 'SibSp', 'Parch'], aggfunc = 'median')


# We will fill missing values based on Pclass and SibSp.

# In[ ]:


df.Age.isnull().sum()


# In[ ]:


age_null = df.Age.isnull()
group_med_age = df.pivot_table(values = 'Age', index = ['Pclass', 'SibSp'], aggfunc = 'median')
df.loc[age_null, 'Age'] = df.loc[age_null, ['Pclass', 'SibSp']].apply(lambda x: group_med_age.loc[(group_med_age.index.get_level_values('Pclass') == x.Pclass) & (group_med_age.index.get_level_values('SibSp') == x.SibSp)].Age.values[0], axis = 1)


# In[ ]:


df.Age.isnull().sum()


# #### Cabin

# In[ ]:


print("Cabin has", df.Cabin.isnull().sum(), "missing values out of", len(df))


# So instead of filling those values, form their cluster. We will assume that those people don't have cabin.

# In[ ]:


df['Cabin'] = df.Cabin.str[0]
df.Cabin.unique()


# In[ ]:


df.Cabin.fillna('O', inplace = True)


# In[ ]:


df.isnull().sum()


# So, we are done with data cleaning part. Missing Survived are from test set.

# In[ ]:


df.sample(2)


# ### Sex

# In[ ]:


sns.factorplot(data = df, x = 'Sex', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Sex').Survived.plot.bar()
plt.ylabel('Survival Probability')


# Females tend to survive more than males.

# #### Age

# In[ ]:


q = sns.kdeplot(df.Age[df.Survived == 1], shade = True, color = 'red')
q = sns.kdeplot(df.Age[df.Survived == 0], shade = True, color = 'blue')
q.set_xlabel("Age")
q.set_ylabel("Frequency")
q = q.legend(['Survived', 'Not Survived'])


# In[ ]:


q = sns.FacetGrid(df, col = 'Survived')
q.map(sns.distplot, 'Age')


# #### Embarked

# In[ ]:


sns.factorplot(data = df, x = 'Embarked', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Embarked').Survived.plot.bar()
plt.ylabel('Survival Probability')


# Cherbourg port is more save as compared to others. Lets look more into this.

# In[ ]:


df.pivot_table(values = 'Survived', index = ['Sex','Embarked']).Survived.plot.bar()
plt.ylabel('Survival Probability')


# We found something interesting! Cherbourg port is very safe for females and Qweenstone and Southampton ports are very dangerous for males.

# In[ ]:


fig, ax =plt.subplots(1,2)
sns.countplot(data = df[df.Sex == 'female'], x = 'Embarked', hue = 'Survived', ax = ax[0])
sns.countplot(data = df[df.Sex == 'male'], x = 'Embarked', hue = 'Survived', ax = ax[1])
fig.show()


# Surprised! Port of embarkation is very safe for females  but more dangerous form males. Similarly other two ports also contradict for males and females.

# #### Parch

# In[ ]:


sns.factorplot(data = df, x = 'Parch', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Parch').Survived.plot.bar()
plt.ylabel('Survival Probability')


# #### Pclass

# In[ ]:


sns.factorplot(data = df, x = 'Pclass', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Pclass').Survived.plot.bar()
plt.ylabel('Survival Probability')


# In[ ]:


df.pivot_table(values = 'Survived', index = ['Sex', 'Pclass']).Survived.plot.bar()
plt.ylabel('Survival Probability')


# Qualty of tickets class assures more safety!

# #### Cabin

# In[ ]:


sns.factorplot(data = df, x = 'Cabin', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Cabin').Survived.plot.bar()
plt.ylabel('Survival Probability')


# #### Fare

# From dataset description, it was clear that Fare values are not skewed. Lets visualize it.

# In[ ]:


plt.boxplot(train.Fare, showmeans = True)
plt.title('Fare Boxplot')
plt.ylabel('Fares')


# In[ ]:


sns.distplot(df.Fare)


# Highly right skewed!

# In[ ]:


df.Fare.skew()    #Measure of skewness level


# We will take log transform. This might help classifier in preditions. Also it will help us to find correlation between variables.

# In[ ]:


df['Fare_log'] = df.Fare.map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


sns.distplot(df.Fare_log)


# In[ ]:


df.Fare_log.skew()


# ### Feature Engineering

# In[ ]:


df['Family_size'] = 1 + df.Parch + df.SibSp
df['Alone'] = np.where(df.Family_size == 1, 1, 0)


# In[ ]:


print(df.Family_size.value_counts())
print(df.Alone.value_counts())


# In[ ]:


sns.factorplot(data = df, x = 'Family_size', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Family_size').Survived.plot.bar()
plt.ylabel('Survival Probability')


# Now, family size with 2 to 4 members are more likely to survive. Thus, we will form bins for this groups.

# In[ ]:


df.loc[df['Family_size'] == 1, 'Family_size_bin'] = 0
df.loc[(df['Family_size'] >= 2) & (df['Family_size'] <= 4), 'Family_size_bin'] = 1
df.loc[df['Family_size'] >=5, 'Family_size_bin'] = 2


# In[ ]:


sns.factorplot(data = df, x = 'Alone', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Alone').Survived.plot.bar()
plt.ylabel('Survival Probability')


# People travelling alone are likely to less survive.

# In[ ]:


df['Title'] = df.Name.str.split(", ", expand = True)[1].str.split(".", expand = True)[0]
df.Title.value_counts()


# In[ ]:


minor_titles = df.Title.value_counts() <= 4
df['Title'] = df.Title.apply(lambda x: 'Others' if minor_titles.loc[x] == True else x)
df.Title.value_counts()


# In[ ]:


sns.factorplot(data = df, x = 'Title', hue = 'Survived', kind = 'count')


# In[ ]:


df.pivot_table(values = 'Survived', index = 'Title').Survived.plot.bar()
plt.ylabel('Survival Probability')


# Lets make bins for age and fare also and visualize them.

# In[ ]:


df['Fare_bin'] = pd.qcut(df.Fare, 4, labels = [0,1,2,3]).astype(int)
df['Age_bin'] = pd.cut(df.Age.astype(int), 5, labels = [0,1,2,3,4]).astype(int)


# In[ ]:


sns.factorplot(data = df, x = 'Age_bin', hue = 'Survived', kind = 'count')


# Youngters are likely to survive more.

# In[ ]:


sns.factorplot(data = df, x = 'Fare_bin', hue = 'Survived', kind = 'count')


# As much you pay, that much you will get security!

# In[ ]:


fig, axs = plt.subplots(1, 3,figsize=(15,5))

sns.pointplot(x = 'Fare_bin', y = 'Survived',  data=df, ax = axs[0])
sns.pointplot(x = 'Age_bin', y = 'Survived',  data=df, ax = axs[1])
sns.pointplot(x = 'Family_size', y = 'Survived', data=df, ax = axs[2])


# ### Handling categorical variables

# In[ ]:


label = LabelEncoder()
df['Title'] = label.fit_transform(df.Title)
df['Sex'] = label.fit_transform(df.Sex)
df['Embarked'] = label.fit_transform(df.Embarked)
df['Cabin'] = label.fit_transform(df.Cabin)


# In[ ]:


df.sample(2)


# In[ ]:


#We will look at correlation between variables. So before working with ticket column, save all variables we worked on yet.
#This id because we will use get_dummies on ticket and not label encoding.
corr_columns = list(df.drop(['Name', 'PassengerId', 'Ticket', 'label'], axis = 1).columns)


# #### Ticket

# In[ ]:


df['Ticket'] = df.Ticket.map(lambda x: re.sub(r'\W+', '', x))   #Remove special characters


# In[ ]:


#If ticket is of digit value, make them a character X
Ticket = []
for i in list(df.Ticket):
    if not i.isdigit():
        Ticket.append(i[:2])
    else:
        Ticket.append("X")
df['Ticket'] = Ticket


# In[ ]:


df.Ticket.unique()


# In[ ]:


df = pd.get_dummies(df, columns = ['Ticket'], prefix = 'T')


# Now we will select features from the dataset for modelling purpose.

# ### Feature Selection

# In[ ]:


cat_variables = [x for x in df.columns if df.dtypes[x] == 'object']
cat_variables


# In[ ]:


df.drop(['Name', 'PassengerId'], axis = 1, inplace = True)


# In[ ]:


df.sample(2)


# In[ ]:


train = df.loc[df.label == 'train'].drop('label', axis = 1)
test = df.loc[df.label == 'test'].drop(['label', 'Survived'], axis = 1)


# Lets look correlation between variables.

# ###### Pearson's

# In[ ]:


plt.figure(figsize = [14,10])
sns.heatmap(train[corr_columns].corr(), cmap = 'RdBu', annot = True)


# 1. Pclass and Cabin has some relations.
# 2. Offcourse Family_size will have relation with Parch and SibSp as it is derived from these two.
# 
# Lets look at spearman's correlation also. Note I'm looking this only for Fare variable as it is not skewed.

# ###### Spearman's

# In[ ]:


plt.figure(figsize = [14,10])
sns.heatmap(train[corr_columns].corr(method = 'spearman'), cmap = 'RdBu', annot = True)


# As expected! Correlation are somewhat more stronger than pearson's (looking at more blue blocks in spearan's). Also, Fare_log and Fare have correlation 1 because one is just the log transformation of another. Thus, rank will be same.[](http://)

# ## Modeling

# Split the data into train and test sets.

# In[ ]:


X_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived'].astype(int)
X_test = test


# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits = 5)


# In[ ]:


classifiers = []
classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(LogisticRegression(random_state = 0))
classifiers.append(LinearSVC(random_state = 0))
classifiers.append(SVC(random_state = 0))
classifiers.append(RandomForestClassifier(random_state = 0))
classifiers.append(ExtraTreesClassifier(random_state = 0))
classifiers.append(XGBClassifier(random_state = 0))
classifiers.append(LGBMClassifier(random_state = 0))
classifiers.append(MLPClassifier())

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y_train, scoring = 'accuracy', cv = kfold, n_jobs = -1))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({'CV_means':cv_means, 'CV_std':cv_std, 'Algorithm':['KNN', 'LinearDiscriminantAnalysis', 'LogisticRegression', 'LinearSVC', 'SVC', 'RandomForest', 'ExtraTrees', 'XGB', 'LGB', 'MLP']})


# In[ ]:


g = sns.barplot("CV_means", "Algorithm", data = cv_res, **{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# We will use Linear Discriminant Analysis, Random Forest, Extra Trees, XGBoost, Light Gradient Boosting and Multi Layer Perceptron. This is because these all performed well and while using voting classifier, I don't want to bias the classifier. So I used two bagging models, two boosting models, linear model and neural network.[](http://)

# #### Hyperparamter Tuning

# Here I'm not using HPT for LDA. Reason behind this is there is no combination which makes some difference. Thus, best parameter will be the default parameters.

# In[ ]:


LDA_best = LinearDiscriminantAnalysis().fit(X_train, y_train)


# In[ ]:


RF = RandomForestClassifier(random_state = 0)
RF_params = {'n_estimators' : [10,50,100],
             'criterion' : ['gini', 'entropy'],
             'max_depth' : [5,8,None],
             'min_samples_split' : [2,5,8],
             'min_samples_leaf' : [1,3,5],
             'max_features' : ['auto', 'log2', None]}
GS_RF = GridSearchCV(RF, param_grid = RF_params, cv = kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_RF.fit(X_train, y_train)
RF_best = GS_RF.best_estimator_
print("Best parameters :", RF_best)
print("Best score :", GS_RF.best_score_)


# In[ ]:


ET = ExtraTreesClassifier(random_state = 0)
ET_params = {'n_estimators' : [10,50,100],
             'criterion' : ['gini', 'entropy'],
             'max_depth' : [5,8,None],
             'min_samples_split' : [2,5,8],
             'min_samples_leaf' : [1,3,5],
             'max_features' : ['auto', 'log2', None]}
GS_ET = GridSearchCV(ET, param_grid = ET_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_ET.fit(X_train, y_train)
ET_best = GS_ET.best_estimator_
print('Best parameters :', ET_best)
print('Best score :', GS_ET.best_score_)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


XGB = XGBClassifier(random_state = 0)
XGB_params = {'n_estimators' : [100,200,500],
              'max_depth' : [3,4,5],
              'learning_rate' : [0.01,0.05,0.1,0.2],
              'booster' : ['gbtree', 'gblinear', 'dart']}
GS_XGB = GridSearchCV(XGB, param_grid = XGB_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_XGB.fit(X_train, y_train)
XGB_best = GS_XGB.best_estimator_
print('Best parameters :', XGB_best)
print('Best score :', GS_XGB.best_score_)


# In[ ]:


LGB = LGBMClassifier(random_state = 0)
LGB_params = {'n_estimators' : [100,200,500],
              'max_depth' : [5,8,-1],
              'learning_rate' : [0.01,0.05,0.1,0.2],
              'boosting_type' : ['gbdt', 'goss', 'dart']}
GS_LGB = GridSearchCV(LGB, param_grid = LGB_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_LGB.fit(X_train, y_train)
LGB_best = GS_LGB.best_estimator_
print('Best parameters :', LGB_best)
print('Best score :', GS_LGB.best_score_)


# In[ ]:


MLP = MLPClassifier(random_state = 0)
MLP_params = {'hidden_layer_sizes' : [[10], [10,10], [10,100], [100,100]],
              'activation' : ['relu', 'tanh', 'logistic'],
              'alpha' : [0.0001,0.001,0.01]}
GS_MLP = GridSearchCV(MLP, param_grid = MLP_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_MLP.fit(X_train, y_train)
MLP_best = GS_MLP.best_estimator_
print('Best parameters :', MLP_best)
print('Best score :', GS_MLP.best_score_)


# ### Plot Feature Importance

# We will plot feature importances of 4 classifiers excluding LDA and MLP. LDA and MLP doesnot have attribute feature_importances_. For linear models, finding feature importance or parameter influence depends on various techniques i.e., p-values, bootstrap scores, various discriminative indices, etc.

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize = [20,10])
fig.subplots_adjust(hspace = 0.7)
classifiers_list = [["RandomForest", RF_best], ["ExtraTrees", ET_best],
                    ["XGBoost", XGB_best], ["LGBoost", LGB_best]]

nClassifier = 0
for row in range(2):
    for col in range(2):
        name = classifiers_list[nClassifier][0]
        classifier = classifiers_list[nClassifier][1]
        feature = pd.Series(classifier.feature_importances_, X_train.columns).sort_values(ascending = False)
        feature.plot.bar(ax = axes[row,col])
        axes[row,col].set_xlabel("Features")
        axes[row,col].set_ylabel("Relative Importance")
        axes[row,col].set_title(name + " Feature Importance")
        nClassifier +=1


# 1. RF and ET used more features for prediction whereas boosters gave Fare and Age variable higher weightage.
# 2. Bins formed were used by trees whereas boosters did not use bins at all inspite of using Fare and Age variable for predictions.
# 
# ###### Reasons to keep linear model and MLP:
# We have created features of variables by forming bins i.e., Fare_bin, Age_bin, etc. Now, linear models, SVC and MLP have tendency to predict better when bins are formed from a feature whereas boosters don't give importance to bins and they use other parameters for classification. Also we saw that trees also used bins features for their prediction. Thus, combining models which classifies the data based on different features is want we want. Thus, I've kept two linear models with trees.

# In[ ]:


LDA_pred = pd.Series(LDA_best.predict(X_test), name = 'LDA')
MLP_pred = pd.Series(MLP_best.predict(X_test), name = "MLP")
RF_pred = pd.Series(RF_best.predict(X_test), name = "RFC")
ET_pred = pd.Series(ET_best.predict(X_test), name = 'ETC')
XGB_pred = pd.Series(XGB_best.predict(X_test), name = "XGB")
LGB_pred = pd.Series(LGB_best.predict(X_test), name = "LGB")

ensemble_results = pd.concat([LDA_pred, MLP_pred, RF_pred, ET_pred, XGB_pred, LGB_pred], axis = 1)
plt.figure(figsize = [8,5])
sns.heatmap(ensemble_results.corr(), annot = True)


# So, we can see that the predictions made by linear models, trees and boosters are quite different. Now, we will make our final output based on VotingClassifier.

# ### Ensemble Modelling

# In[ ]:


voting = VotingClassifier(estimators = [['LDA', LDA_best], ["MLP", MLP_best],
                                        ['RFC', RF_best], ['ETC', ET_best],
                                        ['XGB', XGB_best], ['LGB', LGB_best]], voting = 'soft', n_jobs = -1)
voting = voting.fit(X_train, y_train)


# I've kept voting = 'soft' because for each case, I want the result of that classifier that makes more confident prediction i.e., prediction with more probability.

# In[ ]:


results = pd.DataFrame(test_passengerId, columns = ['PassengerId']).assign(Survived = pd.Series(voting.predict(X_test)))
results.to_csv('models_voting.csv', index = None)


# This was my first kernel. Any kind of suggestion or appreciation is heartily welcomed. Also please UPVOTE it if you liked my work and found helpful to you.
# 
# Thank you!
