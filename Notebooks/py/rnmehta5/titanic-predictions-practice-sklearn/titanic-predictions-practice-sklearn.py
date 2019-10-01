#!/usr/bin/env python
# coding: utf-8

# ## Titanic Survivor Prediction

# Getting started, lets import required libraries & pull in the train and test data

# In[ ]:


#importing required libraries
import numpy as np
import pandas as pd
from sklearn import tree, linear_model, metrics
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

get_ipython().magic(u'matplotlib inline')


# In[ ]:


#read the train and test sets and storing them in pd dataframes
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Lets look at the size of the total data and its split in train and test

# In[ ]:


print(train.shape)
print(test.shape)


# ok look like roughly 70-30% split, lets look at a sample fo what's in the data sets

# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Checking for Missing Data

# In[ ]:


for col in train.columns:
    print('number of null values in ' + col + ': ' + str(train[pd.isnull(train[col])].shape[0]))


# Age, Cabin and Embarked are the 3 columns which have null values, may need to estimate them in order to use that data for predictions

# In[ ]:


for col in test.columns:
    print('number of null values in ' + col + ': ' + str(test[pd.isnull(test[col])].shape[0]))


# Age, fare and cabin have null values in the test set. Now lets look into how the different columns are effecting the survival rates to find the features for an initial model

# ## Picking Predictive Features
# 
# Well, if you have seen the titanic movie it seemed like they were prioritizing women and children to be boarded into lifeboats and ofcourse people from higher classes. Let's see if this hypothesis holds

# In[ ]:


train.pivot_table(index='Sex', values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(index='Sex', values='Survived', aggfunc='mean'))


# Trying to see if the kind of sex distribution in the test data is similar to training data

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
train.pivot_table(index='Sex', values='Survived', aggfunc='count').plot(kind='bar', ax=ax[0])
test.pivot_table(index='Sex', values='PassengerId', aggfunc='count').plot(kind='bar', ax=ax[1])


# so sex is definitely a big factor in survival rate, let's see how it interacts with class, assuming 1st class passengers get preferential treatment and can have higher survival rates

# In[ ]:


train.pivot_table(columns='Sex', index='Pclass',                  values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(columns='Sex', index='Pclass',                  values='Survived', aggfunc='mean'))
print(train.pivot_table(columns='Sex', index='Pclass',                  values='Survived', aggfunc='count'))


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (12,5))
train.pivot_table(index='Pclass', values='Survived', aggfunc='count').plot(kind='bar', ax=ax[0])
test.pivot_table(index='Pclass', values='PassengerId', aggfunc='count').plot(kind='bar', ax=ax[1])


# Interesting! looks like almost all the women in 1st and 2nd class survived. Men on the other hand still survive less but men in the first class have a better chance of surviving. Ok so lets take these 2 features and build a first model. First a logistic regression model as baseline, then, I will be using the decision tree algorithm since it would be well suited to this problem.

# ## First model - with sex and Pclass as the features

# In[ ]:


train = train.join(pd.get_dummies(train.Sex))
test = test.join(pd.get_dummies(test.Sex))


# In[ ]:


class_dummies = pd.get_dummies(train.Pclass)
class_dummies.columns = ['Higher', 'Middle', 'Lower']
train = train.join(class_dummies)


# In[ ]:


X_train = train[['male', 'female','Higher', 'Middle', 'Lower']]
y = train['Survived']

class_dummies = pd.get_dummies(test.Pclass)
class_dummies.columns = ['Higher', 'Middle', 'Lower']
test = test.join(class_dummies)
X_test = test[['male', 'female', 'Higher', 'Middle', 'Lower']]


# In[ ]:


log_reg = linear_model.LogisticRegression()
baseline_log_reg = log_reg.fit(X_train, y)
predicted_survivors = baseline_log_reg.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(train.Survived, baseline_log_reg.predict(X_train)))
sns.heatmap(metrics.confusion_matrix(train.Survived, baseline_log_reg.predict(X_train)),            cmap="Blues", annot=True)
plt.xlabel('Pred Label')
plt.ylabel('True Label')

test_baseline_model = test
test_baseline_model['Survived'] = predicted_survivors
test_baseline_model = test_baseline_model[['PassengerId', 'Survived']]

#test_baseline_model.to_csv('test_baseline_model.csv', index=False)


# In[ ]:


first_tree = tree.DecisionTreeClassifier()
#test['sex_mapped'] = train.Sex.map({'male':1, 'female':0})
first_tree_fit = first_tree.fit(X_train, y)
predicted_survivors = first_tree_fit.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(train.Survived, first_tree_fit.predict(X_train)))
sns.heatmap(metrics.confusion_matrix(train.Survived, first_tree_fit.predict(X_train)),            cmap="Blues", annot=True)
plt.xlabel('Pred Label')
plt.ylabel('True Label')

test_first_model = test
test_first_model['Survived'] = predicted_survivors
test_first_model = test_first_model[['PassengerId', 'Survived']]

#test_first_model.to_csv('test_first_model.csv', index=False)


# Ok looks like both these models seem to have ~78% accuracy on the training set, although the tree model seems to be giving a lot more **false negatives**

# ### Further Feature Exploration 
# 
# Exploration for more features - strarting with the children - let's see if the chances of survival increase if people are travelling with a family for males and females and then based on the class they are in

# In[ ]:


train.pivot_table(columns='Sex', index='Parch',                  values='Survived', aggfunc='mean').plot(kind='bar')
print(train.pivot_table(columns='Sex', index='Parch',                  values='Survived', aggfunc='mean'))
print(train.pivot_table(columns='Sex', index='Parch',                  values='Survived', aggfunc='count'))


# Interesting! women with no childern seem to have a similar survial rate compared to overall surivial rate of women whereas with children have a slightly higher survial rates except those women with exactly 2 children.. which is weird, women with 3 children seem to have a high survival rate again, could it be the women with 2 children stayed back more... doesn't make too much sense. Perhaps where these women come from will offer a better clue. Also men with children have almost a double rate of survival than those who don't!
# 
# Also the tail values of # of children have very few examples, it would be best to combine them so that in best case they don't get completely ignored and in the worst case cause the model to overfit

# In[ ]:


train['new_Parch'] = train.Parch
train['new_Parch'] = train.new_Parch.astype(int)
train.loc[train.new_Parch > 1, 'new_Parch'] = 2
print(train.pivot_table(index=['Sex', 'Pclass'], columns='new_Parch', values='Survived', aggfunc='mean'))
print(train.pivot_table(index=['Sex', 'Pclass'], columns='new_Parch', values='Survived', aggfunc='count'))
train.pivot_table(index=['Sex', 'Pclass'], columns='new_Parch', values='Survived', aggfunc='mean').plot(kind='bar')


# In[ ]:


print(train.pivot_table(index='Sex', columns='SibSp', values='Survived', aggfunc='mean'))
print(train.pivot_table(index='Sex', columns='SibSp', values='Survived', aggfunc='count'))
train.pivot_table(index='Sex', columns='SibSp', values='Survived', aggfunc='mean').plot(kind='bar')


# same issue as the Parent-Children column the higher number of siblings/spouses is pretty sparse, will be merging them together

# In[ ]:


train['new_SibSp'] = train.SibSp
train['new_SibSp'] = train.SibSp.astype(int)
train.loc[train.new_SibSp > 1, 'new_SibSp'] = 2
print(train.pivot_table(index='Sex', columns='new_SibSp', values='Survived', aggfunc='mean'))
print(train.pivot_table(index='Sex', columns='new_SibSp', values='Survived', aggfunc='count'))
train.pivot_table(index='Sex', columns='new_SibSp', values='Survived', aggfunc='mean').plot(kind='bar')


# ### Model 2 with 2 new features
# 
# Trying out the decision tree and logistic regression models with 4 features now to see if there's any improvements

# In[ ]:


X_train = train[['male', 'female', 'Pclass', 'new_Parch', 'new_SibSp']]
y = train['Survived']

test['new_SibSp'] = test.SibSp
test['new_SibSp'] = test.SibSp.astype(int)
test.loc[train.new_SibSp > 1, 'new_SibSp'] = 2

test['new_Parch'] = test.Parch
test['new_Parch'] = test.new_Parch.astype(int)
test.loc[train.new_Parch > 1, 'new_Parch'] = 2

X_test = test[['male', 'female', 'Pclass', 'new_Parch', 'new_SibSp']]


# In[ ]:


second_tree = tree.DecisionTreeClassifier()
#test['sex_mapped'] = train.Sex.map({'male':1, 'female':0})
second_tree_fit = second_tree.fit(X_train, y)
predicted_survivors = second_tree_fit.predict(X_test)

print('Accuracy: ', metrics.accuracy_score(train.Survived, second_tree_fit.predict(X_train)))
sns.heatmap(metrics.confusion_matrix(train.Survived, second_tree_fit.predict(X_train)),            cmap="Blues", annot=True)
plt.xlabel('Pred Label')
plt.ylabel('True Label')

test_second_model = test
test_second_model['Survived'] = predicted_survivors
test_second_model = test_second_model[['PassengerId', 'Survived']]

#test_second_model.to_csv('test_second_model.csv', index=False)


# Nice! thats a bump of about 2% accuracy on the training set, let's see how it does on test data 
# 
# Unfortunately this is not giving any advantages on the test score. Let's also train a couple of other models and see their performance using CV to see if we get even better performance

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

model_names = [
    'KNeighborsClassifier',
    'SVC(kernel="linear")',
    'RandomForestClassifier(max_depth=3)',
    'AdaBoostClassifier()',
    'GaussianNB()',
    'QuadraticDiscriminantAnalysis()'
]

models = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    RandomForestClassifier(max_depth=2),
    AdaBoostClassifier(),
    GaussianNB()
]

fit_models = []

for i in range(len(models)):
    temp_score = cross_val_score(models[i], X_train, y, cv=5)
    fit_models.append(models[i].fit(X_train,y))
    print(model_names[i],' ', temp_score.mean())
    print('+/- ',temp_score.var())


# In[ ]:


test_rf_model = test
test_rf_model['Survived'] = fit_models[2].predict(X_test)
test_rf_model = test_rf_model[['PassengerId', 'Survived']]
test_rf_model.to_csv('test_rf_model.csv', index=False)


# In[ ]:


print('S embarked survival %:',      round((train[train.Embarked == 'S']             .Survived.sum()/train[train.Embarked == 'S'].Survived.count())*100., 3))
print('C embarked survival %:',      round((train[train.Embarked == 'C']             .Survived.sum()/train[train.Embarked == 'C'].Survived.count())*100., 3))
print('Q embarked survival %:',      round((train[train.Embarked == 'Q']             .Survived.sum()/train[train.Embarked == 'Q'].Survived.count())*100., 3))


# In[ ]:


print(train.pivot_table(index='Pclass', columns='Embarked', values='Survived', aggfunc='count'))
print(train.pivot_table(index='Pclass', columns='Embarked', values='Survived', aggfunc='mean'))
train.pivot_table(index='Pclass', columns='Embarked', values='Survived', aggfunc='mean').plot(kind='bar')


# There are 2 passengers missing embarked info in the train set, let's see who they are so that we can figure out how best to handle this

# In[ ]:


train[train.Embarked.isnull()]


# we can potentially interpolate the embarked from port based on cabin and fare information - but let's stick a pin in that one. use the data as is

# In[ ]:


feature_columns = ['Pclass', 'Sex', 'SibSp',                   'Parch', 'Fare', 'Cabin', 'Embarked']
target = ['Survived']


# In[ ]:




