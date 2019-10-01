#!/usr/bin/env python
# coding: utf-8

# #My first kernel

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#d.groupby(['state']).agg({'FATALS':sum})
##pp.sort_values(by='crashTime',ascending=False,inplace=True)
##print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
##dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

# Let's flatten the columns 
#pp.columns = pp.columns.get_level_values(0)
#df.isnull().any()
#df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
#df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
print(train.head())


# In[ ]:


print(train.columns.values)


# In[ ]:


print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# In[ ]:


train['Ticket_len'] = train['Ticket'].apply(lambda x: len(x))
train['Name_len'] = train['Name'].apply(lambda x: len(x))
train = train.drop(['Cabin'],axis=1)
train = train.drop(['Ticket'],axis=1)
print(train.head())


# In[ ]:





# In[ ]:


#train.describe()
train['Sex'].value_counts()


# In[ ]:


train['Family'] = train['SibSp'] + train['Parch'] 


# In[ ]:


train.drop(['SibSp','Parch'],axis=1,inplace=True)
print(train.head())


# In[ ]:


train.isnull().any()
train['Age'].isnull().sum()
#train.shape


# In[ ]:


#train[['Age','Survived']].plot.line(x='Age',y='Survived')
#from ggplot import
import seaborn as sns
sns.countplot(train['Age'], hue=train['Survived'])

#ggplot(aes(x='Age', y='Survived'), data=train) +\
##    geom_point() +\
#    stat_smooth(colour='blue', span=0.2)


# In[ ]:


train['Age'].describe()


# In[ ]:


#train.loc[train['Age'].isnull(),'Age'] = train['Age'].mean()
age_avg = train['Age'].mean()
age_std = train['Age'].std()
age_null_count = train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age'] = train['Age'].astype(int)


# In[ ]:


train['Age'].isnull().any()
train.isnull().any()
train['Embarked'].value_counts()
train['Embarked'].isnull().sum()
train.loc[train['Embarked'].isnull(),'Embarked'] = 's'
train['Embarked'].value_counts()
train.loc[train['Embarked']=='s','Embarked'] = 'S'
train['Embarked'].value_counts()


# In[ ]:


train.isnull().any()


# In[ ]:


test = pd.read_csv('../input/test.csv')
print(test.head())


# In[ ]:


test['Ticket_len'] = test['Ticket'].apply(lambda x: len(x))
test['Name_len'] = test['Name'].apply(lambda x: len(x))
test.drop(['Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


print(test.head())


# In[ ]:


test.loc[test['Embarked'].isnull(),'Embarked'] = 'S'
#test.loc[test['Age'].isnull(),'Age'] = test['Age'].mean()
age_avg = test['Age'].mean()
age_std = test['Age'].std()
age_null_count = test['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
test['Age'][np.isnan(test['Age'])] = age_null_random_list
test['Age'] = test['Age'].astype(int)

test.isnull().any()


# In[ ]:


test.loc[test['Fare'].isnull(),'Fare'] = test['Fare'].mean()


# In[ ]:


test.isnull().any()


# In[ ]:


x = {'male': 1,'female': 0}
#d['state']=d['STATE'].apply(lambda x: states[x])
test['Sex']=test['Sex'].apply(lambda y: x[y])


# In[ ]:


e = {'S': 0,'C': 1,'Q': 2}
test['Embarked']=test['Embarked'].apply(lambda y: e[y])


# In[ ]:


x = {'male': 1,'female': 0}
#d['state']=d['STATE'].apply(lambda x: states[x])
train['Sex']=train['Sex'].apply(lambda y: x[y])

e = {'S': 0,'C': 1,'Q': 2}
train['Embarked']=train['Embarked'].apply(lambda b: e[b])


# In[ ]:


test['Family'] = test['SibSp'] + test['Parch'] 
test.drop(['SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


test['Fare']=test['Fare'].astype(int)
test['Age']=test['Age'].astype(int)


# In[ ]:


train['Fare']=train['Fare'].astype(int)
train['Age']=train['Age'].astype(int)


# In[ ]:


print(test.head())
#train = train.drop(['Name','Fare','Cabin'],axis=1)
#train = train.drop(['Ticket'],axis=1)
print(train.head())


# In[ ]:


#test_cat = test
#test_cat['CategoricalFare'] = pd.qcut(test_cat['Fare'], 4)
#print (test_cat[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# In[ ]:


# Mapping Fare
train.loc[ train['Fare'] <= 7, 'Fare'] 						        = 0
train.loc[(train['Fare'] > 7) & (train['Fare'] <= 14), 'Fare'] = 1
train.loc[(train['Fare'] > 14) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] > 31, 'Fare'] 							        = 3
train['Fare'] = train['Fare'].astype(int)
    
    # Mapping Age
train.loc[ train['Age'] <= 16, 'Age'] 					       = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4
    
    # Mapping Fare
test.loc[test['Fare'] <= 7, 'Fare'] 						        = 0
test.loc[(test['Fare'] > 7) & (test['Fare'] <= 14), 'Fare'] = 1
test.loc[(test['Fare'] > 14) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] 							        = 3
test['Fare'] = test['Fare'].astype(int)
    
    # Mapping Age
test.loc[ test['Age'] <= 16, 'Age'] 					       = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4

    # Mapping Family
train.loc[(train['Family'] > 0) & (train['Family'] <= 3), 'Family'] = 1
train.loc[ train['Family'] > 3, 'Family'] = 2

    # Mapping Family
test.loc[(test['Family'] > 0) & (test['Family'] <= 3), 'Family'] = 1
test.loc[ test['Family'] > 3, 'Family'] = 2

    # Mapping Name_len
test.loc[ test['Name_len'] <= 19, 'Name_len']       = 0
test.loc[(test['Name_len'] > 19) & (test['Name_len'] <= 23), 'Name_len'] = 1
test.loc[(test['Name_len'] > 23) & (test['Name_len'] <= 27), 'Name_len'] = 2
test.loc[(test['Name_len'] > 27) & (test['Name_len'] <= 32), 'Name_len'] = 3
test.loc[ test['Name_len'] > 32, 'Name_len'] = 4
         
         
    # Mapping Name_len
train.loc[ train['Name_len'] <= 19, 'Name_len']       = 0
train.loc[(train['Name_len'] > 19) & (train['Name_len'] <= 23), 'Name_len'] = 1
train.loc[(train['Name_len'] > 23) & (train['Name_len'] <= 27), 'Name_len'] = 2
train.loc[(train['Name_len'] > 27) & (train['Name_len'] <= 32), 'Name_len'] = 3
train.loc[ train['Name_len'] > 32, 'Name_len'] = 4


         


# In[ ]:


import re as re
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

#for dataset in train:
#    print(dataset)
train['Title'] = train['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))


# In[ ]:


train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train['Title'] = train['Title'].map(title_mapping)
train['Title'] = train['Title'].fillna(0)


# In[ ]:


test['Title'] = test['Name'].apply(get_title)
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

#print (test[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
test['Title'] = test['Title'].map(title_mapping)
test['Title'] = test['Title'].fillna(0)


# In[ ]:


train = train.drop("Name",axis=1)
test = test.drop("Name",axis=1)


# In[ ]:


print(test.head())
print(train.head())


# In[ ]:


train['Survived'].groupby(pd.qcut(train['Ticket_len'], 4)).mean()
#train['Ticket_len'].groupby(train['Survived']).mean()


# In[ ]:


X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()


# In[ ]:


X_train = X_train.drop("PassengerId",axis=1)


# In[ ]:


X_train = X_train.drop("Embarked",axis=1)
X_test = X_test.drop("Embarked",axis=1)


# In[ ]:


train['Ticket_len'].value_counts()


# In[ ]:


print(X_train.head())
print(Y_train.head())
print(X_test.head())


# In[ ]:


## Preprocessing
from sklearn import preprocessing
## SCALING
X_train_scale = preprocessing.scale(X_train)
X_test_scale = preprocessing.scale(X_test)

## StandardScaler
#normalizer1 = preprocessing.StandardScaler().fit(X_train)
#normalizer2 = preprocessing.StandardScaler().fit(X_test)

#X_train_stdscale = normalizer1.transform(X_train)
#X_test_stdscale = normalizer1.transform(X_test)



## normalizing
#normalizer1 = preprocessing.Normalizer().fit(X_train)
#normalizer2 = preprocessing.Normalizer().fit(X_test)

#X_train_norm = normalizer1.transform(X_train)
#X_test_norm = normalizer1.transform(X_test)


# In[ ]:





# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5], "min_samples_split" : [2, 4, 10, 12], "n_estimators": [50, 100, 400, 700]}
gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=2, n_jobs=-1)
gs = gs.fit(X_train_scale, Y_train)
print(gs.best_score_)
print(gs.best_params_)
print(gs.cv_results_)


# In[ ]:


print('Hi')


# In[ ]:


#logreg = LogisticRegression()
    
#logreg.fit(X_train_scale, Y_train)

#Y_pred = logreg.predict(X_test_scale)

#logreg.score(X_train_scale, Y_train)


# In[ ]:


# Random Forests
#params = [{'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 100}]
random_forest = RandomForestClassifier(criterion='gini', 
                                      min_samples_leaf=1,
                                      min_samples_split=10,
                                      n_estimators=400,
                                      max_features='auto',
                                      oob_score=True,
                                      random_state=1,
                                      n_jobs=-1)

random_forest.fit(X_train_scale, Y_train)

Y_pred = random_forest.predict(X_test_scale)

random_forest.score(X_train_scale, Y_train)

## after normalization

#random_forest.fit(X_train_stdscale, Y_train)

#Y_pred_stdscale = random_forest.predict(X_test_stdscale)

#random_forest.score(X_train_stdscale, Y_train)
print("%.4f" % random_forest.oob_score_)


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred_orig = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train_scale, Y_train)

Y_pred_scale1 = random_forest.predict(X_test_scale)

random_forest.score(X_train_scale, Y_train)


# In[ ]:


print(X_train.columns)
fimpo = random_forest.feature_importances_
print(fimpo)
testimpo = pd.DataFrame({
        "Columns": X_train.columns,
        "Importance": fimpo
         })
print(testimpo)
#sns.countplot(testimpo['Columns'], hue=testimpo['Importance'])
sns.barplot(x=testimpo['Columns'],y=testimpo['Importance'])
from ggplot import *
#ggplot(aes(x='Columns', y='Importance'), data=testimpo) +\
#    geom_point() +\
#    stat_smooth(colour='blue', span=0.2)


# In[ ]:


#test1 = test
#test1['Survived'] = Y_pred
#test1 = test1['PassengerId','Survived']
#test1 = test[['PassengerId','Survived']]
test2 = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
         })
#test2.reset_index(drop=True,inplace=True)
#test2.set_index('PassengerId', inplace=True)
#print(test2)
test2.to_csv("Predictions_scale1.csv", index=False)

test3 = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_orig
         })

test3.to_csv("Prediction.csv", index=False)

test4 = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred_scale1
         })

test4.to_csv("Prediction_scale2.csv", index=False)

#test3 = test
#test3['Survived'] = Y_pred_norm
#test4 = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": Y_pred_stdscale
#         })
#test4.reset_index(drop=True,inplace=True)
#del test4['index']
#test4.to_csv("Predictions_stdscale2.csv")

#test5 = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": Y_pred_orig
#         })
#test5.reset_index(drop=True,inplace=True)
#del test4['index']
#test5.to_csv("Predictions_categ2.csv")


### Without preprocessing 74%
### after scaling - 77 %
### after normalization - 74% (Doesn't help much)


# In[ ]:




