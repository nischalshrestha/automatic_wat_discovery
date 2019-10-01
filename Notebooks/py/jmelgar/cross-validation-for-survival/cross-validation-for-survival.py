#!/usr/bin/env python
# coding: utf-8

# This is my first Kaggle dataset

# In[ ]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

import os
print(os.listdir("../input"))


# In[ ]:


dataset = pd.read_csv('../input/train.csv')
dataset.head()


# In[ ]:


dataset.info()


#  The columns Age, Cabin and Embarked have null values, we are going to study them

# In[ ]:


dataset['Age'].describe()


# In[ ]:


dataset['Cabin'].describe()


# In[ ]:


print(dataset['Embarked'].describe())
print('/----/')
print(dataset['Embarked'].value_counts())


# has_cabin_T column is created

# In[ ]:


dataset['has_cabin_T'] = dataset['Cabin'].notnull()
dataset['has_cabin_T'] = dataset.apply(lambda x: 1 if x['has_cabin_T'] == True else 0, axis=1)


# In[ ]:


dataset['has_cabin_T'].value_counts()


# In[ ]:


def transform_Sex(df):
    df['sex_T'] = df['Sex']
    df['sex_T'] = df.apply(lambda x: 1 if str(x['Sex']) == 'female' else 0, axis=1)
    return df


# In[ ]:


dataset = transform_Sex(dataset)


# In[ ]:


def transform_Embarked(df):
    df['embarked_T'] = df['Embarked']
    df['embarked_T'] = df.apply(lambda x: 0 if True else x['embarked_T'], axis=1)
    df['embarked_T'] = df.apply(lambda x: 0 if str(x['Embarked']) == 'S' else x['embarked_T'], axis=1)
    df['embarked_T'] = df.apply(lambda x: 1 if str(x['Embarked']) == 'C' else x['embarked_T'], axis=1)
    df['embarked_T'] = df.apply(lambda x: 2 if str(x['Embarked']) == 'Q' else x['embarked_T'], axis=1)
    return df


# In[ ]:


dataset = transform_Embarked(dataset)


# In[ ]:


dataset['embarked_T'].value_counts()


# In[ ]:


dataset.info()


# First we will divide the variable Age in ranges changing the null values ​​by -1

# In[ ]:


def transform_Age(df):
    df['age_T'] = df['Age']
    df['Age'] = df['Age'].fillna(-1)

    df['age_T'] = df.apply(lambda x: -1 if x['Age'] == -1 else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 0 if (x['Age'] >= 0 and x['age_T'] < 5) else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 1 if (x['Age'] >= 5 and x['age_T'] < 12) else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 2 if (x['Age'] >= 12 and x['age_T'] < 18) else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 3 if (x['Age'] >= 18 and x['age_T'] < 25) else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 4 if (x['Age'] >= 25 and x['age_T'] < 35) else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 5 if (x['Age'] >= 35 and x['age_T'] < 60) else x['age_T'], axis=1)
    df['age_T'] = df.apply(lambda x: 6 if x['Age'] >= 60 else x['age_T'], axis=1)

    return df


# In[ ]:


dataset = transform_Age(dataset)


# In[ ]:


dataset.info()


# In[ ]:


dataset['age_T'].value_counts()


# The variable Fare, must be normalized

# In[ ]:


dataset['Fare'].describe()


# In[ ]:


def transform_Fare(df):
    df['fare_T'] = df['Fare'].fillna(df['Fare'].mode().values[0])
    df['fare_T'] = df.apply(lambda x: 0 if  x['Fare'] < 8 else x['fare_T'], axis=1)
    df['fare_T'] = df.apply(lambda x: 1 if (x['Fare'] >= 8 and x['fare_T'] < 15) else x['fare_T'], axis=1)
    df['fare_T'] = df.apply(lambda x: 2 if (x['Fare'] >= 15 and x['fare_T'] < 31) else x['fare_T'], axis=1)
    df['fare_T'] = df.apply(lambda x: 3 if  x['Fare'] >= 31  else x['fare_T'], axis=1)
    
    return df


# In[ ]:


dataset = transform_Fare(dataset)


# In[ ]:


dataset.info()


# A new file will be created that does not contain the variables that we have transformed, also eliminating the PassengerId column

# In[ ]:


train = pd.DataFrame(dataset['Survived'])
train['Pclass'] = dataset['Pclass']
train['SibSp'] = dataset['SibSp']
train['Parch'] = dataset['Parch']
train['has_cabin_T'] = dataset['has_cabin_T']
train['embarked_T'] = dataset['embarked_T']
train['age_T'] = dataset['age_T']
train['fare_T'] = dataset['fare_T']
train['sex_T'] = dataset['sex_T']


# In[ ]:


train.head()


# In[ ]:


train['age_T'].value_counts()


# The variable age_T still has 177 null values, we will proceed to create a model with the rest of the data to be able to impute that value

# In[ ]:


age_data = train[train['age_T'] != -1]


# In[ ]:


len(age_data)


# In[ ]:


age_data.head()


# the Survived variable is deleted, since it can not be used with the Test dataset

# In[ ]:


age_data = age_data.drop(age_data[['Survived']], axis=1)


# In[ ]:


age_data.head()


# the order of the columns is changed so that the Target variable is the last

# In[ ]:


def change_column_order(df, col_name, index):
    cols = df.columns.tolist()
    cols.remove(col_name)
    cols.insert(index, col_name)
    return df[cols]


# In[ ]:


age_data = change_column_order(age_data, 'age_T', 7)


# In[ ]:


age_data.head()


# the file is separated into train and test

# In[ ]:


from sklearn.model_selection import train_test_split
data_split = train_test_split(age_data, train_size = 0.7, test_size=0.3)
age_train = data_split[0] 
age_test = data_split[1]


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor


# In[ ]:


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

selector = DecisionTreeRegressor()
rfecv = RFECV(estimator=selector)

scaler = StandardScaler()

model = RandomForestRegressor()

age_pipe = Pipeline(steps=[("selector",rfecv), ("scaler",scaler), ("model",model)])

hiperparam_pipeline = {"selector__cv":[5,10], 
                       "selector__scoring":["neg_mean_squared_error"], 

                       "scaler__with_mean":[True, False], 
                       "scaler__with_std":[True, False],  
                                                             
                       "model__criterion":["mse"], 
                       "model__n_estimators":[5, 10, 20],
                       "model__max_depth":[3,4,5],
                       "model__min_samples_split":[3, 4, 5],
                       "model__min_samples_leaf":[2,3,4],
                       "model__bootstrap":[True,False]
                      }


age_grid = GridSearchCV(estimator=age_pipe,
                        param_grid=hiperparam_pipeline,
                        scoring="neg_mean_squared_error",
                        cv=5,
                        n_jobs=4,
                        verbose=1
                        )


# In[ ]:


cols = age_test.columns

age_grid.fit(X=age_train[list(cols[0:-1])],
             y=age_train[cols[-1]])


# In[ ]:


age_grid.best_params_


# In[ ]:


# Mean squared error train
from sklearn.metrics import mean_squared_error
mean_squared_error(age_train[cols[-1]], 
                age_grid.predict(X=age_train[list(cols[:-1])]))


# In[ ]:


# Mean squared error test
from sklearn.metrics import mean_squared_error
mean_squared_error(age_test[cols[-1]], 
                age_grid.predict(X=age_test[list(cols[:-1])]))


# In[ ]:


# R² score, the coefficient of determination  train
from sklearn.metrics import r2_score
r2_score(age_train[cols[-1]], 
                age_grid.predict(X=age_train[list(cols[:-1])]))


# In[ ]:


# R² score, the coefficient of determination test
from sklearn.metrics import r2_score
r2_score(age_test[cols[-1]], 
                age_grid.predict(X=age_test[list(cols[:-1])]))


# We verify that the values ​​of the coefficient of determination and the mean squared error are not very different in train and test sets

# now we proceed to impute the null values ​​of the age_T column with the values ​​predicted by the model. 

# In[ ]:


train = change_column_order(train, 'age_T', 7)


# In[ ]:


train.head()


# In[ ]:


def imput_Age(df):
    cols = df.columns
    df['age_T'] = df.apply(lambda x: float(int(age_grid.predict(X=df[list(cols[1:-1])])[0])) 
                           if x['age_T'] == -1 else x['age_T'], axis=1)
    return df


# In[ ]:


train = imput_Age(train)


# In[ ]:


train['age_T'].value_counts()


# In[ ]:


train.info()


# In[ ]:


train.head()


# There are no null values ​​anymore, we are going to analyze the data we have obtained

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


sns.barplot(x="has_cabin_T", y="Survived", hue="Pclass", data=train);


# In[ ]:


sns.barplot(x="Pclass", y="Survived", hue="sex_T", data=train);


# In[ ]:


sns.barplot(x="embarked_T", y="Survived", hue="Pclass", data=train);


# In[ ]:


sns.barplot(x="fare_T", y="Survived", hue="Pclass", data=train);


# In[ ]:


sns.barplot(x="age_T", y="Survived", hue="sex_T", data=train);


# In[ ]:


corr_matrix = train.corr()


# In[ ]:


corr_matrix


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt

names =  corr_matrix.columns.tolist()

correlations = corr_matrix

fig = plt.figure(1, figsize=(9,9))
ax = fig.add_subplot(111)

corr_mat_plot = ax.matshow(corr_matrix)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
cb = fig.colorbar(corr_mat_plot)

cb.ax.tick_params(labelsize='xx-large')
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation='45', ha='left', size = 'xx-large')
ax.set_yticklabels(names, rotation='horizontal', ha='right', size = 'xx-large')
plt.show()


# In[ ]:


list_Grids = []


# In[ ]:


#logistic regression

from sklearn.linear_model import LogisticRegression

selector = DecisionTreeRegressor()
rfecv = RFECV(estimator=selector)

scaler = StandardScaler()

model = LogisticRegression()

titanic_lr_pipe = Pipeline(steps=[("selector",rfecv), ("scaler",scaler), ("model",model)])

hiperparam_pipeline = {"selector__cv":[10, 15, 20], 
                       "selector__scoring":["roc_auc"], 
                       
                       "scaler__with_mean":[True, False], 
                       "scaler__with_std":[True, False],  
                                                             
                       "model__solver":["newton-cg", "lbfgs", "liblinear"], 
                       "model__max_iter":[250, 500, 1000],
                       "model__C":[2.0, 2.5, 3.0]
                      }


list_Grids.append(GridSearchCV(estimator=titanic_lr_pipe,
                               param_grid=hiperparam_pipeline,
                               scoring="roc_auc",
                               cv=6,
                               n_jobs=4,
                               verbose=1
                              )
                 )


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

selector = DecisionTreeRegressor()
rfecv = RFECV(estimator=selector)

scaler = StandardScaler()

model = RandomForestClassifier()

titanic_rf_pipe = Pipeline(steps=[("selector",rfecv), ("scaler",scaler), ("model",model)])

hiperparam_pipeline = {"selector__cv":[5, 10],  
                       "selector__scoring":["roc_auc"], 
                       
                       "scaler__with_mean":[True, False], 
                       "scaler__with_std":[True, False],  
                                                             
                       "model__n_estimators":[5, 10, 20],
                       "model__max_depth":[2, 3, 4, 5, 6, 7],
                       "model__min_samples_split":[2, 3, 4, 5],
                       "model__min_samples_leaf":[2,3,4,5]
                      }


list_Grids.append(GridSearchCV(estimator=titanic_rf_pipe,
                               param_grid=hiperparam_pipeline,
                               scoring="roc_auc",
                               cv=5,
                               n_jobs=4,
                               verbose=1
                              )
                 )


# In[ ]:


#naive Bayes
from sklearn.naive_bayes import GaussianNB

selector = DecisionTreeRegressor()
rfecv = RFECV(estimator=selector)

scaler = StandardScaler()

model = GaussianNB()

titanic_nb_pipe = Pipeline(steps=[("selector",rfecv), ("scaler",scaler), ("model",model)])

hiperparam_pipeline = {"selector__cv":[4, 5, 6], 
                       "selector__scoring":["roc_auc"],
                       
                       "scaler__with_mean":[True, False], 
                       "scaler__with_std":[True, False]
                      }


list_Grids.append(GridSearchCV(estimator=titanic_nb_pipe,
                               param_grid=hiperparam_pipeline,
                               scoring="roc_auc",
                               cv=6,
                               n_jobs=4,
                               verbose=1
                              )
                 )


# In[ ]:


#SVM
from sklearn.svm import SVC

selector = DecisionTreeRegressor()
rfecv = RFECV(estimator=selector)

scaler = StandardScaler()

model = SVC()

titanic_svc_pipe = Pipeline(steps=[("selector",rfecv), ("scaler",scaler), ("model",model)])

hiperparam_pipeline = {"selector__cv":[7, 10, 13], 
                       "selector__scoring":["roc_auc"],
                       
                       "scaler__with_mean":[True, False], 
                       "scaler__with_std":[True, False],  
                                                             
                       "model__kernel":["rbf"],
                       "model__C":[0.01, 0.05, 0.1],
                       "model__degree":[0.5, 1, 1.5]
                      }


list_Grids.append(GridSearchCV(estimator=titanic_svc_pipe,
                               param_grid=hiperparam_pipeline,
                               scoring="roc_auc",
                               cv=6,
                               n_jobs=4,
                               verbose=1
                              )
                 )


# In[ ]:


cols = train.columns


# In[ ]:


for grid in list_Grids:
    grid.fit(X=train[list(cols[1:])],
               y=train[list(cols[0:1])])
    
# bag_pipes will be saved
import pickle
fout = open('bag_grids.pickle','wb')
pickle.dump(list_Grids,fout)
fout.close()


# In[ ]:


estimators_list = []
scores_list = []

for grid in list_Grids:
    estimators_list.append(grid.best_estimator_)
    scores_list.append(grid.best_score_)

print(scores_list)

winner_pipeline = estimators_list[scores_list.index(max(scores_list))]

print(winner_pipeline)


# Now let's check the format of the Test file

# In[ ]:


raw_test = pd.read_csv('../input/test.csv')
raw_test.head()


# In[ ]:


raw_test.info()


# we will process the file to obtain a similar format to the train file, but we will keep the variable PassengerId

# In[ ]:


raw_test['has_cabin_T'] = raw_test['Cabin'].notnull()
raw_test['has_cabin_T'] = raw_test.apply(lambda x: 1 if x['has_cabin_T'] == True else 0, axis=1)

raw_test = transform_Sex(raw_test)
raw_test = transform_Embarked(raw_test)
raw_test = transform_Age(raw_test)
raw_test = transform_Fare(raw_test)

test = pd.DataFrame(raw_test['PassengerId'])
test['Pclass'] = raw_test['Pclass']
test['SibSp'] = raw_test['SibSp']
test['Parch'] = raw_test['Parch']
test['has_cabin_T'] = raw_test['has_cabin_T']
test['embarked_T'] = raw_test['embarked_T']
test['age_T'] = raw_test['age_T']
test['age_T'] = raw_test['age_T']
test['fare_T'] = raw_test['fare_T']
test['sex_T'] = raw_test['sex_T']

test = change_column_order(test, 'age_T', 7)

test.head()


# In[ ]:


test.info()


# In[ ]:


test = imput_Age(test)


# In[ ]:


test.head()


# In[ ]:


cols = test.columns
predictions = winner_pipeline.predict(X=test[list(cols[1:])])


# In[ ]:


PassengerId_list = test['PassengerId'].values


# In[ ]:


print(len(predictions))
print(len(PassengerId_list))


# In[ ]:


d = {'PassengerId':PassengerId_list,'Survived':predictions}
df = pd.DataFrame(d)
df.to_csv('submission.csv', index=False)


# In[ ]:


gender_submission = pd.read_csv('../input/gender_submission.csv')
gender_submission.head()


# In[ ]:


#confusion Matrix
from sklearn.metrics import confusion_matrix

cols = test.columns

conf_matrix = confusion_matrix(y_true=gender_submission['Survived'],
                               y_pred=winner_pipeline.predict(X=test[list(cols[1:])])
                              )

conf_matrix = np.array([(conf_matrix[1]),
                         conf_matrix[0]])[:,::-1]

conf_matrix


# In[ ]:





# In[ ]:




