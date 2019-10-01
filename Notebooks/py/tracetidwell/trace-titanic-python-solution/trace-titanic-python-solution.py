#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
from tensorflow.contrib import skflow


# In[ ]:


titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


# In[ ]:


titanic_df.head()


# In[ ]:


test_df.head()


# In[ ]:


titanic_df['Survived'].mean()


# In[ ]:


titanic_df.groupby('Pclass').mean()


# In[ ]:


class_sex_grouping = titanic_df.groupby(['Pclass', 'Sex']).mean()
print(class_sex_grouping['Survived'])


# In[ ]:


class_sex_grouping['Survived'].plot.bar()


# In[ ]:


group_by_age = pd.cut(titanic_df['Age'], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()


# In[ ]:


titanic_df.count()


# In[ ]:


test_df.count()


# In[ ]:


titanic_df = titanic_df.drop(['Cabin'], axis = 1)


# In[ ]:


test_df = test_df.drop(['Cabin'], axis=1)


# In[ ]:


titanic_df = titanic_df.dropna()


# In[ ]:


#test_df = test_df.dropna()


# In[ ]:


titanic_df.count()


# In[ ]:


test_df.count()


# In[ ]:


def preprocess_titanic_df(df) :
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name', 'Ticket'], axis = 1)
    return processed_df


# In[ ]:


processed_df = preprocess_titanic_df(titanic_df)
processed_df.count()
processed_df


# In[ ]:


processed_test_df = preprocess_titanic_df(test_df)
processed_test_df.count()
processed_test_df


# In[ ]:


median_ages = np.zeros((2,3))
median_ages


# In[ ]:


for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = processed_test_df[(processed_test_df['Sex'] == i) & (processed_test_df['Pclass'] == j+1)]['Age'].dropna().median()
        
median_ages


# In[ ]:


for i in range(0, 2):
    for j in range(0, 3):
        processed_test_df.loc[ (processed_test_df.Age.isnull()) & (processed_test_df.Sex == i) & (processed_test_df.Pclass == j+1),'Age'] = median_ages[i,j]
        
processed_test_df.loc[processed_test_df.Fare.isnull(), 'Fare'] = processed_test_df['Fare'].median()
processed_test_df.count()


# In[ ]:


X = processed_df.drop(['Survived'], axis = 1).values
Y = processed_df['Survived'].values
print(X)


# In[ ]:


X_test = processed_test_df.values


# In[ ]:


#x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)


# In[ ]:


clf_dt = tree.DecisionTreeClassifier(max_depth=10)


# In[ ]:


clf_dt.fit(X, Y)
Y_test = clf_dt.predict(X_test)
clf_dt.score(X_test, Y_test)


# In[ ]:


submission = pd.DataFrame({'PassengerId': processed_test_df['PassengerId'], 'Survived': Y_test})
submission.to_csv('clf_titanic.csv', index=False)


# In[ ]:


#clf_rf = ske.RandomForestClassifier(n_estimators=50)
#test_classifier(clf_rf)


# In[ ]:


#clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
#test_classifier(clf_gb)


# In[ ]:


#eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
#test_classifier(eclf)


# In[ ]:


#def custom_model(X, Y) :
#    layers = skflow.ops.dnn(X, [20, 40, 20], tf.tanh)
#    return skflow.models.logistic_regression(layers, Y)


# In[ ]:


#tf_clf_c = skflow.TensorFlowEstimator(model_fn=custom_model, n_classes=2, batch_size=256, steps=1000, learning_rate=0.05)
#tf_clf_c.fit(x_train, y_train)
#metrics.accuracy_score(y_test, tf_clf_c.predict(x_test))

