#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import tensorflow as tf

import os
print(os.listdir("../input"))


# In[ ]:


Test_df = pd.read_csv("../input/test.csv")
Train_df = pd.read_csv('../input/train.csv')
print(Train_df.shape, Test_df.shape)


# In[ ]:


Train_df.head()


# In[ ]:


Train_df.info()
print('_'*40)
Test_df.info()


# In[ ]:


Train_df.describe()


# In[ ]:


good_feature_list=['Survived','Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']
train_df = Train_df[good_feature_list]
good_feature_list.remove('Survived')
test_df = Test_df[good_feature_list]
print(train_df.columns)
print(test_df.columns)


# In[ ]:


print(train_df['Survived'].sum())
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('----------------------------------------------------------------------------')
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('----------------------------------------------------------------------------')
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('----------------------------------------------------------------------------')
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('----------------------------------------------------------------------------')
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# In[ ]:


# for dataset in combine:
#     dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


print('count of NANs from a total of ', train_df.shape[0])
for col in test_df:
    print(col, train_df[col].isna().sum(), test_df[col].isna().sum())


# In[ ]:


def fillNAN_embarked(df):
    df['Embarked'] = df['Embarked'].fillna(df.Embarked.dropna().mode()[0])
    return df
train_df = fillNAN_embarked(train_df)
test_df = fillNAN_embarked(test_df)
print(train_df.Embarked.isna().sum())
print(test_df.Embarked.isna().sum())


# In[ ]:


def fillNAN_fare(df):
    df['Fare'] = df['Fare'].fillna(df.Fare.dropna().mean())
    return df
train_df = fillNAN_fare(train_df)
test_df = fillNAN_fare(test_df)
print(train_df.Fare.isna().sum())
print(test_df.Fare.isna().sum())


# In[ ]:


def fillNAN_age(df):
    df['Age'] = df['Age'].fillna(df.Age.dropna().median())
    return df
train_df = fillNAN_age(train_df)
test_df = fillNAN_age(test_df)
print(train_df.Age.isna().sum())
print(test_df.Age.isna().sum())


# In[ ]:


print('count of NANs from a total of ', train_df.shape[0])
for col in test_df:
    print(col, train_df[col].isna().sum(), test_df[col].isna().sum())


# In[ ]:


# for dataset in combine:
#     dataset['Age*Class'] = dataset.Age * dataset.Pclass

# train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[ ]:


train_df.head()


# In[ ]:


INPUT_COLUMNS = [
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key='Sex', vocabulary_list=['male','female'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key='Embarked', vocabulary_list=['C','Q','S'])),
    tf.feature_column.numeric_column('Pclass'),
    tf.feature_column.numeric_column('Age'),
    tf.feature_column.numeric_column('SibSp'),
    tf.feature_column.numeric_column('Parch'),
    tf.feature_column.numeric_column('Fare'),
]

def add_more_features(feats):
    # Nothing to add (yet!)
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)


# In[ ]:


def make_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df.drop('Survived', axis=1),
    y = df['Survived'],
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )


# In[ ]:


def make_prediction_input_fn(df, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = None,
    batch_size = 128,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000,
    num_threads = 1
  )


# In[ ]:


model = tf.estimator.LinearRegressor(
      feature_columns = feature_cols)

model.train(input_fn = make_input_fn(train_df, num_epochs = 10))
model.evaluate(input_fn=make_input_fn(train_df,1))


# In[ ]:


model = tf.estimator.DNNClassifier(feature_columns=feature_cols, hidden_units=[256,128,64])
model.train(input_fn = make_input_fn(train_df, num_epochs=50))


# In[ ]:


eval_result = model.evaluate(input_fn = make_input_fn(train_df,1))
eval_result


# In[ ]:


predictions = model.predict(input_fn=make_prediction_input_fn(test_df,1))


# In[ ]:


results = list(predictions)

def x(res,j):
    class_id = res[j]['class_ids'][0]
    probability = int(results[i]['probabilities'][class_id] *100)

    if int(class_id) == 0:
        return ('%s%% probalitity to %s' % (probability,'Not survive'))
    else:
        return ('%s%% probalitity to %s' % (probability,'Survive!'))

print ('Predictions for 10 first records on test(dataset):')

for i in range(0,10):    
    print (x(results,i))


# In[ ]:


# for p in results:
#     print(p.get('class_ids'))
result = [int(p.get('class_ids')[0]) for p in results]
result


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": Test_df["PassengerId"],
        "Survived": result
    })
submission


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": Test_df["PassengerId"],
        "Survived": result
    })
submission.to_csv('gender_submission.csv', index=False)


# In[ ]:


submissions = pd.read_csv('gender_submission.csv')
submissions.head(5)


# In[ ]:




