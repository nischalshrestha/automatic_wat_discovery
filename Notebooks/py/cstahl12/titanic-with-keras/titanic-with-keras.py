#!/usr/bin/env python
# coding: utf-8

# ## Background
# This kernel is intended to use Keras on the classic Titanic survivors dataset.  It is assuming that you are familiar with the titanic survivors data and skips most of the very necessary EDA. <br />
# Specifically I want to see if some of the SibSp and Parch feature engineering can be avoided by using a deep learning architecture and still get a decent enough score.

# ## Load environment

# In[ ]:


from __future__ import print_function
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Activation, Dropout


# In[ ]:


raw_train = pd.read_csv('../input/train.csv', index_col=0)
raw_train['is_test'] = 0
raw_test = pd.read_csv('../input/test.csv', index_col=0)
raw_test['is_test'] = 1


# In[ ]:


all_data = pd.concat((raw_train, raw_test), axis=0)


# ## Functions to preprocess the data

# In[ ]:


def get_title_last_name(name):
    full_name = name.str.split(', ', n=0, expand=True)
    last_name = full_name[0]
    titles = full_name[1].str.split('.', n=0, expand=True)
    titles = titles[0]
    return(titles)

def get_titles_from_names(df):
    df['Title'] = get_title_last_name(df['Name'])
    df = df.drop(['Name'], axis=1)
    return(df)

def get_dummy_cats(df):
    return(pd.get_dummies(df, columns=['Title', 'Pclass', 'Sex', 'Embarked',
                                       'Cabin', 'Cabin_letter']))

def get_cabin_letter(df):    
    df['Cabin'].fillna('Z', inplace=True)
    df['Cabin_letter'] = df['Cabin'].str[0]    
    return(df)

def process_data(df):
    # preprocess titles, cabin, embarked
    df = get_titles_from_names(df)    
    df['Embarked'].fillna('S', inplace=True)
    df = get_cabin_letter(df)
    
    # drop remaining features
    df = df.drop(['Ticket', 'Fare'], axis=1)
    
    # create dummies for categorial features
    df = get_dummy_cats(df)
    
    return(df)

proc_data = process_data(all_data)
proc_train = proc_data[proc_data['is_test'] == 0]
proc_test = proc_data[proc_data['is_test'] == 1]


# In[ ]:


proc_data.head()


# ## Build Network to predict missing ages

# In[ ]:


for_age_train = proc_data.drop(['Survived', 'is_test'], axis=1).dropna(axis=0)
X_train_age = for_age_train.drop('Age', axis=1)
y_train_age = for_age_train['Age']


# In[ ]:


# create model
tmodel = Sequential()
tmodel.add(Dense(input_dim=X_train_age.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
tmodel.add(Activation('relu'))

for i in range(0, 8):
    tmodel.add(Dense(units=64, kernel_initializer='normal',
                     bias_initializer='zeros'))
    tmodel.add(Activation('relu'))
    tmodel.add(Dropout(.25))

tmodel.add(Dense(units=1))
tmodel.add(Activation('linear'))

tmodel.compile(loss='mean_squared_error', optimizer='rmsprop')


# In[ ]:


tmodel.fit(X_train_age.values, y_train_age.values, epochs=600, verbose=2)


# In[ ]:


train_data = proc_train
train_data.loc[train_data['Age'].isnull()]


# In[ ]:


to_pred = train_data.loc[train_data['Age'].isnull()].drop(
          ['Age', 'Survived', 'is_test'], axis=1)
p = tmodel.predict(to_pred.values)
train_data['Age'].loc[train_data['Age'].isnull()] = p


# In[ ]:


test_data = proc_test
to_pred = test_data.loc[test_data['Age'].isnull()].drop(
          ['Age', 'Survived', 'is_test'], axis=1)
p = tmodel.predict(to_pred.values)
test_data['Age'].loc[test_data['Age'].isnull()] = p


# In[ ]:


train_data.loc[train_data['Age'].isnull()]


# In[ ]:


y = pd.get_dummies(train_data['Survived'])
y.head()


# In[ ]:


X = train_data.drop(['Survived', 'is_test'], axis=1)


# In[ ]:


# create model
model = Sequential()
model.add(Dense(input_dim=X.shape[1], units=128,
                 kernel_initializer='normal', bias_initializer='zeros'))
model.add(Activation('relu'))

for i in range(0, 15):
    model.add(Dense(units=128, kernel_initializer='normal',
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
    model.add(Dropout(.40))

model.add(Dense(units=2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X.values, y.values, epochs=500, verbose=2)


# In[ ]:


test_data.columns


# In[ ]:


p_survived = model.predict_classes(test_data.drop(['Survived', 'is_test'], axis=1).values)


# In[ ]:


submission = pd.DataFrame()
submission['PassengerId'] = test_data.index
submission['Survived'] = p_survived


# In[ ]:


submission.shape


# In[ ]:


submission.to_csv('titanic_keras_cs.csv', index=False)

