#!/usr/bin/env python
# coding: utf-8

# # 80% accuracy in a dead-simple, fair and non-overfitting model
# 
# ### If you like it, upvote it.  =]

# In[ ]:


# Default imports from Kaggle
import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

# Custom imports
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# Handy in order to display maps inside notebook
get_ipython().magic(u'matplotlib inline')


# In[ ]:


def show_correlation(df, feature_name, target_name, plot=True, plot_kind='bar'):
    ''' Shows the "correlation" (in a not necessarily statistic meaning) between a feature and its target. '''
    try:
        print('corr', df[feature_name].corr(df[target_name]))
    except:
        pass

    gb = df[[feature_name, target_name]].groupby(feature_name).agg('mean')
    print(gb)
    
    if plot:
        gb.plot(kind=plot_kind)


def onehot(df, column_name):
    ''' Transforms a particular categorical field in "onehot" so it can be used for the classification models. '''
    return pd.concat([df, pd.get_dummies(df[column_name], prefix='_OneHot{}'.format(column_name))], axis=1)


# In[ ]:


df = pd.read_csv('../input/train.csv')
df['_NumRelatives'] = df['Parch'] + df['SibSp']
df['_AgeBin'] = pd.cut(df['Age'], 8)

del df['Cabin']  # Just because it contains a lot of NaN/null values
df.dropna(inplace=True)

df = onehot(df, 'Sex')
df = onehot(df, 'Pclass')  # Despite being numeric this is a categorical feature (numeric value per se doesn't make sense here; after all values act more like labels) 

# Brief overview of our dataframe
print('*' * 10)
print(df.info())

# Elaborate about our dataframe so we can decide what to use as our model features
print('*' * 10)
show_correlation(df, 'Pclass', 'Survived')

print('*' * 10)
show_correlation(df, 'Sex', 'Survived')

print('*' * 10)
show_correlation(df, '_AgeBin', 'Survived')

print('*' * 10)
show_correlation(df, '_NumRelatives', 'Survived')


# In[ ]:


relevant_features = [
    '_OneHotPclass_1', '_OneHotPclass_2', '_OneHotPclass_3',
    '_OneHotSex_male', '_OneHotSex_female',
    'Age',
    '_NumRelatives'
]

X = df[relevant_features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # We have too few data, so prioritize this for the training set.

inner_clf = LogisticRegression(penalty='l1', solver='liblinear')
clf = make_pipeline(
    MinMaxScaler(),  # Scale our features to be within 0..1
    SelectFromModel(inner_clf),  # Remove unnecessary features (mitigating risk of overfitting)
    inner_clf,  # Apply the classifier per se
)
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X_test, y_test, cv=10)
print('Accuracy score: ~{:.2f}%'.format(np.mean(scores) * 100))


# In[ ]:




