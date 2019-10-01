#!/usr/bin/env python
# coding: utf-8

# In[ ]:



get_ipython().magic(u'matplotlib inline')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_inp = pd.read_csv('../input/train.csv')
#train_inp[(train_inp['Cabin'].notnull()) & (train_inp['Survived'] == 0)]
#(train_inp[train_inp['Cabin'].isnull()].size - train_inp[(train_inp['Cabin'].isnull()) & (train_inp['Survived'] == 0)].size)/train_inp[train_inp['Cabin'].isnull()].size


# In[ ]:


test_inp = pd.read_csv('../input/test.csv')


# In[ ]:


# adjust gender to be binary values
train = train_inp
train['Sex_Bin'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[ ]:


#split fare values into bins of width $50.
#This will be used later to fill in age gaps, using bins as categories.

binsize = 50
bins = [x for x in range(int(train['Fare'].min()), int(train['Fare'].max()) + binsize, binsize)]
label = [x for x in range(1, len(bins))]
train['Fare_Cat'] = pd.cut(train['Fare'],bins,labels = label)

train['Cabin_isnan'] = 1
train['Cabin_isnan'] = np.where(train['Cabin'].notnull(), 0, train['Cabin_isnan'])
train.head(25)


# In[ ]:


def fillna_mult(col, df, mult, *dne_action):
#col is the column name in a Dataframe in which the user wants to fill in NANs
#df is the input Dataframe
#mult is a multiindexed Series, where the relevant columns in the original dataframe have been grouped
#and aggregated appropriately.
#the optional dne_action signifies what is to be done if the attributes in the row which has an NAN
#that the user is trying to fill do not lead to a valid entry in the grouped multiindexed Series.
#if left empty, or set with any value other than 'mc', all the entries of that tier will be averaged
#if dne_action is assigned as 'mc', the most common value of all the elements at that tier will be used

    dout = df.copy()
    origna_str = col+'_origna'
    fill_str = col+'_fill'
    dout[origna_str] = 0
    dout[fill_str] = df[col]
    mult_index_names = list(mult.index.names)
    print(len(mult_index_names))
    print(mult_index_names)
    for i in df[np.isnan(df[col])].index.tolist():
        dout[origna_str].loc[i] = 1
        dtemp = mult
        for j in range(len(mult_index_names)):
            if df[mult_index_names[j]].loc[i] in dtemp:
                dtemp = dtemp[df[mult_index_names[j]].loc[i]]
            elif dne_action == 'mc':
                dtemp = dtemp.value_counts().idxmax()
                break
            else:
                dtemp = dtemp.mean()
                break
        dout[fill_str].loc[i] = dtemp
        
    return dout
   


# In[ ]:


#Values are give for the embarkation points, ignoring, and thus perpetuating any NANs
train['Embarked_Num'] = train['Embarked'].map( { 'C': 1, 'Q': 2, 'S': 3} )


# In[ ]:


#The multiindexed Series is constructed using values which are not NANs in the newly formed Embarked_Num column
tg = train[train['Embarked_Num'] > 0].groupby(['Survived', 'Pclass', 'Sex_Bin'])['Embarked_Num'].agg(lambda x:x.value_counts().index[0]).astype(int)


# In[ ]:


#Use fillna_mult to fill in NaN entries in the numerated Embarked field
print(train.head(10).columns)
train_new = fillna_mult('Embarked_Num', train, tg)
print(train.head(10).columns)


# In[ ]:


#The multiindexed Series is constructed using values which are not NaNs in the Age column
tg = train[np.isfinite(train['Age'])].groupby(['Survived', 'Pclass', 'Sex_Bin','Embarked_Num_fill'])['Age'].agg('mean')
train = fillna_mult('Age', train, tg)

#Create a binned value for ages, with a bin size of 5, to use as a feature for filling in NaN fare values
binsize = 5
bins = [x for x in range(0, int(train['Age_fill'].max()) + binsize, binsize)]
label = [x for x in range(1, len(bins))]
train['Age_Cat'] = pd.cut(train['Age_fill'],bins,labels = label)

tg = train[np.isfinite(train['Fare'])].groupby(['Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_Cat'])['Fare'].agg('mean')
train = fillna_mult('Fare', train, tg)


# In[ ]:


#Use all of the functions used to fill in the training data set to the test data set to fill in NaNs
test = test_inp
test['Sex_Bin'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test['Embarked_Num'] = test['Embarked'].map( { 'C': 1, 'Q': 2, 'S': 3} )
tg = test[test['Embarked_Num'] > 0].groupby([ 'Pclass', 'Sex_Bin'])['Embarked_Num'].agg(lambda x:x.value_counts().index[0]).astype(int)


# In[ ]:


test = fillna_mult('Embarked_Num', test, tg)

tg = test[np.isfinite(test['Age'])].groupby(['Pclass', 'Sex_Bin','Embarked_Num_fill'])['Age'].agg('mean')
test = fillna_mult('Age', test, tg)


test['Cabin_isnan'] = 1
test['Cabin_isnan'] = np.where(test['Cabin'].notnull(), 0, test['Cabin_isnan'])


# In[ ]:


binsize = 5
bins = [x for x in range(0, int(test['Age_fill'].max()) + binsize, binsize)]
label = [x for x in range(1, len(bins))]
test['Age_Cat'] = pd.cut(test['Age_fill'],bins,labels = label)

tg = test[np.isfinite(test['Fare'])].groupby(['Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_Cat'])['Fare'].agg('mean')

test[['Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_Cat']].loc[test[np.isnan(test['Fare'])].index.tolist()[0]]
test = fillna_mult('Fare', test, tg, 'avg')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


from sklearn.model_selection import train_test_split

#split the training data into a train/test subset
X_train, X_test, y_train, y_test = train_test_split(train[['Cabin_isnan','Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_fill', 'SibSp', 'Parch', 'Fare_fill', 'Fare_origna', 'Age_origna', 'Embarked_Num_origna']].as_matrix(),
                                                   train['Survived'].as_matrix(),
                                                   test_size=0.05,
                                                    random_state=0)

#X = train[['Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_fill', 'SibSp', 'Parch', 'Fare_fill', 'Fare_origna', 'Age_origna', 'Embarked_Num_origna']].as_matrix()
#Y = train['Survived'].as_matrix()


# In[ ]:


#Create a classifier model, based on random forests and train it using the training portion of the data from train.csv

#clf = RandomForestClassifier(n_estimators=100, random_state=7)
clf = ExtraTreesClassifier(n_estimators=100, random_state=7)
#clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.5,
#                                 max_depth=5, random_state=7)
clf = clf.fit(X_train, y_train)


# In[ ]:


clf.predict(X_test)


# In[ ]:


#Validate this model using the testing portion of the train.csv data
clf.score(X_test, y_test)


# In[ ]:


clf.predict(test[['Cabin_isnan', 'Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_fill', 'SibSp', 'Parch', 'Fare_fill', 'Fare_origna', 'Age_origna', 'Embarked_Num_origna']].as_matrix())


# In[ ]:


#Determine the feature importances and plot the weights in a bar graph

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in np.squeeze(clf.estimators_)],
             axis=0)
indices = np.argsort(importances)[::-1]
feature_rank = [['Cabin_isnan', 'Pclass', 'Sex_Bin','Embarked_Num_fill', 'Age_fill', 'SibSp', 'Parch', 'Fare_fill', 'Fare_origna', 'Age_origna', 'Embarked_Num_origna'][x] for x in indices]
# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, feature_rank[f], importances[indices[f]]))

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_rank, rotation='vertical')
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_test, y_test, cv=3)
scores


# In[ ]:


test.head(25)


# In[ ]:





# In[ ]:





# In[ ]:




