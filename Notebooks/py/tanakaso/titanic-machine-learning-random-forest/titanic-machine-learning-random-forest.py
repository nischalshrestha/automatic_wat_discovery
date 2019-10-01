#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


DATA_DIR = '../input/'
train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')


# In[ ]:


train.head(2)


# In[ ]:


train.info()


# 欠損値の確認

# In[ ]:


print ('===== train =====')
for column in train.columns:
    tab = '\t\t: ' if len(column)<8 else '\t: '
    print (column + tab + str(train[column].isnull().sum()))

print ('\n===== test  =====')
for column in test.columns:
    tab = '\t\t: ' if len(column)<8 else '\t: '
    print (column + tab + str(test[column].isnull().sum()))


# 年齢を平均値で埋める

# In[ ]:


#train["Age"].fillna(train.Age.mean(), inplace=True)
#test["Age"].fillna(train.Age.mean(), inplace=True)


# 階級ごとの年齢の中央値

# In[ ]:


train[['Pclass', 'Age']].boxplot(by='Pclass')
plt.show()


# 年齢を階級ごとの中央値で埋める

# In[ ]:


pclass_median = train.groupby('Pclass')['Age'].median()
fill_train_age = pd.DataFrame(train[['Pclass', 'Age']])
fill_test_age = pd.DataFrame(test[['Pclass', 'Age']])
for pclass in [1, 2, 3]:
    fill_train_age.loc[fill_train_age['Pclass']==pclass, 'Age'] = pclass_median[pclass]
    fill_test_age.loc[fill_test_age['Pclass']==pclass, 'Age'] = pclass_median[pclass]
train["Age"].fillna(fill_train_age['Age'], inplace=True)
test["Age"].fillna(fill_test_age['Age'], inplace=True)


# 乗船港の欠損値を最大数の乗船港で埋める

# In[ ]:


embarked = train.Embarked.value_counts()
train["Embarked"].fillna(embarked.index.max(), inplace=True)


# 運賃の欠損値を、船室階級ごとの運賃の平均値で埋める

# In[ ]:


pclass_mean = train.groupby('Pclass')['Fare'].mean()
fill_fare = pd.DataFrame(test[['Pclass', 'Fare']])
for pclass in [1, 2, 3]:
    fill_fare.loc[fill_fare['Pclass']==pclass, 'Fare'] = pclass_mean[pclass]
test["Fare"].fillna(fill_fare['Fare'], inplace=True)


# 文字列型のデータを整数化する

# In[ ]:


train = pd.concat((train, pd.get_dummies(train['Sex'])),axis=1)
train = pd.concat((train, pd.get_dummies(train['Embarked'])),axis=1)


# In[ ]:


train['Family'] = train['SibSp'] + train['Parch'] + 1


# In[ ]:


train.head(1)


# In[ ]:


X_train = train[[
    'Pclass', 'Age', 'SibSp', 'Parch', 'Family', 
    'Fare', 'C', 'Q', 'S', 'female', 'male'
]]
Y_train = train['Survived']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
clf.score(X_train, Y_train)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# In[ ]:


X_train = np.array(X_train)
Y_train = np.array(Y_train)
skf = StratifiedKFold(n_splits=3)
for fold, (train_index, test_index) in enumerate(skf.split(X_train, Y_train)):
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ('K-Fold: %d, Accuracy: %f' % (fold, accuracy))


# In[ ]:


test = pd.concat((test, pd.get_dummies(test['Sex'])),axis=1)
test = pd.concat((test, pd.get_dummies(test['Embarked'])),axis=1)
test['Family'] = test['SibSp'] + test['Parch'] + 1

X_test = test[[
    'Pclass', 'Age', 'SibSp', 'Parch', 'Family', 
    'Fare', 'C', 'Q', 'S', 'female', 'male'
]]


# In[ ]:


clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, Y_train)
Y_prediction = clf.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': Y_prediction
})
submission.to_csv('./submission_2.csv', index=False)


# 特徴量の重要度を表示する

# In[ ]:


fti = clf.feature_importances_   
for i, feat in enumerate(X_test.columns):
    print('{0:20s} : {1:>.6f}'.format(feat, fti[i]))

